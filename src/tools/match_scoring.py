"""Request-aware apartment result scoring.

This module computes per-row component match scores and an overall match score
using router filters from the current user query. Requested dimensions receive
higher weight in the overall score, while UI metadata lists only requested
component scores for display.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

REQUESTED_BOOST_DEFAULT: float = 2.0
BASE_WEIGHT: float = 1.0
OVERALL_SCORE_COLUMN = "overall_match_score"

# New component-score columns.
PRICE_SCORE_COLUMN = "price_match_score"
SAFETY_SCORE_COLUMN = "safety_match_score"
TRANSIT_SCORE_COLUMN = "transit_match_score"
PROXIMITY_SCORE_COLUMN = "proximity_match_score"
AMENITY_SCORE_COLUMN = "amenity_match_score"

DIMENSION_TO_COLUMN: dict[str, str] = {
    "price": PRICE_SCORE_COLUMN,
    "safety": SAFETY_SCORE_COLUMN,
    "transit": TRANSIT_SCORE_COLUMN,
    "proximity": PROXIMITY_SCORE_COLUMN,
    "amenity": AMENITY_SCORE_COLUMN,
}


def _clip_0_100(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(lower=0.0, upper=100.0)


def _distance_to_score(distances: pd.Series, max_miles: float) -> pd.Series:
    """Convert distance in miles to a 0..100 score (closer is better)."""
    d = pd.to_numeric(distances, errors="coerce")
    if max_miles <= 0:
        max_miles = 0.25
    score = (1.0 - (d / float(max_miles))) * 100.0
    return _clip_0_100(score)


def _best_distance(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Pick the best (minimum) distance among available columns."""
    available = [c for c in columns if c in df.columns]
    if not available:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="float64")
    return df[available].apply(pd.to_numeric, errors="coerce").min(axis=1)


def _requested_proximity_radius(filters: dict[str, Any]) -> float:
    proximities = filters.get("proximity") or []
    if not proximities:
        return 0.25
    miles: list[float] = []
    for p in proximities:
        try:
            v = float(p.get("max_distance_miles") or 0.25)
        except (TypeError, ValueError):
            v = 0.25
        miles.append(max(v, 0.05))
    return min(miles) if miles else 0.25


def extract_requested_dimensions(filters: dict[str, Any]) -> set[str]:
    """Map router filters to requested scoring dimensions."""
    requested: set[str] = set()
    if filters.get("max_rent") is not None:
        requested.add("price")
    if filters.get("safety_preference"):
        requested.add("safety")
    if filters.get("transit_preference"):
        requested.add("transit")
    if filters.get("amenities"):
        requested.add("amenity")
    proximities = filters.get("proximity") or []
    if proximities:
        requested.add("proximity")
    if any((p or {}).get("kind") == "transit" for p in proximities):
        requested.add("transit")
    return requested


def compute_component_scores(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    """Compute available component scores on a copy of `df`."""
    out = df.copy()

    # Price match score from budget fit.
    max_rent = filters.get("max_rent")
    if max_rent is not None and "rent" in out.columns:
        try:
            budget = float(max_rent)
        except (TypeError, ValueError):
            budget = 0.0
        if budget > 0:
            rent = pd.to_numeric(out["rent"], errors="coerce")
            # Price curve:
            # - <= 75% of budget => 100
            # - == budget => 80
            # - between 75%..100% => linear 100 -> 80
            # - above budget => linear drop from 80 toward 0 by 125% of budget
            ratio = rent / budget
            price = pd.Series(0.0, index=out.index, dtype="float64")

            at_or_below_75 = ratio <= 0.75
            between_75_and_100 = (ratio > 0.75) & (ratio <= 1.0)
            above_budget = ratio > 1.0

            price = price.where(~at_or_below_75, 100.0)
            price_mid = 100.0 - ((ratio - 0.75) / 0.25) * 20.0
            price = price.where(~between_75_and_100, price_mid)
            # Over-budget: 80 at budget+0, linearly to 0 at 125% budget.
            price_high = 80.0 - ((ratio - 1.0) / 0.25) * 80.0
            price = price.where(~above_budget, price_high)
            out[PRICE_SCORE_COLUMN] = _clip_0_100(price)

    # Safety match score from direct safety score or inverse crime score.
    if "safety_score" in out.columns:
        out[SAFETY_SCORE_COLUMN] = _clip_0_100(out["safety_score"])
    elif "crime_score" in out.columns:
        out[SAFETY_SCORE_COLUMN] = _clip_0_100(100.0 - pd.to_numeric(out["crime_score"], errors="coerce"))

    # Transit score prefers distance-derived match when distance columns exist.
    transit_distance = _best_distance(
        out,
        [
            "distance_to_subway_miles",
            "distance_to_bus_miles",
            "nearest_subway_distance",
            "nearest_bus_stop_distance",
        ],
    )
    transit_radius = _requested_proximity_radius(filters)
    if transit_distance.notna().any():
        out[TRANSIT_SCORE_COLUMN] = _distance_to_score(transit_distance, max_miles=max(transit_radius, 1.0))
    elif "transit_score" in out.columns:
        out[TRANSIT_SCORE_COLUMN] = _clip_0_100(out["transit_score"])

    # Amenity score comes from existing amenity aggregate when available.
    if "amenity_score" in out.columns:
        out[AMENITY_SCORE_COLUMN] = _clip_0_100(out["amenity_score"])

    # Proximity score uses best available distance signal.
    proximity_distance = _best_distance(
        out,
        [
            "distance_to_subway_miles",
            "distance_to_bus_miles",
            "nearest_subway_distance",
            "nearest_bus_stop_distance",
            "nearest_park_distance",
        ],
    )
    if proximity_distance.notna().any():
        proximity_radius = _requested_proximity_radius(filters)
        out[PROXIMITY_SCORE_COLUMN] = _distance_to_score(proximity_distance, max_miles=max(proximity_radius, 0.25))

    return out


def compute_overall_match_score(
    df: pd.DataFrame,
    requested_dimensions: set[str],
    requested_boost: float = REQUESTED_BOOST_DEFAULT,
) -> pd.DataFrame:
    """Compute weighted overall match score from available component scores."""
    out = df.copy()
    weighted_sum = pd.Series(0.0, index=out.index, dtype="float64")
    total_weight = pd.Series(0.0, index=out.index, dtype="float64")

    for dim, col in DIMENSION_TO_COLUMN.items():
        if col not in out.columns:
            continue
        score = pd.to_numeric(out[col], errors="coerce")
        weight = requested_boost if dim in requested_dimensions else BASE_WEIGHT
        present = score.notna()
        weighted_sum = weighted_sum + score.fillna(0.0) * weight
        total_weight = total_weight + present.astype("float64") * weight

    out[OVERALL_SCORE_COLUMN] = (weighted_sum / total_weight.where(total_weight > 0.0)).fillna(0.0)
    out[OVERALL_SCORE_COLUMN] = _clip_0_100(out[OVERALL_SCORE_COLUMN])
    return out


def score_results_for_request(
    df: pd.DataFrame,
    filters: dict[str, Any] | None,
    requested_boost: float = REQUESTED_BOOST_DEFAULT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply personalized scoring and return `(scored_df, metadata)`."""
    filters = filters or {}
    requested = extract_requested_dimensions(filters)
    scored = compute_component_scores(df, filters)
    scored = compute_overall_match_score(
        scored,
        requested_dimensions=requested,
        requested_boost=requested_boost,
    )
    if OVERALL_SCORE_COLUMN in scored.columns:
        scored = scored.sort_values(OVERALL_SCORE_COLUMN, ascending=False, na_position="last").reset_index(drop=True)

    visible_score_columns = [
        DIMENSION_TO_COLUMN[d]
        for d in ("price", "safety", "transit", "proximity", "amenity")
        if d in requested and DIMENSION_TO_COLUMN[d] in scored.columns
    ]
    if OVERALL_SCORE_COLUMN in scored.columns:
        visible_score_columns.append(OVERALL_SCORE_COLUMN)

    metadata = {
        "requested_dimensions": sorted(requested),
        "visible_score_columns": visible_score_columns,
        "overall_score_column": OVERALL_SCORE_COLUMN,
    }
    return scored, metadata

