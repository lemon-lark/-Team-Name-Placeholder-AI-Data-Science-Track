"""Metric engineering: derived neighborhood scores from cleaned tables."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.ingestion.geospatial import (
    count_points_within_radius,
    nearest_point_distance,
)


# --- Normalization ------------------------------------------------------------


def normalize_series_to_0_100(
    series: pd.Series,
    higher_is_better: bool = True,
) -> pd.Series:
    """Min-max scale a Series into [0, 100]. Constant series -> 50."""
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series(50.0, index=series.index)
    lo, hi = valid.min(), valid.max()
    if lo == hi:
        return pd.Series(50.0, index=series.index)
    scaled = (s - lo) / (hi - lo) * 100.0
    if not higher_is_better:
        scaled = 100.0 - scaled
    return scaled.fillna(50.0).clip(lower=0.0, upper=100.0)


# --- Per-neighborhood building blocks ----------------------------------------


def aggregate_apartments(apartments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate apartment listings to per-neighborhood metrics."""
    if apartments is None or apartments.empty:
        return pd.DataFrame()
    df = apartments.copy()
    df["rent"] = pd.to_numeric(df["rent"], errors="coerce")
    df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce")
    df["sqft"] = pd.to_numeric(df["sqft"], errors="coerce")
    grouped = df.groupby("neighborhood_id", dropna=True)
    rows: list[dict] = []
    for nb_id, group in grouped:
        if pd.isna(nb_id):
            continue
        rents = group["rent"].dropna()
        rows.append(
            {
                "neighborhood_id": int(nb_id),
                "listing_count": int(len(group)),
                "avg_rent": float(rents.mean()) if not rents.empty else None,
                "median_listing_rent": float(rents.median()) if not rents.empty else None,
                "min_rent": int(rents.min()) if not rents.empty else None,
                "max_rent": int(rents.max()) if not rents.empty else None,
                "studio_count": int((group["bedrooms"] == 0).sum()),
                "one_bed_count": int((group["bedrooms"] == 1).sum()),
                "two_bed_count": int((group["bedrooms"] == 2).sum()),
                "avg_sqft": float(group["sqft"].mean()) if group["sqft"].notna().any() else None,
            }
        )
    return pd.DataFrame(rows)


def aggregate_crime(
    crimes: pd.DataFrame,
    populations: dict[int, int],
) -> pd.DataFrame:
    """Aggregate crime events to per-neighborhood counts and rates."""
    if crimes is None or crimes.empty:
        return pd.DataFrame()
    grouped = crimes.groupby("neighborhood_id", dropna=True)
    rows: list[dict] = []
    for nb_id, group in grouped:
        if pd.isna(nb_id):
            continue
        violent = int((group.get("severity", "") == "violent").sum())
        property_ct = int((group.get("severity", "") == "property").sum())
        total = int(len(group))
        pop = populations.get(int(nb_id), 0) or 0
        rows.append(
            {
                "neighborhood_id": int(nb_id),
                "crime_event_count": total,
                "violent_crime_count": violent,
                "property_crime_count": property_ct,
                "crime_per_1000_residents": (total / pop * 1000) if pop else None,
            }
        )
    return pd.DataFrame(rows)


def compute_transit_metrics(
    neighborhoods: pd.DataFrame,
    bus_stops: Optional[pd.DataFrame],
    subway_stations: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Per-neighborhood transit metrics computed from coordinate proximity."""
    rows: list[dict] = []
    for _, nb in neighborhoods.iterrows():
        center_lat = nb.get("center_lat")
        center_lng = nb.get("center_lng")
        nb_id = int(nb["neighborhood_id"])
        bus_count = int((bus_stops["neighborhood_id"] == nb_id).sum()) if bus_stops is not None and not bus_stops.empty else 0
        subway_count = int((subway_stations["neighborhood_id"] == nb_id).sum()) if subway_stations is not None and not subway_stations.empty else 0
        nearest_bus = nearest_point_distance(center_lat, center_lng, bus_stops) if bus_stops is not None and not bus_stops.empty else None
        nearest_subway = nearest_point_distance(center_lat, center_lng, subway_stations) if subway_stations is not None and not subway_stations.empty else None
        bus_within_025 = count_points_within_radius(center_lat, center_lng, bus_stops, 0.25) if bus_stops is not None and not bus_stops.empty else 0
        bus_within_050 = count_points_within_radius(center_lat, center_lng, bus_stops, 0.5) if bus_stops is not None and not bus_stops.empty else 0
        subway_within_050 = count_points_within_radius(center_lat, center_lng, subway_stations, 0.5) if subway_stations is not None and not subway_stations.empty else 0
        rows.append(
            {
                "neighborhood_id": nb_id,
                "bus_stop_count": bus_count,
                "subway_station_count": subway_count,
                "nearest_bus_stop_distance": nearest_bus,
                "nearest_subway_distance": nearest_subway,
                "bus_stop_count_within_0_25_miles": bus_within_025,
                "bus_stop_count_within_0_5_miles": bus_within_050,
                "subway_station_count_within_0_5_miles": subway_within_050,
            }
        )
    return pd.DataFrame(rows)


def subway_access_score(nearest_subway_distance: Optional[float]) -> float:
    if nearest_subway_distance is None or pd.isna(nearest_subway_distance):
        return 50.0
    d = float(nearest_subway_distance)
    if d <= 0.25:
        return 100.0
    if d <= 0.5:
        return 80.0
    if d <= 0.75:
        return 60.0
    if d <= 1.0:
        return 40.0
    return 20.0


def bus_access_score(
    bus_stop_count: int,
    nearest_bus_distance: Optional[float],
    counts_normalized: float = 50.0,
) -> float:
    if bus_stop_count is None and nearest_bus_distance is None:
        return 50.0
    base = counts_normalized
    if nearest_bus_distance is not None and not pd.isna(nearest_bus_distance):
        if nearest_bus_distance > 0.5:
            base = max(0.0, base - 20.0)
        elif nearest_bus_distance < 0.1:
            base = min(100.0, base + 10.0)
    return float(max(0.0, min(100.0, base)))


def transit_score(
    sub_score: Optional[float],
    bus_score: Optional[float],
    subway_data_present: bool,
    bus_data_present: bool,
) -> float:
    if not subway_data_present and not bus_data_present:
        return 50.0
    if not subway_data_present:
        return float(bus_score if bus_score is not None else 50.0)
    if not bus_data_present:
        return float(sub_score if sub_score is not None else 50.0)
    return float(0.6 * (sub_score or 50.0) + 0.4 * (bus_score or 50.0))


# --- Top-level metric pipeline -----------------------------------------------


def build_neighborhood_metrics(
    neighborhoods: pd.DataFrame,
    apartments: Optional[pd.DataFrame] = None,
    crimes: Optional[pd.DataFrame] = None,
    parks: Optional[pd.DataFrame] = None,
    facilities: Optional[pd.DataFrame] = None,
    amenities: Optional[pd.DataFrame] = None,
    hpd_buildings: Optional[pd.DataFrame] = None,
    bus_stops: Optional[pd.DataFrame] = None,
    subway_stations: Optional[pd.DataFrame] = None,
    population_by_id: Optional[dict[int, int]] = None,
    median_income_by_id: Optional[dict[int, int]] = None,
    median_rent_by_id: Optional[dict[int, int]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute all renter-focused metrics. Returns ``(stats_df, warnings)``."""
    warnings: list[str] = []
    population_by_id = population_by_id or {}
    median_income_by_id = median_income_by_id or {}
    median_rent_by_id = median_rent_by_id or {}

    base = neighborhoods[["neighborhood_id"]].copy()
    base["neighborhood_id"] = base["neighborhood_id"].astype(int)

    apt_metrics = aggregate_apartments(apartments) if apartments is not None and not apartments.empty else pd.DataFrame()
    crime_metrics = aggregate_crime(crimes, population_by_id) if crimes is not None and not crimes.empty else pd.DataFrame()

    if apartments is None or apartments.empty:
        warnings.append("No apartment listings provided; rent and listing metrics use neutral 50 fallback.")
    if crimes is None or crimes.empty:
        warnings.append("No crime data provided; safety_score set to neutral 50.")
    if parks is None or parks.empty:
        warnings.append("No parks data provided; green_space_score set to neutral 50.")
    if facilities is None or facilities.empty:
        warnings.append("No facilities data provided; public_service_score set to neutral 50.")
    if (bus_stops is None or bus_stops.empty) and (subway_stations is None or subway_stations.empty):
        warnings.append("No transit data provided; transit_score set to neutral 50.")
    if hpd_buildings is None or hpd_buildings.empty:
        warnings.append("No HPD building data provided; housing supply uses neutral defaults.")

    df = base.merge(apt_metrics, on="neighborhood_id", how="left") if not apt_metrics.empty else base.copy()
    df = df.merge(crime_metrics, on="neighborhood_id", how="left") if not crime_metrics.empty else df

    # Demographics
    df["population"] = df["neighborhood_id"].map(population_by_id)
    df["median_income"] = df["neighborhood_id"].map(median_income_by_id)
    df["median_rent"] = df["neighborhood_id"].map(median_rent_by_id)

    # Parks counts
    if parks is not None and not parks.empty:
        park_counts = parks.groupby("neighborhood_id").agg(
            park_count=("park_id", "count"),
            total_park_acres=("acres", lambda s: float(pd.to_numeric(s, errors="coerce").sum())),
        ).reset_index()
        df = df.merge(park_counts, on="neighborhood_id", how="left")
    else:
        df["park_count"] = 0
        df["total_park_acres"] = 0.0

    # Park nearest distances
    nearest_park: list[Optional[float]] = []
    for _, nb in neighborhoods.iterrows():
        if parks is None or parks.empty:
            nearest_park.append(None)
        else:
            nearest_park.append(nearest_point_distance(nb["center_lat"], nb["center_lng"], parks))
    df["nearest_park_distance"] = nearest_park

    # Facilities
    if facilities is not None and not facilities.empty:
        fac_pivot = (
            facilities.assign(_one=1)
            .pivot_table(
                index="neighborhood_id",
                columns="facility_type",
                values="_one",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
        )
        for col in ("school", "healthcare", "library", "community_center"):
            if col not in fac_pivot.columns:
                fac_pivot[col] = 0
        fac_pivot["facility_count"] = (
            fac_pivot["school"]
            + fac_pivot["healthcare"]
            + fac_pivot["library"]
            + fac_pivot["community_center"]
        )
        fac_pivot = fac_pivot.rename(
            columns={
                "school": "school_count",
                "healthcare": "healthcare_count",
                "library": "library_count",
                "community_center": "community_resource_count",
            }
        )[
            [
                "neighborhood_id",
                "school_count",
                "healthcare_count",
                "library_count",
                "community_resource_count",
                "facility_count",
            ]
        ]
        df = df.merge(fac_pivot, on="neighborhood_id", how="left")
    else:
        for col in ("facility_count", "school_count", "healthcare_count", "library_count", "community_resource_count"):
            df[col] = 0

    # HPD
    if hpd_buildings is not None and not hpd_buildings.empty:
        hpd_agg = (
            hpd_buildings.groupby("neighborhood_id")
            .agg(
                hpd_building_count=("building_id", "count"),
                hpd_unit_count=("residential_units", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
            )
            .reset_index()
        )
        df = df.merge(hpd_agg, on="neighborhood_id", how="left")
    else:
        df["hpd_building_count"] = 0
        df["hpd_unit_count"] = 0

    # Amenities (worship + grocery counts)
    if amenities is not None and not amenities.empty:
        am = amenities.copy()
        am["type"] = am["type"].fillna("").str.lower()
        worship_mask = am["type"].isin(["mosque", "church", "synagogue", "temple", "worship"])
        worship_counts = am[worship_mask].groupby("neighborhood_id").size().rename("worship_count").reset_index()
        grocery_counts = am[am["type"] == "grocery"].groupby("neighborhood_id").size().rename("grocery_count").reset_index()
        df = df.merge(worship_counts, on="neighborhood_id", how="left").merge(grocery_counts, on="neighborhood_id", how="left")
    else:
        df["worship_count"] = 0
        df["grocery_count"] = 0

    # Transit
    transit_metrics_df = compute_transit_metrics(neighborhoods, bus_stops, subway_stations)
    df = df.merge(transit_metrics_df, on="neighborhood_id", how="left")

    # Fill missing counts with zeros so scoring stays well-defined.
    fill_zero = [
        "park_count",
        "total_park_acres",
        "facility_count",
        "school_count",
        "healthcare_count",
        "library_count",
        "community_resource_count",
        "hpd_building_count",
        "hpd_unit_count",
        "worship_count",
        "grocery_count",
        "bus_stop_count",
        "subway_station_count",
        "bus_stop_count_within_0_25_miles",
        "bus_stop_count_within_0_5_miles",
        "subway_station_count_within_0_5_miles",
        "listing_count",
        "studio_count",
        "one_bed_count",
        "two_bed_count",
        "crime_event_count",
        "violent_crime_count",
        "property_crime_count",
    ]
    for col in fill_zero:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    # --- Scores ---------------------------------------------------------------

    # Crime / safety
    if crimes is not None and not crimes.empty:
        df["crime_score"] = normalize_series_to_0_100(df["crime_event_count"], higher_is_better=True)
        df["safety_score"] = 100.0 - df["crime_score"]
    else:
        df["crime_score"] = 50.0
        df["safety_score"] = 50.0

    # Transit
    df["subway_access_score"] = df["nearest_subway_distance"].map(subway_access_score)
    bus_norm = normalize_series_to_0_100(df["bus_stop_count"], higher_is_better=True)
    df["bus_access_score"] = [
        bus_access_score(int(c), d, n)
        for c, d, n in zip(df["bus_stop_count"], df["nearest_bus_stop_distance"], bus_norm)
    ]
    subway_present = subway_stations is not None and not subway_stations.empty
    bus_present = bus_stops is not None and not bus_stops.empty
    df["transit_score"] = [
        transit_score(s, b, subway_present, bus_present)
        for s, b in zip(df["subway_access_score"], df["bus_access_score"])
    ]

    # Green space
    if parks is not None and not parks.empty:
        park_count_norm = normalize_series_to_0_100(df["park_count"], higher_is_better=True)
        acres_norm = normalize_series_to_0_100(df["total_park_acres"], higher_is_better=True)
        proximity_bonus = df["nearest_park_distance"].fillna(2.0).map(
            lambda d: 10.0 if d <= 0.5 else 0.0
        )
        df["green_space_score"] = (0.5 * park_count_norm + 0.4 * acres_norm + proximity_bonus).clip(0, 100)
    else:
        df["green_space_score"] = 50.0

    # Public services
    if facilities is not None and not facilities.empty:
        weights = {
            "healthcare_count": 0.35,
            "library_count": 0.20,
            "school_count": 0.25,
            "community_resource_count": 0.20,
        }
        score = pd.Series(0.0, index=df.index)
        for col, w in weights.items():
            score = score + w * normalize_series_to_0_100(df[col], higher_is_better=True)
        df["public_service_score"] = score.clip(0, 100)
    else:
        df["public_service_score"] = 50.0

    # Housing supply / affordable signal
    if hpd_buildings is not None and not hpd_buildings.empty:
        if df["hpd_unit_count"].sum() > 0:
            df["housing_supply_score"] = normalize_series_to_0_100(df["hpd_unit_count"], higher_is_better=True)
        else:
            df["housing_supply_score"] = normalize_series_to_0_100(df["hpd_building_count"], higher_is_better=True)
        df["affordable_housing_signal"] = df["housing_supply_score"]
    else:
        df["housing_supply_score"] = 50.0
        df["affordable_housing_signal"] = 50.0

    # Amenity score with weight redistribution.
    weights = {
        "transit_score": 0.30,
        "green_space_score": 0.25,
        "public_service_score": 0.20,
    }
    if df["worship_count"].sum() > 0:
        weights["worship_count_norm"] = 0.15
    if df["grocery_count"].sum() > 0:
        weights["grocery_count_norm"] = 0.10
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    worship_norm = normalize_series_to_0_100(df["worship_count"], higher_is_better=True)
    grocery_norm = normalize_series_to_0_100(df["grocery_count"], higher_is_better=True)
    amenity_score_series = pd.Series(0.0, index=df.index)
    if "transit_score" in weights:
        amenity_score_series = amenity_score_series + weights["transit_score"] * df["transit_score"]
    if "green_space_score" in weights:
        amenity_score_series = amenity_score_series + weights["green_space_score"] * df["green_space_score"]
    if "public_service_score" in weights:
        amenity_score_series = amenity_score_series + weights["public_service_score"] * df["public_service_score"]
    if "worship_count_norm" in weights:
        amenity_score_series = amenity_score_series + weights["worship_count_norm"] * worship_norm
    if "grocery_count_norm" in weights:
        amenity_score_series = amenity_score_series + weights["grocery_count_norm"] * grocery_norm
    df["amenity_score"] = amenity_score_series.clip(0, 100)

    # Affordability (lower rent => higher score)
    rent_for_aff = df["median_listing_rent"].copy() if "median_listing_rent" in df.columns else pd.Series(np.nan, index=df.index)
    rent_for_aff = rent_for_aff.fillna(df["median_rent"]) if "median_rent" in df.columns else rent_for_aff
    if rent_for_aff.notna().any():
        df["affordability_score"] = normalize_series_to_0_100(rent_for_aff, higher_is_better=False)
    else:
        df["affordability_score"] = 50.0

    # Availability
    listing_norm = normalize_series_to_0_100(df["listing_count"], higher_is_better=True) if "listing_count" in df.columns else pd.Series(50.0, index=df.index)
    if (apartments is None or apartments.empty) or df["listing_count"].sum() == 0:
        df["availability_score"] = (
            0.7 * df["housing_supply_score"] + 0.3 * df["affordable_housing_signal"]
        ).clip(0, 100)
    else:
        df["availability_score"] = (
            0.6 * listing_norm + 0.25 * df["housing_supply_score"] + 0.15 * df["affordable_housing_signal"]
        ).clip(0, 100)

    # Composite scores
    df["renter_fit_score"] = (
        0.30 * df["affordability_score"]
        + 0.25 * df["safety_score"]
        + 0.25 * df["amenity_score"]
        + 0.20 * df["availability_score"]
    ).clip(0, 100)
    df["value_score"] = (
        0.50 * df["affordability_score"]
        + 0.30 * df["safety_score"]
        + 0.20 * df["amenity_score"]
    ).clip(0, 100)
    df["safety_affordability_score"] = (
        0.5 * df["affordability_score"] + 0.5 * df["safety_score"]
    ).clip(0, 100)

    # Round score columns for display.
    for col in (
        "crime_score",
        "safety_score",
        "transit_score",
        "subway_access_score",
        "bus_access_score",
        "green_space_score",
        "public_service_score",
        "housing_supply_score",
        "affordable_housing_signal",
        "amenity_score",
        "affordability_score",
        "availability_score",
        "renter_fit_score",
        "value_score",
        "safety_affordability_score",
    ):
        if col in df.columns:
            df[col] = df[col].round(1)

    return df, warnings


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")
