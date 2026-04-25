"""Geospatial helpers: distance + neighborhood matching."""
from __future__ import annotations

import math
import re
from typing import Optional

import pandas as pd

try:
    from rapidfuzz import fuzz, process
except ImportError:  # pragma: no cover
    fuzz = None  # type: ignore
    process = None  # type: ignore


EARTH_RADIUS_MILES = 3958.7613


def haversine_distance_miles(
    lat1: float,
    lng1: float,
    lat2: float,
    lng2: float,
) -> float:
    """Great-circle distance in miles between two points."""
    if any(v is None for v in (lat1, lng1, lat2, lng2)):
        return float("inf")
    try:
        lat1_r = math.radians(float(lat1))
        lat2_r = math.radians(float(lat2))
        d_lat = lat2_r - lat1_r
        d_lng = math.radians(float(lng2) - float(lng1))
    except (TypeError, ValueError):
        return float("inf")
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(d_lng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_MILES * c


def normalize_neighborhood_name(name: object) -> str:
    """Lowercase + strip punctuation for fuzzy/exact matches."""
    if name is None:
        return ""
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def assign_neighborhood_by_name(
    row: pd.Series,
    neighborhoods_df: pd.DataFrame,
    name_field: str = "neighborhood",
) -> Optional[int]:
    """Match a row to a neighborhood by name (exact or fuzzy)."""
    if neighborhoods_df is None or neighborhoods_df.empty:
        return None
    if name_field not in row or pd.isna(row[name_field]):
        return None
    raw = normalize_neighborhood_name(row[name_field])
    if not raw:
        return None
    candidates = neighborhoods_df.copy()
    candidates["_norm"] = candidates["name"].map(normalize_neighborhood_name)
    exact = candidates[candidates["_norm"] == raw]
    if not exact.empty:
        return int(exact.iloc[0]["neighborhood_id"])
    if process is None:
        return None
    choices = candidates["_norm"].tolist()
    match = process.extractOne(raw, choices, scorer=fuzz.token_set_ratio)
    if match and match[1] >= 80:
        idx = choices.index(match[0])
        return int(candidates.iloc[idx]["neighborhood_id"])
    return None


def assign_nearest_neighborhood(
    row: pd.Series,
    neighborhoods_df: pd.DataFrame,
    lat_field: str = "lat",
    lng_field: str = "lng",
) -> Optional[int]:
    """Match a row to its nearest neighborhood center.

    NOTE: This is a centroid fallback; the long-term plan is to use
    NTA polygon shapefiles. See ``TODO`` below.
    """
    # TODO: replace with shapefile polygon lookup once a polygon source ships.
    if neighborhoods_df is None or neighborhoods_df.empty:
        return None
    lat = row.get(lat_field) if isinstance(row, pd.Series) else row.get(lat_field)
    lng = row.get(lng_field) if isinstance(row, pd.Series) else row.get(lng_field)
    if pd.isna(lat) or pd.isna(lng):
        return None
    best_id: Optional[int] = None
    best_dist = float("inf")
    for _, nb in neighborhoods_df.iterrows():
        d = haversine_distance_miles(lat, lng, nb["center_lat"], nb["center_lng"])
        if d < best_dist:
            best_dist = d
            best_id = int(nb["neighborhood_id"])
    return best_id


def match_neighborhood(
    row: pd.Series,
    neighborhoods_df: pd.DataFrame,
    nta_field: Optional[str] = "nta_code",
    name_field: Optional[str] = "neighborhood",
    lat_field: Optional[str] = "lat",
    lng_field: Optional[str] = "lng",
) -> Optional[int]:
    """4-tier match: NTA code -> name (exact) -> name (fuzzy) -> nearest center."""
    if neighborhoods_df is None or neighborhoods_df.empty:
        return None
    if nta_field and nta_field in row and not pd.isna(row[nta_field]):
        code = str(row[nta_field]).strip().upper()
        if code:
            match = neighborhoods_df[neighborhoods_df["nta_code"].astype(str).str.upper() == code]
            if not match.empty:
                return int(match.iloc[0]["neighborhood_id"])
    if name_field and name_field in row and not pd.isna(row[name_field]):
        nb_id = assign_neighborhood_by_name(row, neighborhoods_df, name_field=name_field)
        if nb_id is not None:
            return nb_id
    if lat_field and lng_field and lat_field in row and lng_field in row:
        return assign_nearest_neighborhood(row, neighborhoods_df, lat_field=lat_field, lng_field=lng_field)
    return None


def count_points_within_radius(
    center_lat: float,
    center_lng: float,
    points_df: pd.DataFrame,
    radius_miles: float,
    lat_field: str = "lat",
    lng_field: str = "lng",
) -> int:
    """Count rows in ``points_df`` whose lat/lng falls within ``radius_miles``."""
    if points_df is None or points_df.empty or center_lat is None or center_lng is None:
        return 0
    if lat_field not in points_df.columns or lng_field not in points_df.columns:
        return 0
    df = points_df.dropna(subset=[lat_field, lng_field])
    if df.empty:
        return 0
    count = 0
    for _, row in df.iterrows():
        d = haversine_distance_miles(center_lat, center_lng, row[lat_field], row[lng_field])
        if d <= radius_miles:
            count += 1
    return int(count)


def nearest_point_distance(
    center_lat: float,
    center_lng: float,
    points_df: pd.DataFrame,
    lat_field: str = "lat",
    lng_field: str = "lng",
) -> Optional[float]:
    """Return the minimum distance (miles) from a center to any point row."""
    if points_df is None or points_df.empty or center_lat is None or center_lng is None:
        return None
    if lat_field not in points_df.columns or lng_field not in points_df.columns:
        return None
    df = points_df.dropna(subset=[lat_field, lng_field])
    if df.empty:
        return None
    distances = [
        haversine_distance_miles(center_lat, center_lng, r[lat_field], r[lng_field])
        for _, r in df.iterrows()
    ]
    if not distances:
        return None
    return float(min(distances))
