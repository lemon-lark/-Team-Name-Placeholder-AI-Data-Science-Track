"""Flexible column detection and dataset-type recognition for raw NYC files."""
from __future__ import annotations

import re
from typing import Iterable, Optional

import pandas as pd

# --- Aliases per logical column ----------------------------------------------

COLUMN_ALIASES: dict[str, list[str]] = {
    "rent": ["rent", "price", "monthly_rent", "monthly_price", "rent_price", "asking_rent", "rent_amount"],
    "lat": ["lat", "latitude", "y", "the_geom_lat", "latitude_y", "y_coord", "geom_lat"],
    "lng": ["lng", "lon", "long", "longitude", "x", "the_geom_lon", "longitude_x", "x_coord", "geom_lon"],
    "neighborhood": [
        "neighborhood",
        "area",
        "community",
        "nta_name",
        "ntaname",
        "nta",
        "borough_neighborhood",
        "neighborhood_name",
        "geographic_area_neighborhood",
        "geographicarea",
    ],
    "nta_code": ["nta_code", "ntacode", "nta", "nta_2010", "nta2010", "nta_id"],
    "bedrooms": ["beds", "bedrooms", "br", "bed_count", "no_of_beds", "num_beds"],
    "bathrooms": ["baths", "bathrooms", "ba", "bath_count", "num_baths"],
    "sqft": ["sqft", "square_feet", "sf", "size_sqft", "area_sqft"],
    "address": ["address", "location", "full_address", "addr", "street_address"],
    "house_number": ["house_number", "houseno", "housenumber", "street_number"],
    "street_name": ["street_name", "street", "streetname"],
    "borough": ["borough", "boro", "borocode", "boro_name", "boroughname", "boro_nm"],
    "date": ["date", "created_at", "report_date", "available_date", "incident_date", "occurrence_date", "complaint_date"],
    "offense": ["offense_type", "offense", "complaint_type", "crime", "category", "law_cat_cd", "ofns_desc", "ky_cd"],
    "amenity_type": ["type", "category", "amenity", "place_type", "facility_type", "facsubgrp", "facgroup"],
    "facility_name": ["facility_name", "facname", "name", "fac_name"],
    "stop_id": ["stop_id", "stopid", "stop_code", "gtfs_stop_id", "stoppoint_id"],
    "stop_name": ["stop_name", "stopname", "name", "stop_label"],
    "station_id": ["station_id", "stationid", "complex_id", "gtfs_station_id"],
    "station_name": ["station_name", "stationname", "name", "stop_name"],
    "line": ["line", "subway_line", "trunk_line"],
    "routes": ["routes", "daytime_routes", "route_ids", "service_routes"],
    "route": ["route", "route_id", "route_short_name"],
    "park_name": ["park_name", "name", "signname", "parkname"],
    "acres": ["acres", "area_acres", "park_acres", "acreage"],
    "park_type": ["park_type", "typecategory", "category", "type"],
    "building_id": ["building_id", "buildingid", "bin", "bbl", "registration_id", "regid"],
    "residential_units": ["residential_units", "units", "legal_apartment_units", "unitsres", "total_units", "no_units"],
    "program_type": ["program_type", "program", "program_name", "managementprogram"],
    "status": ["status", "building_status", "active_status"],
    "zip_code": ["zip_code", "zip", "zipcode", "postcode", "postal_code"],
    "agency": ["agency", "operator", "owner", "managing_agency", "overagency"],
    "population": ["population", "pop", "total_population", "tot_pop", "pop1", "people"],
    "median_income": ["median_income", "med_income", "household_median_income", "median_household_income"],
    "median_rent": ["median_rent", "med_rent", "median_gross_rent"],
    "year": ["year", "data_year", "yr"],
    "listing_url": ["listing_url", "url", "link", "permalink"],
    "amenity_name": ["amenity_name", "name", "place_name", "label"],
    "severity": ["severity", "law_cat_cd", "category", "level"],
    "unit": ["unit", "apt", "apartment_number", "unit_number"],
    "hpd_violation_count": ["hpd_violation_count", "hpdviolationcount", "violation_count", "violations"],
}


# Filename-to-dataset_type rules (substring, case-insensitive).
FILENAME_RULES: list[tuple[str, str]] = [
    ("buildings_subject_to_hpd", "housing"),
    ("hpd_jurisdiction", "housing"),
    ("facilities_database", "facilities"),
    ("mta_bus_stops", "transit_bus"),
    ("bus_stops", "transit_bus"),
    ("mta_subway_stations", "transit_subway"),
    ("subway_stations", "transit_subway"),
    ("subway_entrances", "transit_subway"),
    ("population_by_neighborhood", "population"),
    ("population", "population"),
    ("parks_properties", "parks"),
    ("parks_property", "parks"),
    ("park_property", "parks"),
    ("acs_demo", "demographics"),
    ("acs_demograph", "demographics"),
    ("acs", "demographics"),
    ("demographics", "demographics"),
    ("nta_demograph", "demographics"),
    ("apartments", "apartments"),
    ("appartment", "apartments"),
    ("listings", "apartments"),
    ("rentals", "apartments"),
    ("zillow", "apartments"),
    ("crime", "crime"),
    ("nypd", "crime"),
    ("shootings", "crime"),
    ("complaint", "crime"),
    ("worship", "worship"),
    ("religious", "worship"),
    ("amenities", "amenities"),
    ("places", "amenities"),
]


SUPPORTED_DATASET_TYPES = {
    "apartments",
    "demographics",
    "crime",
    "transit",
    "transit_bus",
    "transit_subway",
    "amenities",
    "parks",
    "worship",
    "housing",
    "facilities",
    "population",
}


def normalize_column_name(col: str) -> str:
    """Lowercase, strip non-alphanumerics, collapse underscores.

    Examples
    --------
    >>> normalize_column_name("Rent Price ($)")
    'rent_price'
    """
    if col is None:
        return ""
    text = str(col).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _normalized_columns(df: pd.DataFrame) -> dict[str, str]:
    return {normalize_column_name(c): c for c in df.columns}


def detect_column(df: pd.DataFrame, possible_names: Iterable[str]) -> Optional[str]:
    """Return the original column name that matches one of ``possible_names``."""
    if df is None or df.empty:
        # Even empty dataframes can have headers we want to inspect.
        if df is None:
            return None
    norm_map = _normalized_columns(df)
    candidates = [normalize_column_name(n) for n in possible_names]
    for cand in candidates:
        if cand in norm_map:
            return norm_map[cand]
    # Substring fallback (e.g. 'monthly_rent' contains 'rent').
    for cand in candidates:
        for normed, original in norm_map.items():
            if cand and cand in normed:
                return original
    return None


def map_columns(df: pd.DataFrame, dataset_type: str) -> dict[str, Optional[str]]:
    """Return ``{logical_name: actual_column_or_None}`` for a dataset.

    The returned keys depend on the dataset_type so callers know exactly
    which logical fields they need to consume.
    """
    dt = (dataset_type or "").lower()
    keys: list[str]
    if dt == "apartments":
        keys = ["address", "street_name", "unit", "rent", "bedrooms", "bathrooms",
                "sqft", "lat", "lng", "neighborhood", "borough", "date",
                "listing_url", "hpd_violation_count"]
    elif dt == "demographics":
        keys = ["nta_code", "neighborhood", "borough", "population",
                "median_income", "median_rent", "year"]
    elif dt == "crime":
        keys = ["offense", "severity", "date", "lat", "lng", "neighborhood", "borough"]
    elif dt == "transit_bus" or dt == "transit":
        keys = ["stop_id", "stop_name", "lat", "lng", "borough", "route"]
    elif dt == "transit_subway":
        keys = ["station_id", "station_name", "line", "routes", "lat", "lng", "borough"]
    elif dt == "amenities" or dt == "worship":
        keys = ["amenity_name", "amenity_type", "lat", "lng", "neighborhood", "borough"]
    elif dt == "parks":
        keys = ["park_name", "park_type", "acres", "lat", "lng", "borough", "address"]
    elif dt == "housing":
        keys = ["building_id", "borough", "house_number", "street_name", "address",
                "zip_code", "lat", "lng", "residential_units", "program_type", "status"]
    elif dt == "facilities":
        keys = ["facility_name", "amenity_type", "borough", "address", "lat", "lng", "agency"]
    elif dt == "population":
        keys = ["nta_code", "neighborhood", "borough", "population", "year"]
    else:
        keys = list(COLUMN_ALIASES.keys())

    result: dict[str, Optional[str]] = {}
    for key in keys:
        aliases = COLUMN_ALIASES.get(key, [key])
        result[key] = detect_column(df, aliases)
    return result


def detect_dataset_type_from_filename(filename: str) -> Optional[str]:
    """Return the dataset_type implied by a filename (or None)."""
    if not filename:
        return None
    name = str(filename).lower()
    for needle, dt in FILENAME_RULES:
        if needle in name:
            return dt
    return None
