"""Canonical schema for AI-partments.

This module is the single source of truth for table names, columns, and
column types. The agents, the SQL safety validator, and the database
initializer all read from here so the app never falls out of sync.
"""
from __future__ import annotations

from typing import Iterable


# --- Table definitions --------------------------------------------------------
# Each entry: { column_name: duckdb_type }. Order is preserved for CREATE TABLE.

TABLES: dict[str, dict[str, str]] = {
    "neighborhoods": {
        "neighborhood_id": "INTEGER",
        "nta_code": "TEXT",
        "name": "TEXT",
        "borough": "TEXT",
        "city": "TEXT",
        "state": "TEXT",
        "center_lat": "DOUBLE",
        "center_lng": "DOUBLE",
    },
    "neighborhood_stats": {
        "neighborhood_id": "INTEGER",
        # Demographics
        "population": "INTEGER",
        "median_income": "INTEGER",
        "median_rent": "INTEGER",
        "percent_students": "DOUBLE",
        "percent_families": "DOUBLE",
        # Housing
        "hpd_building_count": "INTEGER",
        "hpd_unit_count": "INTEGER",
        "housing_supply_score": "DOUBLE",
        "affordable_housing_signal": "DOUBLE",
        # Transit
        "bus_stop_count": "INTEGER",
        "subway_station_count": "INTEGER",
        "nearest_bus_stop_distance": "DOUBLE",
        "nearest_subway_distance": "DOUBLE",
        "bus_access_score": "DOUBLE",
        "subway_access_score": "DOUBLE",
        "transit_score": "DOUBLE",
        # Parks
        "park_count": "INTEGER",
        "total_park_acres": "DOUBLE",
        "nearest_park_distance": "DOUBLE",
        "green_space_score": "DOUBLE",
        # Facilities
        "facility_count": "INTEGER",
        "school_count": "INTEGER",
        "healthcare_count": "INTEGER",
        "library_count": "INTEGER",
        "community_resource_count": "INTEGER",
        "public_service_score": "DOUBLE",
        # Worship / amenities
        "worship_count": "INTEGER",
        "grocery_count": "INTEGER",
        # Listings
        "listing_count": "INTEGER",
        "avg_rent": "DOUBLE",
        "median_listing_rent": "DOUBLE",
        "min_rent": "INTEGER",
        "max_rent": "INTEGER",
        "studio_count": "INTEGER",
        "one_bed_count": "INTEGER",
        "two_bed_count": "INTEGER",
        "avg_sqft": "DOUBLE",
        # Renter scores
        "crime_score": "DOUBLE",
        "safety_score": "DOUBLE",
        "affordability_score": "DOUBLE",
        "amenity_score": "DOUBLE",
        "availability_score": "DOUBLE",
        "renter_fit_score": "DOUBLE",
        "value_score": "DOUBLE",
        "safety_affordability_score": "DOUBLE",
    },
    "apartments": {
        "apartment_id": "INTEGER",
        "neighborhood_id": "INTEGER",
        "address": "TEXT",
        "rent": "INTEGER",
        "bedrooms": "INTEGER",
        "bathrooms": "DOUBLE",
        "sqft": "INTEGER",
        "available_date": "TEXT",
        "lat": "DOUBLE",
        "lng": "DOUBLE",
        "source": "TEXT",
        "listing_url": "TEXT",
        "nearest_subway_distance": "DOUBLE",
        "nearest_bus_stop_distance": "DOUBLE",
        "hpd_violation_count": "INTEGER",
    },
    "amenities": {
        "amenity_id": "INTEGER",
        "neighborhood_id": "INTEGER",
        "name": "TEXT",
        "type": "TEXT",
        "lat": "DOUBLE",
        "lng": "DOUBLE",
        "distance_miles": "DOUBLE",
        "source_file": "TEXT",
    },
    "crime_events": {
        "event_id": "INTEGER",
        "neighborhood_id": "INTEGER",
        "offense_type": "TEXT",
        "severity": "TEXT",
        "date": "TEXT",
        "lat": "DOUBLE",
        "lng": "DOUBLE",
    },
    "hpd_buildings": {
        "building_id": "TEXT",
        "neighborhood_id": "INTEGER",
        "borough": "TEXT",
        "address": "TEXT",
        "zip_code": "TEXT",
        "lat": "DOUBLE",
        "lng": "DOUBLE",
        "residential_units": "INTEGER",
        "program_type": "TEXT",
        "status": "TEXT",
        "source_file": "TEXT",
    },
    "facilities": {
        "facility_id": "TEXT",
        "neighborhood_id": "INTEGER",
        "name": "TEXT",
        "facility_type": "TEXT",
        "category": "TEXT",
        "borough": "TEXT",
        "address": "TEXT",
        "lat": "DOUBLE",
        "lng": "DOUBLE",
        "agency": "TEXT",
        "source_file": "TEXT",
    },
    "parks": {
        "park_id": "TEXT",
        "neighborhood_id": "INTEGER",
        "name": "TEXT",
        "borough": "TEXT",
        "address": "TEXT",
        "lat": "DOUBLE",
        "lng": "DOUBLE",
        "acres": "DOUBLE",
        "park_type": "TEXT",
        "source_file": "TEXT",
    },
    "transit_bus_stops": {
        "stop_id": "TEXT",
        "stop_name": "TEXT",
        "lat": "DOUBLE",
        "lng": "DOUBLE",
        "borough": "TEXT",
        "route_count": "INTEGER",
        "route_record_count": "INTEGER",
        "neighborhood_id": "INTEGER",
        "source_file": "TEXT",
    },
    "transit_subway_stations": {
        "station_id": "TEXT",
        "station_name": "TEXT",
        "line": "TEXT",
        "routes": "TEXT",
        "lat": "DOUBLE",
        "lng": "DOUBLE",
        "borough": "TEXT",
        "neighborhood_id": "INTEGER",
        "source_file": "TEXT",
    },
    "raw_file_registry": {
        "file_id": "TEXT",
        "file_name": "TEXT",
        "dataset_type": "TEXT",
        "row_count": "INTEGER",
        "column_count": "INTEGER",
        "ingested_at": "TEXT",
        "status": "TEXT",
        "warnings_json": "TEXT",
    },
    "processed_metrics": {
        "neighborhood_id": "INTEGER",
        "metric_name": "TEXT",
        "metric_value": "DOUBLE",
        "source": "TEXT",
        "computed_at": "TEXT",
    },
}


ALLOWED_TABLES: frozenset[str] = frozenset(TABLES.keys())


# Plain-English purpose strings used in LLM prompts so the model picks the
# right table without guessing.
TABLE_DESCRIPTIONS: dict[str, str] = {
    "neighborhoods": "One row per NYC/NJ neighborhood with name, borough, and center lat/lng.",
    "neighborhood_stats": "Pre-computed renter-focused metrics per neighborhood (rents, scores, counts).",
    "apartments": "Individual apartment listings with rent, bedrooms, sqft, and lat/lng.",
    "amenities": "Points of interest (parks, worship, grocery, schools, healthcare, transit) per neighborhood.",
    "crime_events": "Individual crime incidents with offense_type and severity tags.",
    "hpd_buildings": "Housing Preservation & Development buildings - housing-stock signal, NOT live vacancy.",
    "facilities": "Public facilities (schools, healthcare, libraries, community centers).",
    "parks": "Park properties with acres and location.",
    "transit_bus_stops": "Deduplicated MTA bus stops with route counts.",
    "transit_subway_stations": "MTA subway stations with line and routes.",
    "raw_file_registry": "Audit trail of ingested CSV/XLSX files.",
    "processed_metrics": "Generic key/value processed metric store.",
}


def get_create_statements() -> list[str]:
    """Build CREATE TABLE IF NOT EXISTS statements for every canonical table."""
    statements: list[str] = []
    for table, columns in TABLES.items():
        cols_sql = ",\n  ".join(f'"{c}" {t}' for c, t in columns.items())
        statements.append(f'CREATE TABLE IF NOT EXISTS "{table}" (\n  {cols_sql}\n);')
    return statements


def get_schema_summary() -> str:
    """Compact schema description used in LLM prompts."""
    lines: list[str] = []
    for table, columns in TABLES.items():
        desc = TABLE_DESCRIPTIONS.get(table, "")
        col_list = ", ".join(f"{c} {t}" for c, t in columns.items())
        lines.append(f"TABLE {table} -- {desc}\n  COLUMNS: {col_list}")
    return "\n\n".join(lines)


def get_columns(table: str) -> list[str]:
    """Return the column names for a table (empty list if unknown)."""
    return list(TABLES.get(table, {}).keys())


def is_known_table(name: str) -> bool:
    return name in ALLOWED_TABLES


def known_tables() -> Iterable[str]:
    return TABLES.keys()
