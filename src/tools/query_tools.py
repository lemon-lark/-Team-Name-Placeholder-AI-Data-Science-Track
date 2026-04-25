"""Thin wrapper around DuckDB execution + dashboard helpers."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.database import get_table_row_count, run_query


def run_safe_sql(sql: str) -> pd.DataFrame:
    """Execute a (presumed safe) SQL string and return a DataFrame.

    Re-exported so app code can import directly from ``src.tools``.
    """
    return run_query(sql)


def get_listing_profile(apartment_id: int) -> dict[str, Any] | None:
    """Return a complete listing profile joined with neighborhood score context."""
    try:
        safe_id = int(apartment_id)
    except (TypeError, ValueError):
        return None
    sql = f"""
    SELECT
        a.apartment_id,
        a.neighborhood_id,
        a.address,
        a.rent,
        a.bedrooms,
        a.bathrooms,
        a.sqft,
        a.available_date,
        a.lat,
        a.lng,
        a.source,
        a.listing_url,
        a.nearest_subway_distance AS listing_nearest_subway_distance,
        a.nearest_bus_stop_distance AS listing_nearest_bus_distance,
        a.hpd_violation_count,
        n.name AS neighborhood,
        n.borough,
        n.city,
        n.state,
        n.nta_code,
        n.center_lat AS neighborhood_center_lat,
        n.center_lng AS neighborhood_center_lng,
        ns.crime_score,
        ns.safety_score,
        ns.transit_score,
        ns.subway_access_score,
        ns.bus_access_score,
        ns.green_space_score,
        ns.amenity_score,
        ns.public_service_score,
        ns.affordability_score,
        ns.availability_score,
        ns.housing_supply_score,
        ns.affordable_housing_signal,
        ns.renter_fit_score,
        ns.value_score,
        ns.safety_affordability_score,
        ns.park_count,
        ns.nearest_park_distance,
        ns.worship_count,
        ns.grocery_count,
        ns.listing_count,
        ns.median_listing_rent,
        ns.avg_rent AS neighborhood_avg_rent,
        ns.min_rent AS neighborhood_min_rent,
        ns.max_rent AS neighborhood_max_rent,
        ns.population,
        ns.median_income,
        ns.median_rent AS neighborhood_median_rent
    FROM apartments a
    LEFT JOIN neighborhoods n ON n.neighborhood_id = a.neighborhood_id
    LEFT JOIN neighborhood_stats ns ON ns.neighborhood_id = a.neighborhood_id
    WHERE a.apartment_id = {safe_id}
    LIMIT 1
    """
    try:
        df = run_query(sql)
    except Exception:
        return None
    if df.empty:
        return None
    row = df.iloc[0].to_dict()
    return {k: v for k, v in row.items() if pd.notna(v)}


def get_listing_profile_by_address(
    address: str, neighborhood: str | None = None, rent: float | None = None
) -> dict[str, Any] | None:
    """Return a complete listing profile using address fallback matching."""
    address = (address or "").strip()
    if not address:
        return None
    safe_address = address.replace("'", "''")
    safe_neighborhood = (neighborhood or "").strip().replace("'", "''")
    where = [f"a.address = '{safe_address}'"]
    if safe_neighborhood:
        where.append(f"n.name = '{safe_neighborhood}'")
    order_by = "a.apartment_id DESC"
    if rent is not None:
        try:
            rent_value = float(rent)
            order_by = f"ABS(COALESCE(a.rent, 0) - {rent_value}) ASC, a.apartment_id DESC"
        except (TypeError, ValueError):
            pass
    sql = f"""
    SELECT
        a.apartment_id,
        a.neighborhood_id,
        a.address,
        a.rent,
        a.bedrooms,
        a.bathrooms,
        a.sqft,
        a.available_date,
        a.lat,
        a.lng,
        a.source,
        a.listing_url,
        a.nearest_subway_distance AS listing_nearest_subway_distance,
        a.nearest_bus_stop_distance AS listing_nearest_bus_distance,
        a.hpd_violation_count,
        n.name AS neighborhood,
        n.borough,
        n.city,
        n.state,
        n.nta_code,
        n.center_lat AS neighborhood_center_lat,
        n.center_lng AS neighborhood_center_lng,
        ns.crime_score,
        ns.safety_score,
        ns.transit_score,
        ns.subway_access_score,
        ns.bus_access_score,
        ns.green_space_score,
        ns.amenity_score,
        ns.public_service_score,
        ns.affordability_score,
        ns.availability_score,
        ns.housing_supply_score,
        ns.affordable_housing_signal,
        ns.renter_fit_score,
        ns.value_score,
        ns.safety_affordability_score,
        ns.park_count,
        ns.nearest_park_distance,
        ns.worship_count,
        ns.grocery_count,
        ns.listing_count,
        ns.median_listing_rent,
        ns.avg_rent AS neighborhood_avg_rent,
        ns.min_rent AS neighborhood_min_rent,
        ns.max_rent AS neighborhood_max_rent,
        ns.population,
        ns.median_income,
        ns.median_rent AS neighborhood_median_rent
    FROM apartments a
    LEFT JOIN neighborhoods n ON n.neighborhood_id = a.neighborhood_id
    LEFT JOIN neighborhood_stats ns ON ns.neighborhood_id = a.neighborhood_id
    WHERE {" AND ".join(where)}
    ORDER BY {order_by}
    LIMIT 1
    """
    try:
        df = run_query(sql)
    except Exception:
        return None
    if df.empty:
        return None
    row = df.iloc[0].to_dict()
    return {k: v for k, v in row.items() if pd.notna(v)}


def table_preview(table: str, limit: int = 100) -> pd.DataFrame:
    """Return the first ``limit`` rows of ``table`` for the Explore Data tab."""
    safe_table = "".join(c for c in table if c.isalnum() or c == "_")
    if not safe_table:
        return pd.DataFrame()
    try:
        return run_query(f'SELECT * FROM "{safe_table}" LIMIT {int(limit)}')
    except Exception:
        return pd.DataFrame()


def get_header_metrics() -> dict[str, Any]:
    """Compute the header metric cards shown above the tabs."""
    metrics: dict[str, Any] = {
        "total_listings": 0,
        "avg_rent": None,
        "median_rent": None,
        "avg_safety": None,
        "avg_transit": None,
        "best_availability_neighborhood": None,
        "best_availability_score": None,
        "best_renter_fit_neighborhood": None,
        "best_renter_fit_score": None,
        "best_transit_neighborhood": None,
        "best_transit_score": None,
        "neighborhood_count": 0,
    }

    if get_table_row_count("apartments") > 0:
        try:
            df = run_query(
                "SELECT COUNT(*) AS c, "
                "AVG(rent) FILTER (WHERE rent IS NOT NULL) AS avg_rent, "
                "MEDIAN(rent) FILTER (WHERE rent IS NOT NULL) AS med_rent "
                "FROM apartments"
            )
            metrics["total_listings"] = int(df.iloc[0]["c"])
            metrics["avg_rent"] = float(df.iloc[0]["avg_rent"]) if df.iloc[0]["avg_rent"] is not None else None
            metrics["median_rent"] = float(df.iloc[0]["med_rent"]) if df.iloc[0]["med_rent"] is not None else None
        except Exception:
            pass

    if get_table_row_count("neighborhoods") > 0:
        try:
            df = run_query("SELECT COUNT(*) AS c FROM neighborhoods")
            metrics["neighborhood_count"] = int(df.iloc[0]["c"])
        except Exception:
            pass

    if get_table_row_count("neighborhood_stats") > 0:
        try:
            df = run_query(
                "SELECT AVG(safety_score) AS s, AVG(transit_score) AS t "
                "FROM neighborhood_stats"
            )
            metrics["avg_safety"] = float(df.iloc[0]["s"]) if df.iloc[0]["s"] is not None else None
            metrics["avg_transit"] = float(df.iloc[0]["t"]) if df.iloc[0]["t"] is not None else None
        except Exception:
            pass
        try:
            df = run_query(
                "SELECT n.name AS neighborhood, ns.availability_score "
                "FROM neighborhood_stats ns "
                "JOIN neighborhoods n ON n.neighborhood_id = ns.neighborhood_id "
                "ORDER BY ns.availability_score DESC NULLS LAST LIMIT 1"
            )
            if not df.empty:
                metrics["best_availability_neighborhood"] = str(df.iloc[0]["neighborhood"])
                metrics["best_availability_score"] = float(df.iloc[0]["availability_score"])
        except Exception:
            pass
        try:
            df = run_query(
                "SELECT n.name AS neighborhood, ns.renter_fit_score "
                "FROM neighborhood_stats ns "
                "JOIN neighborhoods n ON n.neighborhood_id = ns.neighborhood_id "
                "ORDER BY ns.renter_fit_score DESC NULLS LAST LIMIT 1"
            )
            if not df.empty:
                metrics["best_renter_fit_neighborhood"] = str(df.iloc[0]["neighborhood"])
                metrics["best_renter_fit_score"] = float(df.iloc[0]["renter_fit_score"])
        except Exception:
            pass
        try:
            df = run_query(
                "SELECT n.name AS neighborhood, ns.transit_score "
                "FROM neighborhood_stats ns "
                "JOIN neighborhoods n ON n.neighborhood_id = ns.neighborhood_id "
                "ORDER BY ns.transit_score DESC NULLS LAST LIMIT 1"
            )
            if not df.empty:
                metrics["best_transit_neighborhood"] = str(df.iloc[0]["neighborhood"])
                metrics["best_transit_score"] = float(df.iloc[0]["transit_score"])
        except Exception:
            pass

    return metrics
