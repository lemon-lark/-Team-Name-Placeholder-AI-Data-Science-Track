"""Schema agent: pick the relevant tables/columns for a question.

This keeps the LLM honest. The SQL agent's prompt only mentions the tables
this agent returned, so the LLM cannot invent table names.
"""
from __future__ import annotations

from typing import Any

from src.schema import TABLE_DESCRIPTIONS, get_columns


def schema_agent(question: str, router_output: dict[str, Any]) -> dict[str, Any]:
    """Choose tables and columns relevant to the user's intent."""
    intent = router_output.get("intent", "general_analysis")
    filters = router_output.get("filters") or {}

    required_tables: list[str] = []
    notes: list[str] = []

    if intent == "rental_search":
        required_tables = ["apartments", "neighborhoods", "neighborhood_stats"]
        notes.append("Apartment search: join apartments to neighborhood_stats for scores.")

    elif intent == "neighborhood_comparison":
        required_tables = ["neighborhoods", "neighborhood_stats"]
        if filters.get("neighborhoods"):
            notes.append("Filter by neighborhood name in the WHERE clause.")

    elif intent == "availability_prediction":
        required_tables = ["neighborhoods", "neighborhood_stats", "hpd_buildings"]
        notes.append(
            "Use availability_score, listing_count, housing_supply_score, and HPD counts."
        )

    elif intent == "amenity_search":
        required_tables = ["neighborhoods", "neighborhood_stats", "amenities"]
        notes.append("Amenities table has type column for park/worship/grocery/etc.")
        if any(a in (filters.get("amenities") or []) for a in ("park",)):
            required_tables.append("parks")

    elif intent == "transit_analysis":
        required_tables = [
            "neighborhoods",
            "neighborhood_stats",
            "transit_subway_stations",
            "transit_bus_stops",
        ]
        notes.append("Use subway_access_score and bus_access_score for proximity.")

    elif intent == "housing_supply_analysis":
        required_tables = ["neighborhoods", "neighborhood_stats", "hpd_buildings"]
        notes.append(
            "HPD buildings are a housing-stock signal, not direct vacancy."
        )

    else:
        required_tables = ["neighborhoods", "neighborhood_stats"]

    # Always make sure neighborhood_stats is available for scoring.
    if "neighborhood_stats" not in required_tables:
        required_tables.append("neighborhood_stats")
    if "neighborhoods" not in required_tables:
        required_tables.append("neighborhoods")

    # Proximity filters can pull in tables the base intent did not require.
    proximities = filters.get("proximity") or []
    has_amenity_proximity = any(p.get("kind") == "amenity" for p in proximities)
    has_subway_proximity = any(
        p.get("kind") == "transit" and p.get("target") == "subway"
        for p in proximities
    )
    has_bus_proximity = any(
        p.get("kind") == "transit" and p.get("target") == "bus"
        for p in proximities
    )
    if has_amenity_proximity and "amenities" not in required_tables:
        required_tables.append("amenities")
        notes.append(
            "Use EXISTS subquery on amenities.distance_miles for amenity proximity."
        )
    if has_subway_proximity:
        notes.append(
            "Filter on apartments.nearest_subway_distance (precomputed miles) "
            "instead of joining transit_subway_stations."
        )
    if has_bus_proximity:
        notes.append(
            "Filter on apartments.nearest_bus_stop_distance (precomputed miles) "
            "instead of joining transit_bus_stops."
        )
    # Apartment-level transit proximity reads from the apartments table only.
    if (has_subway_proximity or has_bus_proximity) and "apartments" not in required_tables:
        required_tables.append("apartments")

    join_keys: list[str] = ["neighborhoods.neighborhood_id = neighborhood_stats.neighborhood_id"]
    if "apartments" in required_tables:
        join_keys.append("apartments.neighborhood_id = neighborhoods.neighborhood_id")
    if "amenities" in required_tables:
        join_keys.append("amenities.neighborhood_id = neighborhoods.neighborhood_id")
    if "transit_subway_stations" in required_tables:
        join_keys.append("transit_subway_stations.neighborhood_id = neighborhoods.neighborhood_id")
    if "transit_bus_stops" in required_tables:
        join_keys.append("transit_bus_stops.neighborhood_id = neighborhoods.neighborhood_id")
    if "hpd_buildings" in required_tables:
        join_keys.append("hpd_buildings.neighborhood_id = neighborhoods.neighborhood_id")
    if "parks" in required_tables:
        join_keys.append("parks.neighborhood_id = neighborhoods.neighborhood_id")

    relevant_columns: dict[str, list[str]] = {
        table: get_columns(table) for table in required_tables
    }

    descriptions = {
        table: TABLE_DESCRIPTIONS.get(table, "") for table in required_tables
    }

    return {
        "intent": intent,
        "required_tables": required_tables,
        "join_keys": join_keys,
        "relevant_columns": relevant_columns,
        "table_descriptions": descriptions,
        "notes": " ".join(notes) if notes else "Standard neighborhood analysis.",
    }
