"""SQL generation agent.

Mock Demo Mode produces hand-crafted SQL templated from the router output so
the app always works. When an LLM provider is configured we ask the LLM for
SQL and run it through a strict pre-validator (single SELECT/WITH statement,
known tables, known columns). If the LLM SQL fails the pre-validator we
silently fall back to the mock SQL - the ``safety_agent`` still runs after
this for the security firewall.
"""
from __future__ import annotations

import re
from typing import Any, Optional

import sqlparse
from sqlparse.sql import Identifier, IdentifierList, Token, TokenList
from sqlparse.tokens import DML, Keyword

from src.agents.router_agent import BOROUGH_KEYWORDS, NEIGHBORHOOD_KEYWORDS
from src.llm_client import LLMClient
from src.schema import TABLES, get_schema_summary, is_known_table


SQL_SYSTEM_PROMPT = (
    "You are a careful data analyst for a renter intelligence dashboard. "
    "Your only job is to produce one valid DuckDB SQL SELECT query using "
    "the provided schema. Do not explain. Do not use markdown. Do not "
    "invent tables or columns. Only use SELECT/WITH. Always include "
    "LIMIT 50 unless the user asks otherwise."
)


# Earth radius in miles - used for haversine distance subqueries.
EARTH_RADIUS_MILES: float = 3958.8

# Maps a transit proximity target to (table, returned_column_name).
TRANSIT_TARGET_TABLES: dict[str, tuple[str, str]] = {
    "subway": ("transit_subway_stations", "distance_to_subway_miles"),
    "bus": ("transit_bus_stops", "distance_to_bus_miles"),
}


def _haversine_miles_sql(table: str, alias: str, ref_lat: str, ref_lng: str) -> str:
    """Return a scalar subquery computing the minimum haversine distance in miles.

    Used for transit proximity: for each apartment row in the outer query we
    find the nearest row in ``table`` (typically ``transit_subway_stations``
    or ``transit_bus_stops``) and return the haversine distance in miles.
    """
    return (
        "(SELECT MIN("
        f"2 * {EARTH_RADIUS_MILES} * ASIN(SQRT("
        f"POWER(SIN(RADIANS(({alias}.lat - {ref_lat}) / 2)), 2) + "
        f"COS(RADIANS({ref_lat})) * COS(RADIANS({alias}.lat)) * "
        f"POWER(SIN(RADIANS(({alias}.lng - {ref_lng}) / 2)), 2)"
        "))"
        f") FROM {table} {alias} "
        f"WHERE {alias}.lat IS NOT NULL AND {alias}.lng IS NOT NULL)"
    )


def _amenity_exists_sql(
    target: str, max_miles: float, neighborhood_alias: str = "n"
) -> str:
    """EXISTS subquery on the pre-populated ``amenities.distance_miles`` column."""
    safe_target = (target or "").replace("'", "''")
    return (
        "EXISTS (SELECT 1 FROM amenities am "
        f"WHERE am.neighborhood_id = {neighborhood_alias}.neighborhood_id "
        f"AND am.type = '{safe_target}' "
        f"AND am.distance_miles <= {float(max_miles):g})"
    )


def _amenity_haversine_miles_sql(target: str, apartment_alias: str) -> str:
    """Apartment-level distance to nearest amenity target using haversine."""
    safe_target = (target or "").replace("'", "''")
    return (
        "(SELECT MIN("
        f"2 * {EARTH_RADIUS_MILES} * ASIN(SQRT("
        f"POWER(SIN(RADIANS((am.lat - {apartment_alias}.lat) / 2)), 2) + "
        f"COS(RADIANS({apartment_alias}.lat)) * COS(RADIANS(am.lat)) * "
        f"POWER(SIN(RADIANS((am.lng - {apartment_alias}.lng) / 2)), 2)"
        "))"
        ") FROM amenities am "
        f"WHERE am.type = '{safe_target}' "
        "AND am.lat IS NOT NULL AND am.lng IS NOT NULL)"
    )


def _proximity_clauses(
    proximities: list[dict[str, Any]],
    apartment_alias: Optional[str] = "a",
    neighborhood_alias: str = "n",
) -> tuple[list[str], list[str]]:
    """Build ``(extra_select_columns, extra_where_clauses)`` from proximity filters.

    When ``apartment_alias`` is None the SQL has no apartments join, so transit
    proximity falls back to the precomputed ``ns.nearest_subway_distance`` /
    ``ns.nearest_bus_stop_distance`` columns instead of a haversine.
    """
    extra_selects: list[str] = []
    extra_wheres: list[str] = []
    for prox in proximities or []:
        target = prox.get("target")
        try:
            max_miles = float(prox.get("max_distance_miles") or 0.25)
        except (TypeError, ValueError):
            max_miles = 0.25
        kind = prox.get("kind") or "amenity"

        if kind == "transit" and target in TRANSIT_TARGET_TABLES:
            _table, col_name = TRANSIT_TARGET_TABLES[target]
            apt_col = (
                "nearest_subway_distance"
                if target == "subway"
                else "nearest_bus_stop_distance"
            )
            if apartment_alias is not None:
                # Apartments now ship precomputed transit distances (refreshed by
                # IngestionPipeline._refresh_apartment_transit_distances), so we
                # can read the column directly instead of running a haversine
                # subquery per row.
                extra_selects.append(
                    f"{apartment_alias}.{apt_col} AS {col_name}"
                )
                extra_wheres.append(
                    f"{apartment_alias}.{apt_col} IS NOT NULL "
                    f"AND {apartment_alias}.{apt_col} <= {max_miles:g}"
                )
            else:
                extra_wheres.append(f"ns.{apt_col} <= {max_miles:g}")
        else:
            safe_target = re.sub(r"[^a-z0-9_]+", "_", str(target).lower()).strip("_")
            if safe_target:
                if apartment_alias is not None:
                    distance_sql = _amenity_haversine_miles_sql(str(target), apartment_alias)
                    extra_selects.append(f"{distance_sql} AS distance_to_{safe_target}_miles")
                    extra_wheres.append(
                        f"{distance_sql} IS NOT NULL AND {distance_sql} <= {max_miles:g}"
                    )
                else:
                    escaped_target = str(target).replace("'", "''")
                    extra_selects.append(
                        "(SELECT MIN(am.distance_miles) FROM amenities am "
                        f"WHERE am.neighborhood_id = {neighborhood_alias}.neighborhood_id "
                        f"AND am.type = '{escaped_target}') "
                        f"AS distance_to_{safe_target}_miles"
                    )
                    extra_wheres.append(
                        _amenity_exists_sql(str(target), max_miles, neighborhood_alias)
                    )
            elif apartment_alias is None:
                extra_wheres.append(
                    _amenity_exists_sql(str(target), max_miles, neighborhood_alias)
                )
    return extra_selects, extra_wheres


# Few-shot examples included verbatim in the LLM prompt so the model can
# learn the desired style and constraints.
FEW_SHOT_EXAMPLES = [
    (
        "Find 1-bedroom apartments under $2,500 in low-crime neighborhoods near parks.",
        """SELECT n.name AS neighborhood, n.borough, a.address, a.rent, a.bedrooms,
       a.bathrooms, a.sqft, ns.crime_score, ns.safety_score, ns.transit_score,
       ns.green_space_score, ns.park_count, ns.availability_score,
       ns.renter_fit_score, a.lat, a.lng
FROM apartments a
JOIN neighborhoods n ON a.neighborhood_id = n.neighborhood_id
JOIN neighborhood_stats ns ON n.neighborhood_id = ns.neighborhood_id
WHERE a.bedrooms = 1
  AND a.rent <= 2500
  AND ns.crime_score <= 40
  AND ns.park_count >= 1
ORDER BY ns.renter_fit_score DESC, a.rent ASC
LIMIT 50;""",
    ),
    (
        "Rank neighborhoods by renter_fit_score.",
        """SELECT n.name AS neighborhood, n.borough, ns.median_listing_rent,
       ns.affordability_score, ns.safety_score, ns.transit_score,
       ns.green_space_score, ns.availability_score, ns.renter_fit_score
FROM neighborhoods n
JOIN neighborhood_stats ns ON n.neighborhood_id = ns.neighborhood_id
ORDER BY ns.renter_fit_score DESC
LIMIT 20;""",
    ),
    (
        "Apartments under $3000 within 0.5 mi of a mosque.",
        """SELECT n.name AS neighborhood, n.borough, a.address, a.rent, a.bedrooms,
       a.bathrooms, a.sqft, ns.renter_fit_score, a.lat, a.lng
FROM apartments a
JOIN neighborhoods n ON a.neighborhood_id = n.neighborhood_id
JOIN neighborhood_stats ns ON n.neighborhood_id = ns.neighborhood_id
WHERE a.rent <= 3000
  AND EXISTS (
    SELECT 1 FROM amenities am
    WHERE am.neighborhood_id = n.neighborhood_id
      AND am.type = 'mosque'
      AND am.distance_miles <= 0.5
  )
ORDER BY ns.renter_fit_score DESC, a.rent ASC
LIMIT 50;""",
    ),
    (
        "1-bedroom apartments near subway under $2800.",
        """SELECT n.name AS neighborhood, n.borough, a.address, a.rent, a.bedrooms,
       a.bathrooms, a.sqft, ns.renter_fit_score, a.lat, a.lng,
       a.nearest_subway_distance AS distance_to_subway_miles
FROM apartments a
JOIN neighborhoods n ON a.neighborhood_id = n.neighborhood_id
JOIN neighborhood_stats ns ON n.neighborhood_id = ns.neighborhood_id
WHERE a.bedrooms = 1
  AND a.rent <= 2800
  AND a.nearest_subway_distance IS NOT NULL
  AND a.nearest_subway_distance <= 0.25
ORDER BY ns.renter_fit_score DESC, a.rent ASC
LIMIT 50;""",
    ),
]


# --- Mock SQL builder ---------------------------------------------------------


def _mock_sql(question: str, router_output: dict[str, Any]) -> str:
    intent = router_output.get("intent", "general_analysis")
    filters = router_output.get("filters") or {}
    q_lower = question.lower()
    proximities = filters.get("proximity") or []
    has_rental_signal = (
        filters.get("max_rent") is not None
        or filters.get("exact_bedrooms") is not None
        or filters.get("min_bedrooms") is not None
    )
    has_transit_proximity = any(p.get("kind") == "transit" for p in proximities)
    has_location_scope = bool((filters.get("neighborhoods") or []) or (filters.get("boroughs") or []))

    if intent == "rental_search" or "apartment" in q_lower or "1-bedroom" in q_lower:
        return _mock_apartment_search(filters)

    # Proximity questions about price/bedrooms or transit imply per-listing
    # search, so route to the apartment builder (only place where the
    # haversine subquery is emitted).
    if proximities and (has_rental_signal or has_transit_proximity or has_location_scope):
        return _mock_apartment_search(filters)

    if intent == "neighborhood_comparison" and filters.get("neighborhoods"):
        return _mock_compare_neighborhoods(
            filters["neighborhoods"], q_lower, filters.get("proximity") or []
        )

    if "subway" in q_lower and ("afford" in q_lower or "rent" in q_lower):
        return _mock_subway_and_affordability()

    if "transit" in q_lower and "afford" in q_lower:
        return _mock_transit_affordability()

    if "availability" in q_lower and ("green" in q_lower or "park" in q_lower):
        return _mock_availability_green()

    if "hpd" in q_lower or "housing stock" in q_lower or "housing supply" in q_lower:
        return _mock_hpd_supply()

    if "rank" in q_lower or "renter_fit" in q_lower or "renter fit" in q_lower:
        return _mock_renter_fit_ranking()

    if intent == "transit_analysis":
        return _mock_transit_overview()

    if intent == "amenity_search":
        return _mock_amenity_overview(
            filters.get("amenities") or [], filters.get("proximity") or []
        )

    if intent == "availability_prediction":
        return _mock_availability_overview()

    if intent == "housing_supply_analysis":
        return _mock_hpd_supply()

    return _mock_renter_fit_ranking()


def _mock_apartment_search(filters: dict[str, Any]) -> str:
    where: list[str] = []
    bedrooms = filters.get("exact_bedrooms")
    if bedrooms is None:
        bedrooms = filters.get("min_bedrooms")
    if bedrooms is not None:
        where.append(f"a.bedrooms = {int(bedrooms)}")
    max_rent = filters.get("max_rent")
    if max_rent:
        where.append(f"a.rent <= {int(max_rent)}")
    boroughs = filters.get("boroughs") or []
    borough_names = {str(v) for v in BOROUGH_KEYWORDS.values()}
    known_neighborhood_names = {str(v) for v in NEIGHBORHOOD_KEYWORDS.values()}
    neighborhoods_raw = filters.get("neighborhoods") or []
    # LLMs sometimes place borough labels in neighborhoods (e.g. "Manhattan").
    # Keep only canonical neighborhood-name candidates for n.name filtering.
    neighborhoods = [
        n for n in neighborhoods_raw
        if str(n) not in borough_names and str(n) in known_neighborhood_names
    ]
    if neighborhoods:
        quoted = ", ".join(
            "'" + str(name).replace("'", "''") + "'" for name in neighborhoods
        )
        where.append(f"n.name IN ({quoted})")
    if boroughs:
        quoted_b = ", ".join(
            "'" + str(b).replace("'", "''") + "'" for b in boroughs
        )
        where.append(f"n.borough IN ({quoted_b})")
    if filters.get("safety_preference") == "low_crime":
        where.append("ns.crime_score <= 40")
    elif filters.get("safety_preference") == "high_safety":
        where.append("ns.safety_score >= 60")
    amenities = filters.get("amenities") or []
    proximities = filters.get("proximity") or []
    proximity_targets = {p.get("target") for p in proximities}
    # Skip the cheap presence checks for any amenity that already has a
    # proximity constraint - the EXISTS / haversine subquery is strictly
    # stronger.
    if "park" in amenities and "park" not in proximity_targets:
        where.append("ns.park_count >= 1")
    if (
        any(a in amenities for a in ("worship", "mosque", "church", "synagogue", "temple"))
        and not proximity_targets & {"worship", "mosque", "church", "synagogue", "temple"}
    ):
        where.append("ns.worship_count >= 1")
    if "subway" in amenities and "subway" not in proximity_targets:
        where.append("ns.subway_station_count >= 1")
    if "bus" in amenities and "bus" not in proximity_targets:
        where.append("ns.bus_stop_count >= 1")

    extra_selects, extra_wheres = _proximity_clauses(proximities, apartment_alias="a")
    where.extend(extra_wheres)

    select_extra = (
        ",\n       " + ",\n       ".join(extra_selects) if extra_selects else ""
    )
    where_clause = "\nWHERE " + "\n  AND ".join(where) if where else ""
    return (
        "SELECT n.name AS neighborhood, n.borough, a.address, a.rent, a.bedrooms,\n"
        "       a.bathrooms, a.sqft, a.available_date,\n"
        "       a.nearest_subway_distance, a.nearest_bus_stop_distance,\n"
        "       ns.crime_score, ns.safety_score, ns.transit_score,\n"
        "       ns.green_space_score, ns.park_count, ns.nearest_park_distance,\n"
        "       ns.availability_score,\n"
        "       ns.renter_fit_score, a.lat, a.lng"
        + select_extra
        + "\nFROM apartments a\n"
        "JOIN neighborhoods n ON a.neighborhood_id = n.neighborhood_id\n"
        "JOIN neighborhood_stats ns ON n.neighborhood_id = ns.neighborhood_id"
        + where_clause
        + "\nORDER BY ns.renter_fit_score DESC, a.rent ASC\nLIMIT 50;"
    )


def _mock_compare_neighborhoods(
    neighborhoods: list[str],
    q_lower: str,
    proximities: Optional[list[dict[str, Any]]] = None,
) -> str:
    quoted = ", ".join(f"'{name}'" for name in neighborhoods) or "'Astoria'"
    where: list[str] = [f"n.name IN ({quoted})"]
    _, extra_wheres = _proximity_clauses(proximities or [], apartment_alias=None)
    where.extend(extra_wheres)
    where_clause = "WHERE " + "\n  AND ".join(where)
    return (
        "SELECT n.name AS neighborhood, n.borough,\n"
        "       ns.median_listing_rent, ns.avg_rent,\n"
        "       ns.affordability_score, ns.safety_score, ns.transit_score,\n"
        "       ns.green_space_score, ns.park_count,\n"
        "       ns.housing_supply_score, ns.hpd_unit_count,\n"
        "       ns.availability_score, ns.renter_fit_score,\n"
        "       n.center_lat, n.center_lng\n"
        "FROM neighborhoods n\n"
        "JOIN neighborhood_stats ns ON n.neighborhood_id = ns.neighborhood_id\n"
        + where_clause
        + "\nORDER BY ns.renter_fit_score DESC\n"
        "LIMIT 50;"
    )


def _mock_subway_and_affordability() -> str:
    return (
        "SELECT n.name AS neighborhood, n.borough,\n"
        "       ns.median_listing_rent, ns.affordability_score,\n"
        "       ns.subway_access_score, ns.subway_station_count,\n"
        "       ns.nearest_subway_distance, ns.transit_score,\n"
        "       ns.renter_fit_score, n.center_lat, n.center_lng\n"
        "FROM neighborhoods n\n"
        "JOIN neighborhood_stats ns ON n.neighborhood_id = ns.neighborhood_id\n"
        "WHERE ns.subway_access_score >= 60\n"
        "ORDER BY (ns.subway_access_score + ns.affordability_score) DESC\n"
        "LIMIT 20;"
    )


def _mock_transit_affordability() -> str:
    return (
        "SELECT n.name AS neighborhood, n.borough,\n"
        "       ns.median_listing_rent, ns.affordability_score,\n"
        "       ns.transit_score, ns.subway_access_score, ns.bus_access_score,\n"
        "       ns.renter_fit_score, n.center_lat, n.center_lng\n"
        "FROM neighborhoods n\n"
        "JOIN neighborhood_stats ns ON n.neighborhood_id = ns.neighborhood_id\n"
        "ORDER BY (ns.transit_score + ns.affordability_score) DESC\n"
        "LIMIT 20;"
    )


def _mock_availability_green() -> str:
    return (
        "SELECT n.name AS neighborhood, n.borough,\n"
        "       ns.availability_score, ns.green_space_score,\n"
        "       ns.park_count, ns.total_park_acres, ns.listing_count,\n"
        "       ns.housing_supply_score, ns.renter_fit_score,\n"
        "       n.center_lat, n.center_lng\n"
        "FROM neighborhoods n\n"
        "JOIN neighborhood_stats ns ON n.neighborhood_id = ns.neighborhood_id\n"
        "WHERE ns.availability_score >= 50 AND ns.green_space_score >= 50\n"
        "ORDER BY (ns.availability_score + ns.green_space_score) DESC\n"
        "LIMIT 20;"
    )


def _mock_hpd_supply() -> str:
    return (
        "SELECT n.name AS neighborhood, n.borough,\n"
        "       ns.hpd_building_count, ns.hpd_unit_count,\n"
        "       ns.housing_supply_score, ns.affordable_housing_signal,\n"
        "       ns.median_listing_rent, ns.affordability_score,\n"
        "       ns.availability_score, ns.renter_fit_score,\n"
        "       n.center_lat, n.center_lng\n"
        "FROM neighborhoods n\n"
        "JOIN neighborhood_stats ns ON n.neighborhood_id = ns.neighborhood_id\n"
        "ORDER BY ns.hpd_unit_count DESC\n"
        "LIMIT 20;"
    )


def _mock_renter_fit_ranking() -> str:
    return (
        "SELECT n.name AS neighborhood, n.borough,\n"
        "       ns.median_listing_rent, ns.affordability_score,\n"
        "       ns.safety_score, ns.transit_score, ns.green_space_score,\n"
        "       ns.availability_score, ns.renter_fit_score,\n"
        "       n.center_lat, n.center_lng\n"
        "FROM neighborhoods n\n"
        "JOIN neighborhood_stats ns ON n.neighborhood_id = ns.neighborhood_id\n"
        "ORDER BY ns.renter_fit_score DESC\n"
        "LIMIT 20;"
    )


def _mock_transit_overview() -> str:
    return (
        "SELECT n.name AS neighborhood, n.borough,\n"
        "       ns.subway_station_count, ns.bus_stop_count,\n"
        "       ns.subway_access_score, ns.bus_access_score, ns.transit_score,\n"
        "       ns.nearest_subway_distance, ns.nearest_bus_stop_distance,\n"
        "       ns.median_listing_rent, ns.renter_fit_score,\n"
        "       n.center_lat, n.center_lng\n"
        "FROM neighborhoods n\n"
        "JOIN neighborhood_stats ns ON n.neighborhood_id = ns.neighborhood_id\n"
        "ORDER BY ns.transit_score DESC\n"
        "LIMIT 20;"
    )


def _mock_amenity_overview(
    amenities: list[str], proximities: Optional[list[dict[str, Any]]] = None
) -> str:
    where_parts: list[str] = []
    types_to_filter = [
        a for a in amenities
        if a in (
            "park", "mosque", "church", "synagogue", "temple", "worship", "grocery",
            "subway", "bus", "school", "healthcare", "library", "community_center",
        )
    ]
    if types_to_filter:
        quoted = ", ".join(f"'{t}'" for t in types_to_filter)
        where_parts.append(f"am.type IN ({quoted})")

    for prox in proximities or []:
        target = (prox.get("target") or "").replace("'", "''")
        try:
            max_miles = float(prox.get("max_distance_miles") or 0.25)
        except (TypeError, ValueError):
            max_miles = 0.25
        if not target:
            continue
        where_parts.append(
            f"(am.type = '{target}' AND am.distance_miles <= {max_miles:g})"
        )

    where_clause = ""
    if where_parts:
        where_clause = "WHERE " + " AND ".join(where_parts) + "\n"
    return (
        "SELECT n.name AS neighborhood, n.borough, am.name AS amenity_name,\n"
        "       am.type, am.lat, am.lng, am.distance_miles,\n"
        "       ns.amenity_score, ns.renter_fit_score\n"
        "FROM amenities am\n"
        "JOIN neighborhoods n ON am.neighborhood_id = n.neighborhood_id\n"
        "JOIN neighborhood_stats ns ON n.neighborhood_id = ns.neighborhood_id\n"
        + where_clause
        + "ORDER BY ns.amenity_score DESC, am.distance_miles ASC\nLIMIT 50;"
    )


def _mock_availability_overview() -> str:
    return (
        "SELECT n.name AS neighborhood, n.borough,\n"
        "       ns.availability_score, ns.listing_count,\n"
        "       ns.hpd_building_count, ns.hpd_unit_count,\n"
        "       ns.housing_supply_score, ns.affordable_housing_signal,\n"
        "       ns.median_listing_rent, ns.renter_fit_score,\n"
        "       n.center_lat, n.center_lng\n"
        "FROM neighborhoods n\n"
        "JOIN neighborhood_stats ns ON n.neighborhood_id = ns.neighborhood_id\n"
        "ORDER BY ns.availability_score DESC\n"
        "LIMIT 20;"
    )


# --- LLM-SQL pre-validator ---------------------------------------------------

# All known column names across every table - used to gate `alias.column`
# references in LLM-produced SQL. Computed once at import time.
_ALL_COLUMNS: frozenset[str] = frozenset(
    col for cols in TABLES.values() for col in cols
)

# Common keywords that can appear immediately after ``FROM <table>`` and which
# must NOT be treated as table aliases by the alias-extraction regex.
_NON_ALIAS_TOKENS: frozenset[str] = frozenset(
    {
        "as",
        "on",
        "where",
        "group",
        "order",
        "having",
        "limit",
        "offset",
        "join",
        "inner",
        "left",
        "right",
        "full",
        "outer",
        "cross",
        "using",
        "qualify",
        "window",
        "union",
        "intersect",
        "except",
    }
)

_FROM_OR_JOIN_RE = re.compile(
    r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)"
    r"(?:\s+(?:as\s+)?([a-z_][a-z0-9_]*))?",
    re.IGNORECASE,
)
_QUALIFIED_COLUMN_RE = re.compile(
    r"\b([a-z_][a-z0-9_]*)\s*\.\s*([a-z_][a-z0-9_]*|\*)",
    re.IGNORECASE,
)
_CTE_DEFINITION_RE = re.compile(
    r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\(", re.IGNORECASE
)


def _strip_string_literals(sql: str) -> str:
    """Remove single-quoted string literals so they don't confuse regex."""
    return re.sub(r"'(?:''|[^'])*'", "''", sql)


def _collect_table_walk(parsed: TokenList) -> set[str]:
    """sqlparse-based extraction of FROM/JOIN target table names."""
    tables: set[str] = set()

    def _names_from(token: Token) -> list[str]:
        if isinstance(token, IdentifierList):
            out: list[str] = []
            for ident in token.get_identifiers():
                out.extend(_names_from(ident))
            return out
        if isinstance(token, Identifier):
            real = token.get_real_name()
            return [real] if real else []
        if isinstance(token, TokenList):
            out = []
            for sub in token.tokens:
                out.extend(_names_from(sub))
            return out
        return []

    def _walk(token_list: list[Token]) -> None:
        i = 0
        while i < len(token_list):
            token = token_list[i]
            if token.ttype is Keyword and token.normalized.upper() in (
                "FROM",
                "JOIN",
                "INNER JOIN",
                "LEFT JOIN",
                "RIGHT JOIN",
                "FULL JOIN",
                "OUTER JOIN",
                "CROSS JOIN",
                "LEFT OUTER JOIN",
                "RIGHT OUTER JOIN",
                "FULL OUTER JOIN",
            ):
                j = i + 1
                while j < len(token_list) and (
                    token_list[j].is_whitespace
                    or token_list[j].ttype is sqlparse.tokens.Comment
                ):
                    j += 1
                if j < len(token_list):
                    for name in _names_from(token_list[j]):
                        tables.add(name)
                i = j
            elif isinstance(token, TokenList):
                _walk(list(token.tokens))
            i += 1

    _walk(list(parsed.tokens))
    return tables


def _build_alias_map(sql_no_strings: str, cte_names: set[str]) -> dict[str, str]:
    """Best-effort ``alias -> table`` map for ``alias.column`` validation.

    Maps both real-table aliases (``FROM apartments a`` -> ``a``) and the
    table name itself (``FROM apartments`` -> ``apartments``) so qualified
    references like ``apartments.rent`` resolve. CTE-defined names map to
    the sentinel ``"_cte"`` so we permit any column reference against them.
    """
    alias_map: dict[str, str] = {}
    for match in _FROM_OR_JOIN_RE.finditer(sql_no_strings):
        table = match.group(1).lower()
        alias = (match.group(2) or "").lower()
        target: Optional[str]
        if table in TABLES:
            target = table
        elif table in cte_names:
            target = "_cte"
        else:
            # Will be caught by the table-existence check; skip alias here.
            continue
        alias_map[table] = target
        if alias and alias not in _NON_ALIAS_TOKENS:
            alias_map[alias] = target
    for cte in cte_names:
        alias_map.setdefault(cte.lower(), "_cte")
    return alias_map


def _validate_llm_sql(sql: str) -> tuple[bool, str]:
    """Strict pre-validator for LLM-produced SQL.

    Returns ``(True, "")`` for SQL that is a single SELECT/WITH statement
    referencing only known tables and (where qualified) known columns.
    Returns ``(False, reason)`` for any obvious shape error so the caller
    can silently fall back to the deterministic template. The
    ``safety_agent`` still runs after this for the security firewall;
    this validator's job is to catch hallucinations early.
    """
    if not sql or not sql.strip():
        return False, "empty SQL"

    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        return False, "empty SQL after trim"

    # Must be exactly one statement.
    statements = [s for s in sqlparse.split(cleaned) if s.strip() and s.strip() != ";"]
    if len(statements) != 1:
        return False, f"expected 1 statement, got {len(statements)}"

    # First non-comment word must be SELECT or WITH.
    leading = re.match(r"^\s*(\w+)", cleaned)
    if not leading or leading.group(1).lower() not in ("select", "with"):
        return False, "statement does not start with SELECT or WITH"

    # Parse for table walk.
    try:
        parsed_statements = sqlparse.parse(cleaned)
    except Exception as exc:
        return False, f"sqlparse error: {exc}"
    if not parsed_statements:
        return False, "sqlparse returned no statements"
    parsed = parsed_statements[0]

    # Confirm parser detected SELECT (not some other DML).
    first_dml = next((t for t in parsed.flatten() if t.ttype is DML), None)
    if first_dml is not None and first_dml.value.upper() != "SELECT":
        return False, f"non-SELECT DML detected: {first_dml.value.upper()}"

    # Validate tables.
    cte_names = {m.group(1) for m in _CTE_DEFINITION_RE.finditer(cleaned)}
    referenced = _collect_table_walk(parsed)
    unknown_tables = sorted(
        t for t in referenced if not is_known_table(t) and t not in cte_names
    )
    if unknown_tables:
        return False, f"unknown table(s): {', '.join(unknown_tables)}"

    # Validate qualified column references.
    sql_no_strings = _strip_string_literals(cleaned)
    alias_map = _build_alias_map(sql_no_strings, {c.lower() for c in cte_names})
    for match in _QUALIFIED_COLUMN_RE.finditer(sql_no_strings):
        alias = match.group(1).lower()
        column = match.group(2).lower()
        if column == "*":
            continue
        target = alias_map.get(alias)
        if target is None:
            # Unknown alias - could be a function/CTE column we can't resolve.
            # The safety_agent will still catch table-level issues; skip.
            continue
        if target == "_cte":
            continue
        if column not in TABLES.get(target, {}):
            # Allow common aggregates / SQL keywords occasionally caught by
            # the regex (e.g. when followed by "AS something").
            if column in _ALL_COLUMNS:
                # Column exists somewhere; could be a join expression
                # like ``a.neighborhood_id = n.neighborhood_id``. Be lenient.
                continue
            return False, f"unknown column: {alias}.{column}"

    return True, ""


def _normalize_str_list(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        return [values.strip()] if values.strip() else []
    out: list[str] = []
    for v in values:
        s = str(v).strip()
        if s:
            out.append(s)
    return out


def _validate_sql_semantics(sql: str, router_output: dict[str, Any]) -> tuple[bool, str]:
    """Validate that SQL preserves required router constraints.

    Structural safety is handled by ``_validate_llm_sql`` + ``safety_agent``.
    This guard focuses on semantic fidelity so high-priority filters (location)
    cannot be silently dropped by LLM SQL.
    """
    filters = (router_output or {}).get("filters") or {}
    boroughs = _normalize_str_list(filters.get("boroughs"))
    neighborhoods = _normalize_str_list(filters.get("neighborhoods"))
    if not boroughs and not neighborhoods:
        return True, ""

    lowered = _strip_string_literals((sql or "").lower())
    has_borough_predicate = bool(
        re.search(r"\bn\.borough\b\s*(?:=|in\s*\(|like\b)", lowered)
    )
    has_neighborhood_predicate = bool(
        re.search(r"\bn\.name\b\s*(?:=|in\s*\(|like\b)", lowered)
    )

    if boroughs and not has_borough_predicate:
        return False, "missing borough predicate"
    if neighborhoods and not has_neighborhood_predicate:
        return False, "missing neighborhood predicate"
    return True, ""


# --- Public agent -------------------------------------------------------------


def sql_agent(
    question: str,
    schema_context: dict[str, Any],
    router_output: dict[str, Any],
    llm: Optional[LLMClient] = None,
) -> str:
    """Generate one DuckDB SELECT query for the user's question."""
    mock_sql = _mock_sql(question, router_output)

    if llm is None or llm.provider == "mock":
        return mock_sql

    schema_summary = get_schema_summary()
    required_tables = schema_context.get("required_tables", [])
    notes = schema_context.get("notes", "")
    examples_text = "\n\n".join(
        f"Question: {q}\nSQL:\n{sql}" for q, sql in FEW_SHOT_EXAMPLES
    )
    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Router output:\n{router_output}\n\n"
        f"Required tables (do not invent others): {', '.join(required_tables)}\n\n"
        f"Schema:\n{schema_summary}\n\n"
        f"Schema agent notes: {notes}\n\n"
        f"Few-shot examples:\n{examples_text}\n\n"
        "Return ONLY the SQL query. No explanation, no markdown."
    )

    try:
        result = llm.generate(user_prompt, system_prompt=SQL_SYSTEM_PROMPT, temperature=0.0)
        sql = (result.text or "").strip()
        # Strip markdown fences if the model added them.
        if sql.startswith("```"):
            sql = sql.strip("`")
            if sql.lower().startswith("sql"):
                sql = sql[3:]
        sql = sql.strip()
        ok, _reason = _validate_llm_sql(sql)
        semantic_ok, _semantic_reason = _validate_sql_semantics(sql, router_output)
        if ok and semantic_ok:
            return sql
    except Exception:
        pass
    return mock_sql
