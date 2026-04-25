"""End-to-end ingestion pipeline for raw NYC CSV/XLSX files.

This is intentionally tolerant: missing datasets do not crash the pipeline,
they just yield neutral metric defaults with explicit warnings.

CLI entrypoint:

    python -m src.ingestion.pipeline
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import duckdb
import pandas as pd

from src.config import DB_PATH, PROCESSED_DIR, RAW_DIR, ensure_directories
from src.database import (
    append_rows,
    get_connection,
    initialize_database,
    replace_table,
)
from src.ingestion.cleaning import (
    clean_bathrooms,
    clean_bedrooms,
    clean_date,
    clean_integer,
    clean_numeric,
    clean_rent,
    normalize_borough_name,
    standardize_text,
)
from src.ingestion.column_mapper import (
    SUPPORTED_DATASET_TYPES,
    detect_column,
    detect_dataset_type_from_filename,
    map_columns,
)
from src.ingestion.csv_loader import list_raw_files, load_dataframe
from src.ingestion.data_profiler import profile_file
from src.ingestion.geospatial import match_neighborhood
from src.ingestion.metric_engineering import build_neighborhood_metrics, now_iso
from src.schema import TABLES


PathLike = Union[str, Path]


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _facility_category(name: Optional[str], facility_type: Optional[str]) -> str:
    """Map an arbitrary NYC facility name/type into one of our amenity categories."""
    text = " ".join(filter(None, [str(name or ""), str(facility_type or "")])).lower()
    if any(k in text for k in ("library", "bookmobile")):
        return "library"
    if any(k in text for k in ("school", "academy", "education", "preschool", "head start")):
        return "school"
    if any(k in text for k in ("hospital", "clinic", "health", "medical", "diagnostic")):
        return "healthcare"
    if any(k in text for k in ("community center", "youth center", "cultural", "neighborhood center")):
        return "community_center"
    if any(k in text for k in ("police", "fire", "emergency")):
        return "emergency_service"
    if any(k in text for k in ("childcare", "daycare", "child care")):
        return "childcare"
    if any(k in text for k in ("senior", "older adult")):
        return "senior_center"
    if any(k in text for k in ("recreation", "rec center")):
        return "recreation"
    if any(k in text for k in ("shelter", "social service", "human services")):
        return "social_service"
    return "facility"


def _worship_type_for(name: str, fallback_type: Optional[str] = None) -> str:
    text = (name or "").lower() + " " + (fallback_type or "").lower()
    if any(k in text for k in ("mosque", "masjid", "islamic")):
        return "mosque"
    if any(k in text for k in ("synagogue", "jewish")):
        return "synagogue"
    if any(k in text for k in ("temple", "hindu", "buddhist", "sikh", "gurudwara")):
        return "temple"
    if any(k in text for k in ("church", "cathedral", "chapel", "parish")):
        return "church"
    return "worship"


# --- The pipeline class -------------------------------------------------------


class IngestionPipeline:
    """Drives the raw-CSV-to-DuckDB pipeline."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path else DB_PATH
        ensure_directories()
        initialize_database(self.db_path)
        self._neighborhoods_cache: Optional[pd.DataFrame] = None

    # --- Public entrypoints -------------------------------------------------

    def profile_csv(
        self,
        file_path: PathLike,
        dataset_type: Optional[str] = None,
    ) -> dict[str, Any]:
        return profile_file(file_path, dataset_type=dataset_type)

    def ingest_csv(
        self,
        file_path: PathLike,
        dataset_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """Ingest a single file into the appropriate raw/staging tables."""
        p = Path(file_path)
        if not p.exists():
            return {"status": "error", "reason": f"File not found: {p}"}
        dt = (dataset_type or detect_dataset_type_from_filename(p.name) or "").lower()
        if dt and dt not in SUPPORTED_DATASET_TYPES:
            dt = ""
        if not dt:
            return {
                "status": "error",
                "reason": "Could not determine dataset_type. Pass it explicitly.",
                "file_name": p.name,
            }

        warnings: list[str] = []
        try:
            if dt == "transit_bus":
                row_count, col_count, ingest_warnings = self._ingest_bus_stops(p)
            else:
                df = load_dataframe(p)
                col_count = int(len(df.columns))
                row_count = int(len(df))
                ingest_warnings = self._ingest_dataframe(df, dt, source_file=p.name)
            warnings.extend(ingest_warnings)
            status = "ingested"
        except Exception as exc:
            return {
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
                "file_name": p.name,
                "dataset_type": dt,
            }

        self._register_file(p, dt, row_count, col_count, status, warnings)
        return {
            "status": status,
            "file_name": p.name,
            "dataset_type": dt,
            "row_count": row_count,
            "column_count": col_count,
            "warnings": warnings,
        }

    def ingest_all_raw_files(self) -> dict[str, Any]:
        """Walk ``data/raw/`` and ingest everything we recognize."""
        results: list[dict[str, Any]] = []
        for f in list_raw_files():
            results.append(self.ingest_csv(f))
        return {"results": results, "ingested_at": _utc_now()}

    def rebuild_processed_tables(self) -> dict[str, Any]:
        """Recompute neighborhood_stats from the data already in DuckDB."""
        df, warnings = self.compute_neighborhood_metrics()
        replace_table("neighborhood_stats", df, self.db_path)
        out_csv = PROCESSED_DIR / "neighborhood_stats.csv"
        df.to_csv(out_csv, index=False)
        distance_warnings = self._refresh_apartment_transit_distances()
        warnings = list(warnings) + list(distance_warnings)
        return {
            "rows": int(len(df)),
            "warnings": warnings,
            "csv_path": str(out_csv),
        }

    def _refresh_apartment_transit_distances(self) -> list[str]:
        """Populate per-apartment nearest subway/bus distances via a DuckDB
        haversine UPDATE. Mirrors the formula used in
        :func:`src.agents.sql_agent._haversine_miles_sql` (3958.8 mi Earth radius).
        """
        warnings: list[str] = []
        con = get_connection(self.db_path)
        try:
            for col in ("nearest_subway_distance", "nearest_bus_stop_distance"):
                con.execute(f"ALTER TABLE apartments ADD COLUMN IF NOT EXISTS {col} DOUBLE")
            con.execute(
                "ALTER TABLE apartments ADD COLUMN IF NOT EXISTS hpd_violation_count INTEGER"
            )

            for col, table in (
                ("nearest_subway_distance", "transit_subway_stations"),
                ("nearest_bus_stop_distance", "transit_bus_stops"),
            ):
                count_row = con.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE lat IS NOT NULL AND lng IS NOT NULL"
                ).fetchone()
                if not count_row or not count_row[0]:
                    warnings.append(
                        f"_refresh_apartment_transit_distances: {table} has no geocoded rows; skipping {col}."
                    )
                    continue
                con.execute(
                    f"""
                    UPDATE apartments AS a
                    SET {col} = (
                        SELECT MIN(2 * 3958.8 * ASIN(SQRT(
                            POWER(SIN(RADIANS((s.lat - a.lat)/2)), 2) +
                            COS(RADIANS(a.lat)) * COS(RADIANS(s.lat)) *
                            POWER(SIN(RADIANS((s.lng - a.lng)/2)), 2)
                        )))
                        FROM {table} AS s
                        WHERE s.lat IS NOT NULL AND s.lng IS NOT NULL
                    )
                    WHERE a.lat IS NOT NULL AND a.lng IS NOT NULL
                    """
                )
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"_refresh_apartment_transit_distances failed: {exc}")
        finally:
            con.close()
        return warnings

    def compute_neighborhood_metrics(self) -> tuple[pd.DataFrame, list[str]]:
        """Read the canonical tables back out and run metric engineering."""
        con = get_connection(self.db_path)
        try:
            neighborhoods = con.execute("SELECT * FROM neighborhoods").fetchdf()
            apartments = con.execute("SELECT * FROM apartments").fetchdf()
            crimes = con.execute("SELECT * FROM crime_events").fetchdf()
            parks = con.execute("SELECT * FROM parks").fetchdf()
            facilities = con.execute("SELECT * FROM facilities").fetchdf()
            amenities = con.execute("SELECT * FROM amenities").fetchdf()
            hpd = con.execute("SELECT * FROM hpd_buildings").fetchdf()
            bus = con.execute("SELECT * FROM transit_bus_stops").fetchdf()
            subway = con.execute("SELECT * FROM transit_subway_stations").fetchdf()
        finally:
            con.close()

        # Pull demographics from processed_metrics (the source of truth that the
        # demographics/population ingest writes to). Picking the latest value per
        # (neighborhood_id, metric_name) makes re-ingests idempotent.
        pop_map: dict[int, int] = {}
        inc_map: dict[int, int] = {}
        rent_map: dict[int, int] = {}
        try:
            con = get_connection(self.db_path)
            try:
                pm = con.execute(
                    """
                    SELECT neighborhood_id, metric_name, metric_value
                    FROM processed_metrics
                    WHERE metric_name IN ('population', 'median_income', 'median_rent')
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY neighborhood_id, metric_name
                        ORDER BY computed_at DESC
                    ) = 1
                    """
                ).fetchdf()
            finally:
                con.close()
            if not pm.empty:
                for name, target in (
                    ("population", pop_map),
                    ("median_income", inc_map),
                    ("median_rent", rent_map),
                ):
                    sub = pm[pm["metric_name"] == name]
                    for r in sub.itertuples():
                        if pd.notna(r.metric_value) and pd.notna(r.neighborhood_id):
                            target[int(r.neighborhood_id)] = int(r.metric_value)
        except Exception:
            pop_map, inc_map, rent_map = {}, {}, {}

        df, warnings = build_neighborhood_metrics(
            neighborhoods=neighborhoods,
            apartments=apartments if not apartments.empty else None,
            crimes=crimes if not crimes.empty else None,
            parks=parks if not parks.empty else None,
            facilities=facilities if not facilities.empty else None,
            amenities=amenities if not amenities.empty else None,
            hpd_buildings=hpd if not hpd.empty else None,
            bus_stops=bus if not bus.empty else None,
            subway_stations=subway if not subway.empty else None,
            population_by_id=pop_map,
            median_income_by_id=inc_map,
            median_rent_by_id=rent_map,
        )
        return df, warnings

    def load_processed_tables_to_duckdb(self) -> dict[str, int]:
        """Refresh ``neighborhood_stats`` (and the processed CSV) in DuckDB."""
        result = self.rebuild_processed_tables()
        return {"neighborhood_stats": result["rows"]}

    # --- Per-dataset ingestion ---------------------------------------------

    def _ingest_dataframe(
        self,
        df: pd.DataFrame,
        dataset_type: str,
        source_file: str,
    ) -> list[str]:
        warnings: list[str] = []
        dt = dataset_type
        if dt == "demographics" or dt == "population":
            warnings.extend(self._ingest_demographics_population(df, source_file, is_population=(dt == "population")))
        elif dt == "housing":
            warnings.extend(self._ingest_hpd(df, source_file))
        elif dt == "facilities":
            warnings.extend(self._ingest_facilities(df, source_file))
        elif dt == "transit_subway":
            warnings.extend(self._ingest_subway(df, source_file))
        elif dt == "parks":
            warnings.extend(self._ingest_parks(df, source_file))
        elif dt == "apartments":
            warnings.extend(self._ingest_apartments(df, source_file))
        elif dt == "crime":
            warnings.extend(self._ingest_crime(df, source_file))
        elif dt in ("amenities", "worship"):
            warnings.extend(self._ingest_amenities(df, source_file, dataset_type=dt))
        else:
            warnings.append(f"Unsupported dataset_type: {dt}")
        return warnings

    # --- Specific dataset handlers -----------------------------------------

    def _neighborhoods_df(self) -> pd.DataFrame:
        if self._neighborhoods_cache is None:
            con = get_connection(self.db_path)
            try:
                self._neighborhoods_cache = con.execute("SELECT * FROM neighborhoods").fetchdf()
            finally:
                con.close()
        return self._neighborhoods_cache

    def _refresh_neighborhoods_cache(self) -> None:
        self._neighborhoods_cache = None

    def _ingest_demographics_population(
        self,
        df: pd.DataFrame,
        source_file: str,
        is_population: bool,
    ) -> list[str]:
        warnings: list[str] = []
        cols = map_columns(df, "population" if is_population else "demographics")
        nta_col = cols.get("nta_code")
        name_col = cols.get("neighborhood")
        boro_col = cols.get("borough")
        pop_col = cols.get("population")
        inc_col = cols.get("median_income") if not is_population else None
        rent_col = cols.get("median_rent") if not is_population else None

        if not (nta_col or name_col):
            warnings.append("Demographics file lacks NTA or neighborhood columns; skipping.")
            return warnings

        # Upsert neighborhoods first (so other ingests can join on them).
        nb_existing = self._neighborhoods_df()
        next_id = int(nb_existing["neighborhood_id"].max()) + 1 if not nb_existing.empty else 1
        new_neighborhoods: list[dict] = []
        pop_map: dict[int, int] = {}
        inc_map: dict[int, int] = {}
        rent_map: dict[int, int] = {}

        for _, row in df.iterrows():
            nta_value = standardize_text(row.get(nta_col)) if nta_col else None
            name_value = standardize_text(row.get(name_col)) if name_col else None
            boro_value = normalize_borough_name(row.get(boro_col)) if boro_col else None
            if not (nta_value or name_value):
                continue

            existing_match = None
            if nta_value and not nb_existing.empty:
                m = nb_existing[nb_existing["nta_code"].astype(str).str.upper() == nta_value.upper()]
                if not m.empty:
                    existing_match = int(m.iloc[0]["neighborhood_id"])
            if existing_match is None and name_value and not nb_existing.empty:
                m = nb_existing[nb_existing["name"].astype(str).str.lower() == name_value.lower()]
                if not m.empty:
                    existing_match = int(m.iloc[0]["neighborhood_id"])
            if existing_match is None:
                new_id = next_id
                next_id += 1
                new_neighborhoods.append(
                    {
                        "neighborhood_id": new_id,
                        "nta_code": nta_value,
                        "name": name_value or nta_value,
                        "borough": boro_value,
                        "city": "New York",
                        "state": "NY",
                        "center_lat": None,
                        "center_lng": None,
                    }
                )
                target_id = new_id
            else:
                target_id = existing_match

            pop_val = clean_integer(row.get(pop_col)) if pop_col else None
            if pop_val:
                pop_map[target_id] = pop_val
            if inc_col:
                inc_val = clean_integer(row.get(inc_col))
                if inc_val:
                    inc_map[target_id] = inc_val
            if rent_col:
                rent_val = clean_integer(row.get(rent_col))
                if rent_val:
                    rent_map[target_id] = rent_val

        if new_neighborhoods:
            append_rows("neighborhoods", pd.DataFrame(new_neighborhoods), self.db_path)
            self._refresh_neighborhoods_cache()

        # Persist demographic facts via processed_metrics so re-runs don't lose them.
        records: list[dict] = []
        ts = now_iso()
        for nb_id, value in pop_map.items():
            records.append(
                {
                    "neighborhood_id": nb_id,
                    "metric_name": "population",
                    "metric_value": float(value),
                    "source": source_file,
                    "computed_at": ts,
                }
            )
        for nb_id, value in inc_map.items():
            records.append(
                {
                    "neighborhood_id": nb_id,
                    "metric_name": "median_income",
                    "metric_value": float(value),
                    "source": source_file,
                    "computed_at": ts,
                }
            )
        for nb_id, value in rent_map.items():
            records.append(
                {
                    "neighborhood_id": nb_id,
                    "metric_name": "median_rent",
                    "metric_value": float(value),
                    "source": source_file,
                    "computed_at": ts,
                }
            )
        if records:
            append_rows("processed_metrics", pd.DataFrame(records), self.db_path)
        return warnings

    def _ingest_hpd(self, df: pd.DataFrame, source_file: str) -> list[str]:
        warnings: list[str] = ["HPD building data is a housing-stock signal, not direct vacancy."]
        cols = map_columns(df, "housing")
        nb_df = self._neighborhoods_df()

        rows: list[dict] = []
        for _, row in df.iterrows():
            lat = clean_numeric(row.get(cols.get("lat")) if cols.get("lat") else None)
            lng = clean_numeric(row.get(cols.get("lng")) if cols.get("lng") else None)
            address = standardize_text(row.get(cols.get("address"))) if cols.get("address") else None
            if not address and cols.get("house_number") and cols.get("street_name"):
                hn = standardize_text(row.get(cols["house_number"])) or ""
                sn = standardize_text(row.get(cols["street_name"])) or ""
                address = (hn + " " + sn).strip()
            search_row = pd.Series(
                {
                    "lat": lat,
                    "lng": lng,
                    "neighborhood": standardize_text(row.get(cols.get("borough"))) if cols.get("borough") else None,
                }
            )
            nb_id = match_neighborhood(search_row, nb_df) if not nb_df.empty else None
            rows.append(
                {
                    "building_id": str(row.get(cols.get("building_id"))) if cols.get("building_id") else f"HPD{uuid.uuid4().hex[:8]}",
                    "neighborhood_id": nb_id,
                    "borough": normalize_borough_name(row.get(cols.get("borough"))) if cols.get("borough") else None,
                    "address": address,
                    "zip_code": standardize_text(row.get(cols.get("zip_code"))) if cols.get("zip_code") else None,
                    "lat": lat,
                    "lng": lng,
                    "residential_units": clean_integer(row.get(cols.get("residential_units"))) if cols.get("residential_units") else None,
                    "program_type": standardize_text(row.get(cols.get("program_type"))) if cols.get("program_type") else None,
                    "status": standardize_text(row.get(cols.get("status"))) if cols.get("status") else None,
                    "source_file": source_file,
                }
            )
        if rows:
            append_rows("hpd_buildings", pd.DataFrame(rows), self.db_path)
        return warnings

    def _ingest_facilities(self, df: pd.DataFrame, source_file: str) -> list[str]:
        warnings: list[str] = []
        cols = map_columns(df, "facilities")
        nb_df = self._neighborhoods_df()

        rows: list[dict] = []
        amenity_rows: list[dict] = []
        for _, row in df.iterrows():
            lat = clean_numeric(row.get(cols.get("lat")) if cols.get("lat") else None)
            lng = clean_numeric(row.get(cols.get("lng")) if cols.get("lng") else None)
            name = standardize_text(row.get(cols.get("facility_name"))) if cols.get("facility_name") else None
            ftype = standardize_text(row.get(cols.get("amenity_type"))) if cols.get("amenity_type") else None
            category = _facility_category(name, ftype)
            search_row = pd.Series({"lat": lat, "lng": lng})
            nb_id = match_neighborhood(search_row, nb_df) if not nb_df.empty else None
            rows.append(
                {
                    "facility_id": f"FAC{uuid.uuid4().hex[:10]}",
                    "neighborhood_id": nb_id,
                    "name": name,
                    "facility_type": category,
                    "category": ftype,
                    "borough": normalize_borough_name(row.get(cols.get("borough"))) if cols.get("borough") else None,
                    "address": standardize_text(row.get(cols.get("address"))) if cols.get("address") else None,
                    "lat": lat,
                    "lng": lng,
                    "agency": standardize_text(row.get(cols.get("agency"))) if cols.get("agency") else None,
                    "source_file": source_file,
                }
            )
            if category in {"library", "school", "healthcare", "community_center", "emergency_service",
                            "childcare", "senior_center", "recreation", "social_service"}:
                amenity_rows.append(
                    {
                        "amenity_id": None,
                        "neighborhood_id": nb_id,
                        "name": name,
                        "type": category,
                        "lat": lat,
                        "lng": lng,
                        "distance_miles": None,
                        "source_file": source_file,
                    }
                )
        if rows:
            append_rows("facilities", pd.DataFrame(rows), self.db_path)
        if amenity_rows:
            self._append_amenities(amenity_rows)
        return warnings

    def _ingest_subway(self, df: pd.DataFrame, source_file: str) -> list[str]:
        warnings: list[str] = []
        cols = map_columns(df, "transit_subway")
        nb_df = self._neighborhoods_df()

        rows: list[dict] = []
        amenity_rows: list[dict] = []
        for _, row in df.iterrows():
            lat = clean_numeric(row.get(cols.get("lat")) if cols.get("lat") else None)
            lng = clean_numeric(row.get(cols.get("lng")) if cols.get("lng") else None)
            station_id = standardize_text(row.get(cols.get("station_id"))) if cols.get("station_id") else None
            station_name = standardize_text(row.get(cols.get("station_name"))) if cols.get("station_name") else None
            search_row = pd.Series({"lat": lat, "lng": lng})
            nb_id = match_neighborhood(search_row, nb_df) if not nb_df.empty else None
            rows.append(
                {
                    "station_id": station_id or f"SUB{uuid.uuid4().hex[:8]}",
                    "station_name": station_name,
                    "line": standardize_text(row.get(cols.get("line"))) if cols.get("line") else None,
                    "routes": standardize_text(row.get(cols.get("routes"))) if cols.get("routes") else None,
                    "lat": lat,
                    "lng": lng,
                    "borough": normalize_borough_name(row.get(cols.get("borough"))) if cols.get("borough") else None,
                    "neighborhood_id": nb_id,
                    "source_file": source_file,
                }
            )
            amenity_rows.append(
                {
                    "amenity_id": None,
                    "neighborhood_id": nb_id,
                    "name": station_name,
                    "type": "subway",
                    "lat": lat,
                    "lng": lng,
                    "distance_miles": None,
                    "source_file": source_file,
                }
            )
        if rows:
            append_rows("transit_subway_stations", pd.DataFrame(rows), self.db_path)
        if amenity_rows:
            self._append_amenities(amenity_rows)
        return warnings

    def _ingest_bus_stops(self, file_path: Path) -> tuple[int, int, list[str]]:
        """Ingest the (potentially huge) MTA bus stops file via DuckDB dedup."""
        warnings = [
            "If this file contains route-stop-direction records, the raw row count is not the number of "
            "unique physical stops. AI-partments deduplicates stops before calculating transit access."
        ]
        con = get_connection(self.db_path)
        try:
            con.execute("DROP TABLE IF EXISTS raw_mta_bus_stops")
            con.execute(
                "CREATE TABLE raw_mta_bus_stops AS "
                "SELECT * FROM read_csv_auto(?, sample_size=-1)",
                [str(file_path)],
            )
            row_count = int(con.execute("SELECT COUNT(*) FROM raw_mta_bus_stops").fetchone()[0])
            col_count = int(con.execute("SELECT COUNT(*) FROM information_schema.columns WHERE table_name='raw_mta_bus_stops'").fetchone()[0])

            # Detect column names case-insensitively. We normalize both raw and
            # candidate names so e.g. "Stop ID" matches the candidate "stop_id".
            raw_cols = [r[0] for r in con.execute("DESCRIBE raw_mta_bus_stops").fetchall()]

            def _norm(col: str) -> str:
                return re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")

            col_lookup = {_norm(c): c for c in raw_cols}

            def pick(*candidates: str) -> Optional[str]:
                for cand in candidates:
                    if _norm(cand) in col_lookup:
                        return col_lookup[_norm(cand)]
                return None

            stop_id_col = pick("stop_id", "stopid", "stop_code", "stoppoint_id")
            stop_name_col = pick("stop_name", "stopname", "name")
            lat_col = pick("stop_lat", "lat", "latitude", "y")
            lng_col = pick("stop_lon", "stop_lng", "lng", "lon", "longitude", "x")
            route_col = pick("route_id", "route", "routeid", "route_short_name")
            borough_col = pick("borough", "boro", "boro_name")

            if lat_col is None or lng_col is None:
                warnings.append("Bus stops file has no recognizable lat/lng columns; deduplication skipped.")
                return row_count, col_count, warnings

            lat_double_expr = f'CAST("{lat_col}" AS DOUBLE)'
            lng_double_expr = f'CAST("{lng_col}" AS DOUBLE)'
            stop_id_expr = f'CAST("{stop_id_col}" AS VARCHAR)' if stop_id_col else "NULL"
            stop_name_expr = f'CAST("{stop_name_col}" AS VARCHAR)' if stop_name_col else "NULL"
            route_count_expr = f'COUNT(DISTINCT "{route_col}")' if route_col else "NULL"
            borough_expr = f'MAX(CAST("{borough_col}" AS VARCHAR))' if borough_col else "NULL"

            if stop_id_col:
                group_by = (
                    f'"{stop_id_col}", "{stop_name_col or stop_id_col}", '
                    f"ROUND({lat_double_expr}, 5), ROUND({lng_double_expr}, 5)"
                )
                dedup_sql = (
                    f"SELECT {stop_id_expr} AS stop_id, "
                    f"{stop_name_expr} AS stop_name, "
                    f"AVG({lat_double_expr}) AS lat, "
                    f"AVG({lng_double_expr}) AS lng, "
                    f"{borough_expr} AS borough, "
                    f"{route_count_expr} AS route_count, "
                    f"COUNT(*) AS route_record_count "
                    f"FROM raw_mta_bus_stops GROUP BY {group_by}"
                )
            else:
                dedup_sql = (
                    f"SELECT NULL AS stop_id, "
                    f"{stop_name_expr} AS stop_name, "
                    f"AVG({lat_double_expr}) AS lat, "
                    f"AVG({lng_double_expr}) AS lng, "
                    f"{borough_expr} AS borough, "
                    f"{route_count_expr} AS route_count, "
                    f"COUNT(*) AS route_record_count "
                    f"FROM raw_mta_bus_stops "
                    f"GROUP BY {stop_name_expr}, ROUND({lat_double_expr}, 5), ROUND({lng_double_expr}, 5)"
                )
            dedup_df = con.execute(dedup_sql).fetchdf()
        finally:
            con.close()

        # Match neighborhoods on the dedup output (smaller).
        nb_df = self._neighborhoods_df()
        rows: list[dict] = []
        for _, row in dedup_df.iterrows():
            search_row = pd.Series({"lat": row["lat"], "lng": row["lng"]})
            nb_id = match_neighborhood(search_row, nb_df) if not nb_df.empty else None
            rows.append(
                {
                    "stop_id": row.get("stop_id") if pd.notna(row.get("stop_id")) else f"BUS{uuid.uuid4().hex[:8]}",
                    "stop_name": row.get("stop_name") if pd.notna(row.get("stop_name")) else None,
                    "lat": float(row["lat"]) if pd.notna(row["lat"]) else None,
                    "lng": float(row["lng"]) if pd.notna(row["lng"]) else None,
                    "borough": normalize_borough_name(row.get("borough")) if pd.notna(row.get("borough")) else None,
                    "route_count": int(row["route_count"]) if pd.notna(row.get("route_count")) else None,
                    "route_record_count": int(row["route_record_count"]) if pd.notna(row.get("route_record_count")) else None,
                    "neighborhood_id": nb_id,
                    "source_file": file_path.name,
                }
            )
        if rows:
            append_rows("transit_bus_stops", pd.DataFrame(rows), self.db_path)
        warnings.append(
            f"Deduplicated {row_count} raw rows down to {len(rows)} unique bus stops."
        )
        return row_count, col_count, warnings

    def _ingest_parks(self, df: pd.DataFrame, source_file: str) -> list[str]:
        warnings: list[str] = []
        cols = map_columns(df, "parks")
        nb_df = self._neighborhoods_df()

        rows: list[dict] = []
        amenity_rows: list[dict] = []
        for _, row in df.iterrows():
            lat = clean_numeric(row.get(cols.get("lat")) if cols.get("lat") else None)
            lng = clean_numeric(row.get(cols.get("lng")) if cols.get("lng") else None)
            name = standardize_text(row.get(cols.get("park_name"))) if cols.get("park_name") else None
            search_row = pd.Series({"lat": lat, "lng": lng})
            nb_id = match_neighborhood(search_row, nb_df) if not nb_df.empty else None
            rows.append(
                {
                    "park_id": f"PARK{uuid.uuid4().hex[:8]}",
                    "neighborhood_id": nb_id,
                    "name": name,
                    "borough": normalize_borough_name(row.get(cols.get("borough"))) if cols.get("borough") else None,
                    "address": standardize_text(row.get(cols.get("address"))) if cols.get("address") else None,
                    "lat": lat,
                    "lng": lng,
                    "acres": clean_numeric(row.get(cols.get("acres"))) if cols.get("acres") else None,
                    "park_type": standardize_text(row.get(cols.get("park_type"))) if cols.get("park_type") else None,
                    "source_file": source_file,
                }
            )
            amenity_rows.append(
                {
                    "amenity_id": None,
                    "neighborhood_id": nb_id,
                    "name": name,
                    "type": "park",
                    "lat": lat,
                    "lng": lng,
                    "distance_miles": None,
                    "source_file": source_file,
                }
            )
        if rows:
            append_rows("parks", pd.DataFrame(rows), self.db_path)
        if amenity_rows:
            self._append_amenities(amenity_rows)
        return warnings

    def _ingest_apartments(self, df: pd.DataFrame, source_file: str) -> list[str]:
        warnings: list[str] = []
        cols = map_columns(df, "apartments")
        nb_df = self._neighborhoods_df()

        rows: list[dict] = []
        next_id_row = None
        try:
            con = get_connection(self.db_path)
            next_id_row = con.execute("SELECT MAX(apartment_id) FROM apartments").fetchone()
            con.close()
        except Exception:
            pass
        next_id = (int(next_id_row[0]) if next_id_row and next_id_row[0] else 0) + 1

        for _, row in df.iterrows():
            lat = clean_numeric(row.get(cols.get("lat")) if cols.get("lat") else None)
            lng = clean_numeric(row.get(cols.get("lng")) if cols.get("lng") else None)
            address = standardize_text(row.get(cols.get("address"))) if cols.get("address") else None
            if not address:
                sn = standardize_text(row.get(cols.get("street_name"))) if cols.get("street_name") else None
                unit = standardize_text(row.get(cols.get("unit"))) if cols.get("unit") else None
                if sn:
                    address = f"{sn} #{unit}" if unit else sn
            search_row = pd.Series(
                {
                    "lat": lat,
                    "lng": lng,
                    "neighborhood": standardize_text(row.get(cols.get("neighborhood"))) if cols.get("neighborhood") else None,
                }
            )
            nb_id = match_neighborhood(search_row, nb_df) if not nb_df.empty else None
            rows.append(
                {
                    "apartment_id": next_id,
                    "neighborhood_id": nb_id,
                    "address": address,
                    "rent": clean_rent(row.get(cols.get("rent"))) if cols.get("rent") else None,
                    "bedrooms": clean_bedrooms(row.get(cols.get("bedrooms"))) if cols.get("bedrooms") else None,
                    "bathrooms": clean_bathrooms(row.get(cols.get("bathrooms"))) if cols.get("bathrooms") else None,
                    "sqft": clean_integer(row.get(cols.get("sqft"))) if cols.get("sqft") else None,
                    "available_date": clean_date(row.get(cols.get("date"))) if cols.get("date") else None,
                    "lat": lat,
                    "lng": lng,
                    "source": source_file,
                    "listing_url": standardize_text(row.get(cols.get("listing_url"))) if cols.get("listing_url") else None,
                    "nearest_subway_distance": None,
                    "nearest_bus_stop_distance": None,
                    "hpd_violation_count": clean_integer(row.get(cols.get("hpd_violation_count"))) if cols.get("hpd_violation_count") else None,
                }
            )
            next_id += 1
        if rows:
            append_rows("apartments", pd.DataFrame(rows), self.db_path)
        return warnings

    def _ingest_crime(self, df: pd.DataFrame, source_file: str) -> list[str]:
        warnings: list[str] = []
        cols = map_columns(df, "crime")
        nb_df = self._neighborhoods_df()
        is_shootings_file = "shooting" in (source_file or "").lower()

        rows: list[dict] = []
        next_id = 1
        try:
            con = get_connection(self.db_path)
            r = con.execute("SELECT MAX(event_id) FROM crime_events").fetchone()
            next_id = (int(r[0]) if r and r[0] else 0) + 1
            con.close()
        except Exception:
            pass

        for _, row in df.iterrows():
            offense = standardize_text(row.get(cols.get("offense"))) if cols.get("offense") else None
            severity = standardize_text(row.get(cols.get("severity"))) if cols.get("severity") else None
            if not offense and is_shootings_file:
                # NYPD Shootings extracts have no offense column; force the
                # category so _infer_severity classifies them as violent.
                offense = "shooting"
            inferred = _infer_severity(offense, severity)
            lat = clean_numeric(row.get(cols.get("lat"))) if cols.get("lat") else None
            lng = clean_numeric(row.get(cols.get("lng"))) if cols.get("lng") else None
            search_row = pd.Series({"lat": lat, "lng": lng})
            nb_id = match_neighborhood(search_row, nb_df) if not nb_df.empty else None
            rows.append(
                {
                    "event_id": next_id,
                    "neighborhood_id": nb_id,
                    "offense_type": offense,
                    "severity": inferred,
                    "date": clean_date(row.get(cols.get("date"))) if cols.get("date") else None,
                    "lat": lat,
                    "lng": lng,
                }
            )
            next_id += 1
        if rows:
            append_rows("crime_events", pd.DataFrame(rows), self.db_path)
        return warnings

    def _ingest_amenities(
        self,
        df: pd.DataFrame,
        source_file: str,
        dataset_type: str,
    ) -> list[str]:
        warnings: list[str] = []
        cols = map_columns(df, dataset_type)
        nb_df = self._neighborhoods_df()
        rows: list[dict] = []
        for _, row in df.iterrows():
            name = standardize_text(row.get(cols.get("amenity_name"))) if cols.get("amenity_name") else None
            raw_type = standardize_text(row.get(cols.get("amenity_type"))) if cols.get("amenity_type") else None
            if dataset_type == "worship":
                amenity_type = _worship_type_for(name or "", raw_type)
            else:
                amenity_type = raw_type or "amenity"
            lat = clean_numeric(row.get(cols.get("lat"))) if cols.get("lat") else None
            lng = clean_numeric(row.get(cols.get("lng"))) if cols.get("lng") else None
            search_row = pd.Series({"lat": lat, "lng": lng})
            nb_id = match_neighborhood(search_row, nb_df) if not nb_df.empty else None
            rows.append(
                {
                    "amenity_id": None,
                    "neighborhood_id": nb_id,
                    "name": name,
                    "type": amenity_type,
                    "lat": lat,
                    "lng": lng,
                    "distance_miles": None,
                    "source_file": source_file,
                }
            )
        if rows:
            self._append_amenities(rows)
        return warnings

    def _append_amenities(self, rows: list[dict]) -> None:
        # Amenity ids: continue from current max + 1 so we never collide.
        next_id = 1
        try:
            con = get_connection(self.db_path)
            r = con.execute("SELECT MAX(amenity_id) FROM amenities").fetchone()
            next_id = (int(r[0]) if r and r[0] else 0) + 1
            con.close()
        except Exception:
            pass
        for row in rows:
            row["amenity_id"] = next_id
            next_id += 1
        append_rows("amenities", pd.DataFrame(rows), self.db_path)

    # --- Helpers ------------------------------------------------------------

    def _load_table(self, table: str) -> pd.DataFrame:
        if table not in TABLES:
            return pd.DataFrame()
        con = get_connection(self.db_path)
        try:
            return con.execute(f'SELECT * FROM "{table}"').fetchdf()
        finally:
            con.close()

    def _register_file(
        self,
        path: Path,
        dataset_type: str,
        row_count: int,
        col_count: int,
        status: str,
        warnings: list[str],
    ) -> None:
        record = {
            "file_id": uuid.uuid4().hex,
            "file_name": path.name,
            "dataset_type": dataset_type,
            "row_count": int(row_count),
            "column_count": int(col_count),
            "ingested_at": _utc_now(),
            "status": status,
            "warnings_json": json.dumps(warnings),
        }
        append_rows("raw_file_registry", pd.DataFrame([record]), self.db_path)


def _infer_severity(offense_type: Optional[str], severity_field: Optional[str]) -> str:
    text = " ".join(filter(None, [offense_type or "", severity_field or ""])).lower()
    if any(k in text for k in ("assault", "robbery", "homicide", "murder", "rape", "felony", "violent", "shooting")):
        return "violent"
    if any(k in text for k in ("burglary", "theft", "larceny", "stolen", "auto", "property")):
        return "property"
    return "other"


def main() -> None:
    pipeline = IngestionPipeline()
    print(f"Ingesting raw files from {RAW_DIR} ...")
    summary = pipeline.ingest_all_raw_files()
    for r in summary["results"]:
        status = r.get("status", "?")
        print(f"  [{status}] {r.get('file_name')} -> {r.get('dataset_type')} (rows={r.get('row_count')})")
        for w in r.get("warnings", []) or []:
            print(f"     warning: {w}")
    print("Computing neighborhood metrics ...")
    rebuilt = pipeline.rebuild_processed_tables()
    print(f"  neighborhood_stats rows: {rebuilt['rows']}")
    for w in rebuilt["warnings"]:
        print(f"  warning: {w}")
    print("Done.")


if __name__ == "__main__":
    main()
