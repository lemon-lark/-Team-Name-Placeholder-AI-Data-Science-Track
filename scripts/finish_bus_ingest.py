"""One-shot helper: dedup the already-loaded raw_mta_bus_stops table into
transit_bus_stops, then rebuild processed_metrics + apartment distances.

Run after a pipeline run that loaded raw_mta_bus_stops successfully but
errored during dedup (no need to re-read the 763 MB CSV).
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path

import pandas as pd

from src.config import DB_PATH
from src.database import append_rows, get_connection
from src.ingestion.cleaning import normalize_borough_name
from src.ingestion.geospatial import match_neighborhood
from src.ingestion.pipeline import IngestionPipeline


def _norm(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")


def main() -> None:
    pipeline = IngestionPipeline()
    con = get_connection(DB_PATH)
    try:
        raw_count = con.execute("SELECT COUNT(*) FROM raw_mta_bus_stops").fetchone()[0]
        print(f"raw_mta_bus_stops rows: {raw_count}")

        raw_cols = [r[0] for r in con.execute("DESCRIBE raw_mta_bus_stops").fetchall()]
        col_lookup = {_norm(c): c for c in raw_cols}

        def pick(*candidates: str) -> str | None:
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
        print(
            f"detected: stop_id={stop_id_col}, stop_name={stop_name_col}, "
            f"lat={lat_col}, lng={lng_col}, route={route_col}, borough={borough_col}"
        )
        if lat_col is None or lng_col is None:
            print("Cannot proceed: no lat/lng columns")
            return

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

        print("Running dedup ...")
        dedup_df = con.execute(dedup_sql).fetchdf()
        print(f"deduplicated rows: {len(dedup_df)}")

        # Wipe stale transit_bus_stops first to avoid mixing seed + real data.
        con.execute("DELETE FROM transit_bus_stops")
    finally:
        con.close()

    # Match neighborhoods + insert.
    nb_df = pipeline._neighborhoods_df()  # type: ignore[attr-defined]
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
                "source_file": "MTA_Bus_Stops_20260424.csv",
            }
        )
    append_rows("transit_bus_stops", pd.DataFrame(rows), DB_PATH)
    print(f"appended {len(rows)} unique bus stops")

    print("Rebuilding processed tables ...")
    summary = pipeline.rebuild_processed_tables()
    print(f"  neighborhood_stats rows: {summary['rows']}")
    for w in summary["warnings"]:
        print(f"  warning: {w}")


if __name__ == "__main__":
    main()
