"""Profile a CSV/XLSX before ingestion: rows, columns, missing values, warnings."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from src.ingestion.column_mapper import (
    detect_column,
    detect_dataset_type_from_filename,
    map_columns,
)


PathLike = Union[str, Path]


def _load_sample(file_path: PathLike, sample_rows: int = 200) -> tuple[pd.DataFrame, int, int]:
    """Read a small head sample plus the full row count for the file."""
    p = Path(file_path)
    suffix = p.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(p, nrows=sample_rows)
        full = pd.read_excel(p, usecols=[df.columns[0]] if len(df.columns) else None)
        full_rows = int(len(full))
        full_cols = int(len(df.columns))
        return df, full_rows, full_cols
    # CSV path - try DuckDB for fast row counting.
    try:
        import duckdb

        con = duckdb.connect(":memory:")
        con.execute(
            f"CREATE OR REPLACE VIEW raw_view AS SELECT * FROM read_csv_auto(?, sample_size=-1)",
            [str(p)],
        )
        full_rows = int(con.execute("SELECT COUNT(*) FROM raw_view").fetchone()[0])
        df = con.execute(f"SELECT * FROM raw_view LIMIT {sample_rows}").fetchdf()
        full_cols = int(len(df.columns))
        con.close()
        return df, full_rows, full_cols
    except Exception:
        # Fall back to pandas chunked read.
        df = pd.read_csv(p, nrows=sample_rows, low_memory=False)
        try:
            full_rows = int(sum(1 for _ in open(p, encoding="utf-8", errors="replace"))) - 1
        except Exception:
            full_rows = -1
        return df, max(0, full_rows), int(len(df.columns))


def profile_file(
    file_path: PathLike,
    dataset_type: Optional[str] = None,
    sample_rows: int = 200,
) -> dict[str, Any]:
    """Inspect a raw CSV/XLSX and return a profiling report dict."""
    p = Path(file_path)
    if not p.exists():
        return {"error": f"File not found: {p}"}

    detected_dt = detect_dataset_type_from_filename(p.name)
    dataset = (dataset_type or detected_dt or "").lower() or None

    try:
        df, total_rows, total_cols = _load_sample(p, sample_rows=sample_rows)
    except Exception as exc:
        return {
            "file_name": p.name,
            "error": f"Could not read file: {exc}",
            "dataset_type": dataset,
        }

    column_map = map_columns(df, dataset or "amenities")
    important_columns = {k: v for k, v in column_map.items() if v}

    has_lat = bool(detect_column(df, ["lat", "latitude", "y"]))
    has_lng = bool(detect_column(df, ["lng", "lon", "longitude", "x"]))
    has_neighborhood = bool(
        detect_column(df, ["neighborhood", "nta_name", "ntaname", "community", "area", "nta"])
    )

    missing_summary = {
        col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()
    }
    sample_records = df.head(10).to_dict(orient="records")

    warnings: list[str] = []
    if not has_lat or not has_lng:
        warnings.append("No latitude/longitude columns found.")
    if not has_neighborhood:
        warnings.append("No neighborhood column found - rows will be matched by lat/lng.")
    rent_col = detect_column(df, ["rent", "price", "monthly_rent"])
    if rent_col is not None and not pd.api.types.is_numeric_dtype(df[rent_col]):
        warnings.append("Rent column detected but contains non-numeric values; cleaner will parse.")
    if dataset == "crime":
        date_col = detect_column(df, ["date", "report_date", "incident_date"])
        sev_col = detect_column(df, ["severity", "law_cat_cd", "category"])
        if date_col and not sev_col:
            warnings.append("Crime data has dates but no severity column; severity will be inferred.")
    if dataset == "transit_bus" and total_rows > 1_000_000:
        warnings.append(
            "Transit file has over 1 million rows; deduplication is required and handled by the pipeline."
        )
    if dataset == "housing":
        warnings.append("HPD building data is a housing-stock signal, not direct vacancy.")

    return {
        "file_name": p.name,
        "file_path": str(p),
        "dataset_type": dataset,
        "detected_dataset_type": detected_dt,
        "row_count": int(total_rows),
        "column_count": int(total_cols),
        "columns": list(df.columns),
        "important_columns": important_columns,
        "has_lat_lng": has_lat and has_lng,
        "has_neighborhood": has_neighborhood,
        "missing_value_summary": missing_summary,
        "sample_rows": sample_records,
        "warnings": warnings,
    }
