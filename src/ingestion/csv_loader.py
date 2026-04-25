"""Unified CSV/XLSX loader.

Strategy:
* Excel files (``.xlsx`` / ``.xls``) -> pandas.
* Small CSVs -> pandas.
* Large CSVs -> DuckDB ``read_csv_auto`` (much faster, handles weird dialects).
* Streamlit ``UploadedFile`` objects are written to a temp file first.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd

from src.config import RAW_DIR


PathLike = Union[str, Path]
LARGE_CSV_THRESHOLD_BYTES = 50 * 1024 * 1024  # 50 MB


def list_raw_files(extensions: Iterable[str] = (".csv", ".xlsx", ".xls")) -> list[Path]:
    """Recursively list raw files inside ``data/raw/``."""
    if not RAW_DIR.exists():
        return []
    files: list[Path] = []
    for ext in extensions:
        files.extend(RAW_DIR.rglob(f"*{ext}"))
    files = [f for f in files if f.is_file()]
    return sorted(files)


def load_dataframe(
    file_path: PathLike,
    sample_rows: Optional[int] = None,
    use_duckdb_for_large_csv: bool = True,
) -> pd.DataFrame:
    """Load a CSV/XLSX into a pandas DataFrame.

    Parameters
    ----------
    file_path : str | Path
        Path to the file.
    sample_rows : int | None
        If set, only read this many rows from the head.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        if sample_rows:
            return pd.read_excel(p, nrows=sample_rows)
        return pd.read_excel(p)

    if suffix == ".csv":
        if sample_rows:
            return pd.read_csv(p, nrows=sample_rows, low_memory=False)
        if use_duckdb_for_large_csv:
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            if size >= LARGE_CSV_THRESHOLD_BYTES:
                try:
                    import duckdb

                    con = duckdb.connect(":memory:")
                    df = con.execute(
                        "SELECT * FROM read_csv_auto(?, sample_size=-1)",
                        [str(p)],
                    ).fetchdf()
                    con.close()
                    return df
                except Exception:
                    pass
        return pd.read_csv(p, low_memory=False)

    raise ValueError(f"Unsupported file extension: {suffix}")


def save_uploaded_file(uploaded_file, destination_subfolder: str = "uploads") -> Path:
    """Persist a Streamlit ``UploadedFile`` to ``data/raw/<sub>/`` and return the path."""
    target_dir = RAW_DIR / destination_subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = os.path.basename(getattr(uploaded_file, "name", "upload.csv"))
    out = target_dir / safe_name
    with open(out, "wb") as fh:
        fh.write(uploaded_file.getbuffer() if hasattr(uploaded_file, "getbuffer") else uploaded_file.read())
    return out


def staged_csv_into_duckdb(
    file_path: PathLike,
    duck_con,
    staging_table: str,
) -> int:
    """Load a CSV directly into DuckDB as a staging table. Returns row count."""
    p = Path(file_path)
    duck_con.execute(
        f'CREATE OR REPLACE TABLE "{staging_table}" AS '
        f"SELECT * FROM read_csv_auto(?, sample_size=-1)",
        [str(p)],
    )
    return int(duck_con.execute(f'SELECT COUNT(*) FROM "{staging_table}"').fetchone()[0])
