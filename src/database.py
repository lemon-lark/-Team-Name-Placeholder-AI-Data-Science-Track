"""DuckDB connection helpers and safe query execution."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from src.config import DB_PATH, ensure_directories
from src.schema import TABLES, get_create_statements


def get_connection(db_path: Optional[Path] = None) -> duckdb.DuckDBPyConnection:
    """Open (or create) a DuckDB database file."""
    ensure_directories()
    target = Path(db_path) if db_path else DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(target))


def initialize_database(db_path: Optional[Path] = None) -> None:
    """Create every canonical table if it is missing.

    Also runs idempotent ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` migrations
    so older DuckDB files pick up new schema columns (e.g. apartment-level
    transit distances) without forcing a rebuild.
    """
    con = get_connection(db_path)
    try:
        for stmt in get_create_statements():
            con.execute(stmt)
        for table, columns in TABLES.items():
            for col, dtype in columns.items():
                con.execute(
                    f'ALTER TABLE "{table}" ADD COLUMN IF NOT EXISTS "{col}" {dtype}'
                )
    finally:
        con.close()


def get_available_tables(db_path: Optional[Path] = None) -> list[str]:
    """Return tables that exist in the DuckDB file."""
    con = get_connection(db_path)
    try:
        rows = con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


def get_table_row_count(table: str, db_path: Optional[Path] = None) -> int:
    """Return the row count for a table, or 0 if it does not exist."""
    if table not in TABLES:
        return 0
    con = get_connection(db_path)
    try:
        existing = {r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()}
        if table not in existing:
            return 0
        return int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    finally:
        con.close()


def run_query(sql: str, db_path: Optional[Path] = None) -> pd.DataFrame:
    """Execute a (presumed safe) read-only SQL statement and return a DataFrame.

    The SQL safety agent should validate the query BEFORE this is called. This
    helper still wraps the call in a try/finally and lets the caller surface
    any errors with a friendly message.
    """
    con = get_connection(db_path)
    try:
        return con.execute(sql).fetchdf()
    finally:
        con.close()


def replace_table(
    table: str,
    df: pd.DataFrame,
    db_path: Optional[Path] = None,
) -> int:
    """Replace a table's contents with the rows in ``df``.

    Used by the seed/ingestion code paths. Aligns the dataframe columns to the
    canonical schema before inserting so missing columns become NULL.
    """
    if table not in TABLES:
        raise ValueError(f"Unknown table: {table}")
    columns = list(TABLES[table].keys())

    aligned = df.copy()
    for col in columns:
        if col not in aligned.columns:
            aligned[col] = None
    aligned = aligned[columns]

    con = get_connection(db_path)
    try:
        for stmt in get_create_statements():
            con.execute(stmt)
        con.register("incoming_df", aligned)
        con.execute(f'DELETE FROM "{table}"')
        col_list = ", ".join(f'"{c}"' for c in columns)
        con.execute(
            f'INSERT INTO "{table}" ({col_list}) SELECT {col_list} FROM incoming_df'
        )
        con.unregister("incoming_df")
        return len(aligned)
    finally:
        con.close()


def append_rows(
    table: str,
    df: pd.DataFrame,
    db_path: Optional[Path] = None,
) -> int:
    """Append rows to a table without truncating it."""
    if table not in TABLES:
        raise ValueError(f"Unknown table: {table}")
    columns = list(TABLES[table].keys())

    aligned = df.copy()
    for col in columns:
        if col not in aligned.columns:
            aligned[col] = None
    aligned = aligned[columns]

    con = get_connection(db_path)
    try:
        for stmt in get_create_statements():
            con.execute(stmt)
        con.register("incoming_df", aligned)
        col_list = ", ".join(f'"{c}"' for c in columns)
        con.execute(
            f'INSERT INTO "{table}" ({col_list}) SELECT {col_list} FROM incoming_df'
        )
        con.unregister("incoming_df")
        return len(aligned)
    finally:
        con.close()
