"""Build the local DuckDB database used by the SQL analytics layer.

Reads the PRIMARY analytical dataset (and, for the sensitivity comparison,
the one-record-per-track/artist dataset) and materializes them as DuckDB
tables. The database file itself is gitignored -- it is a reproducible build
artifact, not source-controlled state.

Run directly to (re)build the database from scratch:

    python -m src.build_database

This will:
    1. load + validate the raw CSV (src.load_data)
    2. build the primary and sensitivity datasets (src.clean_data)
    3. write both to data/processed/ as Parquet (for reuse without re-parsing
       the 47MB raw CSV every time)
    4. load both into a fresh DuckDB database file
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.clean_data import build_primary_dataset, build_sensitivity_dataset
from src.load_data import load_raw

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_DB_PATH = PROCESSED_DIR / "spotify.duckdb"

PRIMARY_PARQUET_PATH = PROCESSED_DIR / "primary.parquet"
SENSITIVITY_PARQUET_PATH = PROCESSED_DIR / "sensitivity.parquet"

PRIMARY_TABLE = "tracks_primary"
SENSITIVITY_TABLE = "tracks_sensitivity"


def build_processed_datasets(
    raw_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the full loader -> cleaning pipeline and return both datasets.

    Returns (primary_df, sensitivity_df, dedup_stats).
    """
    raw = load_raw(path=raw_path) if raw_path else load_raw()
    primary, dedup_stats = build_primary_dataset(raw)
    sensitivity = build_sensitivity_dataset(primary)
    return primary, sensitivity, dedup_stats


def write_processed_parquet(
    primary: pd.DataFrame, sensitivity: pd.DataFrame
) -> None:
    """Persist both datasets as Parquet under data/processed/ (gitignored)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    primary.to_parquet(PRIMARY_PARQUET_PATH, index=False)
    sensitivity.to_parquet(SENSITIVITY_PARQUET_PATH, index=False)


def build_database(
    primary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    db_path: Path = DEFAULT_DB_PATH,
) -> duckdb.DuckDBPyConnection:
    """Create (overwriting) a DuckDB database containing both tables.

    Returns an open connection to the built database (caller should close
    it, or use it immediately and let it go out of scope).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    con = duckdb.connect(str(db_path))
    con.register("primary_df", primary)
    con.register("sensitivity_df", sensitivity)
    con.execute(f"CREATE TABLE {PRIMARY_TABLE} AS SELECT * FROM primary_df")
    con.execute(
        f"CREATE TABLE {SENSITIVITY_TABLE} AS SELECT * FROM sensitivity_df"
    )
    con.unregister("primary_df")
    con.unregister("sensitivity_df")
    return con


def main() -> None:
    print("Loading raw data and building analytical datasets...")
    primary, sensitivity, dedup_stats = build_processed_datasets()
    print(f"  primary dataset:     {len(primary):,} rows")
    print(f"  sensitivity dataset: {len(sensitivity):,} rows")
    print(f"  exact-duplicate dedup stats: {dedup_stats}")

    print("Writing processed Parquet files...")
    write_processed_parquet(primary, sensitivity)
    print(f"  wrote {PRIMARY_PARQUET_PATH}")
    print(f"  wrote {SENSITIVITY_PARQUET_PATH}")

    print("Building DuckDB database...")
    con = build_database(primary, sensitivity)
    n_primary = con.execute(f"SELECT COUNT(*) FROM {PRIMARY_TABLE}").fetchone()[0]
    n_sensitivity = con.execute(
        f"SELECT COUNT(*) FROM {SENSITIVITY_TABLE}"
    ).fetchone()[0]
    con.close()
    print(f"  wrote {DEFAULT_DB_PATH}")
    print(f"  {PRIMARY_TABLE}: {n_primary:,} rows")
    print(f"  {SENSITIVITY_TABLE}: {n_sensitivity:,} rows")


if __name__ == "__main__":
    main()
