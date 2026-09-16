"""Tests for src/build_database.py, plus a small number of targeted
DuckDB-query tests for sql/*.sql files that don't warrant their own test
module."""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.build_database import (
    PRIMARY_TABLE,
    SENSITIVITY_TABLE,
    build_database,
)


@pytest.fixture(scope="module")
def small_primary_df():
    return pd.DataFrame(
        {
            "track": ["a", "b", "c"],
            "artist_1": ["x", "y", "z"],
            "popularity_score": [10, 50, 90],
            "artist_1_genre_1": [None, "pop", "rock"],
        }
    )


@pytest.fixture(scope="module")
def small_sensitivity_df():
    return pd.DataFrame(
        {
            "track": ["a", "b"],
            "artist_1": ["x", "y"],
            "popularity_score": [10, 50],
            "artist_1_genre_1": [None, "pop"],
        }
    )


def test_database_builds_successfully(tmp_path, small_primary_df, small_sensitivity_df):
    db_path = tmp_path / "test.duckdb"
    con = build_database(small_primary_df, small_sensitivity_df, db_path=db_path)
    tables = {
        row[0]
        for row in con.execute("SELECT table_name FROM information_schema.tables").fetchall()
    }
    con.close()
    assert PRIMARY_TABLE in tables
    assert SENSITIVITY_TABLE in tables


def test_database_row_counts_match_input_dataframes(
    tmp_path, small_primary_df, small_sensitivity_df
):
    db_path = tmp_path / "test.duckdb"
    con = build_database(small_primary_df, small_sensitivity_df, db_path=db_path)
    n_primary = con.execute(f"SELECT COUNT(*) FROM {PRIMARY_TABLE}").fetchone()[0]
    n_sensitivity = con.execute(f"SELECT COUNT(*) FROM {SENSITIVITY_TABLE}").fetchone()[0]
    con.close()
    assert n_primary == len(small_primary_df)
    assert n_sensitivity == len(small_sensitivity_df)


def test_database_rebuild_overwrites_existing_file(
    tmp_path, small_primary_df, small_sensitivity_df
):
    db_path = tmp_path / "test.duckdb"
    con1 = build_database(small_primary_df, small_sensitivity_df, db_path=db_path)
    con1.close()

    shrunk = small_primary_df.iloc[:1]
    con2 = build_database(shrunk, small_sensitivity_df, db_path=db_path)
    n_primary = con2.execute(f"SELECT COUNT(*) FROM {PRIMARY_TABLE}").fetchone()[0]
    con2.close()
    assert n_primary == 1


def test_full_database_contains_no_impossible_popularity_values(
    tmp_path, primary_df, sensitivity_df
):
    """End-to-end check against the real primary/sensitivity datasets."""
    db_path = tmp_path / "full.duckdb"
    con = build_database(primary_df, sensitivity_df, db_path=db_path)
    min_score, max_score = con.execute(
        f"SELECT MIN(popularity_score), MAX(popularity_score) FROM {PRIMARY_TABLE}"
    ).fetchone()
    con.close()
    assert min_score >= 0
    assert max_score <= 100


def test_full_database_row_counts_match_processed_datasets(
    tmp_path, primary_df, sensitivity_df
):
    db_path = tmp_path / "full.duckdb"
    con = build_database(primary_df, sensitivity_df, db_path=db_path)
    n_primary = con.execute(f"SELECT COUNT(*) FROM {PRIMARY_TABLE}").fetchone()[0]
    n_sensitivity = con.execute(f"SELECT COUNT(*) FROM {SENSITIVITY_TABLE}").fetchone()[0]
    con.close()
    assert n_primary == len(primary_df)
    assert n_sensitivity == len(sensitivity_df)


def _extract_q10_statement() -> str:
    """Pull Q10's exact SELECT statement text out of
    sql/03_segment_analysis.sql (the last statement in the file), so this
    test protects the actual file content rather than a reimplementation
    of its logic."""
    sql_path = (
        Path(__file__).resolve().parent.parent / "sql" / "03_segment_analysis.sql"
    )
    sql_text = sql_path.read_text()
    lines = [l for l in sql_text.splitlines() if not l.strip().startswith("--")]
    statements = [s.strip() for s in "\n".join(lines).split(";") if s.strip()]
    return statements[-1] + ";"


def test_q10_release_year_boundary_behavior():
    """Regression guard for the release_decade_clean -> release_year fix:
    the literal cutoff must include exactly 2000 and earlier, exclude 2001
    and later, and exclude rows with an unknown release_year from both the
    numerator and the denominator."""
    con = duckdb.connect(":memory:")
    con.register(
        "tracks_primary",
        pd.DataFrame(
            {
                "album_type": ["album", "album", "album"],
                "release_year": [2000.0, 2001.0, None],
            }
        ),
    )
    result = con.execute(_extract_q10_statement()).fetchdf()
    con.close()

    assert len(result) == 1  # one row (album_type = 'album')
    row = result.iloc[0]
    # year 2000 and year 2001 rows both counted in the denominator; the
    # null-year row is excluded entirely (track_count == 2, not 3).
    assert row["track_count"] == 2
    # exactly the year-2000 row (1 of 2) counts as "released 2000 or
    # earlier" -- year 2001 must NOT be included.
    assert row["pct_released_2000_or_earlier"] == pytest.approx(50.0)
