"""Tests for src/load_data.py."""

import shutil

import pandas as pd
import pytest

from src.load_data import (
    EXPECTED_COLUMNS,
    EXPECTED_ROW_COUNT,
    EXPECTED_SHA256,
    DEFAULT_RAW_PATH,
    HashMismatchError,
    SchemaError,
    compute_sha256,
    load_raw,
    malformed_row_report,
    validate_raw_file_hash,
)


def test_raw_loader_returns_expected_row_count(raw_df):
    assert len(raw_df) == EXPECTED_ROW_COUNT == 277_938


def test_raw_loader_returns_34_raw_columns(raw_df):
    assert raw_df.shape[1] == 34
    assert list(raw_df.columns) == EXPECTED_COLUMNS


def test_required_columns_exist(raw_df):
    for col in [
        "track",
        "artist_1",
        "artist_1_pop",
        "artist_1_genre_1",
        "release_date",
        "popularity",
        "popularity_score",
        "album_type",
        "tempo",
        "duration_ms",
        "time_signature",
    ]:
        assert col in raw_df.columns


def test_raw_file_sha256_equals_documented_hash():
    actual = validate_raw_file_hash(DEFAULT_RAW_PATH, EXPECTED_SHA256)
    assert actual == EXPECTED_SHA256


def test_hash_mismatch_raises_on_tampered_file(tmp_path):
    tampered = tmp_path / "spotify_tampered.csv"
    shutil.copy(DEFAULT_RAW_PATH, tampered)
    with open(tampered, "a") as f:
        f.write("\nnot,a,real,row\n")
    with pytest.raises(HashMismatchError):
        validate_raw_file_hash(tampered, EXPECTED_SHA256)


def test_load_raw_rejects_wrong_row_count(tmp_path):
    """A CSV with the right columns but wrong row count should fail shape
    validation rather than being silently accepted."""
    bad_csv = tmp_path / "spotify_short.csv"
    header = ",".join(EXPECTED_COLUMNS)
    type_row = ",".join(["string"] * len(EXPECTED_COLUMNS))
    data_row = ",".join([""] * len(EXPECTED_COLUMNS))
    bad_csv.write_text(f"{header}\n{type_row}\n{data_row}\n")
    with pytest.raises(SchemaError):
        load_raw(path=bad_csv, validate_hash=False, validate_shape=True)


def test_popularity_score_within_0_100(raw_df):
    scores = raw_df["popularity_score"].dropna()
    assert scores.min() >= 0
    assert scores.max() <= 100


def test_raw_file_never_altered_by_loading(raw_df):
    """Loading must be read-only: the file's hash must be unchanged after
    load_raw() has run (raw_df is produced by the session fixture)."""
    assert compute_sha256(DEFAULT_RAW_PATH) == EXPECTED_SHA256


def test_type_declaration_row_is_skipped_not_loaded_as_data(raw_df):
    """Row 2 of the file ('string,string,numeric,...') must never appear as
    an observation -- e.g. popularity_score must be numeric everywhere, not
    contain the literal string 'numeric'."""
    assert pd.api.types.is_numeric_dtype(raw_df["popularity_score"])


def test_malformed_row_report_finds_no_malformed_rows():
    report = malformed_row_report(DEFAULT_RAW_PATH)
    assert report["count"] == 0
