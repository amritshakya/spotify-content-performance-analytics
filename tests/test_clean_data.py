"""Tests for src/clean_data.py."""

import numpy as np
import pandas as pd

from src.clean_data import (
    HIGH_POPULARITY_MIN,
    POPULARITY_TIER_BOUNDS,
    add_derived_fields,
    build_primary_dataset,
    build_sensitivity_dataset,
    drop_exact_duplicates,
)


def test_add_derived_fields_preserves_row_count(raw_df):
    """Adding flags/derived columns must never drop or add rows -- only
    dropping exact duplicates (a separate, explicit step) may change count."""
    flagged = add_derived_fields(raw_df)
    assert len(flagged) == len(raw_df)


def test_exact_duplicate_flag_identifies_known_duplicates(raw_df):
    flagged = add_derived_fields(raw_df)
    # The raw extract is known (via reports/data_quality_report.md) to
    # contain exactly one exact-duplicate pair (2 rows).
    assert flagged["exact_duplicate_flag"].sum() == 2


def test_drop_exact_duplicates_removes_exactly_the_known_duplicate(raw_df):
    deduped, stats = drop_exact_duplicates(raw_df)
    assert stats["rows_removed"] == 1
    assert stats["row_count_before"] == len(raw_df)
    assert stats["row_count_after"] == len(raw_df) - 1
    assert len(deduped) == len(raw_df) - 1
    # No exact duplicates should remain.
    assert deduped.duplicated(keep=False).sum() == 0


def test_primary_dataset_exact_duplicate_flag_is_all_false(primary_df):
    """By construction, build_primary_dataset() removes exact duplicates
    before computing flags, so no row in the primary dataset should ever be
    flagged as an exact duplicate of another surviving row."""
    assert primary_df["exact_duplicate_flag"].sum() == 0


def test_track_artist_repeats_remain_in_primary_dataset(primary_df):
    """(track, artist_1) repeats must be RETAINED, not deduplicated, in the
    primary analytical dataset."""
    dup_mask = primary_df.duplicated(subset=["track", "artist_1"], keep=False)
    assert dup_mask.sum() > 0
    # The flag column should agree with a fresh recomputation.
    assert (primary_df["track_artist_duplicate_flag"] == dup_mask).all()


def test_sensitivity_dataset_has_one_record_per_track_artist(sensitivity_df):
    dup_count = sensitivity_df.duplicated(subset=["track", "artist_1"]).sum()
    assert dup_count == 0


def test_sensitivity_dataset_is_strict_subset_of_primary_by_row_count(
    primary_df, sensitivity_df
):
    assert len(sensitivity_df) <= len(primary_df)
    assert len(sensitivity_df) > 0


def test_sensitivity_selection_does_not_use_popularity():
    """Construct a synthetic (track, artist_1) group where the highest
    popularity row is NOT the one that would win a stable sort on
    (track, artist_1, release_date, album_type). The sensitivity dataset
    must keep the sort-order winner, not the highest-popularity row --
    proving selection does not look at popularity_score."""
    df = pd.DataFrame(
        {
            "track": ["Song A", "Song A"],
            "artist_1": ["Artist X", "Artist X"],
            "release_date": [2010.0, 2005.0],  # row 1 (index 1) sorts first
            "album_type": ["single", "album"],
            "popularity_score": [99, 1],  # row 0 has the higher popularity
            "artist_1_pop": [0.5, 0.5],
            "artist_1_genre_1": [None, None],
            "duration_ms": [200_000, 200_000],
        }
    )
    result = build_sensitivity_dataset(df)
    assert len(result) == 1
    # The kept row must be the one with the earlier release_date (2005),
    # i.e. popularity_score == 1, NOT the higher-popularity row (99).
    assert result.iloc[0]["popularity_score"] == 1


def test_tempo_quality_flag_captures_known_zero_tempo_records(raw_df):
    flagged = add_derived_fields(raw_df)
    n_zero = (raw_df["tempo"] == 0).sum()
    assert n_zero == 90  # documented in reports/data_quality_report.md
    assert (flagged["tempo_quality_flag"] == "zero_tempo").sum() == n_zero
    # Raw tempo value must be preserved unchanged, not overwritten with NaN.
    zero_rows = flagged["tempo_quality_flag"] == "zero_tempo"
    assert (flagged.loc[zero_rows, "tempo"] == 0).all()


def test_suspicious_release_date_flag_behaves_consistently(raw_df):
    flagged = add_derived_fields(raw_df)
    expected = raw_df["release_date"] == 1899
    assert (flagged["suspicious_release_date_flag"] == expected).all()
    # Rows must be preserved, not dropped.
    assert flagged["suspicious_release_date_flag"].sum() == 13


def test_popularity_tier_thresholds_match_recovered_bounds():
    assert POPULARITY_TIER_BOUNDS["low"] == (0, 33)
    assert POPULARITY_TIER_BOUNDS["moderate"] == (34, 66)
    assert POPULARITY_TIER_BOUNDS["high"] == (67, 100)
    assert HIGH_POPULARITY_MIN == 67


def test_derived_fields_are_internally_consistent(primary_df):
    # is_high_popularity must exactly agree with popularity_tier == 'high'.
    assert (
        primary_df["is_high_popularity"] == (primary_df["popularity_tier"] == "high")
    ).all()
    # has_primary_genre must exactly agree with artist_1_genre_1 non-null.
    assert (
        primary_df["has_primary_genre"] == primary_df["artist_1_genre_1"].notna()
    ).all()
    # track_duration_minutes must be duration_ms / 60000.
    expected_minutes = primary_df["duration_ms"] / 60_000
    assert np.allclose(primary_df["track_duration_minutes"], expected_minutes)


def test_processed_dataset_has_no_impossible_popularity_values(primary_df):
    assert primary_df["popularity_score"].between(0, 100).all()


def test_build_primary_dataset_row_count(raw_df, primary_df, dedup_stats):
    assert len(primary_df) == len(raw_df) - dedup_stats["rows_removed"]
    assert dedup_stats["rows_removed"] == 1
