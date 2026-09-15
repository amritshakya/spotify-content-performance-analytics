"""Tests for src/metrics.py.

These tests check each metric's output against an independent manual
recomputation from the same data, rather than re-asserting the
implementation's own arithmetic -- so a bug that silently changed a
formula's grain or exclusions would be caught.
"""

import pandas as pd

from src.metrics import (
    high_popularity_share,
    kpi_summary,
    median_artist_popularity,
    median_popularity_score,
    median_track_duration_minutes,
    primary_genre_coverage,
    repeat_track_artist_share,
    track_observation_count,
)


def test_track_observation_count_matches_len(primary_df):
    assert track_observation_count(primary_df) == len(primary_df)


def test_median_popularity_score_is_within_observed_range(primary_df):
    median = median_popularity_score(primary_df)
    assert primary_df["popularity_score"].min() <= median <= primary_df["popularity_score"].max()
    assert median == primary_df["popularity_score"].median()


def test_high_popularity_share_matches_independent_threshold_calc(primary_df):
    manual = float((primary_df["popularity_score"] >= 67).mean())
    assert high_popularity_share(primary_df) == manual


def test_primary_genre_coverage_matches_independent_calc(primary_df):
    manual = float(primary_df["artist_1_genre_1"].notna().mean())
    assert primary_genre_coverage(primary_df) == manual


def test_repeat_track_artist_share_matches_independent_calc(primary_df):
    manual = float(
        primary_df.duplicated(subset=["track", "artist_1"], keep=False).mean()
    )
    assert repeat_track_artist_share(primary_df) == manual


def test_median_track_duration_matches_manual_calc(primary_df):
    manual = float((primary_df["duration_ms"] / 60_000).median())
    assert median_track_duration_minutes(primary_df) == manual


def test_median_artist_popularity_is_bounded_0_1(primary_df):
    med = median_artist_popularity(primary_df)
    assert 0.0 <= med <= 1.0


def test_kpi_summary_contains_all_core_metrics(primary_df):
    summary = kpi_summary(primary_df)
    expected_keys = {
        "track_observation_count",
        "median_popularity_score",
        "mean_popularity_score",
        "high_popularity_share",
        "primary_genre_coverage",
        "median_track_duration_minutes",
        "median_artist_popularity",
        "repeat_track_artist_share",
    }
    assert set(summary.keys()) == expected_keys


def test_metrics_work_without_derived_columns():
    """Metrics that have a fallback path (no derived flag columns present)
    must still compute correctly from raw columns alone."""
    df = pd.DataFrame(
        {
            "track": ["a", "a", "b"],
            "artist_1": ["x", "x", "y"],
            "popularity_score": [10, 90, 50],
            "artist_1_genre_1": [None, "pop", "rock"],
            "artist_1_pop": [0.1, 0.9, 0.5],
            "duration_ms": [60_000, 120_000, 180_000],
        }
    )
    assert high_popularity_share(df) == 1 / 3  # only the 90 qualifies
    assert primary_genre_coverage(df) == 2 / 3
    assert repeat_track_artist_share(df) == 2 / 3  # rows 0 and 1 repeat


def test_share_metrics_are_fractions_between_0_and_1(primary_df):
    summary = kpi_summary(primary_df)
    for key in [
        "high_popularity_share",
        "primary_genre_coverage",
        "repeat_track_artist_share",
    ]:
        assert 0.0 <= summary[key] <= 1.0
