"""Documented KPI / metric layer for the Spotify Content Performance
Analytics project.

Every function here computes exactly one metric and carries its full
definition (formula, grain, exclusions, interpretation, caveats) in its
docstring, per the project's metric-definition standard. `kpi_summary()`
bundles them for convenience (used by the notebook's primary-vs-sensitivity
KPI comparison).

Global caveats that apply to every metric in this module, restated here so
they travel with the code and not just the README:

- This is a fixed, non-random-sampling-frame extract. Every percentage/share
  value returned by this module describes "tracks in this dataset," never
  "Spotify tracks" generally.
- `popularity_score` is a snapshot at an unknown collection time. Metrics
  built on it describe popularity "at the dataset's collection time," not a
  track's popularity in general or over time.
- `artist_1_pop` is entangled with track-level popularity (popular artists
  tend to have popular tracks, and vice versa via the same underlying
  algorithmic signals) -- it is reported as a descriptive figure, never as
  an independent predictor of `popularity_score`.
"""

from __future__ import annotations

import pandas as pd


def track_observation_count(df: pd.DataFrame) -> int:
    """Track Observation Count.

    Formula: COUNT(*)
    Grain: one row per track observation in the given dataset.
    Exclusions: none -- reflects whatever dataset (primary or sensitivity)
        is passed in.
    Interpretation: the size of the analytical dataset. Not a count of
        unique songs -- `(track, artist_1)` repeats are retained in the
        primary dataset (see cleaning policy), so this number can exceed
        the count of distinct track/artist pairs.
    Caveats: not a census of Spotify's catalog; specific to this extract.
    """
    return int(len(df))


def median_popularity_score(df: pd.DataFrame) -> float:
    """Median Popularity Score.

    Formula: MEDIAN(popularity_score)
    Grain: one row per track observation.
    Exclusions: rows with missing popularity_score (none observed in this
        extract, but NaNs are excluded by pandas' median by default).
    Interpretation: the preferred central-tendency summary for this metric
        because the distribution is right-of-center skewed with a large
        mass at 0 (see reports/data_quality_report.md) -- the mean is more
        sensitive to that skew.
    Caveats: reflects popularity at the dataset's unknown collection time,
        not a track's all-time or current popularity.
    """
    return float(df["popularity_score"].median())


def mean_popularity_score(df: pd.DataFrame) -> float:
    """Mean Popularity Score (secondary comparison to the median).

    Formula: MEAN(popularity_score)
    Grain: one row per track observation.
    Exclusions: rows with missing popularity_score.
    Interpretation: reported alongside the median, not in place of it,
        because of distribution skew. A meaningfully larger mean than
        median indicates a right-skewing high-popularity tail.
    Caveats: same collection-time snapshot caveat as median_popularity_score.
    """
    return float(df["popularity_score"].mean())


def high_popularity_share(df: pd.DataFrame) -> float:
    """High-Popularity Share.

    Formula: COUNT(popularity_score >= 67) / COUNT(*)
    Grain: one row per track observation.
    Exclusions: none.
    Interpretation: share of tracks *in this dataset* in the recovered
        "high" popularity tier (see src/clean_data.py POPULARITY_TIER_BOUNDS
        for how this threshold was recovered from the data rather than
        assumed).
    Caveats: "share of tracks in this dataset," not a share of Spotify's
        catalog -- the sampling frame is not established as representative.
    """
    if "is_high_popularity" in df.columns:
        return float(df["is_high_popularity"].mean())
    return float((df["popularity_score"] >= 67).mean())


def primary_genre_coverage(df: pd.DataFrame) -> float:
    """Primary Genre Coverage.

    Formula: COUNT(artist_1_genre_1 IS NOT NULL) / COUNT(*)
    Grain: one row per track observation.
    Exclusions: none.
    Interpretation: the fraction of tracks in this dataset with an observed
        primary-artist genre. Genre-conditioned analyses elsewhere in this
        project only cover this fraction of the data.
    Caveats: genre availability is NOT independent of popularity in this
        dataset -- tracks with an observed genre have systematically higher
        median/mean popularity_score and higher artist_1_pop than tracks
        without one (see the genre-availability bias analysis in
        reports/data_quality_report.md). Any genre-based conclusion
        elsewhere is conditioned on "tracks in this dataset with an observed
        primary genre," not all tracks.
    """
    if "has_primary_genre" in df.columns:
        return float(df["has_primary_genre"].mean())
    return float(df["artist_1_genre_1"].notna().mean())


def median_track_duration_minutes(df: pd.DataFrame) -> float:
    """Median Track Duration (minutes).

    Formula: MEDIAN(duration_ms) / 60000
    Grain: one row per track observation.
    Exclusions: rows with missing duration_ms (none observed in this
        extract).
    Interpretation: uses duration_ms (continuous, raw) rather than the
        course-provided binned `duration` category, to avoid re-deriving a
        median from pre-binned categorical levels.
    Caveats: none beyond the general dataset-scope caveat.
    """
    if "track_duration_minutes" in df.columns:
        return float(df["track_duration_minutes"].median())
    return float((df["duration_ms"] / 60_000).median())


def median_artist_popularity(df: pd.DataFrame) -> float:
    """Median Artist Popularity (primary artist).

    Formula: MEDIAN(artist_1_pop)
    Grain: one row per track observation (i.e., this is a track-level
        median of the primary artist's popularity score, NOT a median
        computed over distinct artists -- an artist with many tracks in the
        dataset is implicitly weighted by its track count).
    Exclusions: rows with missing artist_1_pop (none observed in this
        extract).
    Interpretation: descriptive only.
    Caveats: `artist_1_pop` is entangled with track-level popularity and is
        never treated as an independent predictor of `popularity_score` in
        this project (see module docstring).
    """
    return float(df["artist_1_pop"].median())


def repeat_track_artist_share(df: pd.DataFrame) -> float:
    """Repeat Track/Artist Share.

    Formula: COUNT(rows where (track, artist_1) appears more than once in
        this dataset) / COUNT(*)
    Grain: ROW-level share -- every row that belongs to a repeated
        `(track, artist_1)` group counts, not a per-group count. A group of
        3 repeats contributes 3 to the numerator, not 1.
    Exclusions: none.
    Interpretation: describes how much of the primary dataset consists of
        `(track, artist_1)` combinations that appear more than once (singles
        / album cuts / compilations / remasters / rereleases -- this extract
        has no reliable identifier to distinguish these cases). This is why
        the primary dataset retains repeats rather than deduplicating them;
        see the sensitivity dataset for the one-record-per-track/artist
        comparison.
    Caveats: "share of rows in this dataset," not an estimate of true
        catalog-wide re-release rates.
    """
    if "track_artist_duplicate_flag" in df.columns:
        return float(df["track_artist_duplicate_flag"].mean())
    dup_mask = df.duplicated(subset=["track", "artist_1"], keep=False)
    return float(dup_mask.mean())


def kpi_summary(df: pd.DataFrame) -> dict:
    """Compute all core KPIs for a given dataset and return them as a dict.

    Used to build the primary-vs-sensitivity KPI comparison table in
    reports/data_quality_report.md and notebooks/01_data_quality_and_cleaning.ipynb.
    """
    return {
        "track_observation_count": track_observation_count(df),
        "median_popularity_score": median_popularity_score(df),
        "mean_popularity_score": mean_popularity_score(df),
        "high_popularity_share": high_popularity_share(df),
        "primary_genre_coverage": primary_genre_coverage(df),
        "median_track_duration_minutes": median_track_duration_minutes(df),
        "median_artist_popularity": median_artist_popularity(df),
        "repeat_track_artist_share": repeat_track_artist_share(df),
    }
