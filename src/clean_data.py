"""Cleaning pipeline and derived-field construction.

Implements the conservative cleaning policy documented in the root README and
``reports/data_quality_report.md``. The guiding rule throughout: never
silently delete observations, preserve raw values, and record data-quality
issues as flags rather than removing the rows they describe.

Two analytical datasets come out of this module:

1. ``build_primary_dataset`` -- the PRIMARY analytical dataset. Only exact
   full-row duplicates are removed. `(track, artist_1)` repeats are retained
   because this extract has no reliable track identifier that would let
   singles/albums/remasters/rereleases be distinguished from true duplicates.

2. ``build_sensitivity_dataset`` -- a secondary "one-record-per-track/artist
   sensitivity specification" built from the primary dataset by keeping one
   row per `(track, artist_1)` group, chosen by a rule that does NOT look at
   `popularity_score` (selecting the highest-popularity row would select on
   the outcome variable and bias results upward). This is a sensitivity
   check, not a "cleaner" or "canonical" dataset.

Popularity tier thresholds are not invented here -- they are recovered from
the data itself. The raw `popularity` categorical column turns out to be a
deterministic banding of `popularity_score`:
    0low       -> popularity_score in [0, 33]
    1moderate  -> popularity_score in [34, 66]
    2high      -> popularity_score in [67, 100]
This was established by grouping the raw data by `popularity` and taking the
min/max of `popularity_score` per group (see
`reports/data_quality_report.md`), not assumed in advance.
"""

from __future__ import annotations

import pandas as pd

# Popularity-tier thresholds recovered empirically from the raw `popularity`
# categorical column's relationship to `popularity_score` (see module
# docstring and the data-quality report). These are the dataset's own
# boundaries, not an analyst-invented business rule.
POPULARITY_TIER_BOUNDS = {
    "low": (0, 33),
    "moderate": (34, 66),
    "high": (67, 100),
}

# Threshold used for the `is_high_popularity` boolean convenience flag.
# Reuses the same boundary as the recovered "high" tier above rather than
# introducing a second, competing definition.
HIGH_POPULARITY_MIN = POPULARITY_TIER_BOUNDS["high"][0]


def _popularity_tier(score: float) -> str | float:
    if pd.isna(score):
        return pd.NA
    if score <= POPULARITY_TIER_BOUNDS["low"][1]:
        return "low"
    if score <= POPULARITY_TIER_BOUNDS["moderate"][1]:
        return "moderate"
    return "high"


def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add documented derived/quality-flag columns without removing rows.

    Adds:
        release_year               -- int copy of release_date, for clarity
        release_decade_clean       -- int decade derived from release_date
                                       (floor to nearest 10); falls back to
                                       the raw `release_decade` column when
                                       release_date is missing but
                                       release_decade is present
        track_duration_minutes     -- duration_ms / 60000
        popularity_tier            -- 'low' / 'moderate' / 'high', recovered
                                       thresholds (see module docstring)
        is_high_popularity         -- bool, popularity_score >= 67
        has_primary_genre          -- bool, artist_1_genre_1 not missing
        exact_duplicate_flag       -- bool, row belongs to an exact full-row
                                       duplicate group (marks ALL copies,
                                       including the first, so the flag alone
                                       identifies the group). In the PRIMARY
                                       dataset this is always False by
                                       construction, because
                                       build_primary_dataset() removes exact
                                       duplicates before calling this
                                       function -- call add_derived_fields()
                                       directly on a raw/pre-dedup frame to
                                       see the flag identify the group (used
                                       by the data-quality audit and tests)
        track_artist_duplicate_flag-- bool, (track, artist_1) appears more
                                       than once in the dataset
        tempo_quality_flag         -- 'ok' / 'zero_tempo' / 'missing'; raw
                                       tempo value is preserved unchanged
        suspicious_release_date_flag -- bool, release_date == 1899 (the
                                       dataset's most common implausible-
                                       looking placeholder year); not treated
                                       as proven-invalid
    """
    df = df.copy()

    df["release_year"] = df["release_date"].astype("Int64")

    derived_decade = (df["release_date"] // 10 * 10)
    df["release_decade_clean"] = derived_decade.combine_first(
        df["release_decade"]
    ).astype("Int64")

    df["track_duration_minutes"] = df["duration_ms"] / 60_000

    df["popularity_tier"] = df["popularity_score"].apply(_popularity_tier)
    df["is_high_popularity"] = df["popularity_score"] >= HIGH_POPULARITY_MIN

    df["has_primary_genre"] = df["artist_1_genre_1"].notna()

    df["exact_duplicate_flag"] = df.duplicated(keep=False)

    df["track_artist_duplicate_flag"] = df.duplicated(
        subset=["track", "artist_1"], keep=False
    )

    df["tempo_quality_flag"] = "ok"
    df.loc[df["tempo"].isna(), "tempo_quality_flag"] = "missing"
    df.loc[df["tempo"] == 0, "tempo_quality_flag"] = "zero_tempo"

    df["suspicious_release_date_flag"] = df["release_date"] == 1899

    return df


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove exact full-row duplicates, keeping the first occurrence.

    Operates on the RAW column set only (ignores any already-added derived
    columns) so that a duplicate is judged by original observed values, not
    by derived fields that are themselves computed from those values.

    Returns (deduplicated_df, stats) where stats records how many groups/
    rows were identified and removed.
    """
    raw_cols = [c for c in df.columns if c not in _DERIVED_COLUMNS]
    is_dup_any = df.duplicated(subset=raw_cols, keep=False)
    is_dup_remove = df.duplicated(subset=raw_cols, keep="first")
    stats = {
        "rows_involved_in_duplicate_groups": int(is_dup_any.sum()),
        "rows_removed": int(is_dup_remove.sum()),
        "row_count_before": len(df),
        "row_count_after": len(df) - int(is_dup_remove.sum()),
    }
    return df.loc[~is_dup_remove].reset_index(drop=True), stats


_DERIVED_COLUMNS = [
    "release_year",
    "release_decade_clean",
    "track_duration_minutes",
    "popularity_tier",
    "is_high_popularity",
    "has_primary_genre",
    "exact_duplicate_flag",
    "track_artist_duplicate_flag",
    "tempo_quality_flag",
    "suspicious_release_date_flag",
]


def build_primary_dataset(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build the PRIMARY analytical dataset from a raw-loaded DataFrame.

    Policy: remove exact full-row duplicates only. Retain `(track,
    artist_1)` repeats (see module docstring). Add all documented derived
    fields and quality flags.

    Returns (primary_df, dedup_stats).
    """
    deduped, stats = drop_exact_duplicates(raw_df)
    primary = add_derived_fields(deduped)
    return primary, stats


def build_sensitivity_dataset(primary_df: pd.DataFrame) -> pd.DataFrame:
    """Build the one-record-per-track/artist sensitivity specification.

    Selection rule: a documented **stable sort by (track, artist_1,
    release_date, album_type)** followed by keeping the first row per
    (track, artist_1) group. This rule does not reference
    `popularity_score` (or anything derived from it) anywhere, so it cannot
    select on the outcome variable.

    This dataset is a SENSITIVITY CHECK, not a canonical or "clean truth"
    dataset -- see reports/data_quality_report.md for how its KPIs compare
    to the primary dataset.

    `track_artist_duplicate_flag` is recomputed fresh on the resulting
    dataset (not carried over from `primary_df`) -- by construction, one
    record per (track, artist_1) means this flag must be all-False here.
    The value inherited from `primary_df` would otherwise describe each
    row's duplicate status in the PRIMARY dataset, which is stale and
    incorrect once read as a statement about the sensitivity dataset (the
    row survived precisely because its group was deduplicated).
    """
    sort_keys = ["track", "artist_1", "release_date", "album_type"]
    stable_sorted = primary_df.sort_values(
        by=sort_keys, kind="stable", na_position="last"
    )
    sensitivity = stable_sorted.drop_duplicates(
        subset=["track", "artist_1"], keep="first"
    ).reset_index(drop=True)
    sensitivity["track_artist_duplicate_flag"] = sensitivity.duplicated(
        subset=["track", "artist_1"], keep=False
    )
    return sensitivity
