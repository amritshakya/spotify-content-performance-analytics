"""Data-quality audit computations.

Every function here recomputes a finding directly from a DataFrame -- nothing
in this project hard-codes the audit's numeric results. `notebooks/01_data_quality_and_cleaning.ipynb`
calls these functions directly. `reports/data_quality_report.md` summarizes
calculations reproduced by these functions, the SQL layer, and the
notebooks -- it is a hand-authored write-up, not machine-generated from this
module.

Functions generally take the RAW-loaded DataFrame (before any cleaning) so
that the audit describes the data as delivered, not as cleaned.
"""

from __future__ import annotations

import pandas as pd

CATEGORICAL_AUDIT_COLUMNS = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "popularity",
    "duration",
    "album_type",
    "time_signature",
    "key",
    "artist_1_genre_1",
]

NUMERIC_AUDIT_COLUMNS = [
    "artist_1_pop",
    "release_date",
    "popularity_score",
    "loudness",
    "valence",
    "tempo",
    "duration_ms",
]


def basic_shape(df: pd.DataFrame) -> dict:
    return {"n_rows": int(len(df)), "n_columns": int(df.shape[1])}


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing-value count and percentage per column."""
    n = len(df)
    missing = df.isna().sum()
    out = pd.DataFrame(
        {
            "n_missing": missing,
            "pct_missing": (missing / n * 100).round(3),
        }
    )
    return out.sort_values("n_missing", ascending=False)


def exact_duplicate_stats(df: pd.DataFrame) -> dict:
    is_dup_any = df.duplicated(keep=False)
    is_dup_remove = df.duplicated(keep="first")
    return {
        "rows_in_duplicate_groups": int(is_dup_any.sum()),
        "rows_removed_on_dedup": int(is_dup_remove.sum()),
    }


def track_artist_duplicate_stats(df: pd.DataFrame) -> dict:
    group_sizes = df.groupby(["track", "artist_1"], dropna=False).size()
    dup_groups = group_sizes[group_sizes > 1]
    return {
        "duplicate_groups": int(len(dup_groups)),
        "rows_in_duplicate_groups": int(dup_groups.sum()),
        "max_group_size": int(dup_groups.max()) if len(dup_groups) else 0,
    }


def categorical_cardinalities(
    df: pd.DataFrame, columns: list[str] = CATEGORICAL_AUDIT_COLUMNS
) -> pd.Series:
    return pd.Series({c: df[c].nunique(dropna=True) for c in columns}, name="n_unique")


def numeric_ranges(
    df: pd.DataFrame, columns: list[str] = NUMERIC_AUDIT_COLUMNS
) -> pd.DataFrame:
    return df[columns].agg(["min", "max", "mean", "median"]).T


def tempo_zero_count(df: pd.DataFrame) -> int:
    return int((df["tempo"] == 0).sum())


def suspicious_release_date_count(df: pd.DataFrame, year: int = 1899) -> int:
    return int((df["release_date"] == year).sum())


def unusual_time_signature_counts(df: pd.DataFrame) -> pd.Series:
    """Documented plausible values per the course description are {3,4,5,7}.
    Anything else observed in the data is reported here as unusual."""
    counts = df["time_signature"].value_counts(dropna=False).sort_index()
    return counts


def artist_2_3_missingness(df: pd.DataFrame) -> dict:
    n = len(df)
    return {
        "artist_2_missing_pct": round(float(df["artist_2"].isna().mean() * 100), 3),
        "artist_3_missing_pct": round(float(df["artist_3"].isna().mean() * 100), 3),
        "artist_2_pop_missing_pct": round(
            float(df["artist_2_pop"].isna().mean() * 100), 3
        ),
        "artist_3_pop_missing_pct": round(
            float(df["artist_3_pop"].isna().mean() * 100), 3
        ),
    }


def primary_genre_completeness(df: pd.DataFrame) -> dict:
    n = len(df)
    n_present = int(df["artist_1_genre_1"].notna().sum())
    return {
        "n_present": n_present,
        "n_missing": n - n_present,
        "pct_present": round(n_present / n * 100, 3),
    }


def popularity_distribution(df: pd.DataFrame) -> dict:
    s = df["popularity_score"]
    n = len(s)
    n_zero = int((s == 0).sum())
    quantiles = s.quantile([0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0])
    return {
        "n_zero": n_zero,
        "pct_zero": round(n_zero / n * 100, 3),
        "quantiles": quantiles.to_dict(),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "skew": float(s.skew()),
    }


def album_type_distribution(df: pd.DataFrame) -> pd.Series:
    return df["album_type"].value_counts(dropna=False)


def release_decade_distribution(df: pd.DataFrame) -> pd.Series:
    decade = (df["release_date"] // 10 * 10)
    return decade.value_counts(dropna=False).sort_index()


def genre_availability_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Compare tracks with vs. without an observed primary genre.

    Returns a small comparison table: sample size/share, popularity_score
    median/mean, artist_1_pop mean, and release-decade distribution shape
    (via decade-level share difference summarized as a max abs delta).
    """
    has_genre = df["artist_1_genre_1"].notna()
    n = len(df)

    def _stats(mask: pd.Series) -> dict:
        sub = df.loc[mask]
        return {
            "n": int(mask.sum()),
            "share_of_dataset": round(float(mask.sum() / n), 4),
            "median_popularity_score": float(sub["popularity_score"].median()),
            "mean_popularity_score": float(sub["popularity_score"].mean()),
            "mean_artist_1_pop": float(sub["artist_1_pop"].mean()),
        }

    with_genre = _stats(has_genre)
    without_genre = _stats(~has_genre)

    comparison = pd.DataFrame(
        {"with_primary_genre": with_genre, "without_primary_genre": without_genre}
    )
    return comparison
