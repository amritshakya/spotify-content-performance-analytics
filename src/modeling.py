"""Phase 3 — interpretable popularity modeling.

Answers: how much variation in `popularity_score` is associated with
observable track/content/context characteristics (Model A), and how much
additional predictive power appears when `artist_1_pop` is added (Model B)?

This is an OBSERVATIONAL dataset. Nothing in this module supports or implies
a causal claim -- every model here is descriptive/associational, evaluated
on held-out data, and Model B in particular is documented as being close in
concept to the target rather than an independent causal driver.

Does not modify or depend on any change to the frozen Phase 2 pipeline
(src/load_data.py, src/clean_data.py, src/build_database.py, src/metrics.py,
src/quality_audit.py, or the sql/ files) -- it only reads the already-built
primary/sensitivity DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "popularity_score"

# Fields excluded from every model because they are derived directly (and,
# for `popularity`, exactly -- see src/clean_data.py POPULARITY_TIER_BOUNDS)
# from the target itself. Including any of these would not be predicting
# popularity from independent characteristics -- it would be reconstructing
# the target from its own recoding.
TARGET_DERIVED_COLUMNS = ["popularity", "popularity_tier", "is_high_popularity"]

# ---------------------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------------------
# Model A: content / context only.
#
#   loudness (dB)              -- Spotify-measured overall loudness. Belongs
#                                  as a genuine audio-content characteristic;
#                                  no missingness in this dataset.
#   valence (0-1)               -- Spotify's measure of musical positivity.
#                                  Genuine content characteristic; no
#                                  missingness.
#   tempo (BPM)                 -- Genuine content characteristic. Caveat:
#                                  90 rows have tempo == 0, flagged in Phase 2
#                                  as almost certainly a placeholder rather
#                                  than a real 0-BPM track (see
#                                  reports/data_quality_report.md sec. 7).
#                                  Treated as missing here and imputed with
#                                  the TRAINING-set median, consistent with
#                                  the Phase 2 policy of treating zero tempo
#                                  as missing for analyses that need a
#                                  meaningful tempo value.
#   track_duration_minutes      -- Derived in Phase 2 from duration_ms
#                                  (continuous, not target-derived); genuine
#                                  content characteristic, no missingness.
#   album_type                  -- album / single / compilation. Genuine
#                                  release-context characteristic; no
#                                  missingness (Phase 2 audit).
#   artist_1_genre_1            -- Primary genre. Genuine content
#                                  characteristic where observed, BUT 60.3%
#                                  missing, and Phase 2 established that
#                                  genre availability itself correlates with
#                                  popularity, artist popularity, and release
#                                  period (reports/data_quality_report.md
#                                  sec. 15). Missing genre is therefore kept
#                                  as its own explicit "Unknown" level rather
#                                  than imputed or dropped -- if genre
#                                  availability itself carries information,
#                                  that should show up as the "Unknown"
#                                  category's own coefficient, not be
#                                  silently discarded.
#   release_decade_clean        -- Used as a CATEGORICAL feature (not a
#                                  continuous year trend), because Phase 2's
#                                  decade-level analysis showed a non-monotonic
#                                  popularity pattern across decades (e.g. a
#                                  1990s/2000s dip relative to neighboring
#                                  decades) that a single linear year
#                                  coefficient would not capture. IMPORTANT:
#                                  this is NOT treated as a pure content
#                                  attribute -- it is entangled with recency,
#                                  survivorship, and catalog composition (an
#                                  old track still present in this dataset is
#                                  a curated survivor, not a random sample of
#                                  everything released that decade). 4 rows
#                                  with unknown release_date/decade are
#                                  dropped from the modeling sample only
#                                  (documented count below); they remain in
#                                  the primary/sensitivity analytical
#                                  datasets untouched.
#
# Model B = Model A + artist_1_pop.
#   artist_1_pop                -- Spotify-computed popularity of the primary
#                                  artist. Included ONLY in Model B.
#                                  IMPORTANT: this is conceptually very close
#                                  to the outcome -- popular artists
#                                  mechanically tend to have popular tracks
#                                  via correlated underlying algorithmic
#                                  signals (see README "Measurement
#                                  Caveats"). Model B is therefore labeled a
#                                  CONTEXTUAL/PREDICTIVE comparison, not
#                                  evidence that artist popularity causes
#                                  track popularity, and its performance gain
#                                  over Model A is discussed together with a
#                                  target-proximity / mechanical-relationship
#                                  risk, not presented as a clean causal or
#                                  even a clean "new information" result.
CONTINUOUS_FEATURES_A = ["loudness", "valence", "tempo", "track_duration_minutes"]
CATEGORICAL_FEATURES = ["album_type", "artist_1_genre_1_filled", "release_decade_clean"]
ARTIST_POP_FEATURE = "artist_1_pop"

CONTINUOUS_FEATURES_B = CONTINUOUS_FEATURES_A + [ARTIST_POP_FEATURE]

GENRE_UNKNOWN_LABEL = "Unknown"

# Reference (dropped/baseline) category per categorical feature, chosen to be
# the largest and most substantively meaningful level in each -- NOT chosen
# by alphabetical accident. Coefficients for every other level in that
# feature are then read as "vs. this reference," which is documented
# wherever coefficients are reported.
CATEGORY_REFERENCES = {
    "album_type": "album",  # largest group, 161,872 / 277,937 rows
    "artist_1_genre_1_filled": GENRE_UNKNOWN_LABEL,  # largest group, 60.3% of rows
    "release_decade_clean": "2010",  # largest decade, 108,479 rows
}

ARTIST_FREQUENCY_BINS = [0, 1, 2, 5, 20, 100, np.inf]
ARTIST_FREQUENCY_LABELS = ["1", "2", "3-5", "6-20", "21-100", "100+"]

RANDOM_SEED = 42
TEST_SIZE = 0.2


def build_modeling_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build the modeling sample from a primary- or sensitivity-shaped
    DataFrame (as produced by src/clean_data.py).

    Returns (modeling_df, sample_notes) where modeling_df has one row per
    modeled track observation, containing the target, all Model A/B raw
    feature columns (before preprocessing), `artist_1` (kept for grouping,
    NOT used as a model feature), and a `group_id` column for use with
    GroupShuffleSplit (see make_group_ids). sample_notes documents every
    row dropped and why -- this function never silently drops rows.
    """
    notes = {"input_rows": len(df)}

    missing_decade = df["release_decade_clean"].isna()
    notes["rows_dropped_missing_release_decade"] = int(missing_decade.sum())

    working = df.loc[~missing_decade].copy()

    working["tempo_for_model"] = working["tempo"].where(working["tempo"] != 0, np.nan)
    notes["tempo_values_treated_as_missing"] = int((working["tempo"] == 0).sum())

    working["artist_1_genre_1_filled"] = working["artist_1_genre_1"].fillna(
        GENRE_UNKNOWN_LABEL
    )
    notes["rows_with_unknown_genre"] = int(
        (working["artist_1_genre_1_filled"] == GENRE_UNKNOWN_LABEL).sum()
    )

    working["release_decade_clean"] = working["release_decade_clean"].astype(
        "Int64"
    ).astype(str)

    modeling_df = pd.DataFrame(
        {
            TARGET: working[TARGET].astype(float),
            "loudness": working["loudness"],
            "valence": working["valence"],
            "tempo": working["tempo_for_model"],
            "track_duration_minutes": working["track_duration_minutes"],
            "album_type": working["album_type"],
            "artist_1_genre_1_filled": working["artist_1_genre_1_filled"],
            "release_decade_clean": working["release_decade_clean"],
            "artist_1_pop": working["artist_1_pop"],
            "artist_1": working["artist_1"],
        }
    )

    notes["output_rows"] = len(modeling_df)
    notes["rows_missing_artist_1"] = int(modeling_df["artist_1"].isna().sum())

    # Sanity: the target and its exact derivatives must never leak in as
    # features. This is a hard invariant, not just documentation.
    feature_cols = set(modeling_df.columns) - {TARGET, "artist_1"}
    assert TARGET not in feature_cols
    assert not (feature_cols & set(TARGET_DERIVED_COLUMNS))

    return modeling_df, notes


def make_group_ids(artist_1: pd.Series) -> pd.Series:
    """Build grouping keys for GroupShuffleSplit from the raw `artist_1`
    column.

    Rows with a missing artist_1 (19 in the primary dataset) are each
    assigned their OWN unique group id rather than being lumped into one
    "missing artist" group -- we have no basis to believe two tracks with
    an unrecorded artist_1 are by the same artist, and grouping them
    together would create a large, meaningless synthetic group that could
    distort the split.

    Known limitation, documented and NOT worked around here: `artist_1` is
    not a perfect canonical artist identifier in this extract. In
    particular, "Various Artists" is a placeholder used for compilation
    tracks (14,841 rows in the primary dataset, ~5.3% of it) rather than a
    genuine shared artist identity -- grouping on it forces all of those
    unrelated tracks into the same split side. This is a concrete instance
    of the general caveat that the grouped split reduces obvious same-artist
    leakage but does not guarantee perfect artist-level isolation.
    """
    ids = artist_1.astype("object").copy()
    missing_mask = ids.isna()
    ids.loc[missing_mask] = [
        f"__missing_artist_row_{i}__" for i in ids.index[missing_mask]
    ]
    return ids


def get_category_levels(modeling_df: pd.DataFrame) -> dict[str, list[str]]:
    """Enumerate the observed category levels for each categorical feature
    from the FULL modeling frame (train + test).

    This fixes the *schema* the one-hot encoder should expect (which
    category names exist at all) -- it does not fit any statistic on test
    data, unlike the imputer/scaler, which are fit on the training split
    only inside build_pipeline()'s caller. Any category that were somehow
    still unseen (not possible by construction here) safely falls back to
    handle_unknown='ignore' rather than raising.
    """
    levels = {}
    for col in CATEGORICAL_FEATURES:
        col_levels = sorted(modeling_df[col].dropna().unique().tolist())
        reference = CATEGORY_REFERENCES[col]
        if reference not in col_levels:
            raise ValueError(f"Reference category {reference!r} not observed in {col}")
        levels[col] = col_levels
    return levels


def build_pipeline(
    continuous_features: list[str], category_levels: dict[str, list[str]]
) -> Pipeline:
    """Build the preprocessing + linear regression pipeline.

    Continuous features: median-imputed (guards against the tempo==0 ->
    NaN recoding; no other continuous feature has missingness in this
    dataset) then standardized, so coefficients are comparable to each
    other in standard-deviation units.

    Categorical features: one-hot encoded with an explicit, documented
    reference category dropped per feature (see CATEGORY_REFERENCES) --
    never an arbitrary alphabetical-first drop. `category_levels` (from
    get_category_levels) fixes which category names the encoder expects;
    `handle_unknown="ignore"` means a category never seen in training
    (possible for rare genres or decades near a split boundary) safely
    encodes as all-zeros for that feature at test time, i.e. it is treated
    the same as the reference category, rather than raising an error.

    Preprocessing statistics (the imputer's median, the scaler's mean/std)
    are fit exclusively inside pipeline.fit(X_train, ...) by the caller --
    this function only builds the (unfit) pipeline object, so there is no
    way for it to see test data before evaluation.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_transformer = OneHotEncoder(
        categories=[category_levels[col] for col in CATEGORICAL_FEATURES],
        drop=[CATEGORY_REFERENCES[col] for col in CATEGORICAL_FEATURES],
        handle_unknown="ignore",
        sparse_output=False,
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, continuous_features),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )


def grouped_split(
    modeling_df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Primary split design: GroupShuffleSplit grouped on artist_1 (via
    make_group_ids), so no artist_1 value appears on both sides of the
    split. Returns (train_idx, test_idx) as positional integer arrays into
    modeling_df."""
    groups = make_group_ids(modeling_df["artist_1"])
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(
        splitter.split(modeling_df, groups=groups)
    )
    return train_idx, test_idx


def random_split(
    modeling_df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Secondary split design: conventional reproducible random row split,
    no grouping. Returns (train_idx, test_idx) as positional integer arrays."""
    idx = np.arange(len(modeling_df))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state
    )
    return train_idx, test_idx


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Core regression metrics on RAW (unclipped) predictions."""
    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def bounded_prediction_diagnostic(y_pred: np.ndarray) -> dict:
    """popularity_score is bounded [0, 100]; a linear model is not. Reports
    how often, and how far, raw predictions fall outside that range."""
    y_pred = np.asarray(y_pred)
    n = len(y_pred)
    below = y_pred < 0
    above = y_pred > 100
    return {
        "n": int(n),
        "min_prediction": float(y_pred.min()),
        "max_prediction": float(y_pred.max()),
        "n_below_0": int(below.sum()),
        "pct_below_0": round(float(below.mean() * 100), 3),
        "n_above_100": int(above.sum()),
        "pct_above_100": round(float(above.mean() * 100), 3),
    }


def clip_predictions(y_pred: np.ndarray) -> np.ndarray:
    """Clip raw predictions to the valid [0, 100] range. Used ONLY for
    presentation-clarity figures, never for headline metrics -- see
    bounded_prediction_diagnostic and the module docstring."""
    return np.clip(y_pred, 0, 100)


def artist_frequency_bucket(track_counts: pd.Series) -> pd.Series:
    """Map per-artist track counts (computed over the FULL dataset, not
    just train or test) to the documented frequency buckets."""
    return pd.cut(
        track_counts, bins=ARTIST_FREQUENCY_BINS, labels=ARTIST_FREQUENCY_LABELS
    )


def baseline_pipeline() -> DummyRegressor:
    """Baseline: predicts the TRAINING-set mean popularity_score for every
    test observation."""
    return DummyRegressor(strategy="mean")


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Human-readable output feature names for a fit pipeline's coefficients,
    in the same order as pipeline.named_steps['regressor'].coef_."""
    return list(
        pipeline.named_steps["preprocess"].get_feature_names_out()
    )


def run_model(
    modeling_df: pd.DataFrame,
    continuous_features: list[str],
    category_levels: dict[str, list[str]],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    """Fit one model (Model A or Model B, depending on continuous_features)
    on modeling_df[train_idx] and evaluate on modeling_df[test_idx].

    Preprocessing (imputer median, scaler mean/std) is fit only on the
    training rows, via pipeline.fit(train) -- sklearn Pipelines never let a
    downstream .transform() call on test data influence a fit already
    performed on train data, so there is no leakage path here as long as
    .fit() is only ever called once, on the training rows (which is what
    happens below).

    Returns a dict with the fit pipeline, raw (unclipped) test predictions,
    y_test, evaluate_predictions() metrics, and bounded_prediction_diagnostic().
    """
    feature_cols = continuous_features + CATEGORICAL_FEATURES
    X_train = modeling_df.iloc[train_idx][feature_cols]
    y_train = modeling_df.iloc[train_idx][TARGET].to_numpy()
    X_test = modeling_df.iloc[test_idx][feature_cols]
    y_test = modeling_df.iloc[test_idx][TARGET].to_numpy()

    pipeline = build_pipeline(continuous_features, category_levels)
    pipeline.fit(X_train, y_train)
    # np.errstate here suppresses a known cosmetic RuntimeWarning
    # ("divide by zero encountered in matmul") from numpy's Accelerate BLAS
    # backend on macOS during the X @ coef_ prediction step. Verified this
    # is purely cosmetic: predictions are confirmed 100% finite (no NaN/Inf)
    # with or without it -- it does not mask a real numerical problem.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        y_pred = pipeline.predict(X_test)

    return {
        "pipeline": pipeline,
        "y_test": y_test,
        "y_pred": y_pred,
        "test_idx": test_idx,
        "metrics": evaluate_predictions(y_test, y_pred),
        "bounded": bounded_prediction_diagnostic(y_pred),
    }


def run_baseline(
    modeling_df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    """Fit the training-mean baseline and evaluate it the same way as
    run_model(), for a like-for-like comparison."""
    y_train = modeling_df.iloc[train_idx][TARGET].to_numpy()
    y_test = modeling_df.iloc[test_idx][TARGET].to_numpy()

    model = baseline_pipeline()
    model.fit(np.zeros((len(y_train), 1)), y_train)
    y_pred = model.predict(np.zeros((len(y_test), 1)))

    return {
        "pipeline": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "test_idx": test_idx,
        "metrics": evaluate_predictions(y_test, y_pred),
        "bounded": bounded_prediction_diagnostic(y_pred),
    }
