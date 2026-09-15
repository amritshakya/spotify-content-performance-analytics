"""Tests for src/modeling.py (Phase 3 / Phase 3.1).

Focused on the modeling pipeline's own invariants (no target leakage, no
train/test preprocessing leakage -- including categorical vocabulary, a
grouping key that does not treat placeholder artist labels as one shared
artist, deterministic/isolated splits, correct diagnostics) -- not on
re-testing sklearn/statsmodels internals.
"""

import numpy as np
import pandas as pd
import pytest

from src import modeling as M


# ---------------------------------------------------------------------------
# Modeling dataset construction
# ---------------------------------------------------------------------------


def test_modeling_frame_construction_documents_every_dropped_row(modeling_notes):
    assert modeling_notes["rows_dropped_missing_release_decade"] == 4
    assert modeling_notes["tempo_values_treated_as_missing"] == 90
    assert modeling_notes["rows_missing_artist_1"] == 19
    assert modeling_notes["rows_with_placeholder_artist_1"] > 0
    assert (
        modeling_notes["output_rows"]
        == modeling_notes["input_rows"] - modeling_notes["rows_dropped_missing_release_decade"]
    )


def test_modeling_frame_has_no_missing_decade(modeling_df):
    assert modeling_df["release_decade_clean"].notna().all()


def test_modeling_frame_genre_has_explicit_unknown_level_not_dropped(modeling_df):
    assert "Unknown" in modeling_df["artist_1_genre_1_filled"].unique()
    assert modeling_df["artist_1_genre_1_filled"].notna().all()


def test_modeling_frame_tempo_zero_recoded_to_missing(modeling_df, primary_df):
    zero_tempo_tracks = primary_df.loc[primary_df["tempo"] == 0, "tempo"]
    assert len(zero_tempo_tracks) == 90
    assert modeling_df["tempo"].isna().sum() == 90


# ---------------------------------------------------------------------------
# Target absent from feature matrix / target-derived fields excluded
# ---------------------------------------------------------------------------


def test_target_absent_from_feature_columns():
    feature_cols = set(M.CONTINUOUS_FEATURES_B) | set(M.CATEGORICAL_FEATURES)
    assert M.TARGET not in feature_cols


def test_target_derived_columns_never_used_as_features():
    feature_cols = set(M.CONTINUOUS_FEATURES_B) | set(M.CATEGORICAL_FEATURES)
    assert not (feature_cols & set(M.TARGET_DERIVED_COLUMNS))


def test_target_derived_columns_are_the_documented_ones():
    assert set(M.TARGET_DERIVED_COLUMNS) == {
        "popularity",
        "popularity_tier",
        "is_high_popularity",
    }


def test_build_modeling_frame_asserts_no_target_leakage_defensively():
    df = pd.DataFrame(
        {
            "popularity_score": [10, 20, 30],
            "popularity": ["0low", "0low", "1moderate"],
            "popularity_tier": ["low", "low", "moderate"],
            "is_high_popularity": [False, False, False],
            "release_date": [2000.0, 2001.0, 2002.0],
            "release_decade_clean": [2000, 2000, 2000],
            "tempo": [100.0, 0.0, 120.0],
            "loudness": [-5.0, -6.0, -7.0],
            "valence": [0.5, 0.4, 0.3],
            "track_duration_minutes": [3.0, 3.5, 4.0],
            "album_type": ["album", "single", "album"],
            "artist_1_genre_1": ["pop", None, "rock"],
            "artist_1_pop": [0.5, 0.6, 0.7],
            "artist_1": ["A", "B", "C"],
        }
    )
    modeling_df, notes = M.build_modeling_frame(df)
    assert M.TARGET in modeling_df.columns
    assert not (set(modeling_df.columns) & set(M.TARGET_DERIVED_COLUMNS))


# ---------------------------------------------------------------------------
# Grouping key: missing artists AND placeholder labels each get unique
# synthetic groups; normal named artists stay grouped together
# ---------------------------------------------------------------------------


def test_missing_artist_rows_each_get_a_unique_group_id():
    artist_col = pd.Series([None, None, "Real Artist", None])
    ids = M.make_group_ids(artist_col)
    missing_ids = ids[artist_col.isna()]
    assert missing_ids.nunique() == 3
    assert "Real Artist" in ids.values


def test_placeholder_artist_labels_each_get_a_unique_group_id():
    """Various Artists (and the other documented placeholder labels) must
    NOT be treated as one shared artist -- each occurrence gets its own
    synthetic group, same treatment as a missing artist_1."""
    artist_col = pd.Series(
        ["Various Artists", "Various Artists", "Original Cast", "Unknown", "Unknown Artist"]
    )
    ids = M.make_group_ids(artist_col)
    assert ids.nunique() == len(artist_col)  # every row is its own group


def test_normal_named_artists_remain_grouped_together():
    artist_col = pd.Series(["Real Artist", "Real Artist", "Other Artist"])
    ids = M.make_group_ids(artist_col)
    assert ids.iloc[0] == ids.iloc[1] == "Real Artist"
    assert ids.iloc[2] == "Other Artist"


def test_various_artists_does_not_form_one_giant_group_in_real_data(modeling_df):
    group_ids = M.make_group_ids(modeling_df["artist_1"])
    va_mask = modeling_df["artist_1"] == "Various Artists"
    va_groups = group_ids[va_mask]
    assert va_mask.sum() > 1000  # sanity: this label is genuinely common
    assert va_groups.nunique() == va_mask.sum()  # every occurrence is its own group


def test_various_artists_artist_1_pop_is_constant(modeling_df):
    """Documented finding, exercised as a regression guard: 'Various
    Artists' has a single, constant artist_1_pop value in this dataset."""
    va_pop = modeling_df.loc[modeling_df["artist_1"] == "Various Artists", "artist_1_pop"]
    assert va_pop.nunique() == 1
    assert va_pop.iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Grouped train/test artist separation
# ---------------------------------------------------------------------------


def test_grouped_split_has_no_shared_artist_between_train_and_test(
    modeling_df, grouped_split_idx
):
    train_idx, test_idx = grouped_split_idx
    group_ids = M.make_group_ids(modeling_df["artist_1"])
    train_groups = set(group_ids.iloc[train_idx])
    test_groups = set(group_ids.iloc[test_idx])
    assert train_groups.isdisjoint(test_groups)


def test_grouped_split_keeps_a_frequent_normal_artist_entirely_on_one_side(modeling_df):
    """A real, non-placeholder artist with many tracks (e.g. the most
    frequent named artist other than the placeholder labels) must never
    have rows on both sides of the grouped split."""
    named = modeling_df.loc[
        ~modeling_df["artist_1"].isin(M.PLACEHOLDER_ARTIST_LABELS), "artist_1"
    ]
    frequent_artist = named.value_counts().index[0]
    train_idx, test_idx = M.grouped_split(modeling_df, random_state=42)
    in_train = (modeling_df.iloc[train_idx]["artist_1"] == frequent_artist).sum()
    in_test = (modeling_df.iloc[test_idx]["artist_1"] == frequent_artist).sum()
    assert in_train == 0 or in_test == 0
    assert in_train + in_test > 1  # sanity: this artist has multiple tracks


def test_grouped_split_covers_every_row_exactly_once(modeling_df, grouped_split_idx):
    train_idx, test_idx = grouped_split_idx
    all_idx = np.concatenate([train_idx, test_idx])
    assert len(all_idx) == len(modeling_df)
    assert len(set(all_idx)) == len(all_idx)


# ---------------------------------------------------------------------------
# Deterministic split under fixed seed
# ---------------------------------------------------------------------------


def test_grouped_split_is_deterministic_under_fixed_seed(modeling_df):
    train1, test1 = M.grouped_split(modeling_df, random_state=7)
    train2, test2 = M.grouped_split(modeling_df, random_state=7)
    assert np.array_equal(train1, train2)
    assert np.array_equal(test1, test2)


def test_random_split_is_deterministic_under_fixed_seed(modeling_df):
    train1, test1 = M.random_split(modeling_df, random_state=7)
    train2, test2 = M.random_split(modeling_df, random_state=7)
    assert np.array_equal(train1, train2)
    assert np.array_equal(test1, test2)


def test_different_seeds_produce_different_grouped_splits(modeling_df):
    train1, _ = M.grouped_split(modeling_df, random_state=1)
    train2, _ = M.grouped_split(modeling_df, random_state=2)
    assert not np.array_equal(train1, train2)


# ---------------------------------------------------------------------------
# Preprocessing fit only on training data -- including categorical
# vocabulary (Phase 3.1 fix: this used to be learned from the full dataset)
# ---------------------------------------------------------------------------


def test_preprocessing_statistics_come_from_training_data_only(modeling_df, grouped_split_idx):
    train_idx, test_idx = grouped_split_idx
    feature_cols = M.CONTINUOUS_FEATURES_A + M.CATEGORICAL_FEATURES
    X_train = modeling_df.iloc[train_idx][feature_cols]
    y_train = modeling_df.iloc[train_idx][M.TARGET].to_numpy()

    pipeline = M.build_pipeline(M.CONTINUOUS_FEATURES_A)
    pipeline.fit(X_train, y_train)

    scaler = pipeline.named_steps["preprocess"].named_transformers_["num"].named_steps["scale"]
    expected_mean = (
        X_train[M.CONTINUOUS_FEATURES_A]
        .assign(tempo=X_train["tempo"].fillna(X_train["tempo"].median()))
        .mean()
        .to_numpy()
    )
    assert np.allclose(scaler.mean_, expected_mean, rtol=1e-6)

    full_mean = modeling_df[M.CONTINUOUS_FEATURES_A].mean().to_numpy()
    assert not np.allclose(scaler.mean_, full_mean, rtol=1e-6)


def test_categorical_vocabulary_learned_from_training_data_only():
    """Core Phase 3.1 fix: a category value present ONLY in the held-out
    rows (never in training) must not appear in the fit encoder's learned
    vocabulary."""
    train_df = pd.DataFrame(
        {
            "loudness": [-5.0, -6.0, -7.0, -8.0],
            "valence": [0.5, 0.4, 0.3, 0.2],
            "tempo": [100.0, 110.0, 120.0, 130.0],
            "track_duration_minutes": [3.0, 3.5, 4.0, 4.5],
            # "album" and "Unknown" (the documented reference categories)
            # must be present in training data -- OneHotEncoder(drop=...)
            # requires the dropped category to exist in the fit data.
            "album_type": ["album", "single", "album", "single"],
            "artist_1_genre_1_filled": ["Unknown", "rock", "Unknown", "rock"],
            "release_decade_clean": ["2010", "2010", "2010", "2010"],
        }
    )
    y_train = np.array([10.0, 20.0, 30.0, 40.0])
    pipeline = M.build_pipeline(M.CONTINUOUS_FEATURES_A)
    pipeline.fit(train_df, y_train)

    encoder = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    decade_col_idx = M.CATEGORICAL_FEATURES.index("release_decade_clean")
    learned_decade_categories = set(encoder.categories_[decade_col_idx])
    assert learned_decade_categories == {"2010"}
    assert "1990" not in learned_decade_categories  # never in training


def test_run_model_never_calls_fit_on_test_rows(modeling_df, grouped_split_idx):
    train_idx, test_idx = grouped_split_idx
    result = M.run_model(modeling_df, M.CONTINUOUS_FEATURES_A, train_idx, test_idx)
    imputer = (
        result["pipeline"]
        .named_steps["preprocess"]
        .named_transformers_["num"]
        .named_steps["impute"]
    )
    tempo_idx = M.CONTINUOUS_FEATURES_A.index("tempo")
    train_tempo_median = modeling_df.iloc[train_idx]["tempo"].median()
    assert np.isclose(imputer.statistics_[tempo_idx], train_tempo_median)


# ---------------------------------------------------------------------------
# Unseen categorical handling
# ---------------------------------------------------------------------------


def test_unseen_categorical_at_predict_time_does_not_raise():
    """A category never seen during training (because the vocabulary is now
    learned strictly from training data) must be handled safely
    (handle_unknown='ignore'), not raise at .transform()/.predict() time."""
    train_df = pd.DataFrame(
        {
            "loudness": [-5.0, -6.0, -7.0, -8.0],
            "valence": [0.5, 0.4, 0.3, 0.2],
            "tempo": [100.0, 110.0, 120.0, 130.0],
            "track_duration_minutes": [3.0, 3.5, 4.0, 4.5],
            "album_type": ["album", "single", "album", "single"],
            "artist_1_genre_1_filled": ["Unknown", "rock", "Unknown", "rock"],
            "release_decade_clean": ["2010", "2010", "2020", "2020"],
        }
    )
    y_train = np.array([10.0, 20.0, 30.0, 40.0])
    pipeline = M.build_pipeline(M.CONTINUOUS_FEATURES_A)
    pipeline.fit(train_df, y_train)

    unseen_df = train_df.copy()
    unseen_df.loc[0, "artist_1_genre_1_filled"] = "jazz"  # never seen in training
    unseen_df.loc[1, "release_decade_clean"] = "1990"  # never seen in training
    preds = pipeline.predict(unseen_df)
    assert np.isfinite(preds).all()


def test_unseen_test_only_category_does_not_appear_in_learned_vocabulary():
    """Complementary to the 'learned from training only' test: simulate a
    real train/test split where a category exists only in test, and confirm
    it truly never entered the fit encoder's vocabulary."""
    df = pd.DataFrame(
        {
            "loudness": [-5.0, -6.0, -7.0, -8.0, -9.0],
            "valence": [0.5, 0.4, 0.3, 0.2, 0.1],
            "tempo": [100.0, 110.0, 120.0, 130.0, 140.0],
            "track_duration_minutes": [3.0, 3.5, 4.0, 4.5, 5.0],
            "album_type": ["album", "single", "album", "single", "album"],
            "artist_1_genre_1_filled": ["Unknown", "rock", "Unknown", "rock", "jazz"],
            "release_decade_clean": ["2010", "2010", "2010", "2010", "2010"],
        }
    )
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    train_idx, test_idx = np.array([0, 1, 2, 3]), np.array([4])  # "jazz" only in test

    pipeline = M.build_pipeline(M.CONTINUOUS_FEATURES_A)
    pipeline.fit(df.iloc[train_idx], y[train_idx])
    encoder = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    genre_col_idx = M.CATEGORICAL_FEATURES.index("artist_1_genre_1_filled")
    assert "jazz" not in set(encoder.categories_[genre_col_idx])

    preds = pipeline.predict(df.iloc[test_idx])
    assert np.isfinite(preds).all()


# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------


def test_evaluate_predictions_perfect_prediction_gives_r2_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = M.evaluate_predictions(y, y)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)


def test_evaluate_predictions_constant_mean_prediction_gives_r2_zero():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    pred = np.full_like(y, y.mean())
    metrics = M.evaluate_predictions(y, pred)
    assert metrics["r2"] == pytest.approx(0.0, abs=1e-9)


def test_evaluate_predictions_matches_manual_mae_rmse():
    y = np.array([0.0, 10.0, 20.0])
    pred = np.array([5.0, 5.0, 25.0])
    metrics = M.evaluate_predictions(y, pred)
    expected_mae = np.mean(np.abs(y - pred))
    expected_rmse = np.sqrt(np.mean((y - pred) ** 2))
    assert metrics["mae"] == pytest.approx(expected_mae)
    assert metrics["rmse"] == pytest.approx(expected_rmse)


# ---------------------------------------------------------------------------
# Bounded-prediction diagnostics
# ---------------------------------------------------------------------------


def test_bounded_prediction_diagnostic_counts_out_of_range():
    preds = np.array([-10.0, -0.1, 0.0, 50.0, 100.0, 100.1, 150.0])
    diag = M.bounded_prediction_diagnostic(preds)
    assert diag["n"] == 7
    assert diag["n_below_0"] == 2
    assert diag["n_above_100"] == 2
    assert diag["pct_below_0"] == pytest.approx(2 / 7 * 100, abs=1e-3)
    assert diag["pct_above_100"] == pytest.approx(2 / 7 * 100, abs=1e-3)
    assert diag["min_prediction"] == -10.0
    assert diag["max_prediction"] == 150.0


def test_bounded_prediction_diagnostic_all_in_range():
    preds = np.array([0.0, 50.0, 100.0])
    diag = M.bounded_prediction_diagnostic(preds)
    assert diag["n_below_0"] == 0
    assert diag["n_above_100"] == 0


def test_clip_predictions_bounds_to_0_100():
    preds = np.array([-10.0, 50.0, 150.0])
    clipped = M.clip_predictions(preds)
    assert clipped.min() >= 0
    assert clipped.max() <= 100
    assert list(clipped) == [0.0, 50.0, 100.0]


def test_grouped_model_bounded_diagnostic_is_consistent_with_predictions(
    modeling_df, grouped_split_idx
):
    train_idx, test_idx = grouped_split_idx
    result = M.run_model(modeling_df, M.CONTINUOUS_FEATURES_A, train_idx, test_idx)
    manual = M.bounded_prediction_diagnostic(result["y_pred"])
    assert manual == result["bounded"]


# ---------------------------------------------------------------------------
# Group-size structure
# ---------------------------------------------------------------------------


def test_group_size_summary_shape(modeling_df):
    summary = M.group_size_summary(modeling_df)
    assert summary["n_groups"] > 0
    assert len(summary["ten_largest_group_sizes"]) == 10
    assert summary["ten_largest_group_sizes"] == sorted(
        summary["ten_largest_group_sizes"], reverse=True
    )
    assert 0 < summary["largest_group_pct_of_rows"] < 100


def test_group_size_summary_reflects_corrected_grouping_key(modeling_df):
    """After the Phase 3.1 fix, no single group should dominate the dataset
    the way the un-split 'Various Artists' group once did (~5.3% of rows)."""
    summary = M.group_size_summary(modeling_df)
    assert summary["largest_group_pct_of_rows"] < 1.0


# ---------------------------------------------------------------------------
# Repeated grouped-split evaluation summary calculations
# ---------------------------------------------------------------------------


def test_summarize_sweep_computes_expected_aggregates():
    sweep_df = pd.DataFrame(
        {
            "seed": [1, 2, 3],
            "n_test": [100, 110, 90],
            "n_test_groups": [50, 55, 45],
            "test_mean_popularity": [27.0, 28.0, 29.0],
            "model_a_r2": [0.10, 0.12, 0.14],
            "model_a_mae": [16.0, 16.5, 17.0],
            "model_a_rmse": [19.0, 19.5, 20.0],
            "model_b_r2": [0.25, 0.30, 0.20],
            "model_b_mae": [14.0, 14.5, 15.0],
            "model_b_rmse": [17.0, 17.5, 18.0],
        }
    )
    summary = M.summarize_sweep(sweep_df)
    assert summary.loc["model_a", "median_r2"] == pytest.approx(0.12)
    assert summary.loc["model_a", "min_r2"] == pytest.approx(0.10)
    assert summary.loc["model_a", "max_r2"] == pytest.approx(0.14)
    assert summary.loc["model_b", "median_r2"] == pytest.approx(0.25)
    assert summary.loc["model_b", "min_r2"] == pytest.approx(0.20)
    assert summary.loc["model_b", "max_r2"] == pytest.approx(0.30)
    assert summary.loc["split_composition", "n_test_min"] == 90
    assert summary.loc["split_composition", "n_test_max"] == 110


def test_grouped_split_sweep_runs_all_declared_seeds_on_small_frame():
    """Cheap end-to-end smoke test of the sweep machinery on a small
    synthetic dataset (full 278K-row sweep is exercised in the notebook,
    not in the test suite, for speed)."""
    n = 400
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            M.TARGET: rng.uniform(0, 100, n),
            "loudness": rng.normal(-8, 3, n),
            "valence": rng.uniform(0, 1, n),
            "tempo": rng.uniform(60, 180, n),
            "track_duration_minutes": rng.uniform(1, 6, n),
            "album_type": rng.choice(["album", "single", "compilation"], n),
            "artist_1_genre_1_filled": rng.choice(["pop", "rock", "Unknown"], n),
            "release_decade_clean": rng.choice(["2010", "2020"], n).astype(str),
            "artist_1_pop": rng.uniform(0, 1, n),
            "artist_1": [f"artist_{i % 80}" for i in range(n)],
        }
    )
    seeds = [1, 2, 3]
    sweep = M.grouped_split_sweep(df, seeds=seeds)
    assert len(sweep) == len(seeds)
    assert set(sweep["seed"]) == set(seeds)
    for col in ["model_a_r2", "model_b_r2", "n_train_groups", "n_test_groups"]:
        assert sweep[col].notna().all()


# ---------------------------------------------------------------------------
# Sensitivity model execution
# ---------------------------------------------------------------------------


def test_sensitivity_model_executes_and_produces_finite_metrics(sensitivity_df):
    sens_modeling_df, _notes = M.build_modeling_frame(sensitivity_df)
    train_idx, test_idx = M.grouped_split(sens_modeling_df)

    result_a = M.run_model(sens_modeling_df, M.CONTINUOUS_FEATURES_A, train_idx, test_idx)
    result_b = M.run_model(sens_modeling_df, M.CONTINUOUS_FEATURES_B, train_idx, test_idx)
    for result in (result_a, result_b):
        assert np.isfinite(result["metrics"]["r2"])
        assert np.isfinite(result["metrics"]["mae"])
        assert np.isfinite(result["metrics"]["rmse"])
        assert result["metrics"]["n"] == len(test_idx)


def test_sensitivity_and_primary_results_are_close(modeling_df, sensitivity_df):
    """Primary vs. sensitivity dataset deltas at a fixed seed should be
    small -- comparable to the seed-to-seed variability already observed
    across the primary dataset's own grouped-split sweep, not a distinct
    substantive difference."""
    p_train, p_test = M.grouped_split(modeling_df, random_state=42)
    primary_result = M.run_model(modeling_df, M.CONTINUOUS_FEATURES_A, p_train, p_test)

    sens_modeling_df, _ = M.build_modeling_frame(sensitivity_df)
    s_train, s_test = M.grouped_split(sens_modeling_df, random_state=42)
    sens_result = M.run_model(sens_modeling_df, M.CONTINUOUS_FEATURES_A, s_train, s_test)

    assert abs(primary_result["metrics"]["r2"] - sens_result["metrics"]["r2"]) < 0.05


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def test_baseline_predicts_training_mean_for_every_test_row(modeling_df, grouped_split_idx):
    train_idx, test_idx = grouped_split_idx
    result = M.run_baseline(modeling_df, train_idx, test_idx)
    train_mean = modeling_df.iloc[train_idx][M.TARGET].mean()
    assert np.allclose(result["y_pred"], train_mean)


# ---------------------------------------------------------------------------
# Artist-frequency bucketing
# ---------------------------------------------------------------------------


def test_artist_frequency_bucket_labels_match_expected_ranges():
    counts = pd.Series([1, 2, 4, 15, 50, 500])
    buckets = M.artist_frequency_bucket(counts)
    assert list(buckets.astype(str)) == ["1", "2", "3-5", "6-20", "21-100", "100+"]
