"""Tests for src/modeling.py (Phase 3).

Focused on the modeling pipeline's own invariants (no target leakage, no
train/test preprocessing leakage, deterministic/isolated splits, correct
diagnostics) -- not on re-testing sklearn/statsmodels internals.
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
    assert (
        modeling_notes["output_rows"]
        == modeling_notes["input_rows"] - modeling_notes["rows_dropped_missing_release_decade"]
    )


def test_modeling_frame_has_no_missing_decade(modeling_df):
    assert modeling_df["release_decade_clean"].notna().all()


def test_modeling_frame_genre_has_explicit_unknown_level_not_dropped(modeling_df):
    assert "Unknown" in modeling_df["artist_1_genre_1_filled"].unique()
    # Every row has SOME genre value -- missing genre was recoded, not dropped.
    assert modeling_df["artist_1_genre_1_filled"].notna().all()


def test_modeling_frame_tempo_zero_recoded_to_missing(modeling_df, primary_df):
    # Rows that had tempo == 0 in the primary dataset must show up as NaN in
    # the modeling frame's tempo column (to be imputed inside the pipeline).
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
    """build_modeling_frame() has an internal assertion guarding this
    invariant even if the module-level constants were ever edited
    carelessly -- exercise it directly on a tiny synthetic frame."""
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


def test_grouped_split_covers_every_row_exactly_once(modeling_df, grouped_split_idx):
    train_idx, test_idx = grouped_split_idx
    all_idx = np.concatenate([train_idx, test_idx])
    assert len(all_idx) == len(modeling_df)
    assert len(set(all_idx)) == len(all_idx)  # no duplicates/overlap


def test_missing_artist_rows_each_get_a_unique_group_id():
    artist_col = pd.Series([None, None, "Real Artist", None])
    ids = M.make_group_ids(artist_col)
    missing_ids = ids[artist_col.isna()]
    assert missing_ids.nunique() == 3  # each missing row is its own group
    assert "Real Artist" in ids.values


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
# Preprocessing fit only on training data / no leakage
# ---------------------------------------------------------------------------


def test_preprocessing_statistics_come_from_training_data_only(
    modeling_df, category_levels, grouped_split_idx
):
    """The imputer's median and scaler's mean/std must match the TRAINING
    slice, not the full dataset or the test slice."""
    train_idx, test_idx = grouped_split_idx
    feature_cols = M.CONTINUOUS_FEATURES_A + M.CATEGORICAL_FEATURES
    X_train = modeling_df.iloc[train_idx][feature_cols]
    y_train = modeling_df.iloc[train_idx][M.TARGET].to_numpy()

    pipeline = M.build_pipeline(M.CONTINUOUS_FEATURES_A, category_levels)
    pipeline.fit(X_train, y_train)

    scaler = pipeline.named_steps["preprocess"].named_transformers_["num"].named_steps["scale"]
    expected_mean = (
        X_train[M.CONTINUOUS_FEATURES_A]
        .assign(tempo=X_train["tempo"].fillna(X_train["tempo"].median()))
        .mean()
        .to_numpy()
    )
    assert np.allclose(scaler.mean_, expected_mean, rtol=1e-6)

    # And it must NOT match the full (train+test) dataset's mean, since that
    # would indicate the scaler saw test data.
    full_mean = modeling_df[M.CONTINUOUS_FEATURES_A].mean().to_numpy()
    assert not np.allclose(scaler.mean_, full_mean, rtol=1e-6)


def test_run_model_never_calls_fit_on_test_rows(modeling_df, category_levels, grouped_split_idx):
    """Regression guard: run_model's pipeline.fit call must only ever see
    training rows. Verified by checking the fitted imputer's median exactly
    equals the training-only tempo median."""
    train_idx, test_idx = grouped_split_idx
    result = M.run_model(
        modeling_df, M.CONTINUOUS_FEATURES_A, category_levels, train_idx, test_idx
    )
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


def test_unseen_categorical_at_predict_time_does_not_raise(category_levels):
    """A category the encoder was never fit on must be handled safely
    (handle_unknown='ignore'), not raise at .transform()/.predict() time."""
    train_df = pd.DataFrame(
        {
            "loudness": [-5.0, -6.0, -7.0, -8.0],
            "valence": [0.5, 0.4, 0.3, 0.2],
            "tempo": [100.0, 110.0, 120.0, 130.0],
            "track_duration_minutes": [3.0, 3.5, 4.0, 4.5],
            "album_type": ["album", "single", "album", "single"],
            "artist_1_genre_1_filled": ["pop", "rock", "pop", "rock"],
            "release_decade_clean": ["2010", "2010", "2020", "2020"],
        }
    )
    y_train = np.array([10.0, 20.0, 30.0, 40.0])
    levels = {
        "album_type": ["album", "single"],
        "artist_1_genre_1_filled": ["pop", "rock"],
        "release_decade_clean": ["2010", "2020"],
    }
    # Use the module's real reference categories, temporarily narrowed to
    # only the levels present in this tiny synthetic example.
    import src.modeling as mod

    original_refs = dict(mod.CATEGORY_REFERENCES)
    try:
        mod.CATEGORY_REFERENCES.update(
            {"album_type": "album", "artist_1_genre_1_filled": "pop", "release_decade_clean": "2010"}
        )
        pipeline = M.build_pipeline(M.CONTINUOUS_FEATURES_A, levels)
        pipeline.fit(train_df, y_train)

        unseen_df = train_df.copy()
        unseen_df.loc[0, "artist_1_genre_1_filled"] = "jazz"  # never seen in training
        unseen_df.loc[1, "release_decade_clean"] = "1990"  # never seen in training
        preds = pipeline.predict(unseen_df)
        assert np.isfinite(preds).all()
    finally:
        mod.CATEGORY_REFERENCES.clear()
        mod.CATEGORY_REFERENCES.update(original_refs)


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
    # bounded_prediction_diagnostic rounds percentages to 3 decimal places.
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
    modeling_df, category_levels, grouped_split_idx
):
    train_idx, test_idx = grouped_split_idx
    result = M.run_model(
        modeling_df, M.CONTINUOUS_FEATURES_A, category_levels, train_idx, test_idx
    )
    manual = M.bounded_prediction_diagnostic(result["y_pred"])
    assert manual == result["bounded"]


# ---------------------------------------------------------------------------
# Sensitivity model execution
# ---------------------------------------------------------------------------


def test_sensitivity_model_executes_and_produces_finite_metrics(sensitivity_df):
    sens_modeling_df, _notes = M.build_modeling_frame(sensitivity_df)
    sens_levels = M.get_category_levels(sens_modeling_df)
    train_idx, test_idx = M.grouped_split(sens_modeling_df)

    result_a = M.run_model(
        sens_modeling_df, M.CONTINUOUS_FEATURES_A, sens_levels, train_idx, test_idx
    )
    result_b = M.run_model(
        sens_modeling_df, M.CONTINUOUS_FEATURES_B, sens_levels, train_idx, test_idx
    )
    for result in (result_a, result_b):
        assert np.isfinite(result["metrics"]["r2"])
        assert np.isfinite(result["metrics"]["mae"])
        assert np.isfinite(result["metrics"]["rmse"])
        assert result["metrics"]["n"] == len(test_idx)


def test_sensitivity_and_primary_results_are_close(modeling_df, category_levels, sensitivity_df):
    """Regression guard for the headline Phase 3 finding: primary vs.
    sensitivity dataset deltas should be trivial, same as Phase 2."""
    p_train, p_test = M.grouped_split(modeling_df)
    primary_result = M.run_model(
        modeling_df, M.CONTINUOUS_FEATURES_A, category_levels, p_train, p_test
    )

    sens_modeling_df, _ = M.build_modeling_frame(sensitivity_df)
    sens_levels = M.get_category_levels(sens_modeling_df)
    s_train, s_test = M.grouped_split(sens_modeling_df)
    sens_result = M.run_model(
        sens_modeling_df, M.CONTINUOUS_FEATURES_A, sens_levels, s_train, s_test
    )

    assert abs(primary_result["metrics"]["r2"] - sens_result["metrics"]["r2"]) < 0.01


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
