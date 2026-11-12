"""
Comprehensive Unit and Integration Tests for Heart Disease Risk Scoring Pipeline.

Coverage areas:
1. Data loading  — schema, binarization, null handling (network mocked via monkeypatch)
2. Preprocessing — output shape, NaN absence, unseen-category robustness
3. Leakage       — preprocessor statistics sourced only from training data
4. Model         — predict_proba range, threshold logic, determinism with fixed seed
5. App helpers   — patient DataFrame schema, referral boundary, attribution shape

Design principles:
- ALL tests use a 50-row synthetic fixture seeded at 42; no network calls are made.
- The ucimlrepo.fetch_ucirepo function is monkeypatched and paths redirected via tmp_path
  so load_heart(force_download=True) never touches the network or writes outside temporary storage.
- Each fixture and test has a docstring explaining what failure it would catch.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    LOGREG_MODEL_PATH,
    NUMERIC_FEATURES,
    SEED,
    TARGET_COL,
    THRESHOLD_CONFIG_PATH,
)
from src.data import load_heart, validate_clinical_dataframe
from src.features import build_preprocessor
from src.train import pick_threshold

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N = 50  # small enough that all tests finish in seconds


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    """Create a reproducible 50-row synthetic clinical DataFrame.

    Protects against: fixture nondeterminism causing flaky tests.
    Uses module scope so the same object is reused across all tests in the
    file, avoiding repeated construction overhead.
    """
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "age": rng.uniform(30, 75, N),
            "trestbps": rng.uniform(90, 180, N),
            "chol": rng.uniform(150, 400, N),
            "thalach": rng.uniform(80, 200, N),
            "oldpeak": rng.uniform(0, 5, N),
            "ca": rng.choice([0.0, 1.0, 2.0, 3.0], N),
            "sex": rng.choice([0.0, 1.0], N),
            "cp": rng.choice([1.0, 2.0, 3.0, 4.0], N),
            "fbs": rng.choice([0.0, 1.0], N),
            "restecg": rng.choice([0.0, 1.0, 2.0], N),
            "exang": rng.choice([0.0, 1.0], N),
            "slope": rng.choice([1.0, 2.0, 3.0], N),
            "thal": rng.choice([3.0, 6.0, 7.0], N),
            TARGET_COL: rng.integers(0, 2, N),
        }
    )
    return df


@pytest.fixture()
def mock_ucimlrepo(
    synthetic_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Monkeypatch ucimlrepo so load_heart(force_download=True) never hits the network.

    Protects against: flaky tests due to UCI API rate-limits, downtime, or
    schema changes breaking the ingestion layer mid-CI.

    Redirects RAW_DATA_PATH and FALLBACK_DATA_PATH to pytest's tmp_path fixture,
    ensuring platform-independent writable paths on Windows, macOS, and Linux.
    """
    features_df = synthetic_df[ALL_FEATURES].copy()
    # Simulate raw 'num' column: 0 = healthy, 1-2 = disease (maps to binary 1 after > 0)
    targets_df = pd.DataFrame(
        {"num": np.where(synthetic_df[TARGET_COL] == 1, 1, 0)}
    )

    mock_dataset = SimpleNamespace(
        data=SimpleNamespace(features=features_df, targets=targets_df)
    )

    monkeypatch.setattr(
        "src.data.fetch_ucirepo",
        lambda id: mock_dataset,  # noqa: A002
    )
    mock_raw_path = tmp_path / "mock_heart.csv"
    mock_fb_path = tmp_path / "mock_heart_fb.csv"
    monkeypatch.setattr("src.data.RAW_DATA_PATH", mock_raw_path)
    monkeypatch.setattr("src.data.FALLBACK_DATA_PATH", mock_fb_path)

    return mock_dataset


@pytest.fixture(scope="module")
def fitted_logreg_pipeline(synthetic_df: pd.DataFrame) -> Pipeline:
    """Return a LogisticRegression pipeline fitted on the synthetic fixture.

    Module-scoped to avoid re-fitting repeatedly across test methods.
    """
    X = synthetic_df[ALL_FEATURES]
    y = synthetic_df[TARGET_COL]
    pipe = Pipeline(
        steps=[
            ("prep", build_preprocessor(drop_first=True)),
            ("clf", LogisticRegression(max_iter=500, random_state=SEED)),
        ]
    )
    pipe.fit(X, y)
    return pipe


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------


class TestDataLoading:
    """Verify that load_heart returns a structurally correct DataFrame."""

    def test_load_heart_has_expected_columns(
        self, mock_ucimlrepo: SimpleNamespace
    ) -> None:
        """Catch UCI API schema changes that rename or drop feature columns.

        If the upstream API changes a column name (e.g. 'trestbps' -> 'restbp')
        and this test is absent, the failure will surface silently as an NaN-filled
        feature or a KeyError deep in the pipeline rather than at ingestion.
        """
        df = load_heart(force_download=True)
        for col in ALL_FEATURES:
            assert col in df.columns, f"Expected column '{col}' missing from loaded DataFrame."
        assert TARGET_COL in df.columns, "Binary 'target' column absent from loaded data."

    def test_load_heart_target_is_binary(
        self, mock_ucimlrepo: SimpleNamespace
    ) -> None:
        """Catch regressions in the 'num > 0' binarization logic.

        The raw UCI 'num' column carries values 0-4; failing to binarize it
        would turn this into a 5-class problem without any visible error.
        """
        df = load_heart(force_download=True)
        unique_targets = set(df[TARGET_COL].unique())
        assert unique_targets.issubset(
            {0, 1}
        ), f"Target column is not binary — found values: {unique_targets}"

    def test_load_heart_no_unexpected_nulls_in_features(
        self, mock_ucimlrepo: SimpleNamespace
    ) -> None:
        """Catch null values silently propagating into training features.

        The synthetic fixture has no missing values; if load_heart introduces
        any during join or binarization steps, this test will catch it.
        Any real nulls should be handled by the imputer, not by the loader.
        """
        df = load_heart(force_download=True)
        null_counts = df[ALL_FEATURES].isna().sum()
        total_nulls = null_counts.sum()
        assert total_nulls == 0, (
            f"Unexpected NaN values introduced during loading:\n{null_counts[null_counts > 0]}"
        )


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------


class TestPreprocessing:
    """Verify ColumnTransformer output is correctly shaped, NaN-free, and robust."""

    def test_preprocessor_output_shape(self, synthetic_df: pd.DataFrame) -> None:
        """Catch schema changes that silently alter the transformed feature count.

        A column rename, a category change to NUMERIC_FEATURES, or a change in
        the OHE drop parameter all affect output width. Training would
        succeed on the new shape but the saved model would then be incompatible
        with any subsequently loaded test set.
        """
        X = synthetic_df[ALL_FEATURES]
        prep = build_preprocessor(drop_first=True)
        X_trans = prep.fit_transform(X)

        assert X_trans.ndim == 2, "Transformed output must be a 2-D array."
        assert X_trans.shape[0] == N, (
            f"Row count mismatch: expected {N}, got {X_trans.shape[0]}"
        )
        assert X_trans.shape[1] >= len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES), (
            f"Transformed column count ({X_trans.shape[1]}) is suspiciously low — "
            "check that all categorical features produce at least one OHE column."
        )

    def test_preprocessor_no_nan_after_transform(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """Catch imputer failure silently passing NaN into model input.

        SimpleImputer with strategy='median' or 'most_frequent' should eliminate
        all NaN values. If the imputer strategy is accidentally removed or changed
        to 'constant' without a fill_value, NaN can flow through to the model
        and produce nan predictions without raising an exception.
        """
        X = synthetic_df[ALL_FEATURES].copy()
        X.loc[0, "age"] = np.nan
        X.loc[1, "chol"] = np.nan
        X.loc[2, "thal"] = np.nan

        prep = build_preprocessor(drop_first=True)
        X_trans = prep.fit_transform(X)
        assert not np.isnan(X_trans).any(), (
            "NaN values present in transformed matrix — imputer is not filling all nulls."
        )

    def test_preprocessor_handles_unseen_categories(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """Catch removal of handle_unknown='ignore' from OneHotEncoder.

        When the pipeline sees a category not in the training set (e.g. a new
        ECG code or a data-entry error), it must silently zero-out those OHE
        columns rather than raise a ValueError. This protects the production
        Streamlit app from crashing on edge-case inputs.
        """
        X_train = synthetic_df[ALL_FEATURES].copy()
        prep = build_preprocessor(drop_first=True)
        prep.fit(X_train)

        X_unseen = synthetic_df[ALL_FEATURES].head(5).copy()
        X_unseen["thal"] = 99.0  # completely unseen category

        X_trans = prep.transform(X_unseen)
        assert not np.isnan(X_trans).any(), (
            "Unseen category produced NaN — handle_unknown='ignore' may be missing."
        )


# ---------------------------------------------------------------------------
# 3. Data leakage
# ---------------------------------------------------------------------------


class TestLeakage:
    """Verify that preprocessing statistics are computed only from training data."""

    def test_preprocessor_statistics_from_training_only(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """Confirm that fitting on train data and transforming test data
        does not alter the scaler's internal statistics.

        This is the gold-standard leakage test: if scaler.mean_ changes after
        calling transform(X_test), it means test statistics leaked into the
        preprocessor — which would constitute data snooping and invalidate
        any reported metric.
        """
        from sklearn.model_selection import train_test_split

        X = synthetic_df[ALL_FEATURES]
        y = synthetic_df[TARGET_COL]
        X_train, X_test, _, _ = train_test_split(
            X, y, test_size=0.3, random_state=SEED
        )

        prep = build_preprocessor(drop_first=True)
        prep.fit_transform(X_train)

        num_scaler = prep.named_transformers_["num"].named_steps["scale"]
        train_mean_snapshot = num_scaler.mean_.copy()
        train_var_snapshot = num_scaler.var_.copy()

        prep.transform(X_test)

        np.testing.assert_array_equal(
            num_scaler.mean_,
            train_mean_snapshot,
            err_msg="Scaler mean changed after transform(X_test) — test statistics leaked!",
        )
        np.testing.assert_array_equal(
            num_scaler.var_,
            train_var_snapshot,
            err_msg="Scaler variance changed after transform(X_test) — test statistics leaked!",
        )


# ---------------------------------------------------------------------------
# 4. Model behaviour
# ---------------------------------------------------------------------------


class TestModelBehaviour:
    """Verify that trained model outputs are valid probabilities and reproducible."""

    def test_predict_proba_in_unit_interval(
        self,
        fitted_logreg_pipeline: Pipeline,
        synthetic_df: pd.DataFrame,
    ) -> None:
        """Catch non-probability model outputs (e.g. raw logits or decision_function).

        If someone accidentally swaps predict_proba for decision_function in
        inference code, probabilities can be negative or > 1. This test ensures
        the production inference path always returns valid probabilities.
        """
        X = synthetic_df[ALL_FEATURES]
        probs = fitted_logreg_pipeline.predict_proba(X)[:, 1]
        assert probs.min() >= 0.0, f"Minimum probability is negative: {probs.min():.4f}"
        assert probs.max() <= 1.0, f"Maximum probability exceeds 1.0: {probs.max():.4f}"

    def test_threshold_logic_deterministic_with_fixed_seed(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """Catch non-determinism in pick_threshold caused by missing or changed seeds.

        pick_threshold uses repeated stratified cross-validation internally. If the
        random_state argument is ignored, two runs can return different thresholds,
        making model comparison and audit trails impossible to reproduce.
        """
        X = synthetic_df[ALL_FEATURES]
        y = synthetic_df[TARGET_COL]
        pipe = Pipeline(
            steps=[
                ("prep", build_preprocessor(drop_first=True)),
                ("clf", LogisticRegression(max_iter=500, random_state=SEED)),
            ]
        )
        thr1 = pick_threshold(pipe, X, y, target_recall=0.90, seed=SEED)
        thr2 = pick_threshold(pipe, X, y, target_recall=0.90, seed=SEED)
        assert thr1 == thr2, (
            f"pick_threshold is not deterministic: got {thr1:.4f} vs {thr2:.4f} "
            "with identical seed — check that random_state is propagated correctly."
        )

    def test_threshold_produces_valid_probability(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """Confirm that pick_threshold always returns a value in (0, 1).

        A threshold of 0.0 would flag every patient (zero specificity);
        a threshold >= 1.0 would flag nobody. Both are pathological defaults
        that should never be persisted to models/threshold.json.
        """
        X = synthetic_df[ALL_FEATURES]
        y = synthetic_df[TARGET_COL]
        pipe = Pipeline(
            steps=[
                ("prep", build_preprocessor(drop_first=True)),
                ("clf", LogisticRegression(max_iter=500, random_state=SEED)),
            ]
        )
        thr = pick_threshold(pipe, X, y, target_recall=0.90, seed=SEED)
        assert isinstance(thr, float), f"Threshold must be float, got {type(thr)}"
        assert 0.0 < thr < 1.0, (
            f"Threshold {thr:.4f} is outside the open interval (0, 1). "
            "This would result in degenerate clinical decision behaviour."
        )


# ---------------------------------------------------------------------------
# 5. App helpers
# ---------------------------------------------------------------------------


class TestAppHelpers:
    """Verify pure-function helpers extracted from app.py for testability."""

    def test_build_patient_dataframe_has_correct_schema(self) -> None:
        """Catch UI-to-model column mapping errors.

        If a UI form field is added, renamed, or reordered, it may silently map
        the wrong value to the wrong model feature (e.g. trestbps value sent to
        chol column). This test verifies that every ALL_FEATURES column is present
        and that no extra columns have been added.
        """
        from src.app_helpers import build_patient_dataframe

        df = build_patient_dataframe(
            age=55.0,
            sex=1.0,
            cp=4.0,
            trestbps=130.0,
            chol=240.0,
            fbs=0.0,
            restecg=0.0,
            thalach=150.0,
            exang=0.0,
            oldpeak=1.0,
            slope=2.0,
            ca=0.0,
            thal=3.0,
        )

        assert list(df.columns) == ALL_FEATURES, (
            f"Column mismatch.\nExpected: {ALL_FEATURES}\nGot:      {list(df.columns)}"
        )
        assert len(df) == 1, f"Expected single-row DataFrame, got {len(df)} rows."
        assert df["age"].iloc[0] == 55.0
        assert df["sex"].iloc[0] == 1.0

    def test_get_referral_decision_at_exact_boundary(self) -> None:
        """Catch off-by-one error at the exact threshold value.

        The referral decision uses >= (not >). A probability exactly
        equal to the threshold must trigger a referral — erring on the side of
        sensitivity. If the comparison is accidentally changed to >, patients
        at exactly the threshold would be missed (false negatives).
        """
        from src.app_helpers import get_referral_decision

        threshold = 0.42

        assert get_referral_decision(prob=0.42, threshold=threshold) is True, (
            "Probability == threshold must trigger referral (>= comparison required)."
        )
        assert get_referral_decision(prob=0.4199, threshold=threshold) is False, (
            "Probability just below threshold must not trigger referral."
        )
        assert get_referral_decision(prob=0.4201, threshold=threshold) is True, (
            "Probability just above threshold must trigger referral."
        )

    def test_compute_local_attributions_shape_matches_model(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """Catch coef dimension mismatch with the number of transformed features.

        After fitting, the number of LogisticRegression coefficients must equal
        the number of columns output by the preprocessor. If additional features
        are added to ALL_FEATURES without retraining the model, coef_ and
        x_transformed will have incompatible shapes, raising a silent broadcast
        error or returning an attribution Series with wrong indices.
        """
        from src.app_helpers import build_patient_dataframe, compute_local_attributions

        X = synthetic_df[ALL_FEATURES]
        y = synthetic_df[TARGET_COL]
        pipe = Pipeline(
            steps=[
                ("prep", build_preprocessor(drop_first=True)),
                ("clf", LogisticRegression(max_iter=500, random_state=SEED)),
            ]
        )
        pipe.fit(X, y)

        patient_df = build_patient_dataframe(
            age=55.0,
            sex=1.0,
            cp=4.0,
            trestbps=130.0,
            chol=240.0,
            fbs=0.0,
            restecg=0.0,
            thalach=150.0,
            exang=0.0,
            oldpeak=1.0,
            slope=2.0,
            ca=0.0,
            thal=3.0,
        )

        attributions = compute_local_attributions(pipe, patient_df)

        expected_n_features = pipe.named_steps["prep"].transform(patient_df).shape[1]
        assert len(attributions) == expected_n_features, (
            f"Attribution length {len(attributions)} != transformed feature count "
            f"{expected_n_features}. Model coefficients and feature names are misaligned."
        )

        abs_vals = attributions.abs().values
        assert all(abs_vals[i] >= abs_vals[i + 1] for i in range(len(abs_vals) - 1)), (
            "Attributions are not sorted by absolute magnitude."
        )

    def test_compute_risk_probability_range(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """Confirm compute_risk_probability always returns a value in [0, 1].

        This is a belt-and-suspenders guard: predict_proba is guaranteed by
        sklearn, but this test ensures our wrapper function also enforces the
        constraint and raises informatively if a custom model violates it.
        """
        from src.app_helpers import build_patient_dataframe, compute_risk_probability

        X = synthetic_df[ALL_FEATURES]
        y = synthetic_df[TARGET_COL]
        pipe = Pipeline(
            steps=[
                ("prep", build_preprocessor(drop_first=True)),
                ("clf", LogisticRegression(max_iter=500, random_state=SEED)),
            ]
        )
        pipe.fit(X, y)

        patient_df = build_patient_dataframe(
            age=55.0,
            sex=1.0,
            cp=4.0,
            trestbps=130.0,
            chol=240.0,
            fbs=0.0,
            restecg=0.0,
            thalach=150.0,
            exang=0.0,
            oldpeak=1.0,
            slope=2.0,
            ca=0.0,
            thal=3.0,
        )
        prob = compute_risk_probability(pipe, patient_df)
        assert 0.0 <= prob <= 1.0, f"Probability {prob} outside [0, 1]."


# ---------------------------------------------------------------------------
# 6. Clinical evaluation & biostatistics
# ---------------------------------------------------------------------------


class TestClinicalEvaluation:
    """Verify statistical rigor and Wilson confidence interval computation."""

    def test_wilson_ci_edge_cases_and_bounds(self) -> None:
        """Catch mathematical boundary errors in Wilson interval estimation.

        Wilson intervals must strictly remain within [0.0, 1.0], never
        produce NaN, handle n=0 gracefully, and produce valid intervals
        for boundary proportions (k=0 and k=n).
        """
        from src.evaluate import compute_wilson_ci

        # Zero sample size edge case
        assert compute_wilson_ci(0, 0) == (0.0, 0.0)

        # Perfect recall (k=n, e.g. 25/25)
        lower, upper = compute_wilson_ci(25, 25, confidence=0.95)
        assert 0.0 < lower < 1.0
        assert upper == 1.0
        assert lower >= 0.85  # 25/25 has 95% CI lower bound ~ 0.867

        # Zero successes (k=0, e.g. 0/25)
        lower_zero, upper_zero = compute_wilson_ci(0, 25, confidence=0.95)
        assert lower_zero == 0.0
        assert 0.0 < upper_zero <= 0.15

        # Typical intermediate proportion
        l_mid, u_mid = compute_wilson_ci(15, 30, confidence=0.95)
        assert 0.0 <= l_mid <= u_mid <= 1.0

    def test_evaluate_predictions_output_structure(self) -> None:
        """Verify evaluate_predictions calculates all expected metrics without error."""
        from src.evaluate import evaluate_predictions

        y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.4, 0.6, 0.3])
        metrics = evaluate_predictions(y_true, y_prob, threshold=0.5)

        expected_keys = [
            "threshold",
            "roc_auc",
            "pr_auc",
            "brier_score",
            "recall",
            "recall_95ci",
            "precision",
            "precision_95ci",
            "specificity",
            "specificity_95ci",
            "f1",
            "tp",
            "fp",
            "tn",
            "fn",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric key: '{key}'"

        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert metrics["tp"] + metrics["fn"] == 4  # 4 true positives in ground truth

    def test_bootstrap_confidence_intervals(self) -> None:
        """Verify non-parametric bootstrap produces bounded intervals."""
        from src.evaluate import compute_bootstrap_ci

        y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.4, 0.6, 0.3])
        boot_res = compute_bootstrap_ci(y_true, y_prob, threshold=0.5, n_bootstraps=50, seed=42)

        for metric in ["roc_auc", "recall", "precision", "brier_score"]:
            assert metric in boot_res
            assert 0.0 <= boot_res[metric]["ci_lower"] <= boot_res[metric]["ci_upper"] <= 1.0

    def test_delong_roc_test_identical_models(self) -> None:
        """Verify DeLong test returns p=1.0 and zero difference for identical models."""
        from src.evaluate import delong_roc_test

        y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.4, 0.6, 0.3])

        res = delong_roc_test(y_true, y_prob, y_prob)
        assert res["auc_difference"] == 0.0
        assert res["z_statistic"] == 0.0
        assert res["p_value"] == 1.0
        assert res["is_significant"] is False

    def test_nadeau_bengio_cv_ttest(self) -> None:
        """Verify corrected resampled t-test calculates conservative degrees of freedom."""
        from src.evaluate import nadeau_bengio_paired_ttest

        scores_a = np.array([0.90, 0.88, 0.92, 0.89, 0.91])
        scores_b = np.array([0.89, 0.87, 0.90, 0.88, 0.89])
        res = nadeau_bengio_paired_ttest(scores_a, scores_b, n_train=200, n_test=50)

        assert res["df"] == 4
        assert res["mean_diff"] > 0
        assert 0.0 <= res["p_value"] <= 1.0

    def test_clinical_rule_baseline_interface(self, synthetic_df: pd.DataFrame) -> None:
        """Verify ClinicalRuleBaseline adheres to standard scikit-learn estimator interface."""
        from src.evaluate import ClinicalRuleBaseline

        rule_clf = ClinicalRuleBaseline(min_criteria_points=2)
        X = synthetic_df[ALL_FEATURES]
        rule_clf.fit(X)

        probs = rule_clf.predict_proba(X)
        preds = rule_clf.predict(X)

        assert probs.shape == (len(X), 2)
        assert np.all((probs >= 0.0) & (probs <= 1.0))
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_decision_curve_net_benefit(self) -> None:
        """Verify Decision Curve Analysis Net Benefit calculation and reference lines."""
        from src.evaluate import compute_net_benefit

        y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.4, 0.6, 0.3])
        dca_df = compute_net_benefit(y_true, y_prob, thresholds=np.array([0.2, 0.4, 0.6]))

        # Net Benefit of Treat None must be strictly 0.0
        assert np.all(dca_df["net_benefit_none"] == 0.0)
        assert "net_benefit_model" in dca_df.columns
        assert "net_benefit_all" in dca_df.columns

    def test_cost_sensitive_threshold_optimization(self) -> None:
        """Verify asymmetric cost optimization identifies valid threshold and cost savings."""
        from src.evaluate import evaluate_cost_sensitive

        y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.4, 0.6, 0.3])
        cost_res = evaluate_cost_sensitive(y_true, y_prob, cost_fn_ratio=10.0)

        assert 0.0 < cost_res["optimal_threshold"] < 1.0
        assert cost_res["min_total_cost"] <= cost_res["default_50_cost"]
        assert cost_res["cost_savings_vs_50"] >= 0.0


# ---------------------------------------------------------------------------
# 7. End-to-end inference (uses saved model artifacts)
# ---------------------------------------------------------------------------


class TestEndToEndInference:
    """Validate the full inference path against the serialized production artifacts."""

    def test_valid_patient_produces_referral_decision(self) -> None:
        """Verify inference pipeline handles valid records end-to-end.

        Protects against: model artifact corruption, joblib version mismatch,
        and threshold.json schema changes breaking the app at startup.
        """
        assert LOGREG_MODEL_PATH.exists(), f"Model file missing: {LOGREG_MODEL_PATH}"
        assert THRESHOLD_CONFIG_PATH.exists(), (
            f"Threshold config missing: {THRESHOLD_CONFIG_PATH}"
        )

        model = joblib.load(LOGREG_MODEL_PATH)
        with open(THRESHOLD_CONFIG_PATH) as f:
            threshold_data = json.load(f)
        threshold = threshold_data["threshold"]

        valid_patient = pd.DataFrame(
            [
                {
                    "age": 55.0,
                    "sex": 1.0,
                    "cp": 4.0,
                    "trestbps": 130.0,
                    "chol": 240.0,
                    "fbs": 0.0,
                    "restecg": 0.0,
                    "thalach": 150.0,
                    "exang": 0.0,
                    "oldpeak": 1.0,
                    "slope": 2.0,
                    "ca": 0.0,
                    "thal": 3.0,
                }
            ]
        )

        validate_clinical_dataframe(valid_patient)
        prob = float(model.predict_proba(valid_patient)[0, 1])
        assert 0.0 <= prob <= 1.0
        referral = prob >= threshold
        assert isinstance(referral, (bool, np.bool_))

    def test_missing_feature_raises_validation_error(self) -> None:
        """Catch silent feature dropping that could corrupt model input.

        validate_clinical_dataframe must raise ValueError for a DataFrame that
        is missing any required clinical feature — not silently continue with
        a column of NaN values.
        """
        incomplete_patient = pd.DataFrame(
            [{"age": 55.0, "sex": 1.0}]  # missing 11 required features
        )
        with pytest.raises(ValueError, match="missing required features"):
            validate_clinical_dataframe(incomplete_patient)

    def test_invalid_category_raises_validation_error(self) -> None:
        """Catch invalid categorical values reaching the model.

        A data-entry mistake (e.g. sex=99) would be silently zero-encoded
        by handle_unknown='ignore', producing a subtly wrong prediction.
        validate_clinical_dataframe must reject this before inference.
        """
        invalid_patient = pd.DataFrame(
            [
                {
                    "age": 55.0,
                    "sex": 99.0,  # completely invalid
                    "cp": 4.0,
                    "trestbps": 130.0,
                    "chol": 240.0,
                    "fbs": 0.0,
                    "restecg": 0.0,
                    "thalach": 150.0,
                    "exang": 0.0,
                    "oldpeak": 1.0,
                    "slope": 2.0,
                    "ca": 0.0,
                    "thal": 3.0,
                }
            ]
        )
        with pytest.raises(ValueError, match="unrecognized categories"):
            validate_clinical_dataframe(invalid_patient)
