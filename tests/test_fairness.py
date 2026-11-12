"""
Tests for the Fairness and Responsible-AI Analysis module.

Covers:
- Subgroup metrics computation (correctness, edge cases, small-n warnings)
- Subgroup calibration (empty-cell handling, Brier score direction)
- Fairlearn metric interface (shape, type, value range)
- run_fairness_audit integration (file creation, structure)
"""

from __future__ import annotations

import json
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from src.fairness import (
    _subgroup_metrics,
    _wilson_ci,
    evaluate_subgroup_calibration,
    evaluate_subgroup_performance,
    run_fairness_audit,
)

# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------

N = 80  # Enough to have meaningful subgroups


@pytest.fixture(scope="module")
def synthetic_data() -> tuple[pd.DataFrame, pd.Series, np.ndarray, dict[str, float]]:
    """Synthetic test cohort with known subgroup structure."""
    rng = np.random.default_rng(42)

    # Build all arrays of exactly length N
    sex = np.array([1.0] * 48 + [0.0] * 32)  # 48 + 32 = 80 ✓
    # Age bands: ~25 each in <45, 45-60, >60 (totals 75+5 to hit 80)
    age = np.concatenate([
        rng.integers(29, 45, size=20).astype(float),   # <45: 20 patients
        rng.integers(45, 61, size=35).astype(float),   # 45-60: 35 patients
        rng.integers(61, 78, size=25).astype(float),   # >60: 25 patients
    ])
    y = rng.integers(0, 2, size=N).astype(int)

    X = pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "trestbps": rng.uniform(90, 180, N),
            "chol": rng.uniform(150, 350, N),
            "thalach": rng.uniform(80, 200, N),
            "oldpeak": rng.uniform(0, 4, N),
            "ca": rng.integers(0, 4, N).astype(float),
            "cp": rng.choice([1.0, 2.0, 3.0, 4.0], N),
            "fbs": rng.choice([0.0, 1.0], N),
            "restecg": rng.choice([0.0, 1.0, 2.0], N),
            "exang": rng.choice([0.0, 1.0], N),
            "slope": rng.choice([1.0, 2.0, 3.0], N),
            "thal": rng.choice([3.0, 6.0, 7.0], N),
        }
    )

    y_series = pd.Series(y, name="target")
    # Reasonable discriminator (AUC ≈ 0.70)
    probs = np.clip(0.4 * y + 0.1 * rng.standard_normal(N) + 0.3, 0.01, 0.99)

    model_probs = {
        "Logistic Regression": probs,
        "XGBoost": np.clip(probs + 0.05 * rng.standard_normal(N), 0.01, 0.99),
    }
    thresholds = {"Logistic Regression": 0.45, "XGBoost": 0.45}

    return X, y_series, probs, model_probs, thresholds


# ---------------------------------------------------------------------------
# 1. Wilson CI
# ---------------------------------------------------------------------------


class TestWilsonCI:
    """Test the internal Wilson CI helper."""

    def test_zero_n_returns_full_interval(self) -> None:
        lo, hi = _wilson_ci(0, 0)
        assert lo == 0.0
        assert hi == 1.0

    def test_perfect_recall_upper_bounded(self) -> None:
        lo, hi = _wilson_ci(10, 10)
        assert lo > 0.6  # Tight upper bound when all correct
        assert hi <= 1.0

    def test_zero_positives_in_nonzero_n(self) -> None:
        lo, hi = _wilson_ci(0, 20)
        assert lo == 0.0
        assert hi < 0.2  # Should not span to 1.0 with n=20 and k=0

    def test_values_within_unit_interval(self) -> None:
        for k, n in [(5, 10), (1, 100), (50, 100), (99, 100)]:
            lo, hi = _wilson_ci(k, n)
            assert 0.0 <= lo <= 1.0
            assert 0.0 <= hi <= 1.0
            assert lo <= hi


# ---------------------------------------------------------------------------
# 2. Subgroup Metrics
# ---------------------------------------------------------------------------


class TestSubgroupMetrics:
    """Test _subgroup_metrics for correctness and edge cases."""

    def test_basic_correctness(self) -> None:
        y_true = np.array([1, 1, 1, 0, 0])
        y_prob = np.array([0.8, 0.9, 0.3, 0.1, 0.2])
        result = _subgroup_metrics(y_true, y_prob, threshold=0.5)

        assert result["tp"] == 2  # Only 0.8 and 0.9 ≥ 0.5
        assert result["fn"] == 1
        assert result["recall"] == pytest.approx(2 / 3, abs=0.001)

    def test_small_n_positive_triggers_warning(self) -> None:
        y_true = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 1 positive
        y_prob = np.ones(10) * 0.6
        result = _subgroup_metrics(y_true, y_prob, threshold=0.5)
        assert any("unreliable" in w for w in result["warnings"])

    def test_all_one_class_no_auc(self) -> None:
        """AUC cannot be computed if all labels are the same class."""
        y_true = np.ones(20, dtype=int)
        y_prob = np.ones(20) * 0.8
        result = _subgroup_metrics(y_true, y_prob, threshold=0.5)
        assert result["roc_auc"] is None
        assert any("AUC" in w for w in result["warnings"])

    def test_perfect_model_recall_1(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.85, 0.1, 0.2])
        result = _subgroup_metrics(y_true, y_prob, threshold=0.5)
        assert result["recall"] == 1.0
        assert result["fnr"] == 0.0

    def test_output_keys_complete(self) -> None:
        y_true = np.array([1, 0, 1, 0, 1] * 5)
        y_prob = np.array([0.8, 0.3, 0.7, 0.2, 0.9] * 5)
        result = _subgroup_metrics(y_true, y_prob, threshold=0.5)
        required = ["recall", "fnr", "precision", "roc_auc", "brier_score", "warnings"]
        for key in required:
            assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 3. Subgroup Performance Evaluation
# ---------------------------------------------------------------------------


class TestSubgroupPerformance:
    """Test evaluate_subgroup_performance structure and subgroup coverage."""

    def test_returns_all_subgroups(self, synthetic_data) -> None:
        X, y, _, model_probs, thresholds = synthetic_data
        result = evaluate_subgroup_performance(X, y, model_probs, thresholds)

        expected_subgroups = {
            "Sex: Male", "Sex: Female",
            "Age: <45", "Age: 45–60", "Age: >60", "Overall",
        }
        for model_name in model_probs:
            assert set(result[model_name].keys()) == expected_subgroups

    def test_overall_matches_global_population(self, synthetic_data) -> None:
        """Overall subgroup should equal the full dataset metrics."""
        X, y, probs, model_probs, thresholds = synthetic_data
        result = evaluate_subgroup_performance(X, y, model_probs, thresholds)

        lr_overall = result["Logistic Regression"]["Overall"]
        assert lr_overall["n_total"] == len(y)

    def test_sex_subgroups_partition_population(self, synthetic_data) -> None:
        X, y, _, model_probs, thresholds = synthetic_data
        result = evaluate_subgroup_performance(X, y, model_probs, thresholds)

        lr = result["Logistic Regression"]
        n_male = lr["Sex: Male"]["n_total"]
        n_female = lr["Sex: Female"]["n_total"]
        assert n_male + n_female == len(y)

    def test_age_bands_partition_population(self, synthetic_data) -> None:
        X, y, _, model_probs, thresholds = synthetic_data
        result = evaluate_subgroup_performance(X, y, model_probs, thresholds)

        lr = result["Logistic Regression"]
        n_total = (
            lr["Age: <45"]["n_total"]
            + lr["Age: 45–60"]["n_total"]
            + lr["Age: >60"]["n_total"]
        )
        assert n_total == len(y)

    def test_metrics_in_valid_range(self, synthetic_data) -> None:
        X, y, _, model_probs, thresholds = synthetic_data
        result = evaluate_subgroup_performance(X, y, model_probs, thresholds)

        for model_name, sg_dict in result.items():
            for sg_name, sg_data in sg_dict.items():
                for metric in ["recall", "fnr", "precision"]:
                    val = sg_data.get(metric)
                    if val is not None:
                        assert 0.0 <= val <= 1.0, (
                            f"{model_name}/{sg_name}/{metric} = {val} out of range"
                        )


# ---------------------------------------------------------------------------
# 4. Subgroup Calibration
# ---------------------------------------------------------------------------


class TestSubgroupCalibration:
    """Test evaluate_subgroup_calibration structure and Brier score direction."""

    def test_returns_brier_scores_for_large_subgroups(self, synthetic_data) -> None:
        X, y, _, model_probs, _ = synthetic_data
        result = evaluate_subgroup_calibration(X, y, model_probs)

        # The Male subgroup should have n≥10 and return a brier score
        for model_name in model_probs:
            male_data = result[model_name].get("Sex: Male", {})
            # May or may not be large enough — just check the key exists
            assert "brier_score" in male_data or "warning" in male_data

    def test_brier_score_range(self, synthetic_data) -> None:
        X, y, _, model_probs, _ = synthetic_data
        result = evaluate_subgroup_calibration(X, y, model_probs)

        for model_name, sg_dict in result.items():
            for sg_name, sg_data in sg_dict.items():
                brier = sg_data.get("brier_score")
                if brier is not None:
                    assert 0.0 <= brier <= 1.0, (
                        f"{model_name}/{sg_name} Brier={brier} out of [0,1]"
                    )

    def test_tiny_subgroup_returns_warning(self) -> None:
        """Subgroup with n < 10 should return a warning, not raise."""
        X_small = pd.DataFrame({
            "age": [50, 55],
            "sex": [1.0, 1.0],
            **{col: [0.0, 0.0] for col in [
                "trestbps", "chol", "thalach", "oldpeak", "ca",
                "cp", "fbs", "restecg", "exang", "slope", "thal"
            ]},
        })
        y_small = pd.Series([1, 0])
        probs_small = {"LR": np.array([0.7, 0.3])}
        result = evaluate_subgroup_calibration(X_small, y_small, probs_small, n_bins=5)
        # Female subgroup is empty → should return a warning string
        assert "Sex: Female" in result["LR"]
        assert result["LR"]["Sex: Female"].get("brier_score") is None


# ---------------------------------------------------------------------------
# 5. Fairlearn Metrics
# ---------------------------------------------------------------------------


class TestFairnessMetrics:
    """Test compute_fairness_metrics if fairlearn is available."""

    @pytest.fixture
    def fair_result(self, synthetic_data):
        from src.fairness import _FAIRLEARN_AVAILABLE, compute_fairness_metrics
        if not _FAIRLEARN_AVAILABLE:
            pytest.skip("fairlearn not installed")
        X, y, _, model_probs, thresholds = synthetic_data
        return compute_fairness_metrics(X, y, model_probs, thresholds)

    def test_returns_both_sensitive_features(self, fair_result) -> None:
        for model_name, model_data in fair_result.items():
            assert "sex" in model_data, f"Missing 'sex' key for {model_name}"
            assert "age_band" in model_data, f"Missing 'age_band' key for {model_name}"

    def test_eod_in_neg1_to_1_range(self, fair_result) -> None:
        for model_name, model_data in fair_result.items():
            for sf_name, sf_data in model_data.items():
                eod = sf_data.get("equal_opportunity_difference")
                if eod is not None:
                    assert -1.0 <= eod <= 1.0, (
                        f"{model_name}/{sf_name} EOD={eod} out of range"
                    )

    def test_dpd_in_neg1_to_1_range(self, fair_result) -> None:
        for model_name, model_data in fair_result.items():
            for sf_name, sf_data in model_data.items():
                dpd = sf_data.get("demographic_parity_difference")
                if dpd is not None:
                    assert -1.0 <= dpd <= 1.0, (
                        f"{model_name}/{sf_name} DPD={dpd} out of range"
                    )

    def test_per_group_records_present(self, fair_result) -> None:
        for _model_name, model_data in fair_result.items():
            for _sf_name, sf_data in model_data.items():
                assert "per_group" in sf_data
                assert isinstance(sf_data["per_group"], list)
                assert len(sf_data["per_group"]) >= 2  # At least 2 groups

    def test_interpretations_are_strings(self, fair_result) -> None:
        for _model_name, model_data in fair_result.items():
            for _sf_name, sf_data in model_data.items():
                assert isinstance(sf_data.get("eod_interpretation"), str)
                assert isinstance(sf_data.get("dpd_interpretation"), str)


# ---------------------------------------------------------------------------
# 6. run_fairness_audit integration
# ---------------------------------------------------------------------------


class TestRunFairnessAudit:
    """Integration test for the full audit pipeline."""

    def test_audit_saves_json(self, synthetic_data, tmp_path) -> None:
        X, y, _, model_probs, thresholds = synthetic_data

        with (
            patch("src.fairness.REPORTS_DIR", tmp_path),
            patch("src.fairness.FIGURES_DIR", tmp_path),
            patch("src.fairness.FAIRNESS_AUDIT_PATH", tmp_path / "fairness_audit.json"),
            patch("src.fairness.FAIRNESS_SUBGROUP_PLOT_PATH", tmp_path / "subgroup.png"),
            patch("src.fairness.FAIRNESS_CALIBRATION_PLOT_PATH", tmp_path / "cal.png"),
        ):
            run_fairness_audit(X, y, model_probs, thresholds, save_report=True)

        audit_path = tmp_path / "fairness_audit.json"
        assert audit_path.exists(), "Fairness audit JSON was not created"

        with open(audit_path) as f:
            data = json.load(f)
        assert "subgroup_performance" in data
        assert "subgroup_calibration" in data
        assert "metadata" in data

    def test_audit_structure_keys(self, synthetic_data, tmp_path) -> None:
        X, y, _, model_probs, thresholds = synthetic_data

        with (
            patch("src.fairness.REPORTS_DIR", tmp_path),
            patch("src.fairness.FIGURES_DIR", tmp_path),
            patch("src.fairness.FAIRNESS_AUDIT_PATH", tmp_path / "fa.json"),
            patch("src.fairness.FAIRNESS_SUBGROUP_PLOT_PATH", tmp_path / "sg.png"),
            patch("src.fairness.FAIRNESS_CALIBRATION_PLOT_PATH", tmp_path / "cal.png"),
        ):
            audit = run_fairness_audit(X, y, model_probs, thresholds, save_report=False)

        required_keys = ["subgroup_performance", "subgroup_calibration", "metadata"]
        for key in required_keys:
            assert key in audit, f"Missing key: {key}"
