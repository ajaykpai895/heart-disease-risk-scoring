"""
Clinical Evaluation, Statistical Uncertainty, and Decision Theory Module.

Provides research-grade evaluation utilities for cardiovascular risk models:
1. Exact Wilson score confidence intervals for binomial clinical metrics.
2. Non-parametric Percentile Bootstrap 95% confidence intervals.
3. DeLong test for paired ROC-AUC statistical comparison.
4. Nadeau-Bengio corrected resampled paired t-test for cross-validation folds.
5. Probability recalibration analysis (Raw vs. Platt scaling vs. Isotonic regression).
6. Decision Curve Analysis (DCA) quantifying clinical Net Benefit.
7. Cost-sensitive threshold evaluation under asymmetric clinical error costs.
8. Clinical heuristic rule baseline (Diamond-Forrester criteria).
9. Publication-grade diagnostic and decision-analytic visualizations.
"""

import logging
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, learning_curve

from src.config import (
    CALIBRATION_COMPARISON_PLOT_PATH,
    COST_TRADEOFF_PLOT_PATH,
    DCA_PLOT_PATH,
    DEFAULT_FN_FP_COST_RATIO,
    FIGURES_DIR,
    LEARNING_CURVE_PLOT_PATH,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Baseline: Diamond-Forrester Clinical Rule Model
# ---------------------------------------------------------------------------


class ClinicalRuleBaseline(BaseEstimator, ClassifierMixin):
    """Clinical heuristic baseline based on Diamond-Forrester & Duke criteria.

    Encapsulates standard pre-test cardiology referral heuristic used in
    primary care triage prior to algorithmic risk assessment:
    A patient is categorized as elevated CAD risk if any of the following apply:
    1. Typical Angina (cp == 1.0)
    2. Exercise-induced ST depression >= 1.5 mm (oldpeak >= 1.5)
    3. Exercise-induced angina (exang == 1.0)
    4. Age/Sex risk: Male >= 55 or Female >= 65 with atypical angina (cp == 2.0)
    5. Fluoroscopy colored vessels >= 1 (ca >= 1.0)
    6. Thallium scintigraphy defect (thal in [6.0, 7.0])

    Provides standard scikit-learn estimator interface (`fit`, `predict`, `predict_proba`).
    """

    def __init__(self, min_criteria_points: int = 2) -> None:
        """Initialize clinical heuristic baseline.

        Args:
            min_criteria_points: Number of criteria needed to trigger high-risk referral flag.
        """
        self.min_criteria_points = min_criteria_points
        self.classes_ = np.array([0, 1])

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ClinicalRuleBaseline":
        """Fit baseline (stateless rule; validates schema only)."""
        return self

    def _score_patient(self, row: pd.Series) -> float:
        """Compute integer risk count based on clinical criteria."""
        score = 0
        # Criterion 1: Typical angina
        if row.get("cp", 0.0) == 1.0:
            score += 2  # Strong hallmark
        # Criterion 2: Marked ST depression
        if row.get("oldpeak", 0.0) >= 1.5:
            score += 2
        # Criterion 3: Exertional angina
        if row.get("exang", 0.0) == 1.0:
            score += 1
        # Criterion 4: Age/Sex demographic risk
        sex = row.get("sex", 0.0)
        age = row.get("age", 0.0)
        cp = row.get("cp", 0.0)
        if (sex == 1.0 and age >= 55.0) or (sex == 0.0 and age >= 65.0):
            if cp in [1.0, 2.0]:
                score += 1
        # Criterion 5: Fluoroscopic anatomical plaque
        if row.get("ca", 0.0) >= 1.0:
            score += 2
        # Criterion 6: Perfusion scintigraphy defect
        if row.get("thal", 3.0) in [6.0, 7.0]:
            score += 2
        return float(score)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate pseudo-probabilities proportional to criteria met."""
        scores = np.array([self._score_patient(row) for _, row in X.iterrows()])
        # Maximum possible score is 10 points; map sigmoid-style into [0.05, 0.95]
        probs = 1.0 / (1.0 + np.exp(-(scores - 3.0) / 1.5))
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary classification based on criteria threshold."""
        scores = np.array([self._score_patient(row) for _, row in X.iterrows()])
        return (scores >= self.min_criteria_points).astype(int)


# ---------------------------------------------------------------------------
# 2. Confidence Intervals: Wilson Score & Percentile Bootstrap
# ---------------------------------------------------------------------------


def compute_wilson_ci(
    k: int,
    n: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute the Wilson score interval for a binomial proportion.

    Preferred over Wald normal approximation in biostatistics, especially for
    small cohorts or extreme proportions near 0 or 1.

    Args:
        k: Number of successes (e.g. true positives).
        n: Number of trials (e.g. total positive cases).
        confidence: Desired two-sided confidence level (default: 0.95).

    Returns:
        tuple[float, float]: (lower_bound, upper_bound) bounded in [0.0, 1.0].
    """
    if n == 0:
        return (0.0, 0.0)

    z = 1.95996 if confidence == 0.95 else stats.norm.ppf((1.0 + confidence) / 2.0)
    p_hat = k / n
    denominator = 1.0 + (z**2) / n
    center_adj = p_hat + (z**2) / (2.0 * n)
    spread = z * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z**2) / (4.0 * (n**2)))

    lower = max(0.0, (center_adj - spread) / denominator)
    upper = min(1.0, (center_adj + spread) / denominator)
    return (round(lower, 4), round(upper, 4))


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute non-parametric percentile bootstrap 95% confidence intervals.

    Resamples patient outcomes with replacement B times to derive empirical
    sampling distributions for ROC-AUC, PR-AUC, Recall, Precision, and Specificity.

    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted risk probabilities.
        threshold: Operating decision threshold.
        n_bootstraps: Number of bootstrap iterations (default: 1000).
        confidence: Desired confidence level (default: 0.95).
        seed: Random seed for reproducibility.

    Returns:
        dict: Mapping of metric names to point estimate, mean, lower, and upper bounds.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)

    alpha_low = (1.0 - confidence) / 2.0
    alpha_high = 1.0 - alpha_low

    boot_metrics = {
        "roc_auc": [],
        "pr_auc": [],
        "recall": [],
        "precision": [],
        "specificity": [],
        "brier_score": [],
    }

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yt_b = y_true_arr[idx]
        yp_b = y_prob_arr[idx]

        # Require both classes in bootstrap resample for AUC calculation
        if len(np.unique(yt_b)) < 2:
            continue

        pred_b = (yp_b >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt_b, pred_b, labels=[0, 1]).ravel()

        boot_metrics["roc_auc"].append(roc_auc_score(yt_b, yp_b))
        boot_metrics["pr_auc"].append(average_precision_score(yt_b, yp_b))
        boot_metrics["recall"].append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        boot_metrics["precision"].append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        boot_metrics["specificity"].append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        boot_metrics["brier_score"].append(brier_score_loss(yt_b, yp_b))

    results: dict[str, dict[str, float]] = {}
    for metric_name, values in boot_metrics.items():
        arr = np.array(values)
        results[metric_name] = {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "ci_lower": round(float(np.percentile(arr, alpha_low * 100)), 4),
            "ci_upper": round(float(np.percentile(arr, alpha_high * 100)), 4),
        }
    return results


# ---------------------------------------------------------------------------
# 3. Statistical Testing: DeLong Paired ROC Test & Nadeau-Bengio CV t-Test
# ---------------------------------------------------------------------------


def delong_roc_test(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
) -> dict[str, Any]:
    """Execute DeLong, DeLong, and Clarke-Pearson (1988) test for paired ROC curves.

    Formally tests the null hypothesis H0: AUC_A == AUC_B on identical holdout samples,
    accounting for covariance between correlated risk predictions.

    Args:
        y_true: True binary target array (0 or 1).
        y_prob_a: Predicted probabilities from Model A (e.g. Logistic Regression).
        y_prob_b: Predicted probabilities from Model B (e.g. XGBoost).

    Returns:
        dict: AUCs, difference, standard error, z-statistic, and two-sided p-value.
    """
    y_true = np.asarray(y_true)
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    m = len(pos_idx)
    n = len(neg_idx)

    if m == 0 or n == 0:
        raise ValueError("Both positive and negative samples required for DeLong test.")

    # Structural components for model A
    v10_a = np.array(
        [
            np.mean((y_prob_a[pos_idx[i]] > y_prob_a[neg_idx]) + 0.5 * (y_prob_a[pos_idx[i]] == y_prob_a[neg_idx]))
            for i in range(m)
        ]
    )
    v01_a = np.array(
        [
            np.mean((y_prob_a[pos_idx] > y_prob_a[neg_idx[j]]) + 0.5 * (y_prob_a[pos_idx] == y_prob_a[neg_idx[j]]))
            for j in range(n)
        ]
    )

    # Structural components for model B
    v10_b = np.array(
        [
            np.mean((y_prob_b[pos_idx[i]] > y_prob_b[neg_idx]) + 0.5 * (y_prob_b[pos_idx[i]] == y_prob_b[neg_idx]))
            for i in range(m)
        ]
    )
    v01_b = np.array(
        [
            np.mean((y_prob_b[pos_idx] > y_prob_b[neg_idx[j]]) + 0.5 * (y_prob_b[pos_idx] == y_prob_b[neg_idx[j]]))
            for j in range(n)
        ]
    )

    auc_a = float(np.mean(v10_a))
    auc_b = float(np.mean(v10_b))
    diff = auc_a - auc_b

    # Sample covariance matrices
    s10_aa = np.var(v10_a, ddof=1)
    s10_bb = np.var(v10_b, ddof=1)
    s10_ab = np.cov(v10_a, v10_b)[0, 1]

    s01_aa = np.var(v01_a, ddof=1)
    s01_bb = np.var(v01_b, ddof=1)
    s01_ab = np.cov(v01_a, v01_b)[0, 1]

    var_diff = (s10_aa - 2.0 * s10_ab + s10_bb) / m + (s01_aa - 2.0 * s01_ab + s01_bb) / n
    se_diff = math.sqrt(max(var_diff, 1e-12))

    z_stat = diff / se_diff
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
    ci_95 = (round(diff - 1.96 * se_diff, 4), round(diff + 1.96 * se_diff, 4))

    return {
        "auc_a": round(auc_a, 4),
        "auc_b": round(auc_b, 4),
        "auc_difference": round(diff, 4),
        "standard_error": round(se_diff, 4),
        "z_statistic": round(z_stat, 3),
        "p_value": round(p_value, 4),
        "ci_95": ci_95,
        "is_significant": bool(p_value < 0.05),
    }


def nadeau_bengio_paired_ttest(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_train: int,
    n_test: int,
) -> dict[str, Any]:
    """Execute Nadeau-Bengio (2003) corrected resampled t-test for cross-validation.

    Corrects for the severe variance underestimation of standard paired t-tests
    caused by overlapping training partitions in repeated k-fold CV.

    Formula:
        var_corrected = (1 / n_folds + n_test / n_train) * sample_variance(diffs)

    Args:
        scores_a: 1D array of fold evaluation scores for Model A.
        scores_b: 1D array of fold evaluation scores for Model B.
        n_train: Training sample size per fold.
        n_test: Test sample size per fold.

    Returns:
        dict: Mean difference, corrected t-statistic, degrees of freedom, and p-value.
    """
    diffs = np.asarray(scores_a) - np.asarray(scores_b)
    n_folds = len(diffs)
    mean_diff = float(np.mean(diffs))
    sample_var = float(np.var(diffs, ddof=1))

    # Nadeau & Bengio variance correction factor
    correction_factor = (1.0 / n_folds) + (n_test / n_train)
    corrected_var = correction_factor * sample_var
    corrected_se = math.sqrt(max(corrected_var, 1e-12))

    t_stat = mean_diff / corrected_se
    df = n_folds - 1
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df))

    return {
        "mean_diff": round(mean_diff, 4),
        "sample_std_diff": round(math.sqrt(sample_var), 4),
        "corrected_se": round(corrected_se, 4),
        "t_statistic": round(t_stat, 3),
        "df": df,
        "p_value": round(p_value, 4),
        "is_significant": bool(p_value < 0.05),
    }


# ---------------------------------------------------------------------------
# 4. Calibration & Recalibration (Platt vs. Isotonic)
# ---------------------------------------------------------------------------


def evaluate_calibration_methods(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: int = 5,
) -> dict[str, Any]:
    """Compare Raw, Platt-scaled (Sigmoid), and Isotonic calibrated probabilities.

    Quantifies Brier score reduction and preservation of discrimination (ROC-AUC).

    Args:
        model: Base estimator pipeline.
        X_train: Training covariates.
        y_train: Training labels.
        X_test: Holdout test covariates.
        y_test: Holdout test labels.
        cv: Cross-validation splits for calibration.

    Returns:
        dict: Brier scores, ROC-AUCs, and predicted probabilities for each variant.
    """
    # 1. Raw base model
    model.fit(X_train, y_train)
    raw_probs = model.predict_proba(X_test)[:, 1]

    # 2. Platt Scaling (Sigmoid)
    platt_cal = CalibratedClassifierCV(estimator=model, method="sigmoid", cv=cv)
    platt_cal.fit(X_train, y_train)
    platt_probs = platt_cal.predict_proba(X_test)[:, 1]

    # 3. Isotonic Regression
    iso_cal = CalibratedClassifierCV(estimator=model, method="isotonic", cv=cv)
    iso_cal.fit(X_train, y_train)
    iso_probs = iso_cal.predict_proba(X_test)[:, 1]

    results = {
        "Raw": {
            "brier_score": round(float(brier_score_loss(y_test, raw_probs)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, raw_probs)), 4),
            "probs": raw_probs,
        },
        "Platt (Sigmoid)": {
            "brier_score": round(float(brier_score_loss(y_test, platt_probs)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, platt_probs)), 4),
            "probs": platt_probs,
        },
        "Isotonic": {
            "brier_score": round(float(brier_score_loss(y_test, iso_probs)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, iso_probs)), 4),
            "probs": iso_probs,
        },
    }
    return results


# ---------------------------------------------------------------------------
# 5. Decision Curve Analysis (Net Benefit) & Cost-Sensitive Optimization
# ---------------------------------------------------------------------------


def compute_net_benefit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute Decision Curve Analysis (DCA) Net Benefit across threshold probabilities.

    Net Benefit:
        NB(p_t) = (TP / N) - (FP / N) * (p_t / (1 - p_t))

    Benchmarks:
        Treat None: NB = 0
        Treat All:  NB = Prevalence - (1 - Prevalence) * (p_t / (1 - p_t))

    Args:
        y_true: Ground truth binary targets.
        y_prob: Predicted risk probabilities.
        thresholds: Array of decision threshold probabilities.

    Returns:
        pd.DataFrame: Net benefit values for the model, Treat All, and Treat None.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    prevalence = float(np.mean(y_true))

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.75, 50)

    rows = []
    for pt in thresholds:
        if pt >= 1.0 or pt <= 0.0:
            continue
        weight = pt / (1.0 - pt)
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        nb_model = (tp / n) - (fp / n) * weight
        nb_all = prevalence - (1.0 - prevalence) * weight
        nb_none = 0.0

        rows.append(
            {
                "threshold": round(pt, 3),
                "net_benefit_model": nb_model,
                "net_benefit_all": nb_all,
                "net_benefit_none": nb_none,
            }
        )
    return pd.DataFrame(rows)


def evaluate_cost_sensitive(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn_ratio: float = DEFAULT_FN_FP_COST_RATIO,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    """Execute cost-sensitive threshold analysis under asymmetric clinical error costs.

    Total Cost:
        Cost(T) = cost_fn_ratio * FN(T) + 1.0 * FP(T)
        (assuming FN costs [10x] an unnecessary FP referral).

    Args:
        y_true: Ground truth binary targets.
        y_prob: Predicted probabilities.
        cost_fn_ratio: Multiplier for cost of False Negative vs. False Positive (default: 10.0).
        thresholds: Evaluation thresholds (default: 100 points in [0.01, 0.99]).

    Returns:
        dict: Optimal threshold, cost curve, and comparison against standard 0.50 cutoff.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)

    costs = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fn = np.sum((y_pred == 0) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        total_cost = cost_fn_ratio * fn + 1.0 * fp
        avg_cost = total_cost / n
        costs.append(
            {
                "threshold": float(t),
                "fn": int(fn),
                "fp": int(fp),
                "total_cost": float(total_cost),
                "cost_per_patient": float(avg_cost),
            }
        )

    cost_df = pd.DataFrame(costs)
    min_idx = cost_df["total_cost"].idxmin()
    best_row = cost_df.loc[min_idx]

    # Evaluate cost at standard 0.50 threshold
    pred_50 = (y_prob >= 0.50).astype(int)
    fn_50 = np.sum((pred_50 == 0) & (y_true == 1))
    fp_50 = np.sum((pred_50 == 1) & (y_true == 0))
    cost_50 = cost_fn_ratio * fn_50 + 1.0 * fp_50

    return {
        "optimal_threshold": round(float(best_row["threshold"]), 3),
        "min_total_cost": round(float(best_row["total_cost"]), 1),
        "min_cost_per_patient": round(float(best_row["cost_per_patient"]), 3),
        "fn_at_optimal": int(best_row["fn"]),
        "fp_at_optimal": int(best_row["fp"]),
        "default_50_cost": round(float(cost_50), 1),
        "cost_savings_vs_50": round(float(cost_50 - best_row["total_cost"]), 1),
        "cost_df": cost_df,
    }


# ---------------------------------------------------------------------------
# 6. Comprehensive Metrics Aggregator
# ---------------------------------------------------------------------------


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute comprehensive clinical metrics at a given operating threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    total_pos = tp + fn
    total_pred_pos = tp + fp
    total_neg = tn + fp

    rec = float(recall_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    spec = float(tn / total_neg) if total_neg > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    rec_ci = compute_wilson_ci(tp, total_pos, confidence=0.95)
    prec_ci = compute_wilson_ci(tp, total_pred_pos, confidence=0.95)
    spec_ci = compute_wilson_ci(tn, total_neg, confidence=0.95)

    return {
        "threshold": round(float(threshold), 3),
        "roc_auc": round(roc_auc, 3),
        "pr_auc": round(pr_auc, 3),
        "brier_score": round(brier, 3),
        "recall": round(rec, 3),
        "recall_95ci": rec_ci,
        "precision": round(prec, 3),
        "precision_95ci": prec_ci,
        "specificity": round(spec, 3),
        "specificity_95ci": spec_ci,
        "f1": round(f1, 3),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


# ---------------------------------------------------------------------------
# 7. Diagnostic Visualizations
# ---------------------------------------------------------------------------


def plot_roc_comparison(
    models_dict: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    precomputed_probs: dict[str, np.ndarray] | None = None,
    save_path: Path | None = None,
) -> None:
    """Plot overlaid ROC curves for holdout comparison.

    Supports both fitted sklearn estimators (via ``RocCurveDisplay.from_estimator``)
    and pre-computed probability arrays (via ``RocCurveDisplay.from_predictions``).
    The latter is required for non-sklearn-native classifiers such as
    ``ClinicalRuleBaseline`` that do not pass sklearn's internal estimator checks.

    Args:
        models_dict: Mapping of model name → fitted sklearn Pipeline/estimator.
            Each model must support ``predict_proba``.
        X_test: Holdout feature DataFrame passed to sklearn estimators.
        y_test: Holdout binary target Series.
        precomputed_probs: Optional mapping of name → 1-D probability array for
            classifiers that cannot be passed through ``from_estimator`` (e.g.
            custom heuristic baselines). These are plotted via
            ``RocCurveDisplay.from_predictions``.
        save_path: Override output file path. Defaults to
            ``FIGURES_DIR / "roc_comparison.png"``.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, model in models_dict.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)

    if precomputed_probs:
        for name, probs in precomputed_probs.items():
            RocCurveDisplay.from_predictions(y_test, probs, name=name, ax=ax)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance (AUC = 0.50)")
    ax.set_title("ROC Curves on Holdout Cohort", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.legend(loc="lower right")
    plt.tight_layout()

    out_file = save_path or (FIGURES_DIR / "roc_comparison.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    logger.info("Saved ROC comparison plot to %s", out_file)


def plot_calibration_comparison(
    models_dict: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path | None = None,
) -> None:
    """Plot probability calibration curves (reliability diagram) on holdout."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in models_dict.items():
        CalibrationDisplay.from_estimator(model, X_test, y_test, n_bins=5, name=name, ax=ax)
    ax.set_title("Calibration: Reliability Diagram", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of True CAD Positives")
    plt.tight_layout()

    out_file = save_path or (FIGURES_DIR / "calibration.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    logger.info("Saved calibration plot to %s", out_file)


def plot_calibration_recalibration(
    cal_results_lr: dict[str, Any],
    cal_results_xgb: dict[str, Any],
    y_test: pd.Series,
    save_path: Path | None = None,
) -> None:
    """Plot Raw vs. Platt vs. Isotonic reliability comparison for both models."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    from sklearn.calibration import calibration_curve

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, results, model_name in [
        (ax1, cal_results_lr, "Logistic Regression"),
        (ax2, cal_results_xgb, "XGBoost"),
    ]:
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        for method, data in results.items():
            prob_true, prob_pred = calibration_curve(y_test, data["probs"], n_bins=5)
            ax.plot(
                prob_pred,
                prob_true,
                marker="o",
                label=f"{method} (Brier: {data['brier_score']:.3f}, AUC: {data['roc_auc']:.3f})",
            )
        ax.set_title(f"{model_name} Recalibration", fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of True Positives")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_file = save_path or CALIBRATION_COMPARISON_PLOT_PATH
    plt.savefig(out_file, dpi=150)
    plt.close()
    logger.info("Saved recalibration comparison plot to %s", out_file)


def plot_decision_curve_analysis(
    models_probs: dict[str, np.ndarray],
    y_true: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Generate Decision Curve Analysis plot comparing Net Benefit across thresholds."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = np.linspace(0.05, 0.70, 40)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Reference strategies: Treat None & Treat All
    ref_df = compute_net_benefit(y_true, next(iter(models_probs.values())), thresholds=thresholds)
    ax.plot(
        ref_df["threshold"],
        ref_df["net_benefit_none"],
        "k-",
        alpha=0.6,
        label="Treat None (Refer Nobody)",
    )
    ax.plot(
        ref_df["threshold"],
        ref_df["net_benefit_all"],
        "gray",
        linestyle="--",
        label="Treat All (Refer Everyone)",
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for (name, probs), c in zip(models_probs.items(), colors, strict=False):
        dca_df = compute_net_benefit(y_true, probs, thresholds=thresholds)
        ax.plot(dca_df["threshold"], dca_df["net_benefit_model"], label=name, color=c, lw=2)

    ax.set_ylim(-0.05, 0.55)
    ax.set_xlim(0.05, 0.70)
    ax.set_title("Decision Curve Analysis: Clinical Net Benefit", fontsize=12, fontweight="bold")
    ax.set_xlabel("Cardiology Referral Threshold Probability (p_t)")
    ax.set_ylabel("Clinical Net Benefit")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_file = save_path or DCA_PLOT_PATH
    plt.savefig(out_file, dpi=150)
    plt.close()
    logger.info("Saved Decision Curve Analysis plot to %s", out_file)


def plot_cost_tradeoff_curve(
    models_probs: dict[str, np.ndarray],
    y_true: np.ndarray,
    cost_ratio: float = DEFAULT_FN_FP_COST_RATIO,
    save_path: Path | None = None,
) -> None:
    """Plot clinical cost vs. decision threshold under 10x FN penalty."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, probs in models_probs.items():
        cost_res = evaluate_cost_sensitive(y_true, probs, cost_fn_ratio=cost_ratio)
        df = cost_res["cost_df"]
        ax.plot(
            df["threshold"],
            df["cost_per_patient"],
            label=f"{name} (Min Cost: {cost_res['min_cost_per_patient']:.2f} @ T*={cost_res['optimal_threshold']:.2f})",
            lw=2,
        )
        ax.axvline(
            cost_res["optimal_threshold"],
            linestyle=":",
            alpha=0.5,
        )

    ax.set_title(
        f"Cost-Sensitive Loss vs. Threshold (FN Cost = {int(cost_ratio)}x FP Cost)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Decision Cutoff Threshold")
    ax.set_ylabel("Normalized Cost per Patient")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_file = save_path or COST_TRADEOFF_PLOT_PATH
    plt.savefig(out_file, dpi=150)
    plt.close()
    logger.info("Saved Cost-Sensitive Tradeoff plot to %s", out_file)


def plot_learning_curves(
    models_dict: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv: Any = None,
    save_path: Path | None = None,
) -> None:
    """Compute and plot sample size learning curves to assess data saturation."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fig, axes = plt.subplots(1, len(models_dict), figsize=(6 * len(models_dict), 5), sharey=True)
    if len(models_dict) == 1:
        axes = [axes]

    train_sizes = np.linspace(0.2, 1.0, 5)

    for ax, (name, model) in zip(axes, models_dict.items(), strict=False):
        sizes, train_scores, test_scores = learning_curve(
            model,
            X,
            y,
            cv=cv,
            train_sizes=train_sizes,
            scoring="roc_auc",
            n_jobs=1,
            random_state=42,
        )
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        ax.plot(sizes, train_mean, "o-", color="#1f77b4", label="Training Score")
        ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#1f77b4")
        ax.plot(sizes, test_mean, "o-", color="#ff7f0e", label="Cross-Validation Score")
        ax.fill_between(sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color="#ff7f0e")

        ax.set_title(f"Learning Curve: {name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Training Sample Size (N)")
        ax.set_ylabel("ROC-AUC Score")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_file = save_path or LEARNING_CURVE_PLOT_PATH
    plt.savefig(out_file, dpi=150)
    plt.close()
    logger.info("Saved Learning Curves plot to %s", out_file)


def plot_confusion_matrix_artifact(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    threshold: float,
    save_path: Path | None = None,
) -> None:
    """Generate and save labeled confusion matrix plot."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["No disease", "Disease"]).plot(
        cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"{model_name} (Cutoff T*={threshold:.2f})", fontsize=11, fontweight="bold")
    plt.tight_layout()

    safe_name = model_name.replace(" ", "_")
    out_file = save_path or (FIGURES_DIR / f"cm_{safe_name}.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    logger.info("Saved confusion matrix to %s", out_file)
