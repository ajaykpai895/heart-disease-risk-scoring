"""
Fairness, Subgroup Analysis, and Responsible-AI Evaluation Module.

Implements Mitchell et al. (2019) Model Card methodology and Gebru et al. (2018)
Datasheet principles through quantitative subgroup analysis of a cardiovascular
risk classification model.

Analysis dimensions:
1. Subgroup performance (Recall, Precision, AUC, FNR) for sex and age bands
   with Wilson 95% confidence intervals and sample-size warnings.
2. Subgroup calibration: observed vs. predicted event rate (reliability).
3. Fairlearn-based algorithmic fairness metrics: Equal Opportunity Difference
   and Demographic Parity Difference with honest interpretation.
4. Full audit report serialized to reports/fairness_audit.json.
5. Publication-grade fairness visualisation saved to reports/figures/.

Limitations stated explicitly:
- The Cleveland dataset (n=303) produces very small subgroup cells.
  Many CIs span the full [0, 1] range; conclusions should not be drawn
  from cells with n < 20 positive labels.
- The dataset was collected from a single US academic medical center in
  the 1980s. Any measured disparities may reflect data artefacts, not
  true population differences.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from src.config import FIGURES_DIR, REPORTS_DIR

# Optional fairlearn import — fail gracefully so the module still loads
# even if fairlearn is not installed, surfacing a clear error only at call time.
try:
    from fairlearn.metrics import (
        MetricFrame,
        demographic_parity_difference,
        equalized_odds_difference,
        false_negative_rate,
        selection_rate,
        true_positive_rate,
    )

    _FAIRLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FAIRLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FAIRNESS_AUDIT_PATH: Path = REPORTS_DIR / "fairness_audit.json"
FAIRNESS_SUBGROUP_PLOT_PATH: Path = FIGURES_DIR / "fairness_subgroup_metrics.png"
FAIRNESS_CALIBRATION_PLOT_PATH: Path = FIGURES_DIR / "fairness_calibration_subgroup.png"

# Minimum number of *positive* labels in a subgroup to trust recall/FNR estimates.
MIN_POSITIVE_CELL_SIZE: int = 15
# Minimum total subgroup size for AUC to be estimable at all.
MIN_AUC_CELL_SIZE: int = 20

AGE_BINS: list[int] = [0, 45, 60, 100]
AGE_LABELS: list[str] = ["<45", "45–60", ">60"]

# ---------------------------------------------------------------------------
# 1. Subgroup Performance Analysis
# ---------------------------------------------------------------------------


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion k/n.

    Returns (0.0, 1.0) when n == 0 (vacuous interval — no data).
    """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return float(np.clip(centre - half, 0.0, 1.0)), float(np.clip(centre + half, 0.0, 1.0))


def _subgroup_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute clinical performance metrics for one subgroup slice.

    Args:
        y_true: Binary ground-truth labels (0/1) for this subgroup.
        y_prob: Predicted positive-class probabilities for this subgroup.
        threshold: Decision threshold for hard predictions.

    Returns:
        dict with recall, precision, fnr, auc, pr_auc, brier, CIs, and
        a 'warnings' list of human-readable caveats about sample size.
    """
    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    warnings: list[str] = []

    if n_pos < MIN_POSITIVE_CELL_SIZE:
        warnings.append(
            f"Only {n_pos} positive labels — Recall and FNR estimates are unreliable. "
            "Do NOT draw clinical conclusions from this cell."
        )
    if n < MIN_AUC_CELL_SIZE:
        warnings.append(
            f"Total n={n} — AUC estimate is highly unstable. "
            "Treat as indicative only."
        )

    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    recall = tp / n_pos if n_pos > 0 else float("nan")
    fnr = fn / n_pos if n_pos > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    recall_ci = _wilson_ci(tp, n_pos) if n_pos > 0 else (float("nan"), float("nan"))
    fnr_ci = _wilson_ci(fn, n_pos) if n_pos > 0 else (float("nan"), float("nan"))
    prec_ci = _wilson_ci(tp, tp + fp) if (tp + fp) > 0 else (float("nan"), float("nan"))

    # AUC requires both classes present
    if n_pos > 0 and n_neg > 0:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
            pr_auc = float(average_precision_score(y_true, y_prob))
        except ValueError:
            auc = float("nan")
            pr_auc = float("nan")
    else:
        auc = float("nan")
        pr_auc = float("nan")
        warnings.append(
            "Cannot compute AUC — subgroup contains only one class. "
            "Stratified split may be necessary."
        )

    brier = float(brier_score_loss(y_true, y_prob)) if n > 0 else float("nan")

    return {
        "n_total": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "threshold": round(threshold, 3),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": round(recall, 3) if not np.isnan(recall) else None,
        "recall_95ci": [round(recall_ci[0], 3), round(recall_ci[1], 3)],
        "fnr": round(fnr, 3) if not np.isnan(fnr) else None,
        "fnr_95ci": [round(fnr_ci[0], 3), round(fnr_ci[1], 3)],
        "precision": round(precision, 3) if not np.isnan(precision) else None,
        "precision_95ci": [round(prec_ci[0], 3), round(prec_ci[1], 3)],
        "roc_auc": round(auc, 3) if not np.isnan(auc) else None,
        "pr_auc": round(pr_auc, 3) if not np.isnan(pr_auc) else None,
        "brier_score": round(brier, 3) if not np.isnan(brier) else None,
        "warnings": warnings,
    }


def evaluate_subgroup_performance(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_probs: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Evaluate performance broken down by sex and age band for each model.

    Subgroups:
    - Sex: Male (sex == 1) vs Female (sex == 0).
    - Age band: <45 / 45–60 / >60.

    Args:
        X_test: Holdout feature DataFrame (original scale, not preprocessed).
        y_test: Binary target series aligned with X_test.
        model_probs: Mapping of model name → positive-class probability array.
        thresholds: Mapping of model name → operating decision threshold.

    Returns:
        Nested dict: {model_name: {subgroup_key: metrics_dict}}
    """
    y_arr = y_test.values.astype(int)

    # Build age band membership
    age_band = pd.cut(
        X_test["age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=True,
    )

    subgroup_masks: dict[str, np.ndarray] = {
        "Sex: Male": (X_test["sex"] == 1.0).values,
        "Sex: Female": (X_test["sex"] == 0.0).values,
        "Age: <45": (age_band == "<45").values,
        "Age: 45–60": (age_band == "45–60").values,
        "Age: >60": (age_band == ">60").values,
        "Overall": np.ones(len(y_arr), dtype=bool),
    }

    results: dict[str, Any] = {}
    for model_name, probs in model_probs.items():
        thr = thresholds.get(model_name, 0.5)
        model_results: dict[str, Any] = {}
        for sg_name, mask in subgroup_masks.items():
            if mask.sum() == 0:
                model_results[sg_name] = {"warnings": ["Empty subgroup — no data."]}
                continue
            model_results[sg_name] = _subgroup_metrics(y_arr[mask], probs[mask], thr)
        results[model_name] = model_results

    return results


# ---------------------------------------------------------------------------
# 2. Subgroup Calibration Check
# ---------------------------------------------------------------------------


def evaluate_subgroup_calibration(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_probs: dict[str, np.ndarray],
    n_bins: int = 5,
) -> dict[str, Any]:
    """Assess calibration per subgroup via mean predicted vs. observed event rate.

    Uses a reduced bin count (5 instead of the standard 10) because the full
    Cleveland test set has only ~61 observations — 10-bin calibration curves
    would have ~6 patients per bin, making individual bin estimates meaningless.

    Args:
        X_test: Holdout feature DataFrame.
        y_test: Binary target series.
        model_probs: Model name → predicted probability array.
        n_bins: Number of probability bins for calibration assessment.

    Returns:
        Nested dict: {model_name: {subgroup: {bin_data, brier, calibration_gap}}}
    """
    y_arr = y_test.values.astype(int)
    age_band = pd.cut(X_test["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)

    subgroup_masks: dict[str, np.ndarray] = {
        "Sex: Male": (X_test["sex"] == 1.0).values,
        "Sex: Female": (X_test["sex"] == 0.0).values,
        "Age: <45": (age_band == "<45").values,
        "Age: 45–60": (age_band == "45–60").values,
        "Age: >60": (age_band == ">60").values,
    }

    cal_results: dict[str, Any] = {}
    for model_name, probs in model_probs.items():
        model_cal: dict[str, Any] = {}
        for sg_name, mask in subgroup_masks.items():
            n_sg = mask.sum()
            if n_sg < 10:
                model_cal[sg_name] = {
                    "warning": f"n={n_sg} — calibration estimate is meaningless. Skipping.",
                    "brier_score": None,
                    "mean_calibration_gap": None,
                }
                continue
            sg_probs = probs[mask]
            sg_true = y_arr[mask]

            brier = float(brier_score_loss(sg_true, sg_probs))

            # Bin by predicted probability
            bins = np.linspace(0.0, 1.0, n_bins + 1)
            bin_indices = np.digitize(sg_probs, bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            bin_data = []
            for b in range(n_bins):
                mask_b = bin_indices == b
                n_b = mask_b.sum()
                if n_b == 0:
                    continue
                mean_pred = float(sg_probs[mask_b].mean())
                mean_obs = float(sg_true[mask_b].mean())
                bin_data.append(
                    {
                        "bin": b,
                        "n": int(n_b),
                        "mean_predicted": round(mean_pred, 3),
                        "mean_observed": round(mean_obs, 3),
                        "gap": round(abs(mean_pred - mean_obs), 3),
                    }
                )

            gaps = [bd["gap"] for bd in bin_data]
            mean_gap = float(np.mean(gaps)) if gaps else float("nan")

            model_cal[sg_name] = {
                "n": int(n_sg),
                "brier_score": round(brier, 3),
                "mean_calibration_gap": round(mean_gap, 3),
                "bin_data": bin_data,
                "warning": (
                    f"n={n_sg} is small; calibration bins may contain < 5 patients."
                    if n_sg < 25
                    else None
                ),
            }
        cal_results[model_name] = model_cal

    return cal_results


# ---------------------------------------------------------------------------
# 3. Fairlearn Metrics
# ---------------------------------------------------------------------------


def compute_fairness_metrics(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_probs: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Compute algorithmic fairness metrics via the fairlearn library.

    Metrics computed:
    - **Equal Opportunity Difference (EOD)**: Difference in True Positive Rate
      (Recall) between the most- and least-advantaged group.
      Value 0 = perfect equal opportunity.
      Positive = privileged group has higher recall (fewer missed cases).
    - **Demographic Parity Difference (DPD)**: Difference in positive prediction
      rate across groups, regardless of ground truth.
      Value 0 = model refers patients at equal rates across groups.
    - **MetricFrame**: Per-group breakdown of recall, FNR, precision, selection
      rate used as audit trail.

    Sensitive features evaluated: sex (binary), age band (3-way).

    Interpretation guidance is embedded in the returned dict under the
    'interpretation' key. Numbers are reported without overclaiming.

    Args:
        X_test: Holdout feature DataFrame (original scale).
        y_test: Binary target series.
        model_probs: Model name → probability array.
        thresholds: Model name → decision threshold.

    Returns:
        dict with fairness metrics for each model and sensitive feature.

    Raises:
        RuntimeError: If fairlearn is not installed.
    """
    if not _FAIRLEARN_AVAILABLE:
        raise RuntimeError(
            "fairlearn is not installed. Run: pip install fairlearn"
        )

    y_arr = y_test.values.astype(int)
    age_band = pd.cut(
        X_test["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True
    ).astype(str)
    sex_labels = X_test["sex"].map({1.0: "Male", 0.0: "Female"}).fillna("Unknown")

    sensitive_features: dict[str, pd.Series] = {
        "sex": sex_labels,
        "age_band": age_band,
    }

    fairness_results: dict[str, Any] = {}

    for model_name, probs in model_probs.items():
        thr = thresholds.get(model_name, 0.5)
        y_pred = (probs >= thr).astype(int)
        model_fairness: dict[str, Any] = {}

        for sf_name, sf_values in sensitive_features.items():
            sf_arr = sf_values.values

            try:
                eod = float(
                    equalized_odds_difference(
                        y_arr,
                        y_pred,
                        sensitive_features=sf_arr,
                    )
                )
                dpd = float(
                    demographic_parity_difference(
                        y_arr,
                        y_pred,
                        sensitive_features=sf_arr,
                    )
                )

                mf = MetricFrame(
                    metrics={
                        "recall": true_positive_rate,
                        "fnr": false_negative_rate,
                        "selection_rate": selection_rate,
                    },
                    y_true=y_arr,
                    y_pred=y_pred,
                    sensitive_features=sf_arr,
                )

                per_group_df = mf.by_group.reset_index()
                per_group_records = per_group_df.to_dict(orient="records")

                # Build interpretation
                eod_abs = abs(eod)
                dpd_abs = abs(dpd)

                if eod_abs < 0.05:
                    eod_interp = (
                        f"EOD = {eod:.3f}: Negligible. The model's recall differs by "
                        f"less than 5 percentage points across {sf_name} groups. "
                        "However, with n≈61 test observations this cannot be "
                        "distinguished from sampling noise."
                    )
                elif eod_abs < 0.10:
                    eod_interp = (
                        f"EOD = {eod:.3f}: Moderate gap. One group's recall is "
                        f"~{eod_abs:.0%} lower. With n≈61 this estimate is "
                        "highly uncertain — a single misclassification shifts it "
                        "materially. Do not over-interpret."
                    )
                else:
                    eod_interp = (
                        f"EOD = {eod:.3f}: Substantial gap detected. This warrants "
                        "investigation but cannot be confirmed as systematic bias "
                        f"without a larger validation cohort stratified by {sf_name}."
                    )

                if dpd_abs < 0.05:
                    dpd_interp = (
                        f"DPD = {dpd:.3f}: Negligible referral rate difference across "
                        f"{sf_name} groups."
                    )
                else:
                    dpd_interp = (
                        f"DPD = {dpd:.3f}: Referral rates differ by {dpd_abs:.0%} "
                        f"across {sf_name} groups. Note: this dataset has "
                        f"a severe sex imbalance (206M vs 97F) — a DPD driven "
                        "primarily by prevalence differences is expected and "
                        "does NOT by itself indicate algorithmic bias."
                    )

                model_fairness[sf_name] = {
                    "equal_opportunity_difference": round(eod, 4),
                    "demographic_parity_difference": round(dpd, 4),
                    "per_group": per_group_records,
                    "eod_interpretation": eod_interp,
                    "dpd_interpretation": dpd_interp,
                    "data_limitation": (
                        f"Test set n=61; {sf_name} subgroups are very small. "
                        "All fairness estimates carry wide uncertainty intervals "
                        "and should not be used to certify or condemn the model."
                    ),
                }

            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Fairlearn metric computation failed for %s / %s: %s",
                    model_name,
                    sf_name,
                    exc,
                )
                model_fairness[sf_name] = {"error": str(exc)}

        fairness_results[model_name] = model_fairness

    return fairness_results


# ---------------------------------------------------------------------------
# 4. Visualisations
# ---------------------------------------------------------------------------


def plot_subgroup_performance(
    subgroup_results: dict[str, Any],
    models_to_plot: list[str] | None = None,
    save_path: Path | None = None,
) -> None:
    """Plot side-by-side subgroup Recall and FNR bar charts with 95% CIs.

    Args:
        subgroup_results: Output of :func:`evaluate_subgroup_performance`.
        models_to_plot: Subset of model names to include. Defaults to all.
        save_path: Override output file path.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if models_to_plot is None:
        models_to_plot = list(subgroup_results.keys())

    subgroups = ["Sex: Male", "Sex: Female", "Age: <45", "Age: 45–60", "Age: >60"]
    metrics = ["recall", "fnr"]
    metric_labels = ["Recall (Sensitivity)", "False Negative Rate"]
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels, strict=False)):
        ax = axes[ax_idx]
        x = np.arange(len(subgroups))
        width = 0.8 / max(len(models_to_plot), 1)

        for m_idx, model_name in enumerate(models_to_plot):
            vals: list[float] = []
            lowers: list[float] = []
            uppers: list[float] = []

            for sg in subgroups:
                sg_data = subgroup_results.get(model_name, {}).get(sg, {})
                v = sg_data.get(metric)
                ci = sg_data.get(f"{metric}_95ci", [None, None])

                if v is None or np.isnan(v):
                    vals.append(0.0)
                    lowers.append(0.0)
                    uppers.append(0.0)
                else:
                    vals.append(float(v))
                    low = ci[0] if ci[0] is not None else float(v)
                    high = ci[1] if ci[1] is not None else float(v)
                    lowers.append(float(v) - float(low))
                    uppers.append(float(high) - float(v))

            offset = (m_idx - len(models_to_plot) / 2 + 0.5) * width
            ax.bar(
                x + offset,
                vals,
                width * 0.9,
                label=model_name,
                color=colors[m_idx % len(colors)],
                alpha=0.82,
            )
            ax.errorbar(
                x + offset,
                vals,
                yerr=[lowers, uppers],
                fmt="none",
                color="black",
                capsize=4,
                linewidth=1.2,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(subgroups, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_title(f"Subgroup {metric_label}", fontsize=11, fontweight="bold")
        ax.axhline(0.9, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Target Recall (0.90)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        # Add small-n warning markers
        for m_idx, model_name in enumerate(models_to_plot):
            offset = (m_idx - len(models_to_plot) / 2 + 0.5) * width
            for sg_idx, sg in enumerate(subgroups):
                sg_data = subgroup_results.get(model_name, {}).get(sg, {})
                n_pos = sg_data.get("n_positive", 999)
                if isinstance(n_pos, int) and n_pos < MIN_POSITIVE_CELL_SIZE:
                    ax.text(
                        sg_idx + offset,
                        1.05,
                        "⚠",
                        ha="center",
                        fontsize=9,
                        color="darkorange",
                    )

    plt.suptitle(
        "Subgroup Performance Analysis\n"
        "⚠ = n_positive < 15: estimate unreliable",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_path = save_path or FAIRNESS_SUBGROUP_PLOT_PATH
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved subgroup performance plot to %s", out_path)


def plot_subgroup_calibration(
    cal_results: dict[str, Any],
    save_path: Path | None = None,
) -> None:
    """Plot calibration gap (mean |predicted − observed|) by subgroup and model.

    Args:
        cal_results: Output of :func:`evaluate_subgroup_calibration`.
        save_path: Override output file path.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, sg_dict in cal_results.items():
        for sg_name, sg_data in sg_dict.items():
            if sg_data.get("mean_calibration_gap") is not None:
                rows.append(
                    {
                        "Model": model_name,
                        "Subgroup": sg_name,
                        "Mean Calibration Gap": sg_data["mean_calibration_gap"],
                        "n": sg_data.get("n", 0),
                    }
                )

    if not rows:
        logger.warning("No calibration data to plot (all subgroups too small).")
        return

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="Subgroup", columns="Model", values="Mean Calibration Gap")

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="black", linewidth=0.5)

    ax.set_title(
        "Subgroup Calibration: Mean |Predicted – Observed| Probability\n"
        "(lower = better calibrated; ≥ 0.10 suggests systematic over/under-estimation)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mean Calibration Gap", fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.axhline(0.10, color="red", linestyle="--", linewidth=1, label="Threshold of concern (0.10)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = save_path or FAIRNESS_CALIBRATION_PLOT_PATH
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved subgroup calibration plot to %s", out_path)


# ---------------------------------------------------------------------------
# 5. Full Fairness Audit Runner
# ---------------------------------------------------------------------------


def run_fairness_audit(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_probs: dict[str, np.ndarray],
    thresholds: dict[str, float],
    save_report: bool = True,
) -> dict[str, Any]:
    """Run the complete fairness and responsible-AI audit pipeline.

    Executes subgroup performance, subgroup calibration, and fairlearn
    fairness metrics, then saves plots and a JSON audit report.

    Args:
        X_test: Holdout feature DataFrame.
        y_test: Binary target series.
        model_probs: Model name → predicted probability array.
        thresholds: Model name → operating threshold.
        save_report: Whether to save fairness_audit.json.

    Returns:
        dict containing 'subgroup_performance', 'subgroup_calibration',
        and 'fairness_metrics' sub-dicts.
    """
    logger.info("--- Starting Fairness & Responsible-AI Audit ---")

    # 1. Subgroup performance
    logger.info("Computing subgroup performance metrics...")
    subgroup_perf = evaluate_subgroup_performance(X_test, y_test, model_probs, thresholds)

    # Log any small-n warnings immediately
    for model_name, sg_dict in subgroup_perf.items():
        for sg_name, sg_data in sg_dict.items():
            for w in sg_data.get("warnings", []):
                logger.warning("[%s / %s] %s", model_name, sg_name, w)

    # 2. Subgroup calibration
    logger.info("Computing subgroup calibration...")
    subgroup_cal = evaluate_subgroup_calibration(X_test, y_test, model_probs)

    # 3. Fairlearn metrics
    if _FAIRLEARN_AVAILABLE:
        logger.info("Computing fairlearn algorithmic fairness metrics...")
        fairness_metrics = compute_fairness_metrics(X_test, y_test, model_probs, thresholds)
    else:
        logger.warning(
            "fairlearn not installed — skipping algorithmic fairness metrics. "
            "Run: pip install fairlearn"
        )
        fairness_metrics = {"error": "fairlearn not installed"}

    # 4. Plots
    plot_subgroup_performance(
        subgroup_perf,
        models_to_plot=["Logistic Regression", "XGBoost"],
    )
    plot_subgroup_calibration(subgroup_cal)

    audit = {
        "subgroup_performance": subgroup_perf,
        "subgroup_calibration": subgroup_cal,
        "fairness_metrics": fairness_metrics,
        "metadata": {
            "dataset": "UCI Cleveland Heart Disease (n=303, test n=61)",
            "sensitive_features": ["sex", "age_band"],
            "min_positive_cell_threshold": MIN_POSITIVE_CELL_SIZE,
            "min_auc_cell_threshold": MIN_AUC_CELL_SIZE,
            "caveat": (
                "All fairness estimates are based on a 61-patient holdout set. "
                "Subgroup cells are extremely small. No conclusions about "
                "systematic algorithmic bias can be drawn from this dataset alone."
            ),
        },
    }

    if save_report:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        def _json_serializable(obj: Any) -> Any:
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Not JSON serializable: {type(obj)}")

        with open(FAIRNESS_AUDIT_PATH, "w") as f:
            json.dump(audit, f, indent=2, default=_json_serializable)
        logger.info("Fairness audit report saved to %s", FAIRNESS_AUDIT_PATH)

    logger.info("--- Fairness Audit Complete ---")
    return audit
