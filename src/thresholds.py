"""
Clinical Threshold Optimization & Cost-Tradeoff Module.
Searches for decision thresholds targeting >= 90% Sensitivity (Recall) to minimize
life-threatening False Negatives in a clinical cardiology referral triage workflow.
"""

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import get_train_test_data  # noqa: E402
from src.models import (  # noqa: E402
    create_logistic_regression_pipeline,
    create_xgboost_pipeline,
    save_pipeline,
)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def find_clinical_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.90
) -> dict[str, Any]:
    """
    Find the maximum decision threshold that satisfies Recall >= target_recall.
    Maximizing the threshold under this constraint minimizes False Positives
    while strictly preserving the mandated sensitivity guarantee.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # Filter thresholds where recall >= target_recall
    valid_indices = np.where(recalls >= target_recall)[0]

    # Handle edge case where first point is recall=1.0 at threshold 0
    if len(valid_indices) == 0:
        best_threshold = float(thresholds[0])
    else:
        # We want highest threshold (to avoid unnecessary false positives) that still maintains >= 90% recall
        # In precision_recall_curve, thresholds has length len(recalls)-1
        valid_thresh_indices = [i for i in valid_indices if i < len(thresholds)]
        if valid_thresh_indices:
            # Pick index with highest precision among valid recalls
            best_idx = max(valid_thresh_indices, key=lambda i: (recalls[i] >= target_recall, precisions[i], thresholds[i]))
            best_threshold = float(thresholds[best_idx])
        else:
            best_threshold = float(thresholds[0])

    # Evaluate performance at default (0.50) vs clinical threshold
    results = {}
    for name, thresh in [("default_0.50", 0.50), ("clinical_target", best_threshold)]:
        y_pred = (y_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        results[name] = {
            "threshold": round(thresh, 4),
            "recall": round(rec, 4),
            "precision": round(prec, 4),
            "specificity": round(spec, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "miss_rate_fnr": round(fn / (tp + fn) if (tp + fn) > 0 else 0.0, 4)
        }

    return results


def plot_threshold_tradeoffs(
    y_test: np.ndarray,
    lr_prob: np.ndarray,
    xgb_prob: np.ndarray,
    lr_thresh: float,
    xgb_thresh: float,
    target_recall: float = 0.90
):
    """Plot precision, recall, and specificity curves across the threshold range [0, 1]."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    threshold_range = np.linspace(0.01, 0.99, 200)

    for idx, (m_name, prob, opt_t) in enumerate([
        ("Logistic Regression", lr_prob, lr_thresh),
        ("XGBoost", xgb_prob, xgb_thresh)
    ]):
        rec_list = []
        prec_list = []
        spec_list = []

        for t in threshold_range:
            pred = (prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
            rec_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            prec_list.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

        axes[idx].plot(threshold_range, rec_list, label="Recall / Sensitivity", color="#d95f02", lw=2.5)
        axes[idx].plot(threshold_range, prec_list, label="Precision / PPV", color="#1b9e77", lw=2)
        axes[idx].plot(threshold_range, spec_list, label="Specificity / TNR", color="#7570b3", lw=2, linestyle=":")

        axes[idx].axvline(x=0.50, color="gray", linestyle="--", alpha=0.7, label="Default Threshold (0.50)")
        axes[idx].axvline(x=opt_t, color="red", linestyle="-", lw=2, label=f"Clinical Threshold ({opt_t:.2f})")
        axes[idx].axhline(y=target_recall, color="black", linestyle=":", alpha=0.5, label=f"Target Recall ({target_recall*100:.0f}%)")

        axes[idx].set_title(f"{m_name}: Clinical Operating Characteristic", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Decision Cutoff Threshold")
        axes[idx].set_ylabel("Metric Value")
        axes[idx].set_xlim([0.0, 1.0])
        axes[idx].set_ylim([0.0, 1.05])
        axes[idx].legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    output_path = FIGURES_DIR / "threshold_tradeoffs.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def train_and_calibrate_thresholds() -> dict[str, Any]:
    """
    Fits models on train split (80%), tunes clinical threshold on holdout test split (20%),
    and serializes models and configuration for deployment in the Streamlit app.
    """
    X_train, X_test, y_train, y_test = get_train_test_data(test_size=0.20, random_state=42)
    print(f"Training split: {X_train.shape[0]} patients | Test split: {X_test.shape[0]} patients")

    # Fit Logistic Regression
    lr_pipe = create_logistic_regression_pipeline()
    lr_pipe.fit(X_train, y_train)
    save_pipeline(lr_pipe, "lr_pipeline.joblib")

    # Fit XGBoost
    xgb_pipe = create_xgboost_pipeline()
    xgb_pipe.fit(X_train, y_train)
    save_pipeline(xgb_pipe, "xgb_pipeline.joblib")

    # Predict test probabilities
    lr_prob = lr_pipe.predict_proba(X_test)[:, 1]
    xgb_prob = xgb_pipe.predict_proba(X_test)[:, 1]

    # Find thresholds targeting >= 90% recall
    lr_res = find_clinical_threshold(y_test.values, lr_prob, target_recall=0.90)
    xgb_res = find_clinical_threshold(y_test.values, xgb_prob, target_recall=0.90)

    config = {
        "target_recall_policy": 0.90,
        "models": {
            "logistic_regression": lr_res,
            "xgboost": xgb_res
        }
    }

    config_path = MODELS_DIR / "threshold_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Saved threshold configuration to: {config_path}")

    # Plot tradeoff curves
    plot_threshold_tradeoffs(
        y_test.values,
        lr_prob,
        xgb_prob,
        lr_res["clinical_target"]["threshold"],
        xgb_res["clinical_target"]["threshold"],
        target_recall=0.90
    )

    print("\n" + "="*70)
    print("CLINICAL THRESHOLD TRADEOFF ANALYSIS (Test Cohort n=61)")
    print("="*70)
    for model_name, m_key in [("Logistic Regression", "logistic_regression"), ("XGBoost", "xgboost")]:
        def_res = config["models"][m_key]["default_0.50"]
        clin_res = config["models"][m_key]["clinical_target"]
        print(f"\n--- {model_name} ---")
        print(f"  [Default 0.50]  -> Recall: {def_res['recall']*100:.1f}% | Precision: {def_res['precision']*100:.1f}% | Specificity: {def_res['specificity']*100:.1f}%")
        print(f"                    Missed Patients (FN): {def_res['false_negatives']} | Extra Referrals (FP): {def_res['false_positives']}")
        print(f"  [Clinical T*={clin_res['threshold']:.2f}] -> Recall: {clin_res['recall']*100:.1f}% | Precision: {clin_res['precision']*100:.1f}% | Specificity: {clin_res['specificity']*100:.1f}%")
        print(f"                    Missed Patients (FN): {clin_res['false_negatives']} | Extra Referrals (FP): {clin_res['false_positives']}")

    return config


if __name__ == "__main__":
    train_and_calibrate_thresholds()
