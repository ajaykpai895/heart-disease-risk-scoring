"""
Model Explainability and Clinical Interpretability Suite.

Implements:
1. Logistic Regression: Log-odds coefficients, unaliased Odds Ratios (exp(beta)),
   and directionality plots.
2. XGBoost: Shapley Additive Explanations (SHAP) using TreeExplainer.
3. Cross-model concordance check (Spearman rank correlation of feature importances).

CLI Usage:
    python -m src.explain
"""

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import (
    FIGURES_DIR,
    LOGREG_MODEL_PATH,
    LR_ODDS_RATIOS_PATH,
    X_TEST_PATH,
    XGB_MODEL_PATH,
)

logger = logging.getLogger("src.explain")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def compute_logistic_odds_ratios(
    model_path: Path = LOGREG_MODEL_PATH,
    output_csv: Path = LR_ODDS_RATIOS_PATH,
) -> pd.DataFrame:
    """Extract unaliased coefficients and compute Odds Ratios from Logistic Regression.

    Formula:
        Odds Ratio = exp(beta_j)
    Interpretation:
        OR > 1.0 indicates increased odds of coronary artery disease.
        OR < 1.0 indicates protective effect relative to the reference category.

    Args:
        model_path: Path to serialized Logistic Regression Pipeline.
        output_csv: Path to save generated Odds Ratios CSV.

    Returns:
        pd.DataFrame: Table of features, beta coefficients, and Odds Ratios sorted by |beta|.
    """
    logger.info("Loading Logistic Regression pipeline from %s", model_path)
    logreg = joblib.load(model_path)
    prep = logreg.named_steps["prep"]
    clf = logreg.named_steps["clf"]

    feature_names = prep.get_feature_names_out()
    coefs = clf.coef_[0]

    odds_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coef": coefs,
            "odds_ratio": np.exp(coefs),
        }
    )
    odds_df["abs_coef"] = odds_df["coef"].abs()
    odds_df = odds_df.sort_values("abs_coef", ascending=False).drop(
        columns=["abs_coef"]
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    odds_df.to_csv(output_csv, index=False)
    logger.info("Saved unaliased odds ratios to %s", output_csv)

    # Plot horizontal forest-style coefficient effect bar chart
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top_effects = odds_df.head(12).iloc[::-1]
    plt.figure(figsize=(9, 6.5))
    colors = ["#d62728" if c > 0 else "#1f77b4" for c in top_effects["coef"]]
    plt.barh(top_effects["feature"], top_effects["coef"], color=colors, edgecolor="black", alpha=0.85)
    plt.axvline(0, color="k", lw=0.9, linestyle="--")
    plt.title(
        "Logistic Regression: Feature Risk Attributions (Beta Coefficients)\n"
        "(Red = Increases CAD Risk, Blue = Protective / Lowers Risk)",
        fontsize=11,
        fontweight="bold",
    )
    plt.xlabel("Log-Odds Coefficient (Beta)")
    plt.tight_layout()

    plot_file = FIGURES_DIR / "lr_coefficients.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    logger.info("Saved LR coefficients plot to %s", plot_file)
    return odds_df


def compute_xgboost_shap_explanations(
    model_path: Path = XGB_MODEL_PATH,
    test_path: Path = X_TEST_PATH,
) -> tuple[np.ndarray, list]:
    """Compute exact SHAP values for XGBoost using TreeExplainer.

    Args:
        model_path: Path to serialized XGBoost Pipeline.
        test_path: Path to CSV of holdout features.

    Returns:
        Tuple[np.ndarray, list]: SHAP value matrix and feature names list.
    """
    logger.info("Loading XGBoost pipeline from %s and test data from %s", model_path, test_path)
    xgb_pipe = joblib.load(model_path)
    X_test = pd.read_csv(test_path)

    prep = xgb_pipe.named_steps["prep"]
    clf = xgb_pipe.named_steps["clf"]

    Xt = prep.transform(X_test)
    feat_names = list(prep.get_feature_names_out())

    logger.info("Computing TreeExplainer SHAP values across %d test patients...", len(X_test))
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(Xt)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Beeswarm summary plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, Xt, feature_names=feat_names, show=False)
    plt.title("XGBoost Global Feature Attribution (SHAP Beeswarm)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    summary_plot = FIGURES_DIR / "shap_summary.png"
    plt.savefig(summary_plot, dpi=150)
    plt.close()
    logger.info("Saved SHAP summary beeswarm to %s", summary_plot)

    # Bar plot
    plt.figure(figsize=(9, 6.5))
    shap.summary_plot(shap_values, Xt, feature_names=feat_names, plot_type="bar", show=False)
    plt.title("XGBoost Mean Absolute SHAP Importance", fontsize=11, fontweight="bold")
    plt.tight_layout()
    bar_plot = FIGURES_DIR / "shap_bar.png"
    plt.savefig(bar_plot, dpi=150)
    plt.close()
    logger.info("Saved SHAP bar plot to %s", bar_plot)

    return shap_values, feat_names


def assess_cross_model_agreement(
    odds_df: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: list,
) -> float:
    """Assess whether transparent and black-box models agree on feature importance.

    Computes Spearman rank correlation between Logistic Regression absolute coefficients
    and XGBoost mean absolute SHAP values.

    Args:
        odds_df: DataFrame of Logistic Regression coefficients.
        shap_values: Matrix of XGBoost SHAP values.
        feature_names: List of feature names.

    Returns:
        float: Spearman rank correlation in [-1.0, 1.0].
    """
    shap_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names).rank(
        ascending=False
    )
    lr_imp = odds_df.set_index("feature")["coef"].abs().rank(ascending=False)

    agreement = pd.concat([lr_imp, shap_imp], axis=1, keys=["LR rank", "SHAP rank"]).dropna()
    agreement = agreement.sort_values("LR rank")

    spearman_corr = float(agreement.corr(method="spearman").iloc[0, 1])
    logger.info("Top 8 Feature Concordance (LR Rank vs SHAP Rank):\n%s", agreement.head(8).to_string())
    logger.info("Cross-Model Spearman Rank Agreement: %.2f", spearman_corr)
    return spearman_corr


def run_explainability() -> None:
    """Execute complete explainability workflow."""
    odds_df = compute_logistic_odds_ratios()
    logger.info("Top 6 Risk Factors (Logistic Regression):\n%s", odds_df.head(6).to_string(index=False))

    shap_values, feat_names = compute_xgboost_shap_explanations()
    assess_cross_model_agreement(odds_df, shap_values, feat_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Logistic Regression Odds Ratios and XGBoost SHAP Explanations"
    )
    parser.parse_args()
    run_explainability()
