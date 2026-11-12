"""
Research-Grade Model Training, Validation, and Clinical Decision Analysis Pipeline.

Executes:
1. Nested Cross-Validation (inner loop tuning, outer loop evaluation) vs. Non-Nested CV
   to rigorously quantify and eliminate hyperparameter selection optimism bias.
2. Clinical Baseline Benchmarking: Dummy Classifier & Diamond-Forrester Clinical Heuristic.
3. Statistical Hypothesis Testing: DeLong test on holdout AUCs and Nadeau-Bengio
   corrected resampled t-test on repeated CV fold metrics.
4. Probability Calibration Analysis: Raw vs. Platt Scaling (Sigmoid) vs. Isotonic Regression.
5. Decision Curve Analysis (DCA): Quantifies clinical Net Benefit across referral cutoffs.
6. Asymmetric Cost-Sensitive Optimization: Minimizes clinical loss under 10x False Negative penalty.
7. Non-parametric Bootstrap 95% Confidence Intervals for holdout discrimination and sensitivity.
8. Empirical Learning Curves assessing data saturation across sample size fractions.
9. Artifact serialization and publication-grade Markdown table export.

CLI Usage:
    python -m src.train --target-recall 0.90 --seed 42
"""

import argparse
import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    BOOTSTRAP_ROUNDS,
    CV_REPEATS,
    CV_RESULTS_PATH,
    CV_SPLITS,
    DCA_PLOT_PATH,
    DEFAULT_FN_FP_COST_RATIO,
    DEFAULT_TARGET_RECALL,
    FIGURES_DIR,
    HOLDOUT_RESULTS_PATH,
    LEARNING_CURVE_PLOT_PATH,
    LOGREG_MODEL_PATH,
    LOGREG_PARAM_GRID,
    MODELS_DIR,
    REPORTS_DIR,
    RESEARCH_RESULTS_PATH,
    SEED,
    TEST_SIZE,
    THRESHOLD_CONFIG_PATH,
    TUNING_CV_REPEATS,
    X_TEST_PATH,
    XGB_MODEL_PATH,
    XGB_PARAM_GRID,
    Y_TEST_PATH,
)
from src.data import get_train_test_data
from src.evaluate import (
    ClinicalRuleBaseline,
    compute_bootstrap_ci,
    delong_roc_test,
    evaluate_calibration_methods,
    evaluate_cost_sensitive,
    evaluate_predictions,
    nadeau_bengio_paired_ttest,
    plot_calibration_comparison,
    plot_calibration_recalibration,
    plot_confusion_matrix_artifact,
    plot_cost_tradeoff_curve,
    plot_decision_curve_analysis,
    plot_learning_curves,
    plot_roc_comparison,
)
from src.fairness import run_fairness_audit
from src.features import build_preprocessor

logger = logging.getLogger("src.train")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def pick_threshold(
    model: Pipeline,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    target_recall: float = DEFAULT_TARGET_RECALL,
    seed: int = SEED,
) -> float:
    """Select the decision threshold guaranteeing Recall >= target_recall with max precision.

    Evaluated on out-of-fold cross-validated probability predictions to prevent
    threshold over-fitting to the training set.

    Args:
        model: Unfitted or fitted sklearn Pipeline.
        X_tr: Training features DataFrame.
        y_tr: Training binary target Series.
        target_recall: Mandated minimum sensitivity bound (default: 0.90).
        seed: Random seed for CV splits.

    Returns:
        float: Optimized probability threshold cutoff.
    """
    logger.info("Computing out-of-fold probabilities for threshold selection...")
    proba = cross_val_predict(
        model,
        X_tr,
        y_tr,
        cv=RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=1, random_state=seed),
        method="predict_proba",
    )[:, 1]

    from sklearn.metrics import precision_recall_curve

    prec, rec, thr = precision_recall_curve(y_tr, proba)
    ok = np.where(rec[:-1] >= target_recall)[0]

    if len(ok) == 0:
        logger.warning(
            "No threshold achieved target recall >= %.2f. Defaulting to lowest threshold.",
            target_recall,
        )
        return float(thr[0])

    best_idx = ok[np.argmax(prec[:-1][ok])]
    optimal_thr = float(thr[best_idx])
    logger.info(
        "Selected threshold T*=%.3f (OOF Recall=%.3f, OOF Precision=%.3f)",
        optimal_thr,
        rec[best_idx],
        prec[best_idx],
    )
    return optimal_thr


def run_nested_cv(
    pipeline: Pipeline,
    param_grid: dict[str, list[Any]],
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = SEED,
) -> tuple[np.ndarray, float, float]:
    """Execute Nested Cross-Validation with inner tuning and outer evaluation.

    Outer loop: 5-fold Stratified K-Fold.
    Inner loop: 5-fold Stratified K-Fold inside GridSearchCV.

    Args:
        pipeline: Base pipeline.
        param_grid: Hyperparameter grid dictionary.
        X: Feature matrix.
        y: Target series.
        seed: Random seed.

    Returns:
        tuple: (outer_scores_array, mean_score, std_score).
    """
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv_inner,
        n_jobs=1,
    )
    outer_scores = cross_val_score(grid_search, X, y, cv=cv_outer, scoring="roc_auc")
    return outer_scores, float(np.mean(outer_scores)), float(np.std(outer_scores))


def train_and_evaluate(
    target_recall: float = DEFAULT_TARGET_RECALL,
    seed: int = SEED,
    test_size: float = TEST_SIZE,
    cost_fn_ratio: float = DEFAULT_FN_FP_COST_RATIO,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Execute research-grade model training, statistical audit, and decision analysis.

    Args:
        target_recall: Mandated minimum sensitivity bound (default: 0.90).
        seed: Random seed for deterministic reproducibility.
        test_size: Holdout cohort fraction (default: 0.20).
        cost_fn_ratio: Multiplier for cost of False Negative vs. False Positive (default: 10.0).

    Returns:
        tuple[dict[str, Any], pd.DataFrame]: Holdout metrics dictionary and research summary table.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Partitioning dataset (test_size=%.2f, seed=%d)...", test_size, seed)
    X_train, X_test, y_train, y_test = get_train_test_data(
        test_size=test_size, random_state=seed
    )

    # -------------------------------------------------------------------------
    # 1. Pipeline Definitions & Clinical Baselines
    # -------------------------------------------------------------------------
    logreg = Pipeline(
        steps=[
            ("prep", build_preprocessor(drop_first=True)),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )
    xgb = Pipeline(
        steps=[
            ("prep", build_preprocessor(drop_first=True)),
            (
                "clf",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=seed,
                ),
            ),
        ]
    )
    dummy_baseline = DummyClassifier(strategy="most_frequent")
    clinical_baseline = ClinicalRuleBaseline(min_criteria_points=2)

    # -------------------------------------------------------------------------
    # 2. Nested Cross-Validation (Inner Tuning, Outer Evaluation) vs. Non-Nested
    # -------------------------------------------------------------------------
    logger.info("Executing Nested CV vs. Non-Nested CV comparison...")
    lr_nested_scores, lr_nested_mean, lr_nested_std = run_nested_cv(
        logreg, LOGREG_PARAM_GRID, X_train, y_train, seed=seed
    )
    xgb_nested_scores, xgb_nested_mean, xgb_nested_std = run_nested_cv(
        xgb, XGB_PARAM_GRID, X_train, y_train, seed=seed
    )

    # Non-nested standard hyperparameter search
    cv_inner = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=TUNING_CV_REPEATS, random_state=seed
    )
    logreg_gs = GridSearchCV(logreg, LOGREG_PARAM_GRID, scoring="roc_auc", cv=cv_inner)
    xgb_gs = GridSearchCV(xgb, XGB_PARAM_GRID, scoring="roc_auc", cv=cv_inner)
    logreg_gs.fit(X_train, y_train)
    xgb_gs.fit(X_train, y_train)

    lr_non_nested_score = float(logreg_gs.best_score_)
    xgb_non_nested_score = float(xgb_gs.best_score_)

    lr_optimism_bias = lr_non_nested_score - lr_nested_mean
    xgb_optimism_bias = xgb_non_nested_score - xgb_nested_mean

    logger.info(
        "Nested CV AUC — Logistic Regression: %.3f ± %.3f (Non-nested: %.3f, Optimism Bias: +%.3f)",
        lr_nested_mean,
        lr_nested_std,
        lr_non_nested_score,
        lr_optimism_bias,
    )
    logger.info(
        "Nested CV AUC — XGBoost:             %.3f ± %.3f (Non-nested: %.3f, Optimism Bias: +%.3f)",
        xgb_nested_mean,
        xgb_nested_std,
        xgb_non_nested_score,
        xgb_optimism_bias,
    )

    models: dict[str, Any] = {
        "Dummy Baseline": dummy_baseline,
        "Clinical Rule Baseline": clinical_baseline,
        "Logistic Regression": logreg_gs.best_estimator_,
        "XGBoost": xgb_gs.best_estimator_,
    }

    # -------------------------------------------------------------------------
    # 3. 5x5 Repeated Stratified CV Evaluation & Statistical Fold Comparison
    # -------------------------------------------------------------------------
    logger.info("Executing 5x5 Repeated Stratified CV across %d folds...", CV_SPLITS * CV_REPEATS)
    cv_outer = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=seed
    )
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "recall": "recall",
        "precision": "precision",
        "brier": "neg_brier_score",
    }

    cv_results_dict = {}
    cv_rows = []
    for name, model in models.items():
        if name == "Dummy Baseline":
            # Dummy classifier metrics
            res = cross_validate(model, X_train, y_train, cv=cv_outer, scoring=scoring)
        elif name == "Clinical Rule Baseline":
            res = cross_validate(model, X_train, y_train, cv=cv_outer, scoring=scoring)
        else:
            res = cross_validate(model, X_train, y_train, cv=cv_outer, scoring=scoring)
        cv_results_dict[name] = res
        cv_rows.append(
            {
                "Model": name,
                "ROC-AUC": f"{res['test_roc_auc'].mean():.3f} ± {res['test_roc_auc'].std():.3f}",
                "PR-AUC": f"{res['test_pr_auc'].mean():.3f} ± {res['test_pr_auc'].std():.3f}",
                "Recall": f"{res['test_recall'].mean():.3f} ± {res['test_recall'].std():.3f}",
                "Precision": f"{res['test_precision'].mean():.3f} ± {res['test_precision'].std():.3f}",
                "Brier Score": f"{-res['test_brier'].mean():.3f} ± {res['test_brier'].std():.3f}",
            }
        )
    cv_table = pd.DataFrame(cv_rows)
    cv_table.to_csv(CV_RESULTS_PATH, index=False)
    logger.info("Repeated CV Summary Table:\n%s", cv_table.to_string(index=False))

    # Nadeau-Bengio corrected resampled t-test on CV fold ROC-AUCs
    n_train_fold = int(len(X_train) * (CV_SPLITS - 1) / CV_SPLITS)
    n_test_fold = int(len(X_train) / CV_SPLITS)
    lr_cv_aucs = cv_results_dict["Logistic Regression"]["test_roc_auc"]
    xgb_cv_aucs = cv_results_dict["XGBoost"]["test_roc_auc"]

    cv_ttest = nadeau_bengio_paired_ttest(
        lr_cv_aucs, xgb_cv_aucs, n_train=n_train_fold, n_test=n_test_fold
    )
    logger.info(
        "Nadeau-Bengio CV Paired t-Test (LR vs. XGBoost): Diff=%.3f, t=%.2f, p=%.4f (Statistically Significant: %s)",
        cv_ttest["mean_diff"],
        cv_ttest["t_statistic"],
        cv_ttest["p_value"],
        cv_ttest["is_significant"],
    )

    # -------------------------------------------------------------------------
    # 4. Out-of-fold Clinical Threshold Tuning & Holdout Prediction
    # -------------------------------------------------------------------------
    logger.info("Tuning sensitivity-calibrated thresholds and predicting holdout cohort...")
    thresholds = {}
    holdout_probs = {}
    holdout_preds = {}
    holdout_results: dict[str, Any] = {}

    for name, model in models.items():
        if name in ["Logistic Regression", "XGBoost"]:
            thr = pick_threshold(model, X_train, y_train, target_recall=target_recall, seed=seed)
        elif name == "Clinical Rule Baseline":
            thr = 0.50
        else:
            thr = 0.50
        thresholds[name] = thr

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= thr).astype(int)

        holdout_probs[name] = y_prob
        holdout_preds[name] = y_pred

        metrics = evaluate_predictions(y_test.values, y_prob, threshold=thr)
        holdout_results[name] = metrics

        if name in ["Logistic Regression", "XGBoost"]:
            plot_confusion_matrix_artifact(y_test.values, y_pred, model_name=name, threshold=thr)

    # -------------------------------------------------------------------------
    # 5. Bootstrap 95% Confidence Intervals for Holdout Metrics
    # -------------------------------------------------------------------------
    logger.info("Computing percentile bootstrap 95%% CIs (B=%d resamples)...", BOOTSTRAP_ROUNDS)
    bootstrap_results: dict[str, Any] = {}
    for name in ["Logistic Regression", "XGBoost"]:
        boot_ci = compute_bootstrap_ci(
            y_test.values,
            holdout_probs[name],
            threshold=thresholds[name],
            n_bootstraps=BOOTSTRAP_ROUNDS,
            seed=seed,
        )
        bootstrap_results[name] = boot_ci
        logger.info(
            "%s Bootstrap CIs — ROC-AUC: [%.3f, %.3f], Recall: [%.3f, %.3f], Precision: [%.3f, %.3f]",
            name,
            boot_ci["roc_auc"]["ci_lower"],
            boot_ci["roc_auc"]["ci_upper"],
            boot_ci["recall"]["ci_lower"],
            boot_ci["recall"]["ci_upper"],
            boot_ci["precision"]["ci_lower"],
            boot_ci["precision"]["ci_upper"],
        )

    # -------------------------------------------------------------------------
    # 6. DeLong Paired ROC-AUC Statistical Test on Holdout Cohort
    # -------------------------------------------------------------------------
    delong_res = delong_roc_test(
        y_test.values,
        holdout_probs["Logistic Regression"],
        holdout_probs["XGBoost"],
    )
    logger.info(
        "DeLong Test (LR vs. XGBoost on Holdout): AUC Diff=%.3f, z=%.2f, p=%.4f (Significant: %s)",
        delong_res["auc_difference"],
        delong_res["z_statistic"],
        delong_res["p_value"],
        delong_res["is_significant"],
    )

    # -------------------------------------------------------------------------
    # 7. Calibration Analysis: Raw vs. Platt (Sigmoid) vs. Isotonic
    # -------------------------------------------------------------------------
    logger.info("Evaluating Platt scaling and Isotonic recalibration...")
    cal_lr = evaluate_calibration_methods(
        logreg_gs.best_estimator_, X_train, y_train, X_test, y_test
    )
    cal_xgb = evaluate_calibration_methods(
        xgb_gs.best_estimator_, X_train, y_train, X_test, y_test
    )
    plot_calibration_recalibration(cal_lr, cal_xgb, y_test)
    logger.info(
        "Logistic Regression Brier Score — Raw: %.3f, Platt: %.3f, Isotonic: %.3f",
        cal_lr["Raw"]["brier_score"],
        cal_lr["Platt (Sigmoid)"]["brier_score"],
        cal_lr["Isotonic"]["brier_score"],
    )
    logger.info(
        "XGBoost Brier Score             — Raw: %.3f, Platt: %.3f, Isotonic: %.3f",
        cal_xgb["Raw"]["brier_score"],
        cal_xgb["Platt (Sigmoid)"]["brier_score"],
        cal_xgb["Isotonic"]["brier_score"],
    )

    # -------------------------------------------------------------------------
    # 8. Decision Curve Analysis (Clinical Net Benefit)
    # -------------------------------------------------------------------------
    logger.info("Computing Decision Curve Analysis across referral thresholds...")
    dca_probs = {
        "Logistic Regression": holdout_probs["Logistic Regression"],
        "XGBoost": holdout_probs["XGBoost"],
        "Clinical Rule Baseline": holdout_probs["Clinical Rule Baseline"],
    }
    plot_decision_curve_analysis(dca_probs, y_test.values, save_path=DCA_PLOT_PATH)

    # -------------------------------------------------------------------------
    # 9. Asymmetric Cost-Sensitive Optimization (10x FN Penalty)
    # -------------------------------------------------------------------------
    logger.info("Conducting asymmetric cost optimization (10x FN cost ratio)...")
    cost_lr = evaluate_cost_sensitive(
        y_test.values, holdout_probs["Logistic Regression"], cost_fn_ratio=cost_fn_ratio
    )
    cost_xgb = evaluate_cost_sensitive(
        y_test.values, holdout_probs["XGBoost"], cost_fn_ratio=cost_fn_ratio
    )
    cost_clin = evaluate_cost_sensitive(
        y_test.values, holdout_probs["Clinical Rule Baseline"], cost_fn_ratio=cost_fn_ratio
    )

    plot_cost_tradeoff_curve(dca_probs, y_test.values, cost_ratio=cost_fn_ratio)
    logger.info(
        "Cost-Minimizing Thresholds — LR: T*=%.2f (Cost: %.1f vs. 50%%: %.1f), XGB: T*=%.2f (Cost: %.1f vs. 50%%: %.1f)",
        cost_lr["optimal_threshold"],
        cost_lr["min_total_cost"],
        cost_lr["default_50_cost"],
        cost_xgb["optimal_threshold"],
        cost_xgb["min_total_cost"],
        cost_xgb["default_50_cost"],
    )

    # -------------------------------------------------------------------------
    # 10. Empirical Learning Curves (Assessing Data Saturation)
    # -------------------------------------------------------------------------
    logger.info("Generating learning curves across training cohort subsets...")
    learning_models = {
        "Logistic Regression": logreg_gs.best_estimator_,
        "XGBoost": xgb_gs.best_estimator_,
    }
    plot_learning_curves(learning_models, X_train, y_train, save_path=LEARNING_CURVE_PLOT_PATH)

    # -------------------------------------------------------------------------
    # 11. Final Diagnostic Plots & Artifact Serialization
    # -------------------------------------------------------------------------
    plot_roc_comparison(
        {
            "Logistic Regression": models["Logistic Regression"],
            "XGBoost": models["XGBoost"],
        },
        X_test,
        y_test,
        # ClinicalRuleBaseline is not a standard sklearn estimator so it cannot
        # be passed to RocCurveDisplay.from_estimator; use pre-computed probs instead.
        precomputed_probs={"Clinical Rule Baseline": holdout_probs["Clinical Rule Baseline"]},
    )
    plot_calibration_comparison(
        {
            "Logistic Regression": models["Logistic Regression"],
            "XGBoost": models["XGBoost"],
        },
        X_test,
        y_test,
    )

    # -------------------------------------------------------------------------
    # 12. Fairness & Responsible-AI Audit
    # -------------------------------------------------------------------------
    logger.info("Running fairness and responsible-AI subgroup audit...")
    fairness_model_probs = {
        "Logistic Regression": holdout_probs["Logistic Regression"],
        "XGBoost": holdout_probs["XGBoost"],
    }
    run_fairness_audit(
        X_test=X_test,
        y_test=y_test,
        model_probs=fairness_model_probs,
        thresholds={
            "Logistic Regression": thresholds["Logistic Regression"],
            "XGBoost": thresholds["XGBoost"],
        },
    )

    joblib.dump(models["Logistic Regression"], LOGREG_MODEL_PATH)
    joblib.dump(models["XGBoost"], XGB_MODEL_PATH)
    with open(THRESHOLD_CONFIG_PATH, "w") as f:
        json.dump(
            {
                "threshold": thresholds["Logistic Regression"],
                "threshold_xgb": thresholds["XGBoost"],
                "cost_optimal_threshold_lr": cost_lr["optimal_threshold"],
                "cost_optimal_threshold_xgb": cost_xgb["optimal_threshold"],
                "target_recall": target_recall,
                "cost_fn_ratio": cost_fn_ratio,
            },
            f,
            indent=2,
        )

    # Save comprehensive research results table
    research_summary = pd.DataFrame(
        [
            {
                "Model / Baseline": "Dummy Baseline",
                "Operating Cutoff": "0.50",
                "Repeated CV ROC-AUC": cv_table.loc[cv_table["Model"] == "Dummy Baseline", "ROC-AUC"].values[0],
                "Holdout ROC-AUC [95% CI]": f"{holdout_results['Dummy Baseline']['roc_auc']:.3f}",
                "Holdout Recall [95% CI]": f"{holdout_results['Dummy Baseline']['recall']:.1%}",
                "Holdout Precision [95% CI]": f"{holdout_results['Dummy Baseline']['precision']:.1%}",
                "Brier Score": f"{holdout_results['Dummy Baseline']['brier_score']:.3f}",
                "Clinical Cost (10x FN)": f"{evaluate_cost_sensitive(y_test.values, holdout_probs['Dummy Baseline'], cost_fn_ratio)['default_50_cost']:.0f}",
            },
            {
                "Model / Baseline": "Clinical Rule Baseline (Diamond-Forrester)",
                "Operating Cutoff": "Criteria >= 2",
                "Repeated CV ROC-AUC": cv_table.loc[cv_table["Model"] == "Clinical Rule Baseline", "ROC-AUC"].values[0],
                "Holdout ROC-AUC [95% CI]": f"{holdout_results['Clinical Rule Baseline']['roc_auc']:.3f}",
                "Holdout Recall [95% CI]": f"{holdout_results['Clinical Rule Baseline']['recall']:.1%}",
                "Holdout Precision [95% CI]": f"{holdout_results['Clinical Rule Baseline']['precision']:.1%}",
                "Brier Score": f"{holdout_results['Clinical Rule Baseline']['brier_score']:.3f}",
                "Clinical Cost (10x FN)": f"{cost_clin['default_50_cost']:.0f}",
            },
            {
                "Model / Baseline": "Logistic Regression (Linear, Interpretable)",
                "Operating Cutoff": f"T* = {thresholds['Logistic Regression']:.2f}",
                "Repeated CV ROC-AUC": f"{lr_nested_mean:.3f} ± {lr_nested_std:.3f} (Nested)",
                "Holdout ROC-AUC [95% CI]": (
                    f"{holdout_results['Logistic Regression']['roc_auc']:.3f} "
                    f"[{bootstrap_results['Logistic Regression']['roc_auc']['ci_lower']:.3f} – "
                    f"{bootstrap_results['Logistic Regression']['roc_auc']['ci_upper']:.3f}]"
                ),
                "Holdout Recall [95% CI]": (
                    f"{holdout_results['Logistic Regression']['recall']:.1%} "
                    f"[{bootstrap_results['Logistic Regression']['recall']['ci_lower']:.1%} – "
                    f"{bootstrap_results['Logistic Regression']['recall']['ci_upper']:.1%}]"
                ),
                "Holdout Precision [95% CI]": (
                    f"{holdout_results['Logistic Regression']['precision']:.1%} "
                    f"[{bootstrap_results['Logistic Regression']['precision']['ci_lower']:.1%} – "
                    f"{bootstrap_results['Logistic Regression']['precision']['ci_upper']:.1%}]"
                ),
                "Brier Score": f"{holdout_results['Logistic Regression']['brier_score']:.3f}",
                "Clinical Cost (10x FN)": f"{cost_lr['min_total_cost']:.0f} (Cost-minimized)",
            },
            {
                "Model / Baseline": "XGBoost (Nonlinear Ensemble)",
                "Operating Cutoff": f"T* = {thresholds['XGBoost']:.2f}",
                "Repeated CV ROC-AUC": f"{xgb_nested_mean:.3f} ± {xgb_nested_std:.3f} (Nested)",
                "Holdout ROC-AUC [95% CI]": (
                    f"{holdout_results['XGBoost']['roc_auc']:.3f} "
                    f"[{bootstrap_results['XGBoost']['roc_auc']['ci_lower']:.3f} – "
                    f"{bootstrap_results['XGBoost']['roc_auc']['ci_upper']:.3f}]"
                ),
                "Holdout Recall [95% CI]": (
                    f"{holdout_results['XGBoost']['recall']:.1%} "
                    f"[{bootstrap_results['XGBoost']['recall']['ci_lower']:.1%} – "
                    f"{bootstrap_results['XGBoost']['recall']['ci_upper']:.1%}]"
                ),
                "Holdout Precision [95% CI]": (
                    f"{holdout_results['XGBoost']['precision']:.1%} "
                    f"[{bootstrap_results['XGBoost']['precision']['ci_lower']:.1%} – "
                    f"{bootstrap_results['XGBoost']['precision']['ci_upper']:.1%}]"
                ),
                "Brier Score": f"{holdout_results['XGBoost']['brier_score']:.3f}",
                "Clinical Cost (10x FN)": f"{cost_xgb['min_total_cost']:.0f} (Cost-minimized)",
            },
        ]
    )
    research_summary.to_csv(RESEARCH_RESULTS_PATH, index=False)

    full_audit = {
        "holdout_results": holdout_results,
        "bootstrap_results": bootstrap_results,
        "delong_test": delong_res,
        "nadeau_bengio_cv_ttest": cv_ttest,
        "nested_cv": {
            "logistic_regression": {
                "nested_mean": lr_nested_mean,
                "nested_std": lr_nested_std,
                "non_nested": lr_non_nested_score,
                "optimism_bias": lr_optimism_bias,
            },
            "xgboost": {
                "nested_mean": xgb_nested_mean,
                "nested_std": xgb_nested_std,
                "non_nested": xgb_non_nested_score,
                "optimism_bias": xgb_optimism_bias,
            },
        },
        "calibration": {
            "logistic_regression": {k: {"brier": v["brier_score"], "auc": v["roc_auc"]} for k, v in cal_lr.items()},
            "xgboost": {k: {"brier": v["brier_score"], "auc": v["roc_auc"]} for k, v in cal_xgb.items()},
        },
        "cost_sensitive": {
            "logistic_regression": {
                "optimal_threshold": cost_lr["optimal_threshold"],
                "min_cost": cost_lr["min_total_cost"],
                "default_50_cost": cost_lr["default_50_cost"],
                "savings": cost_lr["cost_savings_vs_50"],
            },
            "xgboost": {
                "optimal_threshold": cost_xgb["optimal_threshold"],
                "min_cost": cost_xgb["min_total_cost"],
                "default_50_cost": cost_xgb["default_50_cost"],
                "savings": cost_xgb["cost_savings_vs_50"],
            },
        },
    }
    with open(HOLDOUT_RESULTS_PATH, "w") as f:
        json.dump(full_audit, f, indent=2)

    X_test.to_csv(X_TEST_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH, index=False)
    logger.info("Artifacts and research report saved successfully.")

    return full_audit, research_summary


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Heart Disease Risk Scoring: Research-Grade Clinical Training & Audit Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=DEFAULT_TARGET_RECALL,
        help="Target clinical sensitivity bound in (0.0, 1.0]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for cross-validation splits and model initialization",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Proportion of data allocated to the holdout test set",
    )
    parser.add_argument(
        "--cost-ratio",
        type=float,
        default=DEFAULT_FN_FP_COST_RATIO,
        help="Cost penalty ratio for False Negative relative to False Positive",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not 0.5 <= args.target_recall <= 1.0:
        raise ValueError(
            f"--target-recall must be between 0.50 and 1.00; got {args.target_recall}."
        )
    train_and_evaluate(
        target_recall=args.target_recall,
        seed=args.seed,
        test_size=args.test_size,
        cost_fn_ratio=args.cost_ratio,
    )
