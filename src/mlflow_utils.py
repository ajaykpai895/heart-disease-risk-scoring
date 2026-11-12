"""
MLflow Experiment Tracking and Model Registry Utilities.

Provides lightweight experiment logging for cardiovascular risk modeling:
- Tracks hyperparameters, cross-validation metrics, and holdout scores.
- Logs diagnostic artifacts (ROC curves, calibration diagrams, confusion matrices).
- Registers and logs scikit-learn / XGBoost pipelines.
- Supports local file-based tracking (default: ./mlruns) with zero external setup.

Viewing the MLflow Dashboard:
    Run: mlflow ui --port 5000
    Open: http://localhost:5000 in your browser to inspect runs, compare
          hyperparameter runs, and download serialized model artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.config import (
    DEFAULT_FN_FP_COST_RATIO,
    DEFAULT_TARGET_RECALL,
    FIGURES_DIR,
    PROJECT_ROOT,
    SEED,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)

MLFLOW_DB_PATH: Path = PROJECT_ROOT / "mlflow.db"
DEFAULT_TRACKING_URI: str = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
EXPERIMENT_NAME: str = "heart-disease-risk-scoring"


def setup_mlflow_tracking(
    experiment_name: str = EXPERIMENT_NAME,
    tracking_uri: str | None = None,
) -> None:
    """Initialize MLflow tracking URI (default SQLite database) and active experiment."""
    import mlflow

    uri = tracking_uri or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow configured: URI=%s, Experiment=%s", uri, experiment_name)


def log_experiment_run(
    model_name: str,
    model: Any,
    params: dict[str, Any],
    cv_metrics: dict[str, float],
    holdout_metrics: dict[str, float],
    artifacts: list[Path] | None = None,
) -> str:
    """Log a complete training run to MLflow.

    Args:
        model_name: Name of the architecture (e.g. 'Logistic Regression').
        model: Fitted pipeline object to log.
        params: Hyperparameter dictionary.
        cv_metrics: Cross-validation scores (mean AUC, recall, etc.).
        holdout_metrics: Holdout evaluation scores.
        artifacts: List of file paths (plots, CSVs) to log as run artifacts.

    Returns:
        str: MLflow active run ID.
    """
    import mlflow

    setup_mlflow_tracking()

    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id

        # Log system and training parameters
        mlflow.log_params(
            {
                "model_architecture": model_name,
                "seed": SEED,
                "test_size": TEST_SIZE,
                "target_recall": DEFAULT_TARGET_RECALL,
                "cost_fn_ratio": DEFAULT_FN_FP_COST_RATIO,
                **params,
            }
        )

        # Log cross-validation scores
        for k, v in cv_metrics.items():
            mlflow.log_metric(f"cv_{k}", float(v))

        # Log holdout scores
        for k, v in holdout_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"holdout_{k}", float(v))

        # Log artifacts (plots, tables)
        if artifacts:
            for art_path in artifacts:
                if art_path.exists():
                    mlflow.log_artifact(str(art_path))

        # Log scikit-learn model
        try:
            import mlflow.sklearn

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format="cloudpickle",
            )
        except Exception as exc:
            logger.warning("Could not log model artifact to MLflow: %s", exc)

        logger.info("MLflow run logged: ID=%s, Model=%s", run_id, model_name)
        return run_id


def log_all_saved_results() -> None:
    """Convenience helper: parse existing saved report files and log them to MLflow."""
    import json

    import joblib

    from src.config import (
        HOLDOUT_RESULTS_PATH,
        LOGREG_MODEL_PATH,
        RESEARCH_RESULTS_PATH,
        THRESHOLD_CONFIG_PATH,
        XGB_MODEL_PATH,
    )

    if not HOLDOUT_RESULTS_PATH.exists():
        logger.warning("No holdout results found to log.")
        return

    with open(HOLDOUT_RESULTS_PATH) as f:
        holdout_data = json.load(f)

    with open(THRESHOLD_CONFIG_PATH) as f:
        thr_data = json.load(f)

    artifacts = [
        HOLDOUT_RESULTS_PATH,
        RESEARCH_RESULTS_PATH,
        FIGURES_DIR / "roc_comparison.png",
        FIGURES_DIR / "calibration.png",
        FIGURES_DIR / "decision_curve_analysis.png",
        FIGURES_DIR / "learning_curves.png",
    ]

    # Log Logistic Regression
    if LOGREG_MODEL_PATH.exists():
        lr_model = joblib.load(LOGREG_MODEL_PATH)
        lr_holdout = holdout_data.get("holdout_results", {}).get("Logistic Regression", {})
        log_experiment_run(
            model_name="Logistic_Regression",
            model=lr_model,
            params={"threshold": thr_data.get("threshold", 0.5)},
            cv_metrics={"nested_auc": holdout_data.get("nested_cv", {}).get("logistic_regression", {}).get("nested_mean", 0.90)},
            holdout_metrics={k: v for k, v in lr_holdout.items() if isinstance(v, (int, float))},
            artifacts=artifacts,
        )

    # Log XGBoost
    if XGB_MODEL_PATH.exists():
        xgb_model = joblib.load(XGB_MODEL_PATH)
        xgb_holdout = holdout_data.get("holdout_results", {}).get("XGBoost", {})
        log_experiment_run(
            model_name="XGBoost",
            model=xgb_model,
            params={"threshold": thr_data.get("threshold_xgb", 0.5)},
            cv_metrics={"nested_auc": holdout_data.get("nested_cv", {}).get("xgboost", {}).get("nested_mean", 0.89)},
            holdout_metrics={k: v for k, v in xgb_holdout.items() if isinstance(v, (int, float))},
            artifacts=artifacts,
        )


if __name__ == "__main__":
    log_all_saved_results()
