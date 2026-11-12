"""
Model Architecture and Pipeline Construction Module.
Defines transparent (Logistic Regression) and black-box (XGBoost) model pipelines,
complete with hyperparameter search grids and serialization utilities.
"""

import sys
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import get_preprocessor  # noqa: E402


def create_logistic_regression_pipeline(
    C: float = 0.5,
    penalty: str = "l2",
    solver: str = "liblinear",
    max_iter: int = 1000,
    random_state: int = 42
) -> Pipeline:
    """
    Construct an end-to-end transparent Logistic Regression pipeline.

    Clinical Rationale:
    Logistic regression provides direct probabilistic outputs with monotonic, additive
    log-odds contributions. Each coefficient exponentiates to an odds ratio, allowing
    clinicians to assess patient-specific risk factors transparently.
    L2 regularization (Ridge) prevents coefficient inflation from collinear features
    (e.g., ST depression and slope).
    """
    preprocessor = get_preprocessor()
    classifier = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )


def create_xgboost_pipeline(
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    gamma: float = 0.1,
    reg_lambda: float = 1.0,
    random_state: int = 42
) -> Pipeline:
    """
    Construct an end-to-end black-box XGBoost pipeline.

    Clinical Rationale:
    XGBoost captures complex non-linear clinical interactions (e.g. how heart rate reserve
    interacts with ST segment depression during stress testing). Shallow tree depth (max_depth=3)
    and subsampling protect against overfitting on small clinical cohorts (n=303).
    """
    preprocessor = get_preprocessor()
    classifier = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_lambda=reg_lambda,
        eval_metric="logloss",
        random_state=random_state
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )


def save_pipeline(pipeline: Pipeline, filename: str) -> Path:
    """Persist trained pipeline to models/ directory."""
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    filepath = models_dir / filename
    joblib.dump(pipeline, filepath)
    print(f"Model saved successfully to: {filepath}")
    return filepath


def load_pipeline(filename: str) -> Pipeline:
    """Load serialized pipeline from models/ directory."""
    filepath = PROJECT_ROOT / "models" / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found at {filepath}")
    return joblib.load(filepath)


if __name__ == "__main__":
    lr_pipe = create_logistic_regression_pipeline()
    xgb_pipe = create_xgboost_pipeline()
    print("Logistic Regression pipeline initialized:\n", lr_pipe)
    print("\nXGBoost pipeline initialized:\n", xgb_pipe)
