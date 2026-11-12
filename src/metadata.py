"""
Model Provenance and Metadata Tracking Module.

Generates and persists audit-grade metadata for production models:
- Framework versions (scikit-learn, xgboost, python)
- Timestamp and environment details
- Cryptographic SHA-256 hash of raw training data
- Operating decision thresholds and optimization targets
- Holdout and cross-validation performance metrics
- Git commit hash for exact source code traceability
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sklearn
import xgboost

from src.config import (
    DEFAULT_FN_FP_COST_RATIO,
    DEFAULT_TARGET_RECALL,
    HOLDOUT_RESULTS_PATH,
    MODELS_DIR,
    RAW_DATA_PATH,
    SEED,
    TEST_SIZE,
    THRESHOLD_CONFIG_PATH,
)

logger = logging.getLogger(__name__)

METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"


def compute_file_sha256(filepath: Path) -> str:
    """Compute cryptographic SHA-256 hash of a file for data lineage tracking."""
    if not filepath.exists():
        return "file_not_found"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit() -> str:
    """Retrieve current Git commit SHA-1 or return fallback string."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
        return commit
    except Exception:
        return "unversioned_git_tree"


def generate_model_metadata(
    save_path: Path = METADATA_PATH,
) -> dict[str, Any]:
    """Compile and serialize production model lineage and audit metadata.

    Args:
        save_path: Destination path for model_metadata.json.

    Returns:
        dict: Complete structured metadata dictionary.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load calibrated thresholds
    thresholds: dict[str, Any] = {}
    if THRESHOLD_CONFIG_PATH.exists():
        with open(THRESHOLD_CONFIG_PATH) as f:
            thresholds = json.load(f)

    # Load holdout metrics
    holdout: dict[str, Any] = {}
    if HOLDOUT_RESULTS_PATH.exists():
        with open(HOLDOUT_RESULTS_PATH) as f:
            holdout = json.load(f)

    metadata: dict[str, Any] = {
        "model_name": "Cardiovascular Risk Screening (Logistic Regression & XGBoost)",
        "model_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "environment": {
            "python_version": sys.version.split()[0],
            "scikit_learn_version": sklearn.__version__,
            "xgboost_version": xgboost.__version__,
            "platform": sys.platform,
        },
        "dataset_lineage": {
            "source": "UCI Cleveland Heart Disease (ID: 45)",
            "file_path": str(RAW_DATA_PATH),
            "sha256_hash": compute_file_sha256(RAW_DATA_PATH),
            "test_size": TEST_SIZE,
            "random_seed": SEED,
        },
        "operating_parameters": {
            "target_recall": DEFAULT_TARGET_RECALL,
            "cost_fn_ratio": DEFAULT_FN_FP_COST_RATIO,
            "threshold_logistic_regression": thresholds.get("threshold"),
            "threshold_xgboost": thresholds.get("threshold_xgb"),
            "cost_optimal_threshold_lr": thresholds.get("cost_optimal_threshold_lr"),
            "cost_optimal_threshold_xgb": thresholds.get("cost_optimal_threshold_xgb"),
        },
        "performance_snapshot": {
            "logistic_regression_holdout": holdout.get("holdout_results", {}).get("Logistic Regression"),
            "xgboost_holdout": holdout.get("holdout_results", {}).get("XGBoost"),
            "delong_test_p_value": holdout.get("delong_test", {}).get("p_value"),
            "nadeau_bengio_p_value": holdout.get("nadeau_bengio_cv_ttest", {}).get("p_value"),
        },
    }

    with open(save_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved model metadata artifact to %s", save_path)
    return metadata


if __name__ == "__main__":
    generate_model_metadata()
