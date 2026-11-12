"""
Data Ingestion, Validation, and Partitioning Module.

Handles robust loading of the Cleveland Heart Disease dataset (UCI ID: 45),
enforces schema and physiological range validation, and provides stratified splits.
"""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from src.config import (
    ALL_FEATURES,
    CLINICAL_RANGES,
    FALLBACK_DATA_PATH,
    RAW_DATA_PATH,
    SEED,
    TARGET_COL,
    TEST_SIZE,
    VALID_CATEGORIES,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def validate_clinical_dataframe(df: pd.DataFrame) -> None:
    """Validate schema, required columns, and physiological ranges of clinical data.

    Args:
        df: Input pandas DataFrame to validate.

    Raises:
        ValueError: If required columns are missing, types are invalid, or values
            exceed biologically plausible boundaries.
    """
    missing_cols = set(ALL_FEATURES) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required features: {missing_cols}")

    if TARGET_COL in df.columns:
        valid_targets = {0, 1}
        actual_targets = set(df[TARGET_COL].dropna().unique())
        if not actual_targets.issubset(valid_targets):
            raise ValueError(
                f"Target column '{TARGET_COL}' contains invalid values: {actual_targets}. "
                f"Expected binary {valid_targets}."
            )

    # Validate numeric continuous boundaries
    for col, bounds in CLINICAL_RANGES.items():
        if col in df.columns:
            series = df[col].dropna()
            out_of_bounds = series[(series < bounds["min"]) | (series > bounds["max"])]
            if not out_of_bounds.empty:
                logger.warning(
                    "Feature '%s' has %d records outside expected clinical range [%.1f, %.1f].",
                    col,
                    len(out_of_bounds),
                    bounds["min"],
                    bounds["max"],
                )

    # Validate categorical domains
    for col, categories in VALID_CATEGORIES.items():
        if col in df.columns:
            invalid_cats = set(df[col].dropna().unique()) - set(categories)
            if invalid_cats:
                raise ValueError(
                    f"Categorical feature '{col}' contains unrecognized categories: {invalid_cats}. "
                    f"Expected subset of {categories}."
                )

    logger.debug("Clinical DataFrame passed schema and range validation.")


def load_heart(force_download: bool = False) -> pd.DataFrame:
    """Load the Cleveland Heart Disease dataset with binarized target.

    Tries local cache first, then ucimlrepo API (id=45), and finally falls back
    to local fallback CSV if the remote API fails.

    Args:
        force_download: If True, bypasses local raw cache and downloads fresh from UCI.

    Returns:
        pd.DataFrame: Validated DataFrame with 13 features and binary 'target' column.

    Raises:
        RuntimeError: If data cannot be loaded from any source.
    """
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not force_download and RAW_DATA_PATH.exists():
        logger.info("Loading cached raw dataset from %s", RAW_DATA_PATH)
        df = pd.read_csv(RAW_DATA_PATH)
        validate_clinical_dataframe(df)
        return df

    # Attempt fetch from ucimlrepo
    try:
        logger.info("Fetching UCI Cleveland dataset (id=45) via ucimlrepo...")
        dataset = fetch_ucirepo(id=45)
        df = dataset.data.features.copy()

        # Target 'num': 0 = healthy, 1-4 = coronary artery disease
        target_series = (
            dataset.data.targets["num"]
            if "num" in dataset.data.targets.columns
            else dataset.data.targets.iloc[:, 0]
        )
        df[TARGET_COL] = (target_series > 0).astype(int)

        # Cache locally
        df.to_csv(RAW_DATA_PATH, index=False)
        df.to_csv(FALLBACK_DATA_PATH, index=False)
        logger.info("Successfully fetched and cached dataset (%d rows).", len(df))
    except Exception as exc:
        logger.warning(
            "ucimlrepo API download failed (%s). Attempting local fallback...", exc
        )
        if FALLBACK_DATA_PATH.exists():
            df = pd.read_csv(FALLBACK_DATA_PATH)
        else:
            raise RuntimeError(
                f"Unable to load data from ucimlrepo or fallback {FALLBACK_DATA_PATH}."
            ) from exc

    validate_clinical_dataframe(df)
    return df


def load_raw_data() -> tuple[pd.DataFrame, pd.Series]:
    """Retrieve features matrix X and binary target series y.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features X (13 columns) and binary target y.
    """
    df = load_heart()
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].copy()
    return X, y


def get_train_test_data(
    test_size: float = TEST_SIZE,
    random_state: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Generate stratified train/test partitions to preserve class balance.

    Args:
        test_size: Proportion of dataset allocated to the holdout test set.
        random_state: Random seed for deterministic reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X, y = load_raw_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    logger.info(
        "Created stratified split: train=%d (%.1f%% pos), test=%d (%.1f%% pos)",
        len(X_train),
        y_train.mean() * 100,
        len(X_test),
        y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    data = load_heart()
    logger.info("Loaded dataset shape: %s", data.shape)
    logger.info("Missing value profile:\n%s", data.isna().sum()[data.isna().sum() > 0])
    logger.info("Class distribution:\n%s", data[TARGET_COL].value_counts(normalize=True))
