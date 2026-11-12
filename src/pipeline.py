"""
Leak-free Preprocessing Pipeline Module.
Constructs ColumnTransformer pipelines for numeric and categorical clinical variables,
ensuring zero data leakage across cross-validation folds and holdout sets.
"""

import sys
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CATEGORICAL_FEATURES, NUMERICAL_FEATURES  # noqa: E402


def get_preprocessor() -> ColumnTransformer:
    """
    Construct the scikit-learn ColumnTransformer.

    - Numeric pipeline:
        1. Median imputation (robust against physiological outliers like extreme cholesterol).
        2. Standard scaling (zero mean, unit variance for stable logistic regression convergence).
    - Categorical pipeline:
        1. Mode imputation (most frequent value).
        2. One-hot encoding with drop='first' (avoids multicollinearity for linear models).
           handle_unknown='ignore' ensures robustness if unobserved levels appear.

    Returns:
        ColumnTransformer: Preprocessor ready to be embedded into an end-to-end Pipeline.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor


def get_feature_names(fitted_preprocessor: ColumnTransformer) -> list[str]:
    """
    Retrieve human-interpretable feature names post-transformation.
    """
    return list(fitted_preprocessor.get_feature_names_out())


if __name__ == "__main__":
    from src.data import load_raw_data
    X, y = load_raw_data()
    preprocessor = get_preprocessor()
    X_trans = preprocessor.fit_transform(X)
    names = get_feature_names(preprocessor)
    print(f"Transformed feature matrix shape: {X_trans.shape}")
    print(f"Features ({len(names)}): {names}")
