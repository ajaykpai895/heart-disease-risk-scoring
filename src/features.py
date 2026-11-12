"""
Feature Preprocessing and Pipeline Construction Module.

Constructs leak-free ColumnTransformer pipelines with:
- Median imputation and standard scaling for continuous physiological features.
- Mode imputation and reference-dropped one-hot encoding for categorical factors,
  avoiding the dummy variable trap and enabling interpretable Odds Ratios.
"""

import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logger = logging.getLogger(__name__)


def build_preprocessor(drop_first: bool = True) -> ColumnTransformer:
    """Build a leak-free scikit-learn ColumnTransformer for clinical tabular data.

    Clinical & Statistical Rationale:
    1. Continuous features: Imputed with median (robust to physiological outliers
       such as extreme cholesterol or blood pressure readings), followed by
       StandardScaler (zero mean, unit variance) to stabilize gradient descent and
       ensure equal regularization penalty in L2-regularized Logistic Regression.
    2. Categorical features: Imputed with mode (most frequent), followed by
       OneHotEncoder with `drop='first'`. Dropping the reference level eliminates
       structural multicollinearity (the dummy variable trap) and ensures that
       Logistic Regression coefficients map to true epidemiological Odds Ratios
       relative to a baseline clinical category (e.g. sex_1.0 vs female baseline).

    Args:
        drop_first: Whether to drop the first category in OneHotEncoder to avoid
            collinearity and produce unaliased odds ratios in linear models.

    Returns:
        ColumnTransformer: Configured preprocessor ready for pipeline embedding.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    drop_param = "first" if drop_first else None
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop=drop_param,
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    logger.debug(
        "Built ColumnTransformer with %d numeric and %d categorical features (drop_first=%s).",
        len(NUMERIC_FEATURES),
        len(CATEGORICAL_FEATURES),
        drop_first,
    )
    return preprocessor


def get_feature_names(fitted_preprocessor: ColumnTransformer) -> list[str]:
    """Retrieve human-readable feature names output by a fitted ColumnTransformer.

    Args:
        fitted_preprocessor: A fitted ColumnTransformer instance.

    Returns:
        List[str]: Clean list of feature names post-transformation.
    """
    return list(fitted_preprocessor.get_feature_names_out())


if __name__ == "__main__":
    from src.data import load_raw_data

    X_raw, _ = load_raw_data()
    prep = build_preprocessor(drop_first=True)
    X_trans = prep.fit_transform(X_raw)
    feat_names = get_feature_names(prep)
    logger.info("Transformed shape: %s", X_trans.shape)
    logger.info("Output feature names (%d): %s", len(feat_names), feat_names)
