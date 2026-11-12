"""
Global Configuration and Constants for Heart Disease Risk Scoring.

Centralizes paths, hyperparameters, clinical definitions, and seed values
to ensure zero magic numbers across the codebase.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_PATH: Path = DATA_DIR / "raw" / "heart_disease_cleveland.csv"
FALLBACK_DATA_PATH: Path = PROJECT_ROOT / "data_raw.csv"

MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# Serialized model artifacts
LOGREG_MODEL_PATH: Path = MODELS_DIR / "logreg.joblib"
XGB_MODEL_PATH: Path = MODELS_DIR / "xgb.joblib"
THRESHOLD_CONFIG_PATH: Path = MODELS_DIR / "threshold.json"

# Evaluation report artifacts
CV_RESULTS_PATH: Path = REPORTS_DIR / "cv_results.csv"
HOLDOUT_RESULTS_PATH: Path = REPORTS_DIR / "holdout_results.json"
LR_ODDS_RATIOS_PATH: Path = REPORTS_DIR / "lr_odds_ratios.csv"
X_TEST_PATH: Path = REPORTS_DIR / "X_test.csv"
Y_TEST_PATH: Path = REPORTS_DIR / "y_test.csv"
RESEARCH_RESULTS_PATH: Path = REPORTS_DIR / "research_evaluation_summary.csv"

# Research diagnostic figure paths
DCA_PLOT_PATH: Path = FIGURES_DIR / "decision_curve_analysis.png"
LEARNING_CURVE_PLOT_PATH: Path = FIGURES_DIR / "learning_curves.png"
COST_TRADEOFF_PLOT_PATH: Path = FIGURES_DIR / "cost_tradeoff.png"
CALIBRATION_COMPARISON_PLOT_PATH: Path = FIGURES_DIR / "calibration_recalibration.png"

# -----------------------------------------------------------------------------
# Reproducibility & Cross-Validation
# -----------------------------------------------------------------------------
SEED: int = 42
TEST_SIZE: float = 0.20
DEFAULT_TARGET_RECALL: float = 0.90
DEFAULT_FN_FP_COST_RATIO: float = 10.0
BOOTSTRAP_ROUNDS: int = 1000
CV_SPLITS: int = 5
CV_REPEATS: int = 5
TUNING_CV_REPEATS: int = 2

# -----------------------------------------------------------------------------
# Clinical Features & Schema Definition
# -----------------------------------------------------------------------------
NUMERIC_FEATURES: list[str] = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "ca",
]

CATEGORICAL_FEATURES: list[str] = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal",
]

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL: str = "target"

# Clinically valid input ranges for input validation
CLINICAL_RANGES: dict[str, dict[str, float]] = {
    "age": {"min": 18.0, "max": 100.0},
    "trestbps": {"min": 60.0, "max": 250.0},
    "chol": {"min": 80.0, "max": 700.0},
    "thalach": {"min": 50.0, "max": 250.0},
    "oldpeak": {"min": 0.0, "max": 10.0},
    "ca": {"min": 0.0, "max": 3.0},
}

VALID_CATEGORIES: dict[str, list[float]] = {
    "sex": [0.0, 1.0],
    "cp": [1.0, 2.0, 3.0, 4.0],
    "fbs": [0.0, 1.0],
    "restecg": [0.0, 1.0, 2.0],
    "exang": [0.0, 1.0],
    "slope": [1.0, 2.0, 3.0],
    "thal": [3.0, 6.0, 7.0],
}

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "age": "Age (years)",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (1 = typical angina, 2 = atypical, 3 = non-anginal, 4 = asymptomatic)",
    "trestbps": "Resting blood pressure (mm Hg upon hospital admission)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "Resting ECG (0 = normal, 1 = ST-T wave abnormality, 2 = LV hypertrophy)",
    "thalach": "Maximum heart rate achieved during exercise stress testing",
    "exang": "Exercise-induced angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise relative to rest (mm)",
    "slope": "Slope of peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping)",
    "ca": "Number of major vessels colored by fluoroscopy (0-3)",
    "thal": "Thallium scintigraphy (3 = normal, 6 = fixed defect, 7 = reversible defect)",
}

# -----------------------------------------------------------------------------
# Hyperparameter Grids
# -----------------------------------------------------------------------------
LOGREG_PARAM_GRID: dict[str, list[float]] = {
    "clf__C": [0.01, 0.1, 1.0, 10.0],
}

XGB_PARAM_GRID: dict[str, list[int]] = {
    "clf__max_depth": [2, 3, 4],
    "clf__n_estimators": [100, 300],
}
