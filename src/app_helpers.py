"""
Pure-function helpers extracted from app.py to enable unit testing.

All Streamlit calls remain in app.py; this module contains only testable
business logic (DataFrame construction, inference, attribution, counterfactuals, decisions).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    ALL_FEATURES,
    LOGREG_MODEL_PATH,
    THRESHOLD_CONFIG_PATH,
    XGB_MODEL_PATH,
)

logger = logging.getLogger(__name__)

# Human-readable labels for transformed feature names
FEATURE_LABEL_MAP: dict[str, str] = {
    "age": "Age",
    "trestbps": "Resting Blood Pressure",
    "chol": "Serum Cholesterol",
    "thalach": "Peak Exercise Heart Rate",
    "oldpeak": "ST Depression (oldpeak)",
    "ca": "Fluoroscopy Vessels Colored",
    "sex_1.0": "Biological Sex: Male",
    "cp_2.0": "Chest Pain: Atypical Angina",
    "cp_3.0": "Chest Pain: Non-Anginal",
    "cp_4.0": "Chest Pain: Asymptomatic Referral",
    "fbs_1.0": "Fasting Blood Sugar > 120 mg/dl",
    "restecg_1.0": "Resting ECG: ST-T Abnormality",
    "restecg_2.0": "Resting ECG: LV Hypertrophy",
    "exang_1.0": "Exercise-Induced Angina: Yes",
    "slope_2.0": "ST Segment Slope: Flat",
    "slope_3.0": "ST Segment Slope: Downsloping",
    "thal_6.0": "Thallium: Fixed Defect (Prior Infarct)",
    "thal_7.0": "Thallium: Reversible Defect (Active Ischemia)",
}


def clean_feature_label(raw_feature_name: str) -> str:
    """Map a raw transformed feature name to a clinician-friendly label.

    Args:
        raw_feature_name: Name output by ColumnTransformer (e.g., 'thal_7.0').

    Returns:
        str: Human-readable clinical label.
    """
    clean_name = raw_feature_name.replace("num__", "").replace("cat__", "")
    return FEATURE_LABEL_MAP.get(clean_name, clean_name)


def load_all_artifacts(
    logreg_path: Path = LOGREG_MODEL_PATH,
    xgb_path: Path = XGB_MODEL_PATH,
    threshold_path: Path = THRESHOLD_CONFIG_PATH,
) -> tuple[Pipeline, Pipeline, float, float]:
    """Load serialized models and calibrated thresholds.

    Args:
        logreg_path: Path to serialized Logistic Regression joblib artifact.
        xgb_path: Path to serialized XGBoost joblib artifact.
        threshold_path: Path to serialized threshold JSON config.

    Returns:
        tuple containing (lr_model, xgb_model, threshold_lr, threshold_xgb).

    Raises:
        FileNotFoundError: If any of the required artifacts do not exist.
    """
    if not logreg_path.exists():
        raise FileNotFoundError(f"Logistic Regression model not found at {logreg_path}")
    if not xgb_path.exists():
        raise FileNotFoundError(f"XGBoost model not found at {xgb_path}")
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold configuration not found at {threshold_path}")

    lr_model = joblib.load(logreg_path)
    xgb_model = joblib.load(xgb_path)

    with open(threshold_path) as f:
        config = json.load(f)

    threshold_lr = float(config.get("threshold", 0.50))
    threshold_xgb = float(config.get("threshold_xgb", 0.50))

    return lr_model, xgb_model, threshold_lr, threshold_xgb


def build_patient_dataframe(
    age: float,
    sex: float,
    cp: float,
    trestbps: float,
    chol: float,
    fbs: float,
    restecg: float,
    thalach: float,
    exang: float,
    oldpeak: float,
    slope: float,
    ca: float,
    thal: float,
) -> pd.DataFrame:
    """Assemble a one-row patient DataFrame with columns in canonical feature order.

    Args:
        age: Patient age in years.
        sex: Biological sex (1.0 = male, 0.0 = female).
        cp: Chest pain type (1=typical, 2=atypical, 3=non-anginal, 4=asymptomatic).
        trestbps: Resting blood pressure (mm Hg).
        chol: Serum cholesterol (mg/dl).
        fbs: Fasting blood sugar > 120 mg/dl (1.0 = true, 0.0 = false).
        restecg: Resting ECG (0=normal, 1=ST-T abnormality, 2=LV hypertrophy).
        thalach: Maximum heart rate achieved during exercise.
        exang: Exercise-induced angina (1.0 = yes, 0.0 = no).
        oldpeak: ST depression induced by exercise (mm).
        slope: Slope of peak exercise ST segment (1=upsloping, 2=flat, 3=downsloping).
        ca: Number of major fluoroscopy vessels colored (0-3).
        thal: Thallium scintigraphy (3=normal, 6=fixed defect, 7=reversible defect).

    Returns:
        pd.DataFrame: Single-row DataFrame matching ALL_FEATURES order.
    """
    record = {
        "age": float(age),
        "sex": float(sex),
        "cp": float(cp),
        "trestbps": float(trestbps),
        "chol": float(chol),
        "fbs": float(fbs),
        "restecg": float(restecg),
        "thalach": float(thalach),
        "exang": float(exang),
        "oldpeak": float(oldpeak),
        "slope": float(slope),
        "ca": float(ca),
        "thal": float(thal),
    }
    return pd.DataFrame([record])[ALL_FEATURES]


def compute_risk_probability(model: Any, patient_df: pd.DataFrame) -> float:
    """Return the predicted CAD risk probability for a single patient.

    Args:
        model: Fitted pipeline exposing ``predict_proba``.
        patient_df: One-row DataFrame matching the feature schema.

    Returns:
        float: Probability in [0.0, 1.0].

    Raises:
        ValueError: If probability output is not in [0.0, 1.0].
    """
    prob = float(model.predict_proba(patient_df)[0, 1])
    if not (0.0 <= prob <= 1.0):
        raise ValueError(f"Model returned invalid probability: {prob}")
    return prob


def get_referral_decision(prob: float, threshold: float) -> bool:
    """Determine whether the patient meets the cardiology referral cutoff.

    Args:
        prob: Predicted risk probability in [0.0, 1.0].
        threshold: Operating decision threshold.

    Returns:
        bool: True if referral recommended (prob >= threshold).
    """
    return prob >= threshold


def compute_local_attributions(
    model: Pipeline,
    patient_df: pd.DataFrame,
) -> pd.Series:
    """Compute per-feature linear log-odds attributions for Logistic Regression.

    Args:
        model: Fitted sklearn Pipeline with 'prep' and 'clf' steps.
        patient_df: One-row DataFrame.

    Returns:
        pd.Series: Signed attributions sorted by absolute magnitude.
    """
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    x_trans = prep.transform(patient_df)[0]
    feat_names = list(prep.get_feature_names_out())
    cleaned_names = [clean_feature_label(name) for name in feat_names]
    contributions = pd.Series(x_trans * clf.coef_[0], index=cleaned_names)
    return contributions.sort_values(key=abs, ascending=False)


def generate_plain_english_explanation(
    contributions: pd.Series,
    top_n: int = 3,
) -> dict[str, Any]:
    """Generate plain-English explanation sentences from local attributions.

    Args:
        contributions: Series of signed feature contributions to log-odds.
        top_n: Maximum number of positive and negative drivers to include.

    Returns:
        dict with 'elevating_factors', 'protective_factors', and 'summary_text'.
    """
    positives = contributions[contributions > 0.05].head(top_n)
    negatives = contributions[contributions < -0.05].head(top_n)

    elevating_items = [
        f"{idx} (+{val:.2f} log-odds)" for idx, val in positives.items()
    ]
    protective_items = [
        f"{idx} ({val:.2f} log-odds)" for idx, val in negatives.items()
    ]

    parts: list[str] = []
    if elevating_items:
        parts.append(
            f"**Key risk-elevating factors:** {', '.join(elevating_items)}."
        )
    else:
        parts.append("No major physiological risk factors markedly elevated above baseline.")

    if protective_items:
        parts.append(
            f"**Key protective factors:** {', '.join(protective_items)}."
        )
    else:
        parts.append("Few protective clinical indicators observed.")

    return {
        "elevating_factors": list(positives.index),
        "protective_factors": list(negatives.index),
        "summary_text": " ".join(parts),
    }


def simulate_what_if(
    model: Any,
    base_patient: pd.DataFrame,
    feature_name: str,
    candidate_values: list[float] | np.ndarray,
) -> pd.DataFrame:
    """Simulate how predicted risk changes when a single feature varies.

    Keeps all other patient characteristics fixed while varying `feature_name`.

    Args:
        model: Fitted estimator with predict_proba.
        base_patient: One-row DataFrame of baseline patient values.
        feature_name: Feature column name to vary.
        candidate_values: Sequence of alternative values to evaluate.

    Returns:
        pd.DataFrame with 'Value' and 'Predicted Risk (%)' columns.
    """
    rows = []
    for val in candidate_values:
        sim_df = base_patient.copy()
        sim_df[feature_name] = float(val)
        prob = compute_risk_probability(model, sim_df)
        rows.append({"Value": float(val), "Predicted Risk (%)": round(prob * 100, 1)})
    return pd.DataFrame(rows)


def compare_model_predictions(
    prob_lr: float,
    threshold_lr: float,
    prob_xgb: float,
    threshold_xgb: float,
) -> dict[str, Any]:
    """Compare predictions from Logistic Regression and XGBoost.

    Identifies consensus and flags when models disagree on referral.

    Args:
        prob_lr: Logistic Regression predicted risk probability.
        threshold_lr: Logistic Regression referral threshold.
        prob_xgb: XGBoost predicted risk probability.
        threshold_xgb: XGBoost referral threshold.

    Returns:
        dict with decisions, agreement status, and clinical advisory.
    """
    flag_lr = get_referral_decision(prob_lr, threshold_lr)
    flag_xgb = get_referral_decision(prob_xgb, threshold_xgb)
    agree = flag_lr == flag_xgb
    diff_prob = abs(prob_lr - prob_xgb)

    if agree:
        if flag_lr:
            advisory = (
                "Both models achieve clinical consensus: **Referral Recommended**. "
                "Both linear and non-linear models agree this patient exceeds risk thresholds."
            )
        else:
            advisory = (
                "Both models achieve clinical consensus: **Routine Follow-Up**. "
                "Both models place this patient below the cardiology referral threshold."
            )
    else:
        advisory = (
            "⚠️ **MODEL DISAGREEMENT DETECTED**: "
            f"Logistic Regression predicts {'REFERRAL' if flag_lr else 'NO REFERRAL'} "
            f"(p={prob_lr:.1%}, T*={threshold_lr:.2f}), whereas XGBoost predicts "
            f"{'REFERRAL' if flag_xgb else 'NO REFERRAL'} (p={prob_xgb:.1%}, T*={threshold_xgb:.2f}). "
            "In clinical practice, when models disagree, priority defaults to the "
            "interpretable model and clinician judgment with mandatory manual review."
        )

    return {
        "lr_decision": flag_lr,
        "xgb_decision": flag_xgb,
        "models_agree": agree,
        "probability_gap": round(diff_prob, 3),
        "advisory": advisory,
    }
