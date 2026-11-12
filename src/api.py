"""
Production-Ready FastAPI Microservice for Cardiovascular Risk Screening.

Provides high-performance, validated HTTP REST inference endpoints:
- GET  /health  : Health check returning model metadata and operating thresholds.
- POST /predict : Pydantic-validated patient risk prediction with both
                  Logistic Regression (transparent) and XGBoost (ensemble),
                  consensus check, and top local risk factor drivers.

Run locally:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

Example cURL:
    curl -X POST "http://localhost:8000/predict" \
         -H "Content-Type: application/json" \
         -d '{
               "age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 240,
               "fbs": 0, "restecg": 1, "thalach": 145, "exang": 1,
               "oldpeak": 2.2, "slope": 2, "ca": 1, "thal": 7
             }'
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.app_helpers import (
    build_patient_dataframe,
    compare_model_predictions,
    compute_local_attributions,
    compute_risk_probability,
    get_referral_decision,
    load_all_artifacts,
)

logger = logging.getLogger("src.api")

# Global cached models
ml_models: dict[str, Any] = {}


def get_models() -> dict[str, Any]:
    """Retrieve or lazily initialize the production ML models and thresholds."""
    global ml_models
    if not ml_models:
        try:
            lr_model, xgb_model, threshold_lr, threshold_xgb = load_all_artifacts()
            ml_models["lr_model"] = lr_model
            ml_models["xgb_model"] = xgb_model
            ml_models["threshold_lr"] = threshold_lr
            ml_models["threshold_xgb"] = threshold_xgb
            logger.info("Loaded models: LR (T*=%.3f), XGB (T*=%.3f)", threshold_lr, threshold_xgb)
        except Exception as exc:
            logger.error("Failed to load artifacts: %s", exc)
            raise RuntimeError(f"Model initialization failed: {exc}") from exc
    return ml_models


app = FastAPI(
    title="Cardiovascular Risk Scoring API",
    description="Cardiology referral triage microservice comparing interpretable vs. black-box ML models.",
    version="1.0.0",
)


class PatientInput(BaseModel):
    """Validated schema for patient clinical, demographic, and diagnostic variables."""

    age: float = Field(..., ge=18.0, le=100.0, description="Age in years", examples=[58.0])
    sex: float = Field(..., ge=0.0, le=1.0, description="Biological sex (1=male, 0=female)", examples=[1.0])
    cp: float = Field(..., ge=1.0, le=4.0, description="Chest pain type (1=typical, 2=atypical, 3=non-anginal, 4=asymptomatic)", examples=[4.0])
    trestbps: float = Field(..., ge=60.0, le=250.0, description="Resting blood pressure (mm Hg)", examples=[140.0])
    chol: float = Field(..., ge=80.0, le=700.0, description="Serum cholesterol (mg/dl)", examples=[240.0])
    fbs: float = Field(..., ge=0.0, le=1.0, description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)", examples=[0.0])
    restecg: float = Field(..., ge=0.0, le=2.0, description="Resting ECG (0=normal, 1=ST-T abnormality, 2=LVH)", examples=[1.0])
    thalach: float = Field(..., ge=50.0, le=250.0, description="Peak exercise heart rate (bpm)", examples=[145.0])
    exang: float = Field(..., ge=0.0, le=1.0, description="Exercise-induced angina (1=yes, 0=no)", examples=[1.0])
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise (mm)", examples=[2.2])
    slope: float = Field(..., ge=1.0, le=3.0, description="Peak exercise ST slope (1=upsloping, 2=flat, 3=downsloping)", examples=[2.0])
    ca: float = Field(..., ge=0.0, le=3.0, description="Major vessels colored by fluoroscopy (0-3)", examples=[1.0])
    thal: float = Field(..., description="Thallium scintigraphy (3=normal, 6=fixed defect, 7=reversible defect)", examples=[7.0])

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 58.0,
                "sex": 1.0,
                "cp": 4.0,
                "trestbps": 140.0,
                "chol": 240.0,
                "fbs": 0.0,
                "restecg": 1.0,
                "thalach": 145.0,
                "exang": 1.0,
                "oldpeak": 2.2,
                "slope": 2.0,
                "ca": 1.0,
                "thal": 7.0,
            }
        }
    }


class ModelPredictionDetail(BaseModel):
    """Detailed score and decision for a single model architecture."""

    risk_probability: float = Field(..., description="Predicted CAD probability in [0, 1]")
    referral_recommended: bool = Field(..., description="Whether risk meets or exceeds the operating threshold")
    operating_threshold: float = Field(..., description="Safety-calibrated decision threshold")


class PredictionResponse(BaseModel):
    """Comprehensive triage response covering both models, consensus, and explainability."""

    logistic_regression: ModelPredictionDetail
    xgboost: ModelPredictionDetail
    models_agree: bool
    clinical_advisory: str
    top_risk_drivers: list[str]
    regulatory_notice: str = (
        "Educational demo only. NOT A MEDICAL DEVICE. Clinical judgment must prevail."
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    logistic_regression_threshold: float
    xgboost_threshold: float
    service_version: str = "1.0.0"


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return microservice health status and active operating thresholds."""
    models = get_models()
    return HealthResponse(
        status="healthy",
        logistic_regression_threshold=models["threshold_lr"],
        xgboost_threshold=models["threshold_xgb"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_cardiac_risk(patient: PatientInput) -> PredictionResponse:
    """Assess patient CAD risk and return cardiology referral triage recommendations.

    Executes both Logistic Regression (interpretable primary) and XGBoost (comparative),
    evaluates agreement, and computes localized risk factor attributions.
    """
    models = get_models()

    # Validate thal categorical value
    if patient.thal not in [3.0, 6.0, 7.0]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid thal value: {patient.thal}. Expected one of [3.0, 6.0, 7.0].",
        )

    patient_df = build_patient_dataframe(
        age=patient.age,
        sex=patient.sex,
        cp=patient.cp,
        trestbps=patient.trestbps,
        chol=patient.chol,
        fbs=patient.fbs,
        restecg=patient.restecg,
        thalach=patient.thalach,
        exang=patient.exang,
        oldpeak=patient.oldpeak,
        slope=patient.slope,
        ca=patient.ca,
        thal=patient.thal,
    )

    lr_model = models["lr_model"]
    xgb_model = models["xgb_model"]
    thr_lr = models["threshold_lr"]
    thr_xgb = models["threshold_xgb"]

    prob_lr = compute_risk_probability(lr_model, patient_df)
    prob_xgb = compute_risk_probability(xgb_model, patient_df)
    dec_lr = get_referral_decision(prob_lr, thr_lr)
    dec_xgb = get_referral_decision(prob_xgb, thr_xgb)
    comparison = compare_model_predictions(prob_lr, thr_lr, prob_xgb, thr_xgb)

    # Local feature attributions from linear model
    attributions = compute_local_attributions(lr_model, patient_df)
    top_drivers = [
        f"{idx} ({val:+.2f} log-odds)"
        for idx, val in attributions.head(5).items()
    ]

    return PredictionResponse(
        logistic_regression=ModelPredictionDetail(
            risk_probability=round(prob_lr, 4),
            referral_recommended=dec_lr,
            operating_threshold=thr_lr,
        ),
        xgboost=ModelPredictionDetail(
            risk_probability=round(prob_xgb, 4),
            referral_recommended=dec_xgb,
            operating_threshold=thr_xgb,
        ),
        models_agree=comparison["models_agree"],
        clinical_advisory=comparison["advisory"],
        top_risk_drivers=top_drivers,
    )
