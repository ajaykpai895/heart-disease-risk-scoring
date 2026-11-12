"""
Tests for the FastAPI Inference Microservice (src/api.py).

Tests:
- health_check endpoint handler status and threshold reporting.
- predict_cardiac_risk endpoint handler with severe CAD patient (asserts referral triggered).
- predict_cardiac_risk endpoint handler with low-risk patient (asserts routine follow-up).
- Pydantic schema validation for out-of-range inputs (ValidationError).
- Invalid categorical domain rejection (HTTPException 422).
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from src.api import (
    HealthResponse,
    PatientInput,
    PredictionResponse,
    health_check,
    predict_cardiac_risk,
)


class TestAPIEndpoints:
    """Test suite for FastAPI endpoints and Pydantic schemas."""

    def test_health_check_returns_healthy(self) -> None:
        """health_check handler must return healthy status with valid thresholds."""
        res: HealthResponse = health_check()
        assert res.status == "healthy"
        assert 0.0 < res.logistic_regression_threshold < 1.0
        assert 0.0 < res.xgboost_threshold < 1.0

    def test_predict_high_risk_patient(self) -> None:
        """predict_cardiac_risk for severe symptomatic CAD patient must recommend referral."""
        patient = PatientInput(
            age=62.0,
            sex=1.0,
            cp=4.0,  # Asymptomatic catheterization referral
            trestbps=160.0,
            chol=286.0,
            fbs=1.0,
            restecg=2.0,
            thalach=108.0,  # Severe chronotropic incompetence
            exang=1.0,
            oldpeak=3.4,  # Deep ST depression
            slope=2.0,
            ca=2.0,  # 2 calcified vessels
            thal=7.0,  # Reversible defect
        )
        res: PredictionResponse = predict_cardiac_risk(patient)
        assert res.logistic_regression.referral_recommended is True
        assert res.xgboost.referral_recommended is True
        assert res.models_agree is True
        assert len(res.top_risk_drivers) >= 3
        assert "NOT A MEDICAL DEVICE" in res.regulatory_notice

    def test_predict_low_risk_patient(self) -> None:
        """predict_cardiac_risk for healthy patient must recommend routine follow-up."""
        patient = PatientInput(
            age=35.0,
            sex=0.0,  # Young female
            cp=1.0,
            trestbps=110.0,
            chol=170.0,
            fbs=0.0,
            restecg=0.0,
            thalach=185.0,  # High exercise heart rate
            exang=0.0,
            oldpeak=0.0,  # No ST depression
            slope=1.0,  # Upsloping ST
            ca=0.0,  # 0 vessels
            thal=3.0,  # Normal perfusion
        )
        res: PredictionResponse = predict_cardiac_risk(patient)
        assert res.logistic_regression.referral_recommended is False
        assert res.xgboost.referral_recommended is False
        assert res.models_agree is True

    def test_schema_rejects_out_of_bounds_feature(self) -> None:
        """Pydantic must raise ValidationError for out-of-range biological input."""
        with pytest.raises(ValidationError):
            PatientInput(
                age=150.0,  # Exceeds max 100.0
                sex=1.0,
                cp=1.0,
                trestbps=120.0,
                chol=200.0,
                fbs=0.0,
                restecg=0.0,
                thalach=150.0,
                exang=0.0,
                oldpeak=0.0,
                slope=1.0,
                ca=0.0,
                thal=3.0,
            )

    def test_predict_rejects_invalid_thal_category(self) -> None:
        """predict_cardiac_risk with unsupported thal category must raise HTTPException 422."""
        patient = PatientInput(
            age=55.0,
            sex=1.0,
            cp=1.0,
            trestbps=120.0,
            chol=200.0,
            fbs=0.0,
            restecg=0.0,
            thalach=150.0,
            exang=0.0,
            oldpeak=0.0,
            slope=1.0,
            ca=0.0,
            thal=99.0,  # Invalid category
        )
        with pytest.raises(HTTPException) as exc_info:
            predict_cardiac_risk(patient)
        assert exc_info.value.status_code == 422
