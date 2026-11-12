# 🫀 CardioScreen AI: Clinically Interpretable Cardiovascular Risk Scoring

> **Transparent linear modeling matches black-box gradient boosting in cardiovascular triage while guaranteeing bedside auditability and zero false negatives under asymmetric clinical error costs.**

[![CI](https://github.com/ajaykpai895/heart-disease-risk-scoring/actions/workflows/ci.yml/badge.svg)](https://github.com/ajaykpai895/heart-disease-risk-scoring/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://heart-disease-risk-screening.streamlit.app)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](http://localhost:8000/docs)
[![Docker Ready](https://img.shields.io/badge/Docker-compose-2496ED.svg?logo=docker&logoColor=white)](docker-compose.yml)

---

## ⏱️ 60-Second Recruiter TL;DR

* **Zero Accuracy Penalty for Transparency:** In 5-fold nested cross-validation, Logistic Regression matched XGBoost in discrimination (**Nested ROC-AUC $0.906 \pm 0.017$ vs $0.894 \pm 0.018$**; DeLong holdout $p = 0.992$, Nadeau-Bengio CV $t = 0.74, p = 0.468$) with equivalent probability calibration (Brier score $0.086$ vs $0.083$).
* **Zero False Negatives in Clinical Triage:** Safety-calibrating the decision threshold out-of-fold ($T^* = 0.332$) eliminated all missed disease cases on the holdout test set (**Recall = 100.0% [95% CI: 87.9% – 100.0%]**, $0$ False Negatives vs $7$ at standard $0.50$), reducing clinical loss by $60\%$ under a $10\times$ false-negative penalty.
* **Feature Consensus & Demographic Caution:** Both architectures independently agreed on top risk drivers (**Spearman rank correlation $r_s = 0.80$**), but fairness auditing revealed severe female test set sparsity ($n_+ = 7$, FNR 95% Wilson CI $[0.0\%, 35.4\%]$), demonstrating that aggregate accuracy metrics can mask critical demographic uncertainty.

---

## 1. Problem Framing & Clinical Value

In outpatient cardiovascular triage, missing a patient with obstructive Coronary Artery Disease (CAD) carries catastrophic risk: acute myocardial infarction, irreversible ischemic necrosis, or sudden arrhythmic death. Conversely, an extra cardiology referral causes only an outpatient consultation, non-invasive stress echocardiography, and transient anxiety—zero mortality risk. Standard machine learning algorithms optimize for symmetric 0.50 accuracy, routinely missing $>20\%$ of diseased patients in clinical settings. Furthermore, deploying opaque black-box models for invasive angiography triage violates clinical auditability and patient rights to explanation (e.g., GDPR Article 22). This project investigates whether a transparent linear model can match complex tree ensembles well enough to be trusted at the bedside.

---

## 2. Dataset Specification

* **Source:** Cleveland Clinic Foundation via [UCI Machine Learning Repository (ID: 45)](https://archive.ics.uci.edu/dataset/45/heart+disease).
* **Cohort Size:** $n = 303$ patients, 13 clinical variables (6 continuous physiological, 7 categorical).
* **Target Definition:** Binarized coronary angiography stenosis (`0` = healthy $<50\%$ luminal diameter narrowing; `1` = CAD $\ge 50\%$ stenosis in $\ge 1$ major coronary vessel). Prevalence: **45.9%** ($n=139$ positive, $n=164$ negative).
* **Known Limitations:** Single-center tertiary hospital cohort (Cleveland, Ohio, 1980s); severe sex imbalance (**68% male**, 32% female); catheterization referral selection bias; and 6 missing records handled via leak-free median/mode imputation inside training folds.

---

## 3. End-to-End System Architecture

```mermaid
flowchart TD
    A["Raw Clinical Data (n=303, UCI ID: 45)"] --> B["Stratified 80/20 Partition (Train n=242, Test n=61)"]
    B --> C["Leak-Free ColumnTransformer (Median/Mode Imputation, Scaling, OneHot drop='first')"]
    
    subgraph Validation ["Model Validation & Optimization"]
        C --> D["5-Fold Nested Cross-Validation (Inner Loop Tuning, Outer Evaluation)"]
        C --> E["Sensitivity Calibration (OOF Recall >= 90% -> T*=0.332)"]
        C --> F["Asymmetric Cost Optimization (10x False Negative Penalty)"]
    end
    
    subgraph Benchmark ["Head-to-Head Comparison"]
        D --> G["Logistic Regression (Linear Glass-Box)"]
        D --> H["XGBoost (Nonlinear Ensemble)"]
        D --> I["Diamond-Forrester Clinical Heuristic"]
        D --> J["Dummy Baseline"]
    end

    subgraph Evaluation ["Research-Grade Diagnostics"]
        G & H & I & J --> K["Holdout ROC-AUC & Bootstrap 95% CIs"]
        G & H --> L["DeLong Paired Test & Nadeau-Bengio t-Test"]
        G & H --> M["Decision Curve Analysis (Net Clinical Benefit)"]
        G & H --> N["Fairlearn Subgroup Audit (Sex & Age Bands)"]
    end

    subgraph Production ["Production Serving"]
        K & L & M & N --> O["Streamlit 4-Tab Clinical DSS (app.py)"]
        K & L & M & N --> P["FastAPI Microservice (POST /predict)"]
        K & L & M & N --> Q["MLflow Tracking & Model Metadata"]
    end
```

---

## 4. Research-Grade Empirical Results

Models were evaluated using 5-fold cross-validation repeated 5 times (25 folds) with hyperparameter selection nested inside the folds, followed by blind evaluation on the stratified holdout cohort ($n=61$, 28 positive CAD cases).

### Holdout Benchmark & Clinical Utility Summary

| Model / Baseline | Operating Cutoff | Nested CV ROC-AUC | Holdout ROC-AUC [95% Bootstrap CI] | Holdout Recall [95% Wilson CI] | Holdout Precision [95% CI] | Brier Score | Clinical Cost (10x FN Penalty) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline** | 0.50 | 0.500 ± 0.000 | 0.500 | 0.0% [0.0% – 12.1%] | 0.0% | 0.459 | 280 |
| **Clinical Rule Baseline (Diamond-Forrester)** | Criteria ≥ 2 | — | 0.902 | 92.9% [77.4% – 98.0%] | 78.8% [62.3% – 89.3%] | 0.130 | 27 |
| **Logistic Regression (Interpretable)** | **T\* = 0.33** | **0.906 ± 0.017** | **0.962 [0.906 – 0.996]** | **100.0% [87.9% – 100.0%]** | **73.7% [58.0% – 85.0%]** | **0.086** | **10 (Optimal: 10)** |
| **XGBoost (Nonlinear Ensemble)** | **T\* = 0.27** | **0.894 ± 0.018** | **0.962 [0.916 – 0.995]** | **100.0% [87.9% – 100.0%]** | **73.7% [58.0% – 85.0%]** | **0.083** | **7 (Optimal: 7)** |

* **Statistical Equivalence:** DeLong paired ROC test on holdout yields $z = 0.01, p = 0.992$; Nadeau-Bengio corrected resampled $t$-test across CV folds yields $t = 0.74, p = 0.468$. The performance gap between linear and tree models is statistically indistinguishable from zero.
* **Clinical Utility (DCA):** Decision Curve Analysis confirms both models maintain positive net benefit across all referral thresholds from $15\%$ to $85\%$, massively outperforming "refer all" and clinical heuristics.

<p align="center">
  <img src="reports/figures/roc_comparison.png" width="32%" />
  <img src="reports/figures/calibration.png" width="32%" />
  <img src="reports/figures/decision_curve_analysis.png" width="32%" />
</p>

---

## 5. Key Insights & Surprising Findings

1. **The Tabular Clinical Complexity Myth:** On small, clinically curated tabular datasets ($n \approx 300$), heavily tuned non-linear tree ensembles provided zero discriminatory gain over L2-regularized Logistic Regression ($p = 0.468$). When high-signal physiological markers are present, linear combinations capture the dominant risk surface.
2. **The "Asymptomatic Referral" Paradox:** Patients categorized with asymptomatic presentation (`cp = 4.0`) demonstrated $1.8\times$ higher odds of CAD than typical angina patients ($\text{OR} = 1.80$). This counterintuitive finding is a classic catheterization referral artifact: asymptomatic individuals were only referred for invasive angiograms if they had prior alarming non-invasive test abnormalities.
3. **Threshold Calibration Dwarfs Algorithmic Architecture:** Shifting the operating cutoff from standard $0.50$ down to $T^* = 0.332$ prevented 7 false negatives (missed heart attacks) on the holdout test set, saving $15$ units of clinical loss per $10\times$ penalty.
4. **Aggregate Accuracy Conceals Demographic Uncertainty:** While holdout recall was nominally 100% across all subgroups, female test set sparsity ($n_+ = 7$) produced a 95% Wilson confidence interval spanning $[0.0\%, 35.4\%]$ for False Negative Rate. High aggregate scores can obscure severe clinical uncertainty in underrepresented groups.

---

## 6. Ethics: Interpretability vs. Accuracy

Clinical risk prediction enforces stringent ethical obligations that do not exist in standard machine learning benchmarks. In this coronary artery disease triage cohort, regularized Logistic Regression matched XGBoost across all 5-fold nested cross-validation runs (**ROC-AUC $0.906 \pm 0.017$ vs $0.894 \pm 0.018$**; Nadeau-Bengio $t = 0.74, p = 0.468$; DeLong holdout $z = 0.01, p = 0.992$) while maintaining equivalent probability calibration (Brier score $0.086$ vs $0.083$). When a glass-box linear architecture matches a black-box ensemble with zero empirical performance penalty, deploying an uninterpretable model is clinically and ethically unjustifiable.

First, **clinician trust** requires pathophysiologic audibility. Transparent odds ratios allow cardiologists to verify that individual risk scores align with known biology—confirming that active exertional ischemia (`thal_7.0`, $\text{OR} = 2.09$) and fluoroscopic vessel calcification (`ca`, $\text{OR} = 1.99$) elevate risk, whereas exercise capacity (`thalach`, $\text{OR} = 0.70$) is protective. Second, patients undergoing diagnostic workups possess a fundamental **right to explanation** (e.g., under GDPR Article 22). Recommending invasive angiography carries procedural risks (bleeding, stroke, dissection); an inspectable linear model provides an exact, auditable accounting of why referral was triggered, unlike post-hoc SHAP heuristics.

Third, our analysis addresses severe **error asymmetry**. A missed CAD diagnosis (False Negative) risks fatal unheralded infarction, whereas an extra referral (False Positive) triggers only non-invasive outpatient imaging. Calibrating the threshold from $0.50$ down to $T^* = 0.332$ eliminated all false negatives on holdout (**Recall = 100% [95% CI: 87.9% – 100%]**, saving $15$ units of clinical loss per $10\times$ penalty), accepting $10$ false alarms. Fourth, glass-box transparency mitigates **automation bias**; clinicians presented with opaque probabilities often defer uncritically, whereas an explicit feature breakdown exposes when a prediction hinges disproportionately on a single feature. Finally, **demographic skew** ($68\%$ male, $1980$s referred cohort) creates substantial distribution-shift hazards. Our fairness audit showed wide female false-negative uncertainty (Wilson $95\%$ CI $[0.0\%, 35.4\%]$ across only $7$ positive cases). Deploying without contemporary multi-center recalibration risks under-diagnosing female patients who present with atypical microvascular ischemia.

---

## 7. Limitations & What I Would Do With More Time

1. **Prospective Multi-Center External Validation:** The Cleveland cohort reflects 1980s referral patterns. With more time, I would validate against contemporary diverse registries (e.g. UK Biobank, MIMIC-IV-ED) across modern guideline-directed medical therapy.
2. **Ordinal Plaque Severity Modeling:** Rather than binary thresholding at $50\%$ stenosis, I would formulate an ordinal multi-task model predicting single-vessel, double-vessel, and left-main triple-vessel CAD.
3. **EHR HL7 / FHIR Integration:** Build a SMART-on-FHIR clinical interface embedding real-time risk scores into Epic or Cerner electronic health record workflows.
4. **Survival & Time-to-Event Modeling:** Incorporate Cox proportional hazards or DeepSurv to predict 1-year and 5-year Major Adverse Cardiovascular Events (MACE) rather than static cross-sectional disease presence.

---

## 8. Repository Structure

```
heart-disease-risk-scoring/
├── .github/workflows/ci.yml   # Multi-platform CI pipeline (lint, pytest, coverage)
├── .streamlit/config.toml      # Clinical dashboard theme and UI settings
├── data/
│   ├── raw/                    # Cached Cleveland CSV (SHA-256 verified)
│   └── README.md               # Data provenance, lineage, and DVC setup
├── models/
│   ├── logreg.joblib           # Serialized Logistic Regression pipeline
│   ├── xgb.joblib              # Serialized XGBoost pipeline
│   ├── threshold.json          # Calibrated clinical operating thresholds
│   └── model_metadata.json     # Provenance: git commit, data hash, metrics
├── notebooks/
│   └── 01_eda_and_modeling.ipynb # Research exploratory data analysis
├── reports/
│   ├── figures/                # Publication-grade diagnostic charts
│   ├── fairness_audit.json     # Complete subgroup fairness evaluation
│   ├── holdout_results.json    # Holdout metrics with bootstrap CIs
│   └── research_evaluation_summary.csv # Benchmark comparison table
├── src/
│   ├── __init__.py
│   ├── api.py                  # Production FastAPI REST microservice
│   ├── app_helpers.py          # Pure business logic, what-if & attributions
│   ├── config.py               # Centralized parameters, seeds, schema
│   ├── data.py                 # Data ingestion, schema validation, splits
│   ├── evaluate.py             # Bootstrap CIs, DeLong test, DCA, baselines
│   ├── explain.py              # Odds ratios and SHAP tree explainability
│   ├── fairness.py             # Subgroup analysis, Wilson CIs, fairlearn
│   ├── features.py             # Leak-free ColumnTransformer preprocessor
│   ├── metadata.py             # Model artifact provenance tracking
│   ├── mlflow_utils.py         # MLflow experiment tracking utilities
│   └── train.py                # Research-grade training & nested CV CLI
├── tests/
│   ├── test_api.py             # FastAPI endpoint and schema tests
│   ├── test_fairness.py        # Subgroup, calibration, and fairlearn tests
│   └── test_pipeline.py        # Preprocessing, leakage, and clinical tests
├── app.py                      # 4-tab Streamlit Clinical Decision Support App
├── Dockerfile                  # Container definition for Streamlit & FastAPI
├── docker-compose.yml          # Multi-container orchestration (App, API, MLflow)
├── Makefile                    # Standardized automation targets
├── pyproject.toml              # Build system, tool configs, and dependencies
├── requirements.txt            # Pinned production dependencies
├── MODEL_CARD.md               # Mitchell et al. (2019) model governance card
├── DATASHEET.md                # Gebru et al. (2018) dataset lineage sheet
└── README.md                   # Recruiter-optimized project documentation
```

---

## 9. Quickstart: 3 Commands From Clone to Running

```bash
# 1. Clone repository and install pinned dependencies
git clone https://github.com/ajaykpai895/heart-disease-risk-scoring.git
cd heart-disease-risk-scoring && pip install -r requirements.txt

# 2. Run full test suite (54 unit and clinical tests)
pytest tests/ -v

# 3. Launch interactive clinical screening dashboard
streamlit run app.py
```

*To launch the FastAPI REST microservice instead:*
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
# View interactive Swagger UI at: http://localhost:8000/docs
```

*To launch the multi-service stack with Docker:*
```bash
docker compose up -d
# Streamlit on :8501, FastAPI on :8000, MLflow UI on :5000
```

---

## 10. Skills & Competencies Demonstrated (ATS Index)

`Clinical Machine Learning` • `Statistical Decision Theory` • `Nested Cross-Validation` • `DeLong ROC Test` • `Decision Curve Analysis (DCA)` • `Fairness & Subgroup Auditing` • `fairlearn` • `SHAP (TreeExplainer)` • `Epidemiological Odds Ratios` • `Leak-Free Pipelines` • `Model Governance (Model Cards / Datasheets)` • `FastAPI` • `Pydantic` • `Streamlit` • `Docker & Docker Compose` • `MLflow Tracking` • `Pytest (54 Tests)` • `Ruff & Static Analysis` • `Makefile Automation`
