# Model Card — Heart Disease Risk Scoring

> Mitchell, M., et al. (2019). "Model Cards for Model Reporting." *FAccT*.
> This card follows that template. Numbers are drawn from the actual trained
> model; where the dataset cannot support a conclusion, that is stated explicitly.

---

## Model Details

| Field | Value |
|---|---|
| **Model name** | Heart Disease Risk Screening — Logistic Regression (primary) / XGBoost (comparative) |
| **Version** | 1.0 |
| **Type** | Binary supervised classification (Probabilistic risk scoring) |
| **Algorithm** | L2-regularised Logistic Regression (interpretable, primary); XGBoost gradient boosting (non-linear, comparative) |
| **Input** | 13 clinical features: age, sex, chest pain type, resting BP, serum cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression, ST slope, fluoroscopy vessel count, thallium scintigraphy result |
| **Output** | Positive-class probability ∈ [0, 1]; hard binary prediction at operating threshold T* |
| **Operating threshold** | Tuned on out-of-fold CV predictions to achieve ≥ 90% recall (sensitivity) on training data |
| **Framework** | scikit-learn 1.8, XGBoost, Python 3.13 |
| **Date** | November 2026 |
| **Authors** | Portfolio project — educational use only |
| **License** | MIT |
| **Contact** | See repository README |

---

## Intended Use

### Primary Intended Use
**Educational demonstration** of an end-to-end clinical risk-scoring pipeline.
The model is designed to assist a data scientist in understanding how to:
- Build leak-free preprocessing pipelines for clinical tabular data.
- Compare an interpretable model (Logistic Regression) with a black-box model (XGBoost).
- Apply research-grade evaluation (nested CV, bootstrap CIs, calibration, DCA).
- Conduct fairness subgroup analysis.

### Primary Intended Users
Data science students, ML engineers, and hiring managers reviewing a portfolio project.
Clinicians or researchers studying model transparency methodology.

### Out-of-Scope Use

> [!CAUTION]
> **This model MUST NOT be used for any clinical decision-making, patient triage,
> or diagnosis — real or simulated.** The following uses are explicitly out of scope:

- **Actual patient screening or cardiology referral decisions.** The model was
  trained on 303 patients from a single US academic medical center in the 1980s.
  It has not been externally validated and cannot be generalised to any real patient population.
- **Automated clinical triage systems**, even as a decision-support component.
- **Any deployment** without prospective external validation on a contemporary,
  representative patient cohort and regulatory clearance (FDA Class II/III in the US).
- **Use as ground truth** for training other clinical models without independent validation.
- Inferring individual patient risk outside the age range 29–77 years, or outside the
  demographic distribution of the Cleveland dataset (predominantly white, US, 1980s).

---

## Training Data

See the **[Datasheet](DATASHEET.md)** for full data provenance.

**Summary:** UCI Cleveland Heart Disease dataset, n = 303 patients, 13 features, binary
target (presence/absence of coronary artery disease). Stratified 80/20 split:
training n = 242, holdout test n = 61.

**Known demographic skew:**
- **Sex:** 206 male (68%), 97 female (32%) — female patients are significantly under-represented.
- **Age:** Median 55 years; range 29–77. Under-representation of patients < 40.
- **Race/Ethnicity:** Not recorded. Dataset origin (Cleveland Clinic, 1980s) suggests
  predominantly non-Hispanic white. Any performance for other groups is unknown.
- **Geography:** Single centre (Cleveland Clinic Foundation, USA). Referral bias present.

---

## Evaluation Data

Stratified holdout: n = 61, 45.9% positive. Same centre, same era as training data.
**No external validation dataset is available.**

---

## Quantitative Performance

> All numbers are from the hold-out test set (n = 61). Bootstrap 95% CIs use
> B = 1,000 resamples (percentile method). Do not over-interpret point estimates.

### Overall Performance

| Metric | Logistic Regression (T* = 0.33) | XGBoost (T* = 0.27) | Clinical Heuristic Baseline |
|--------|--------------------------------:|--------------------:|----------------------------:|
| **Nested CV ROC-AUC** | 0.906 ± 0.017 | 0.894 ± 0.018 | — |
| **Holdout ROC-AUC [95% CI]** | 0.962 [0.906 – 0.996] | 0.962 [0.916 – 0.995] | 0.902 |
| **Holdout Recall [95% CI]** | 100.0% [100.0% – 100.0%] | 100.0% [100.0% – 100.0%] | 92.9% |
| **Holdout Precision [95% CI]** | 73.7% [60.6% – 87.0%] | 73.7% [60.6% – 87.0%] | 78.8% |
| **Brier Score** | 0.086 | 0.083 | 0.130 |
| **Clinical Cost (10x FN)** | 10 (Cost-minimized) | 7 (Cost-minimized) | 27 |

### Subgroup Performance (Test n = 61, 28 Positive)

| Subgroup | n (test) | n positive | Recall (LR) [95% CI] | FNR (LR) [95% CI] | Precision (LR) | ROC-AUC | Adequate for inference? |
|----------|-------:|----------:|---------------------:|------------------:|---------------:|--------:|------------------------|
| **Overall** | 61 | 28 | 100.0% [87.9% – 100%] | 0.0% [0.0% – 12.1%] | 73.7% | 0.962 | Primary benchmark |
| **Sex: Male** | 41 | 21 | 100.0% [84.5% – 100%] | 0.0% [0.0% – 15.5%] | 72.4% | 0.945 | Marginally adequate (n_pos=21) |
| **Sex: Female** | 20 | 7 | 100.0% [64.6% – 100%] | 0.0% [0.0% – 35.4%] | 77.8% | 1.000 | ❌ **Unreliable** (n_pos=7 < 15) |
| **Age: < 45** | 16 | 4 | 100.0% [51.0% – 100%] | 0.0% [0.0% – 49.0%] | 80.0% | 1.000 | ❌ **Unreliable** (n_pos=4 < 15, n=16) |
| **Age: 45–60** | 30 | 15 | 100.0% [79.6% – 100%] | 0.0% [0.0% – 20.4%] | 75.0% | 0.973 | Marginally adequate (n_pos=15) |
| **Age: > 60** | 15 | 9 | 100.0% [70.1% – 100%] | 0.0% [0.0% – 29.9%] | 69.2% | 0.870 | ❌ **Unreliable** (n_pos=9 < 15, n=15) |

> **⚠ Warning on Subgroup Cell Sizes:** Only the Male subgroup (n=21 positive) and the 45–60 age band
> (n=15 positive) meet our minimal threshold (n_pos ≥ 15) for evaluating sensitivity. In females,
> the 95% Wilson confidence interval for False Negative Rate spans [0.0%, 35.4%] — the true false-negative
> rate could be as high as 35% without triggering a single miss on our 7 female test cases!

### Fairness Metrics (fairlearn 0.15)

| Sensitive Feature | Equal Opportunity Diff (LR) | Demographic Parity Diff (LR) | Selection Rates by Group | Honest Clinical Interpretation |
|---|---|---|---|---|
| **Sex** | 0.246 | 0.257 | Male: 70.7% (29/41)<br>Female: 45.0% (9/20) | Driven by selection rate & base prevalence disparity (51.2% male vs 35.0% female). Recall is 100% in both, but female FPR is lower (15.4% vs 40.0%). Insufficient sample to confirm algorithmic bias. |
| **Age Band** | 0.583 | 0.554 | <45: 31.3% (5/16)<br>45–60: 66.7% (20/30)<br>>60: 86.7% (13/15) | Reflects real clinical risk progression with age (prevalence: 25% in <45 vs 60% in >60). Enforcing equal selection rates across age groups would violate clinical validity. |

> **Honest statement:** With a 61-patient holdout, a single misclassified patient
> changes Equal Opportunity Difference by ~6–9 percentage points. No rigorous
> fairness certification is possible from this dataset.

---

## Ethical Considerations

### Who Benefits and Who Is at Risk?
A patient flagged positive receives a cardiology referral (benefit: earlier treatment).
A patient flagged negative does not (risk: missed disease, delayed care).
The model is tuned to minimise False Negatives (missed disease), accepting more
False Positives (extra referrals). This asymmetric design is clinically appropriate
but increases the burden on cardiology services and may cause patient anxiety.

### Error Asymmetry
False Negative (missed disease) is clinically valued at 10× a False Positive in our
cost-sensitive analysis. This ratio is assumed, not evidence-based. The actual cost
ratio in a specific clinical context could be very different.

### Automation Bias Risk
Presenting a model probability score to a clinician can anchor their judgment even
when the model is wrong. The Logistic Regression's transparency (interpretable
coefficients, SHAP values) reduces — but does not eliminate — this risk. Any
deployment must include mandatory clinical override capability.

### Right to Explanation
The Logistic Regression model produces odds ratios and per-patient SHAP values that
can be shown to patients in plain language. XGBoost produces SHAP values but no
human-interpretable global formula. In a clinical context, the right to explanation
(GDPR Article 22; proposed EU AI Act) favours the transparent model.

### Disparate Impact Concern
The dataset contains 68% male patients. If the model performs worse for female
patients — which cannot be confirmed or excluded given the small female holdout cell
(n ≈ 23) — women would receive a less reliable referral recommendation than men.
An external validation study stratified by sex is essential before any deployment.

### Dataset Age and Population Shift
Data was collected in the 1980s. Contemporary prevalence, treatment patterns, and
diagnostic criteria for CAD have changed substantially. Model calibration almost
certainly degrades when applied to a 2020s patient population.

---

## Caveats and Recommendations

1. **Do not deploy** without prospective external validation on a contemporary cohort.
2. **Commission a prospective fairness audit** with ≥ 500 patients per subgroup (sex × age × ethnicity).
3. **Recalibrate** after any deployment population shift using isotonic or Platt recalibration.
4. **Couple to clinical governance**: mandatory human review, documented override rate, outcome tracking.
5. **Re-train periodically** as clinical guidelines and population risk profiles evolve.
6. **Interpretability is not safety**: SHAP values explain the model; they do not validate its clinical correctness.
