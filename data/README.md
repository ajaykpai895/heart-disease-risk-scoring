# Data Architecture & Lineage: Cleveland Heart Disease Cohort

## 1. Provenance and Citation

- **Original Study:** Detrano, R., et al. (1989). "International application of a new probability algorithm for the diagnosis of coronary artery disease." *The American Journal of Cardiology*, 64(5), 304–310.
- **Repository:** UCI Machine Learning Repository (Dataset ID: 45).
- **Collection Site:** Cleveland Clinic Foundation (CCF), Cleveland, Ohio, USA.
- **Acquisition Tool:** `ucimlrepo` Python library (`ucimlrepo.fetch_ucirepo(id=45)`).
- **Local Raw Cache:** `data/raw/heart_disease_cleveland.csv` (SHA-256: `201a892d28eb6b1f7ed22dd80d6113cffcc174ccbcdfdc8ee321a7a54c8bbf6b`).

---

## 2. Cohort Specification

- **Total Sample Size:** $n = 303$ patients.
- **Features:** 13 clinical, electrocardiographic, and scintigraphic predictors (6 numeric continuous, 7 categorical).
- **Target (`num`):** Original angiographic luminal narrowing:
  - `0`: $<50\%$ diameter stenosis (no hemodynamically obstructive disease)
  - `1, 2, 3, 4`: $\ge 50\%$ diameter stenosis in 1, 2, 3, or 4 major coronary vessels.
- **Binary Conversion:** `target = (num > 0).astype(int)`.
- **Prevalence:** $45.9\%$ obstructive CAD ($n=139$ positive, $n=164$ negative).

---

## 3. Known Data Quality & Demographic Skews

1. **Catheterization Referral Selection Bias:** Every patient in this cohort was referred for invasive coronary angiography. Consequently, "asymptomatic" patients (`cp = 4.0`) had severe prior non-invasive findings (e.g. markedly abnormal stress test or ECG), resulting in a high disease prevalence (~$70\%$). This causes an apparent clinical paradox where asymptomatic patients show higher odds of CAD than typical angina patients.
2. **Severe Sex Imbalance:** 68% male ($n=206$), 32% female ($n=97$). In the holdout test set ($n=61$), only 7 females have true positive CAD.
3. **Missing Data Patterns:** 6 total records contain missing values:
   - `ca` (colored vessels): 4 missing entries.
   - `thal` (thallium defect): 2 missing entries.
   Handled via leak-free median (numeric) and mode (categorical) imputation fit strictly inside training folds.
4. **Historical Treatment Era:** Collected in the mid-to-late 1980s prior to modern high-potency statins, dual antiplatelet therapy, and modern drug-eluting coronary stents.

---

## 4. Data Versioning & DVC (Data Version Control) Guide

To version this dataset using DVC for enterprise-grade MLOps pipelines:

```bash
# 1. Initialize DVC in the repository
dvc init

# 2. Track raw dataset file
dvc add data/raw/heart_disease_cleveland.csv

# 3. Commit the generated pointer file to Git
git add data/raw/heart_disease_cleveland.csv.dvc data/.gitignore
git commit -m "chore(data): track raw Cleveland dataset with DVC"

# 4. Configure remote storage (e.g., AWS S3, Google Cloud Storage, Azure Blob, or local network share)
dvc remote add -d myremote s3://my-clinical-bucket/heart-disease/data
dvc push

# 5. On a clean clone / CI runner:
dvc pull
```
