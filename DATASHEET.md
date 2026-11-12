# Datasheet for the UCI Cleveland Heart Disease Dataset

> Gebru, T., et al. (2018). "Datasheets for Datasets." *Communications of the ACM*.
> This datasheet follows that template. Statements are based on the published
> literature and UCI repository metadata. Gaps in documentation are explicitly noted.

---

## Motivation

**For what purpose was the dataset created?**
The Cleveland Heart Disease dataset was collected to study the detection of coronary
artery disease (CAD) using non-invasive and minimally invasive clinical features.
The original research goal was to reduce the need for coronary angiography by
identifying high-risk patients from routine measurements. It was later donated to the
UCI Machine Learning Repository to serve as a benchmark dataset for supervised
classification methods.

**Who created the dataset, and on whose behalf?**
Robert Detrano, M.D., Ph.D., and colleagues at the Cleveland Clinic Foundation (CCF),
in collaboration with researchers from the Hungarian Institute of Cardiology (Budapest),
University Hospital (Zürich, Switzerland), and University Hospital (Basel, Switzerland).
The Cleveland subset is the most complete; the other three sites contain substantial
missing data and are rarely used. The dataset was funded through academic research
grants; no commercial funding is disclosed.

**Who funded the creation of the dataset?**
Not disclosed in the UCI repository metadata. The original paper (Detrano et al., 1989,
*The American Journal of Cardiology*) does not list explicit funding sources in
the publicly available description.

---

## Composition

**What do the instances represent?**
Each row represents one patient who underwent cardiac evaluation at the Cleveland Clinic
Foundation. The evaluation included clinical interview, resting ECG, exercise stress test,
thallium scintigraphy, and fluoroscopy.

**How many instances are there?**
The Cleveland subset contains **303 instances**. After removing rows with missing values
in the original target variable encoding, 297 usable rows remain in the processed version.
Our pipeline uses all 303 rows and imputes the 6 rows with missing values in features
using median/mode imputation.

**What data does each instance consist of?**

| Feature | Type | Description | Missing (n) |
|---------|------|-------------|----------:|
| age | Numeric | Age in years | 0 |
| sex | Binary | 1 = male, 0 = female | 0 |
| cp | Categorical (1–4) | Chest pain type | 0 |
| trestbps | Numeric | Resting blood pressure (mm Hg) | 0 |
| chol | Numeric | Serum cholesterol (mg/dl) | 0 |
| fbs | Binary | Fasting blood sugar > 120 mg/dl | 0 |
| restecg | Categorical (0–2) | Resting ECG result | 0 |
| thalach | Numeric | Maximum heart rate achieved | 0 |
| exang | Binary | Exercise-induced angina | 0 |
| oldpeak | Numeric | ST depression (exercise vs. rest, mm) | 0 |
| slope | Categorical (1–3) | Slope of peak exercise ST segment | 0 |
| ca | Numeric (0–3) | Number of major vessels coloured by fluoroscopy | **4** |
| thal | Categorical | Thallium scintigraphy result | **2** |
| **target** | Binary | 0 = no disease, 1 = presence of CAD | 0 |

**Missing data:** 6 rows have at least one missing value (`ca` or `thal`). The original UCI
encoding uses `?` for missingness. Our pipeline imputes: median for numeric features,
mode for categoricals. This is a reasonable default but **systematic missingness in `thal`
and `ca` may bias imputation** — patients who did not receive fluoroscopy or thallium
testing may be systematically different from those who did.

**Label definition:**
The original target variable `num` takes values 0–4 (0 = healthy; 1–4 = increasing
angiographic stenosis severity). We binarize to `target = (num > 0).astype(int)`,
which treats any detectable coronary artery disease as "positive." This is the standard
convention but introduces ambiguity: patients with mild stenosis (num = 1) are labelled
the same as patients with severe triple-vessel disease (num = 4). Whether this is
appropriate depends on the clinical use case.

**Is there a label for every instance?** Yes, all 303 rows have a valid binary target.

**Is the dataset self-contained?** Yes. All features were measured as part of the same
clinical evaluation episode; there are no external linkages.

**Does the dataset contain sensitive personal information?**
The public version contains **age and sex** but no names, dates of birth, or identifiers.
The original hospital records presumably contained identifiers that were removed before
donation to UCI.

---

## Collection Process

**How was the data collected?**
Data was collected prospectively at the Cleveland Clinic Foundation from patients referred
for cardiac evaluation. Patients underwent:
1. Clinical interview (demographics, symptoms, risk factors)
2. Resting 12-lead ECG
3. Exercise stress test (Bruce protocol)
4. Thallium-201 myocardial scintigraphy
5. Coronary fluoroscopy (vessel counting)

Diagnosis was confirmed by coronary angiography (the gold standard reference), which
defines the ground-truth labels.

**When was the data collected?**
The original Detrano et al. paper was published in 1989. Data collection period is
not explicitly stated in the UCI metadata but is estimated to be in the **mid-to-late
1980s**. This is approximately **40 years ago**. Contemporary CAD prevalence, treatment
(statins, PCI), diagnostic criteria, and referral pathways differ substantially.

**Who was included in the data?**
Patients referred to the Cleveland Clinic cardiology service for evaluation of suspected
coronary artery disease. This is a **referred population**, not a population-based sample.
Patients with mild or no symptoms who never sought care are absent (referral bias).

**Were patients asked for consent?**
Not documented in the UCI metadata. The 1989 publication precedes modern IRB and data
sharing consent standards. No participant consent framework is documented.

---

## Preprocessing, Cleaning, and Labeling

**What preprocessing was applied before the data was made available?**
Fourteen features were extracted from the clinical record. The original `num` variable
(0–4) was simplified to binary by most ML practitioners, though the UCI repository stores
the original ordinal encoding.

**Who performed the preprocessing?**
The Cleveland Clinic research team (Detrano et al.), then subsequent UCI curatorial staff.

**Were there quality control checks?**
Not documented. Known data quality issues:
- Some `chol` values are physiologically implausible (e.g., values of 564 mg/dl exist
  in the dataset). These are retained in this project without winsorising.
- The `thal` variable uses numeric codes (3, 6, 7) with no intermediate values —
  the encoding is sparse and inconsistently documented across sources.
- Some records appear duplicated across the four international sites (Cleveland, Hungary,
  Zürich, Basel); only the Cleveland subset is used here to avoid implicit data leakage.

---

## Uses

**Has the dataset been used for other tasks?**
Extensively. The Cleveland Heart Disease dataset has been cited in thousands of ML papers
as a benchmark. It is commonly used for demonstrating logistic regression, decision trees,
neural networks, ensemble methods, and feature selection. It is **not suitable** for
generalisation claims about cardiovascular medicine.

**Is there anything about the composition that might affect future uses?**

> [!WARNING]
> The following limitations are frequently ignored in papers that use this dataset:

1. **Selection bias:** Referred patients, not a population cohort. Prevalence (~54%) is
   far higher than in general population screening (~5–10% for a primary care cohort of
   middle-aged adults). PPV and NPV estimates from this dataset are not transferable to
   general practice.
2. **Demographic skew:** 68% male; nearly all from a single US centre in the 1980s.
   Performance on women, non-white populations, and contemporary patients is unknown.
3. **Sample size:** n = 303 is insufficient to train a production clinical model.
   External validation requires at minimum ~1,000 events (positive cases).
4. **Label ambiguity:** Binary binarisation collapses severity. A model that correctly
   classifies mild stenosis (num=1) patients is not equivalent to one that reliably
   identifies high-risk surgical patients (num=3-4).
5. **Historical drift:** Lipid-lowering therapy, PCI, and guideline-based management
   have changed cardiovascular risk dramatically since the 1980s. A model trained here
   may systematically under-estimate treated risk.

**Are there tasks for which the dataset should not be used?**
- Direct clinical deployment without external validation.
- As sole evidence for any clinical guideline.
- As a representative benchmark for model performance on a contemporary population.
- Making generalised claims about sex differences in heart disease presentation
  without acknowledging the small female sample and referral bias.

---

## Distribution

**How is the dataset distributed?**
Freely available via the UCI Machine Learning Repository (ID: 45) under a
Creative Commons Attribution 4.0 International licence. Accessible programmatically
via the `ucimlrepo` Python package.

**Is the dataset subject to any copyright or IP restrictions?**
Licensed under CC BY 4.0. Attribution required: Detrano et al. (1989), UCI Machine
Learning Repository (Dua & Graff, 2019).

---

## Maintenance

**Who is responsible for maintaining the dataset?**
The UCI Machine Learning Repository (University of California, Irvine).

**Will the dataset be updated?**
No. The dataset is considered a fixed historical benchmark. No new patient data has
been added since the original donation.

**Known demographic skew summary:**

| Attribute | Statistic |
|-----------|-----------|
| Sex distribution | 68% male, 32% female |
| Age range | 29–77 years (median 55) |
| Race/Ethnicity | Not recorded |
| Geographic origin | Cleveland Clinic Foundation, USA (single centre) |
| Era | Mid-to-late 1980s |
| CAD prevalence in sample | ~54% (selected/referred cohort, not population-level) |
| Missing values | 6 rows (ca: 4 missing, thal: 2 missing) |
