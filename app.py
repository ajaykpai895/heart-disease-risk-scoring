"""
Cardiovascular Risk Screening & Clinical Decision Support System.

A clinical triage application demonstrating interpretable versus black-box
risk prediction for obstructive Coronary Artery Disease (CAD).

Features:
- Clean 4-tab clinical interface: Risk Assessment, Model Comparison, Explainability, About & Limitations.
- Calibrated probability gauge with operating cutoff markers (targeting >= 90% Recall).
- Side-by-side Logistic Regression vs. XGBoost predictions with consensus advisory.
- Interactive "What-If" counterfactual trajectory simulation.
- Plain-English localized attribution explanations for patient communication.
- Full regulatory notices, Model Card links, and demographic caveats.

Launch:
    streamlit run app.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.app_helpers import (
    build_patient_dataframe,
    compare_model_predictions,
    compute_local_attributions,
    compute_risk_probability,
    generate_plain_english_explanation,
    get_referral_decision,
    load_all_artifacts,
    simulate_what_if,
)
from src.config import (
    DCA_PLOT_PATH,
    FIGURES_DIR,
    LR_ODDS_RATIOS_PATH,
    RESEARCH_RESULTS_PATH,
)
from src.data import validate_clinical_dataframe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Streamlit App Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Cardiovascular Risk Screening | Clinical DSS",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_cached_artifacts():
    """Load serialized models and calibrated thresholds with caching."""
    return load_all_artifacts()


# ---------------------------------------------------------------------------
# Sidebar & Header
# ---------------------------------------------------------------------------
st.sidebar.title("🫀 CardioScreen AI")
st.sidebar.caption("Clinical Decision Support • Cardiology Triage v1.0")

st.sidebar.markdown("---")
st.sidebar.info(
    "**Operating Target:** Sensitivity (Recall) $\\ge 90\\%$\n\n"
    "Thresholds are safety-calibrated out-of-fold to prevent false negatives "
    "(missed disease cases) during primary care cardiology triage."
)

st.sidebar.markdown("### 📋 Quick Links")
st.sidebar.markdown("- [Model Card (FAccT 2019)](MODEL_CARD.md)")
st.sidebar.markdown("- [Dataset Datasheet (CACM 2018)](DATASHEET.md)")
st.sidebar.markdown("- [FastAPI Microservice Docs](/docs)")

# Top banner & disclaimer
st.title("🫀 Heart Disease Risk Screening & Clinical Decision Support")
st.caption(
    "Translational Clinical Machine Learning: Interpretable Logistic Regression vs. XGBoost on the Cleveland CAD Cohort"
)

st.warning(
    "⚠️ **REGULATORY NOTICE:** Educational and research demonstration only. "
    "This software is **NOT A MEDICAL DEVICE** and must not be used for diagnosis or patient management. "
    "Cardiology referral decisions must always be made by a licensed healthcare provider based on full clinical evaluation."
)

# Load artifacts
try:
    lr_model, xgb_model, threshold_lr, threshold_xgb = get_cached_artifacts()
except Exception as err:
    st.error(f"❌ Failed to load model artifacts: {err}. Please run `python -m src.train` first.")
    st.stop()

# ---------------------------------------------------------------------------
# Layout: 4 Main Tabs
# ---------------------------------------------------------------------------
tab_assess, tab_compare, tab_explain, tab_about = st.tabs(
    [
        "🩺 Patient Risk Assessment",
        "📊 Model Comparison & Benchmarks",
        "🔍 Plain-English Explainability",
        "ℹ️ About & Clinical Limitations",
    ]
)

# ===========================================================================
# TAB 1: Risk Assessment
# ===========================================================================
with tab_assess:
    st.markdown("### 1. Patient Clinical Presentation")
    st.markdown(
        "Enter physiological, diagnostic, and symptomatic variables from outpatient intake or stress testing:"
    )

    col_input1, col_input2 = st.columns(2)

    with col_input1:
        st.markdown("##### 👤 Demographics & Hemodynamics")
        age = st.slider("Age (years)", min_value=25, max_value=85, value=55, help="Age in years.")
        sex = st.selectbox(
            "Biological Sex",
            options=[1.0, 0.0],
            format_func=lambda x: "Male" if x == 1.0 else "Female",
            help="Biological sex recorded at hospital admission.",
        )
        cp = st.selectbox(
            "Chest Pain Type",
            options=[1.0, 2.0, 3.0, 4.0],
            format_func=lambda x: {
                1.0: "1: Typical Angina",
                2.0: "2: Atypical Angina",
                3.0: "3: Non-Anginal Chest Pain",
                4.0: "4: Asymptomatic Referral",
            }[x],
            index=3,
            help="Clinical categorization of chest discomfort presentation.",
        )
        trestbps = st.slider(
            "Resting Blood Pressure (mm Hg)",
            min_value=90,
            max_value=200,
            value=130,
            help="Resting systolic blood pressure upon admission.",
        )
        chol = st.slider(
            "Serum Cholesterol (mg/dl)",
            min_value=120,
            max_value=560,
            value=240,
            help="Total fasting serum cholesterol.",
        )
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0.0, 1.0],
            format_func=lambda x: "Yes (> 120 mg/dl) — Diabetes / Impaired Fasting" if x == 1.0 else "No (≤ 120 mg/dl)",
            help="Surrogate indicator for impaired glucose tolerance.",
        )

    with col_input2:
        st.markdown("##### ⚡ Stress Testing & Fluoroscopy")
        restecg = st.selectbox(
            "Resting ECG Findings",
            options=[0.0, 1.0, 2.0],
            format_func=lambda x: {
                0.0: "0: Normal",
                1.0: "1: ST-T Wave Abnormality (T-wave inversions)",
                2.0: "2: Left Ventricular Hypertrophy (Estes criteria)",
            }[x],
            help="Baseline 12-lead electrocardiogram.",
        )
        thalach = st.slider(
            "Peak Exercise Heart Rate (bpm)",
            min_value=70,
            max_value=210,
            value=150,
            help="Maximum chronotropic response reached during Bruce treadmill stress test.",
        )
        exang = st.selectbox(
            "Exercise-Induced Angina",
            options=[0.0, 1.0],
            format_func=lambda x: "Yes — Anginal symptoms under exertion" if x == 1.0 else "No",
            help="Patient experienced typical ischemic chest pain during exercise.",
        )
        oldpeak = st.slider(
            "ST Depression (oldpeak, mm)",
            min_value=0.0,
            max_value=6.0,
            value=1.0,
            step=0.1,
            help="Horizontal or downsloping ST-segment depression induced by exercise.",
        )
        slope = st.selectbox(
            "Peak Exercise ST Slope",
            options=[1.0, 2.0, 3.0],
            format_func=lambda x: {
                1.0: "1: Upsloping (Low ischemia risk)",
                2.0: "2: Flat (Equivocal / Ischemia)",
                3.0: "3: Downsloping (High ischemia risk)",
            }[x],
            index=1,
            help="Trajectory of the ST segment during peak exercise.",
        )
        ca = st.selectbox(
            "Fluoroscopy Major Vessels Colored (0–3)",
            options=[0.0, 1.0, 2.0, 3.0],
            index=0,
            help="Number of major coronary arteries showing calcified plaque under fluoroscopy.",
        )
        thal = st.selectbox(
            "Thallium Myocardial Perfusion Scintigraphy",
            options=[3.0, 6.0, 7.0],
            format_func=lambda x: {
                3.0: "3: Normal Perfusion",
                6.0: "6: Fixed Defect (Previous Infarct / Non-viable)",
                7.0: "7: Reversible Defect (Active Exertional Ischemia)",
            }[x],
            index=0,
            help="Thallium-201 nuclear stress scintigraphy result.",
        )

    # Build validated patient DataFrame
    patient_df = build_patient_dataframe(
        age=age,
        sex=sex,
        cp=cp,
        trestbps=trestbps,
        chol=chol,
        fbs=fbs,
        restecg=restecg,
        thalach=thalach,
        exang=exang,
        oldpeak=oldpeak,
        slope=slope,
        ca=ca,
        thal=thal,
    )

    try:
        validate_clinical_dataframe(patient_df)
    except ValueError as val_err:
        st.error(f"Input Validation Error: {val_err}")
        st.stop()

    # Model inference
    prob_lr = compute_risk_probability(lr_model, patient_df)
    prob_xgb = compute_risk_probability(xgb_model, patient_df)
    dec_lr = get_referral_decision(prob_lr, threshold_lr)
    dec_xgb = get_referral_decision(prob_xgb, threshold_xgb)
    comparison = compare_model_predictions(prob_lr, threshold_lr, prob_xgb, threshold_xgb)

    st.markdown("---")
    st.markdown("### 2. Clinical Risk Score & Triage Recommendation")

    # Side-by-side risk score display
    col_res1, col_res2 = st.columns(2)

    with col_res1:
        st.markdown("#### 🔬 Logistic Regression (Primary, Transparent)")
        st.metric(
            label="Predicted CAD Probability",
            value=f"{prob_lr:.1%}",
            delta=f"Threshold: {threshold_lr:.2f}",
            delta_color="inverse" if dec_lr else "normal",
        )
        # Visual colored risk bar
        st.progress(prob_lr)
        if dec_lr:
            st.error(
                f"🚨 **CARDIOLOGY REFERRAL RECOMMENDED**\n\n"
                f"Risk ({prob_lr:.1%}) meets or exceeds the safety-calibrated threshold ($T^* = {threshold_lr:.2f}$). "
                "Recommend diagnostic stress echocardiography or cardiology consult."
            )
        else:
            st.success(
                f"✅ **ROUTINE CLINICAL FOLLOW-UP**\n\n"
                f"Risk ({prob_lr:.1%}) is below the referral cutoff ($T^* = {threshold_lr:.2f}$). "
                "Continue standard cardiovascular prevention and lifestyle modification."
            )

    with col_res2:
        st.markdown("#### 🌲 XGBoost (Comparative Black-Box)")
        st.metric(
            label="Predicted CAD Probability",
            value=f"{prob_xgb:.1%}",
            delta=f"Threshold: {threshold_xgb:.2f}",
            delta_color="inverse" if dec_xgb else "normal",
        )
        st.progress(prob_xgb)
        if dec_xgb:
            st.error(
                f"🚨 **CARDIOLOGY REFERRAL RECOMMENDED**\n\n"
                f"Risk ({prob_xgb:.1%}) meets or exceeds the safety-calibrated threshold ($T^* = {threshold_xgb:.2f}$)."
            )
        else:
            st.success(
                f"✅ **ROUTINE CLINICAL FOLLOW-UP**\n\n"
                f"Risk ({prob_xgb:.1%}) is below the referral cutoff ($T^* = {threshold_xgb:.2f}$)."
            )

    # Model Consensus / Disagreement Banner
    if comparison["models_agree"]:
        st.info(f"🤝 **Consensus**: {comparison['advisory']}")
    else:
        st.warning(comparison["advisory"])

    # -----------------------------------------------------------------------
    # Interactive "What-If" Counterfactual Simulation
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 3. Interactive 'What-If' Risk Simulation")
    st.markdown(
        "Simulate how this patient's predicted risk would change if a single physiological "
        "parameter were modified, holding all other features identical:"
    )

    col_sim1, col_sim2 = st.columns([1, 2])

    with col_sim1:
        sim_feature = st.selectbox(
            "Select Variable to Vary:",
            options=["thalach", "oldpeak", "trestbps", "chol", "ca"],
            format_func=lambda x: {
                "thalach": "Peak Heart Rate Capacity (bpm)",
                "oldpeak": "Exercise ST Depression (mm)",
                "trestbps": "Resting Blood Pressure (mm Hg)",
                "chol": "Serum Cholesterol (mg/dl)",
                "ca": "Fluoroscopy Vessels Colored (0–3)",
            }[x],
        )

        sim_ranges = {
            "thalach": np.linspace(80, 200, 25),
            "oldpeak": np.linspace(0.0, 5.0, 25),
            "trestbps": np.linspace(100, 190, 25),
            "chol": np.linspace(140, 400, 25),
            "ca": [0.0, 1.0, 2.0, 3.0],
        }

    with col_sim2:
        sim_data_lr = simulate_what_if(lr_model, patient_df, sim_feature, sim_ranges[sim_feature])
        sim_data_xgb = simulate_what_if(xgb_model, patient_df, sim_feature, sim_ranges[sim_feature])

        chart_df = pd.DataFrame(
            {
                sim_feature: sim_data_lr["Value"],
                "Logistic Regression Risk (%)": sim_data_lr["Predicted Risk (%)"],
                "XGBoost Risk (%)": sim_data_xgb["Predicted Risk (%)"],
                "Referral Threshold (%)": [threshold_lr * 100] * len(sim_data_lr),
            }
        ).set_index(sim_feature)

        st.line_chart(chart_df)
        st.caption(
            f"Trajectory of risk as **{sim_feature}** varies while keeping all other patient parameters fixed. "
            "Dashed line represents the safety referral threshold."
        )


# ===========================================================================
# TAB 2: Model Comparison & Benchmarks
# ===========================================================================
with tab_compare:
    st.markdown("### 📊 Research-Grade Model Evaluation & Clinical Baselines")
    st.markdown(
        "Rigorous head-to-head comparison between transparent linear modeling, non-linear ensemble trees, "
        "and established pre-test cardiology heuristics (Diamond-Forrester criteria)."
    )

    if RESEARCH_RESULTS_PATH.exists():
        research_df = pd.read_csv(RESEARCH_RESULTS_PATH)
        st.dataframe(research_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run `python -m src.train` to generate the complete benchmark summary table.")

    st.markdown("#### 🔬 Statistical Comparison & Decision Diagnostics")
    col_diag1, col_diag2 = st.columns(2)

    with col_diag1:
        roc_path = FIGURES_DIR / "roc_comparison.png"
        if roc_path.exists():
            st.image(str(roc_path), caption="Holdout ROC Curves with Pre-test Clinical Baseline", use_container_width=True)
        else:
            st.info("ROC comparison plot not found.")

    with col_diag2:
        cal_path = FIGURES_DIR / "calibration.png"
        if cal_path.exists():
            st.image(str(cal_path), caption="Probability Calibration & Reliability Curves", use_container_width=True)
        else:
            st.info("Calibration plot not found.")

    col_diag3, col_diag4 = st.columns(2)

    with col_diag3:
        if DCA_PLOT_PATH.exists():
            st.image(str(DCA_PLOT_PATH), caption="Decision Curve Analysis (Net Clinical Benefit)", use_container_width=True)

    with col_diag4:
        learn_path = FIGURES_DIR / "learning_curves.png"
        if learn_path.exists():
            st.image(str(learn_path), caption="Empirical Learning Curves Across Cohort Fractions", use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📐 Statistical Equivalence Test Results")
    st.markdown(
        "- **DeLong Test on Holdout ROC-AUC:** $z = 0.01, p = 0.992$ (Difference between Logistic Regression and XGBoost is statistically indistinguishable from zero).\n"
        "- **Nadeau-Bengio Corrected CV Paired $t$-Test:** $t = 0.74, p = 0.468$ (No significant difference across 25 cross-validation folds).\n"
        "- **Conclusion:** Transparent Logistic Regression delivers equal discrimination with zero accuracy penalty."
    )


# ===========================================================================
# TAB 3: Explainability
# ===========================================================================
with tab_explain:
    st.markdown("### 🔍 Patient-Specific & Global Model Explainability")
    st.markdown(
        "Under clinical governance frameworks (e.g., GDPR Article 22, Proposed EU AI Act), patients and "
        "clinicians have a right to an auditable pathophysiological justification for automated referrals."
    )

    # 1. Plain English Local Explanation
    st.markdown("#### 🗣️ Plain-English Patient Risk Explanation")
    local_attributions = compute_local_attributions(lr_model, patient_df)
    plain_english = generate_plain_english_explanation(local_attributions, top_n=3)

    st.info(plain_english["summary_text"])

    # 2. Local Waterfall Attribution Chart
    st.markdown("#### 📈 Local Feature Attributions (Log-Odds Contribution)")
    st.caption("Standardized input × regression coefficient for the current patient:")

    top_drivers = local_attributions.head(8)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#E63946" if val > 0 else "#457B9D" for val in top_drivers.values]
    ax.barh(top_drivers.index[::-1], top_drivers.values[::-1], color=colors[::-1], edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linestyle="--", alpha=0.6)
    ax.set_xlabel("Contribution to Predicted Log-Odds")
    ax.set_title("Localized Risk Factor Attributions (Current Patient)", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 3. Global Population-Level Odds Ratios
    st.markdown("---")
    st.markdown("#### 🌐 Population-Level Odds Ratios ($e^\\beta$) from Logistic Regression")
    if LR_ODDS_RATIOS_PATH.exists():
        odds_df = pd.read_csv(LR_ODDS_RATIOS_PATH)
        st.dataframe(odds_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run `python -m src.explain` to generate the complete epidemiological odds ratios table.")

    # 4. Global SHAP Summary
    shap_summary_path = FIGURES_DIR / "shap_summary.png"
    if shap_summary_path.exists():
        st.markdown("#### 🌲 XGBoost Global Feature Importance (SHAP Summary)")
        st.image(str(shap_summary_path), caption="SHAP Beeswarm: Distribution of feature impacts on XGBoost output", use_container_width=True)


# ===========================================================================
# TAB 4: About & Limitations
# ===========================================================================
with tab_about:
    st.markdown("### ℹ️ About, Clinical Governance & Responsible-AI Audit")

    st.markdown(
        """
        #### 🎯 Clinical Problem Formulation
        In cardiovascular primary care triage, missing a patient with obstructive coronary artery disease
        (**False Negative**) carries severe mortality risk (unheralded myocardial infarction, fatal arrhythmia),
        whereas an extra referral (**False Positive**) causes only an outpatient consultation and non-invasive imaging.
        
        To reflect this **asymmetric error cost**, this application operates at a calibrated sensitivity cutoff
        ($T^* = 0.332$ for Logistic Regression, $T^* = 0.269$ for XGBoost) targeting $\\ge 90\\%$ recall.
        """
    )

    st.markdown("---")
    st.markdown("#### ⚖️ Responsible-AI & Demographic Subgroup Audit")

    col_audit1, col_audit2 = st.columns(2)
    with col_audit1:
        subgroup_fig = FIGURES_DIR / "fairness_subgroup_metrics.png"
        if subgroup_fig.exists():
            st.image(str(subgroup_fig), caption="Subgroup Recall and False Negative Rate with 95% Wilson CIs", use_container_width=True)

    with col_audit2:
        cal_subgroup_fig = FIGURES_DIR / "fairness_calibration_subgroup.png"
        if cal_subgroup_fig.exists():
            st.image(str(cal_subgroup_fig), caption="Subgroup Calibration Gaps by Demographic Slice", use_container_width=True)

    st.markdown(
        """
        #### ⚠️ Known Data Skew & Limitations (Do Not Overclaim)
        1. **Demographic Imbalance (68% Male):** The historical Cleveland Clinic dataset ($n=303$) severely under-represents
           female patients ($n=97$, with only $7$ positive cases in the test set). The 95% Wilson CI for female False Negative Rate
           spans $[0.0\\%, 35.4\\%]$—the model cannot be certified as safe for female triage on this data alone.
        2. **Historical Referral Cohort (1980s):** All patients in this dataset underwent invasive catheterization.
           Prevalence (~$46\\%$) is dramatically higher than primary care screening populations (~$5-10\\%$).
        3. **Single-Center Retrospective Origin:** Collected at a single US tertiary medical center prior to modern
           statin and guideline-directed medical therapy. **Prospective multi-center external validation is mandatory
           prior to any real clinical deployment.**
        """
    )

    st.markdown("---")
    st.markdown("#### 📜 Formal Documentation")
    st.markdown("- **Model Card:** Review the full [MODEL_CARD.md](MODEL_CARD.md) following Mitchell et al. (2019).")
    st.markdown("- **Datasheet for Dataset:** Review [DATASHEET.md](DATASHEET.md) following Gebru et al. (2018).")
    st.markdown("- **Audit Artifact:** Complete machine-readable audit serialized to `reports/fairness_audit.json`.")
