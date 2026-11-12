"""
Exploratory Data Analysis & Clinical Statistical Profiling.
Generates publication-quality figures in reports/figures/ and summarizes
clinical interpretations with univariate and bivariate statistics.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw_data, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COL

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set high-quality visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 15
})


def analyze_target_and_demographics(df: pd.DataFrame):
    """Plot target distribution and demographic breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Target distribution
    counts = df[TARGET_COL].value_counts().sort_index()
    labels = ["No Disease (0)", "Disease Present (1)"]
    colors = ["#2b5c8f", "#d95f02"]
    
    axes[0].bar(labels, counts.values, color=colors, width=0.5, edgecolor="black", alpha=0.85)
    axes[0].set_title("Target Distribution (Coronary Artery Disease >= 50% stenosis)", pad=15)
    axes[0].set_ylabel("Patient Count")
    axes[0].set_ylim(0, max(counts.values) + 25)
    for i, v in enumerate(counts.values):
        pct = (v / len(df)) * 100
        axes[0].text(i, v + 3, f"{v} ({pct:.1f}%)", ha="center", fontweight="bold")

    # 2. Disease by Sex
    sex_disease = pd.crosstab(df["sex"], df[TARGET_COL], normalize="index") * 100
    sex_labels = ["Female (0)", "Male (1)"]
    sex_disease.index = sex_labels
    sex_disease.columns = ["No Disease", "Disease Present"]
    
    sex_disease.plot(kind="bar", stacked=True, color=colors, ax=axes[1], edgecolor="black", alpha=0.85)
    axes[1].set_title("Disease Prevalence by Biological Sex", pad=15)
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_xlabel("Sex")
    axes[1].legend(loc="upper right")
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=0)

    plt.tight_layout()
    output_path = FIGURES_DIR / "target_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def analyze_numerical_features(df: pd.DataFrame):
    """Plot distribution and boxplots comparing healthy vs disease for all continuous features."""
    fig, axes = plt.subplots(len(NUMERICAL_FEATURES), 2, figsize=(14, 4 * len(NUMERICAL_FEATURES)))
    
    clinical_names = {
        "age": "Age (years)",
        "trestbps": "Resting Blood Pressure (mm Hg)",
        "chol": "Serum Cholesterol (mg/dl)",
        "thalach": "Max Heart Rate Achieved (bpm)",
        "oldpeak": "ST Depression (mm)"
    }

    p_values = {}
    for idx, feature in enumerate(NUMERICAL_FEATURES):
        # Statistical test: Mann-Whitney U test (non-parametric comparison)
        healthy = df[df[TARGET_COL] == 0][feature].dropna()
        disease = df[df[TARGET_COL] == 1][feature].dropna()
        u_stat, p_val = stats.mannwhitneyu(healthy, disease, alternative="two-sided")
        p_values[feature] = p_val

        # KDE Distribution plot
        sns.kdeplot(data=df, x=feature, hue=TARGET_COL, common_norm=False,
                    fill=True, palette=["#2b5c8f", "#d95f02"], ax=axes[idx, 0], alpha=0.4)
        axes[idx, 0].set_title(f"Density: {clinical_names[feature]}", fontweight="bold")
        axes[idx, 0].set_xlabel(clinical_names[feature])
        axes[idx, 0].legend(title="Target", labels=["Disease (1)", "No Disease (0)"])

        # Boxplot with p-value annotation
        sns.boxplot(data=df, x=TARGET_COL, y=feature, palette=["#2b5c8f", "#d95f02"],
                    ax=axes[idx, 1], width=0.4, boxprops=dict(alpha=0.8))
        axes[idx, 1].set_xticklabels(["No Disease (0)", "Disease Present (1)"])
        axes[idx, 1].set_title(f"Boxplot: {clinical_names[feature]} (Mann-Whitney p={p_val:.2e})", fontweight="bold")
        axes[idx, 1].set_xlabel("Target")
        axes[idx, 1].set_ylabel(clinical_names[feature])

    plt.tight_layout()
    output_path = FIGURES_DIR / "numerical_distributions.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")
    return p_values


def analyze_categorical_features(df: pd.DataFrame):
    """Plot stacked proportions of categorical risk factors conditioned on CAD diagnosis."""
    cat_cols = ["cp", "restecg", "exang", "slope", "ca", "thal"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    titles = {
        "cp": "Chest Pain Type\n(1=Typ, 2=Atyp, 3=Non-ang, 4=Asympt)",
        "restecg": "Resting ECG\n(0=Normal, 1=ST-T Abn, 2=LVH)",
        "exang": "Exercise-Induced Angina\n(0=No, 1=Yes)",
        "slope": "ST Segment Peak Slope\n(1=Upsloping, 2=Flat, 3=Downsloping)",
        "ca": "Major Fluoroscopy Vessels Colored\n(0 to 3 vessels)",
        "thal": "Thallium Stress Test\n(3=Normal, 6=Fixed, 7=Reversible)"
    }

    colors = ["#2b5c8f", "#d95f02"]
    for idx, col in enumerate(cat_cols):
        crosstab = pd.crosstab(df[col], df[TARGET_COL], normalize="index") * 100
        crosstab.columns = ["No Disease", "Disease Present"]
        
        crosstab.plot(kind="bar", stacked=True, color=colors, ax=axes[idx], edgecolor="black", alpha=0.85)
        axes[idx].set_title(titles[col], fontsize=11, fontweight="bold")
        axes[idx].set_ylabel("Percentage (%)")
        axes[idx].set_xlabel("")
        axes[idx].legend(loc="upper left")
        plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=0)

    plt.tight_layout()
    output_path = FIGURES_DIR / "categorical_proportions.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def analyze_correlations(df: pd.DataFrame):
    """Correlation matrix of all features with target."""
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-0.6, vmax=0.6, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Spearman/Pearson Correlation Matrix (Clinical Features & Target)", pad=20, fontsize=14, fontweight="bold")

    plt.tight_layout()
    output_path = FIGURES_DIR / "correlation_heatmap.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def run_full_eda():
    X, y = load_raw_data()
    full_df = pd.concat([X, y], axis=1)

    print("--- Running Full Clinical EDA ---")
    analyze_target_and_demographics(full_df)
    p_vals = analyze_numerical_features(full_df)
    analyze_categorical_features(full_df)
    analyze_correlations(full_df)

    print("\n--- Clinical Significant Features (Mann-Whitney U p-values) ---")
    for feat, p in p_vals.items():
        sig = "*** (p < 0.001)" if p < 0.001 else "** (p < 0.01)" if p < 0.01 else "* (p < 0.05)" if p < 0.05 else "ns"
        print(f"  {feat:10s}: p = {p:.3e} {sig}")


if __name__ == "__main__":
    run_full_eda()
