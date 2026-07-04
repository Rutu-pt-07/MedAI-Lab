# Typhoid Detection Model — Antigravity Project Blueprint
> **agents.md** · Place this file at the root of your project workspace.  
> Antigravity will parse this file to understand the project mission, agent roles, rules, skills, and task sequence.

---

## Project Mission

Build a clinically trustworthy, explainable Typhoid & Rickettsia detection model from Widal/Weil-Felix serology data (`Original_Dataset.csv`).  
The output is not just a model — it is a **Clinical Performance Report** with SHAP explanations, fairness checks, and a doctor-readable summary.

**Priority metric:** Sensitivity (Recall) for the `Positive` class — missing a Typhoid case is worse than a false alarm.  
**Interpretability requirement:** Every prediction must be explainable to a clinician.

---

## Global Rules
> These rules apply to every agent in every task. No agent may violate them.

```
1. NEVER treat titer columns (TO, TH, AH, BH, OX2, OXK, OX9, A, M) as categorical or continuous floats.
   They are strictly ORDINAL: 1:80=1, 1:160=2, 1:320=3.

2. NEVER impute Rickettsia titer NaNs without first creating the `rickettsia_panel_conducted` binary flag.
   The absence of a test is clinical data, not missing data.

3. NEVER use raw Accuracy as the primary evaluation metric.
   Always report: Sensitivity, Specificity, PPV, NPV, F1 (macro), AUC-PR, and AUC-ROC.

4. NEVER use black-box models without a paired SHAP explanation.
   Every model output shown to the user must include a SHAP waterfall plot.

5. ALWAYS verify that monotonicity constraints are respected:
   If titer value increases → predicted probability of disease must NOT decrease.

6. ALWAYS check for demographic shortcuts before finalising:
   Run Partial Dependence Plots (PDP) on Age and Gender to confirm the model
   is using antigen signals — not just demographic patterns.

7. The patient Name column is encrypted. Drop it immediately. Never use it.
```

---

## Folder Structure to Create

```
project-root/
├── agents.md                        ← this file
├── Original_Dataset.csv             ← raw data (do not modify)
├── .agents/
│   ├── rules/
│   │   └── medical_logic.md         ← global clinical constraints (auto-generated from this file)
│   └── skills/
│       ├── titer_converter.py       ← reusable titer encoding skill
│       ├── age_parser.py            ← reusable age parsing skill
│       └── shap_report.py           ← reusable SHAP waterfall generator
├── data/
│   ├── raw/                         ← copy of Original_Dataset.csv
│   └── processed/                   ← cleaned outputs go here
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   └── 03_model_and_explainability.ipynb
├── outputs/
│   ├── plots/                       ← all EDA and model visualisation PNGs
│   ├── models/                      ← saved model artifacts (.pkl / .json)
│   └── reports/
│       └── clinical_summary.md      ← final doctor-readable report
└── requirements.txt
```

---

## important note:
'''
use colab extentions to acess the gpu/cpu for training the models
'''

## Skills (Reusable Code Blocks)

### Skill: `titer_converter.py`
**Purpose:** Strip quotes from titer strings and convert to ordinal integers.  
**Input:** Any pandas DataFrame with raw titer columns.  
**Output:** Same DataFrame with titers as integers (1, 2, 3) and NaN preserved.

```python
# .agents/skills/titer_converter.py

import pandas as pd
import re

TITER_MAP = {
    '"1:80"':  1,
    '"1:160"': 2,
    '"1:320"': 3,
    '1:80':    1,
    '1:160':   2,
    '1:320':   3,
}

TITER_COLS = ['TO', 'TH', 'AH', 'BH', 'OX2', 'OXK', 'OX9', 'A', 'M']

def convert_titers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all titer columns from quoted strings to ordinal integers.
    NaN values are preserved (not-tested patients).
    Validation: asserts only values {1, 2, 3, NaN} remain after conversion.
    """
    df = df.copy()
    for col in TITER_COLS:
        if col in df.columns:
            df[col] = df[col].map(TITER_MAP)
    
    # Validation check
    for col in TITER_COLS:
        if col in df.columns:
            invalid = df[col].dropna()[~df[col].dropna().isin([1, 2, 3])]
            if not invalid.empty:
                raise ValueError(f"Unexpected titer values in {col}: {invalid.unique()}")
    
    print(f"[titer_converter] Converted {len(TITER_COLS)} titer columns to ordinal integers.")
    return df
```

---

### Skill: `age_parser.py`
**Purpose:** Parse messy age strings into decimal years.  
**Input:** A pandas Series with raw Age values like `5y`, `10m`, `9h`, `25d`, ` 35y`, `52 y`.  
**Output:** A float Series in decimal years.

```python
# .agents/skills/age_parser.py

import pandas as pd
import re

def parse_age_to_years(age_series: pd.Series) -> pd.Series:
    """
    Parses mixed-format age strings to decimal years.
    
    Handles: years (y), months (m), days (d), hours (h)
    Strips: leading/trailing spaces, internal spaces between number and unit.
    
    Examples:
        '5y'    → 5.0
        '3.5y'  → 3.5
        '10m'   → 0.833
        '25d'   → 0.068
        '9h'    → 0.001
        ' 25y'  → 25.0   (leading space handled)
        '52 y'  → 52.0   (internal space handled)
    """
    UNIT_TO_YEARS = {
        'y': 1.0,
        'm': 1 / 12,
        'd': 1 / 365,
        'h': 1 / (24 * 365),
    }

    def _parse_single(raw):
        if pd.isna(raw):
            return None
        cleaned = str(raw).strip().replace(' ', '')
        match = re.match(r'^([\d.]+)([ymdhYMDH])$', cleaned)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            return round(value * UNIT_TO_YEARS[unit], 6)
        return None  # unparseable — will appear as NaN for investigation

    parsed = age_series.apply(_parse_single)
    
    # Validation report
    n_failed = parsed.isna().sum() - age_series.isna().sum()
    if n_failed > 0:
        print(f"[age_parser] WARNING: {n_failed} age values could not be parsed. Inspect manually.")
    
    print(f"[age_parser] Age range after parsing: {parsed.min():.3f}y – {parsed.max():.1f}y")
    return parsed
```

---

### Skill: `shap_report.py`
**Purpose:** Generate SHAP waterfall plots for individual patient predictions.  
**Input:** Trained model, preprocessed feature DataFrame, patient indices.  
**Output:** PNG files saved to `outputs/plots/`.

```python
# .agents/skills/shap_report.py

import shap
import matplotlib.pyplot as plt
import pandas as pd
import os

def generate_shap_waterfall(model, X: pd.DataFrame, patient_idx: int,
                             label: str, output_dir: str = "outputs/plots") -> None:
    """
    Generate a SHAP waterfall plot for a single patient.
    
    Args:
        model:       Trained XGBoost / CatBoost / EBM model (SHAP-compatible)
        X:           Preprocessed feature DataFrame (after encoding, no target column)
        patient_idx: Row index of the patient in X
        label:       Human-readable label e.g. 'Positive', 'Negative', 'Minimal'
        output_dir:  Directory to save the PNG
    """
    os.makedirs(output_dir, exist_ok=True)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[patient_idx], max_display=12, show=False)
    plt.title(f"SHAP Explanation — Patient #{patient_idx} (Actual: {label})", fontsize=13)
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, f"shap_waterfall_patient_{patient_idx}_{label}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[shap_report] Saved: {out_path}")
```

---

## Agent Definitions

### Agent 1 — `DataEngineer`
**Role:** Data integrity and preprocessing.  
**Owns:** Phase 1 entirely. Does not train any model.

```yaml
agent: DataEngineer
description: >
  Responsible for transforming the raw clinical dataset into a clean, 
  analysis-ready DataFrame. Applies all biological logic constraints.
  
inputs:
  - Original_Dataset.csv

outputs:
  - data/processed/cleaned_dataset.csv
  - data/processed/preprocessing_report.md    # summary of all transformations + row counts

tasks:

  - id: strip_column_names
    description: >
      Strip all leading/trailing whitespace from column names.
      Specifically: 'Acute_typhoid ' and 'Paratyphoid_A ' have trailing spaces.
    code: |
      df.columns = df.columns.str.strip()

  - id: drop_name_column
    description: >
      The Name column contains encrypted tokens — zero analytical value, potential PII risk.
      Drop it unconditionally.
    code: |
      df.drop(columns=['Name'], inplace=True)

  - id: parse_age
    description: >
      Use the age_parser skill to convert the Age column from mixed-format strings 
      to decimal years (float). Log any values that fail to parse.
    skill: age_parser.py
    verify: |
      assert df['Age_years'].between(0, 120).all(), "Age out of plausible human range"

  - id: rickettsia_flag
    description: >
      BEFORE touching any NaN in titer columns — create a binary flag:
      rickettsia_panel_conducted = 1 if OX2 is not NaN, else 0.
      This captures the clinical decision to not run the Rickettsia panel.
    code: |
      df['rickettsia_panel_conducted'] = df['OX2'].notna().astype(int)
    verify: |
      assert df['rickettsia_panel_conducted'].isin([0, 1]).all()
      assert df['rickettsia_panel_conducted'].sum() == 709  # 1106 - 397

  - id: convert_titers
    description: >
      Use the titer_converter skill to convert all 9 titer columns from quoted 
      strings to ordinal integers. For patients who had the Rickettsia panel 
      (rickettsia_panel_conducted=1), remaining NaNs in OX2/OXK/OX9/A/M 
      should stay NaN (not tested for that specific antigen).
      For patients where rickettsia_panel_conducted=0, fill all Rickettsia 
      titer NaNs with 0 (meaning: panel not conducted, no reaction recorded).
    skill: titer_converter.py
    code: |
      rickettsia_titers = ['OX2', 'OXK', 'OX9', 'A', 'M']
      mask_not_tested = df['rickettsia_panel_conducted'] == 0
      df.loc[mask_not_tested, rickettsia_titers] = df.loc[mask_not_tested, rickettsia_titers].fillna(0)

  - id: encode_gender
    description: Binary encode Gender. Male=1, Female=0.
    code: |
      df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

  - id: handle_6_corrupt_rows
    description: >
      6 rows have NaN across TO/TH/AH/BH and all five diagnosis columns simultaneously.
      These are likely data entry failures. Isolate them to a separate CSV for manual review.
      Drop them from the main dataset.
    code: |
      corrupt_mask = df[['TO', 'TH', 'Typhoid']].isna().all(axis=1)
      df[corrupt_mask].to_csv('data/processed/flagged_corrupt_rows.csv', index=True)
      df = df[~corrupt_mask].reset_index(drop=True)
    verify: |
      assert df.shape[0] == 1100  # 1106 - 6

  - id: encode_targets
    description: >
      Encode the 5 diagnosis columns to integers for modelling.
      Typhoid: Negative=0, Minimal=1, Positive=2  (ordinal — severity increases)
      Acute_typhoid: No=0, Yes=1
      Paratyphoid_A: No=0, Yes=1
      Paratyphoid_B: No=0, Yes=1
      Rickettsia_Suspect: No=0, Yes=1 (only for the 709 patients who were tested)
    code: |
      df['Typhoid'] = df['Typhoid'].map({'Negative': 0, 'Minimal': 1, 'Positive': 2})
      for col in ['Acute_typhoid', 'Paratyphoid_A', 'Paratyphoid_B', 'Rickettsia_Suspect']:
          df[col] = df[col].map({'No': 0, 'Yes': 1})

  - id: save_cleaned_data
    description: Save the final cleaned DataFrame and write a preprocessing report.
    code: |
      df.to_csv('data/processed/cleaned_dataset.csv', index=False)

  - id: write_preprocessing_report
    description: >
      Generate data/processed/preprocessing_report.md with:
      - Before/after row counts
      - Column dtype confirmation
      - NaN counts per column (post-cleaning)
      - Rickettsia flag breakdown (709 tested / 397 not tested)
      - Age range stats (min, max, mean, median)
      - Titer value distribution for each of the 9 columns
      - List of any rows saved to flagged_corrupt_rows.csv
```

---

### Agent 2 — `EDAAgent`
**Role:** Exploratory data analysis — generate all visualisations.  
**Owns:** All plots. Does not modify data. Does not train models.

```yaml
agent: EDAAgent
description: >
  Produces a complete visual EDA of the cleaned dataset.
  Every plot is saved as a high-resolution PNG to outputs/plots/.
  The goal is to prove to a clinician that the data follows biological logic
  before any model is built.

inputs:
  - data/processed/cleaned_dataset.csv

outputs:
  - outputs/plots/  (all plots below)

dependencies:
  - DataEngineer  # must complete first

libraries_required:
  - pandas, numpy, matplotlib, seaborn, missingno, scipy, plotly

plots:

  - id: missing_data_matrix
    type: missingno matrix
    description: >
      Shows the "solid block" of NaN values for OX2/OXK/OX9/A/M/Rickettsia_Suspect 
      for the 397 untested patients. Proves systematic sub-group, not random missingness.
    file: missing_data_matrix.png

  - id: age_distribution
    type: histogram + KDE
    description: >
      Age (decimal years) distribution of all 1100 patients.
      Use bins of 5 years. Annotate peaks. Show pediatric (<15y) / adult / elderly bands.
    file: age_distribution.png

  - id: gender_balance
    type: bar chart
    description: Male vs Female count with percentage labels.
    file: gender_balance.png

  - id: typhoid_class_distribution
    type: donut chart
    description: >
      Typhoid outcome proportions: Negative / Minimal / Positive.
      Annotate with counts AND percentages. Highlight that Positive = ~7%.
      Include a callout: "A naive model guessing Negative gets 42% accuracy — 
      Sensitivity matters more than Accuracy here."
    file: typhoid_class_distribution.png

  - id: all_targets_class_distribution
    type: horizontal grouped bar
    description: >
      One bar group per diagnosis column (Typhoid, Acute_typhoid, Paratyphoid_A, 
      Paratyphoid_B, Rickettsia_Suspect). Shows imbalance across all 5 targets at once.
    file: all_targets_class_distribution.png

  - id: titer_gradient_vs_typhoid
    type: grouped bar (3 bars per antigen, one per titer level)
    description: >
      For each of TO, TH, AH, BH: show % of Typhoid=Positive patients at titer 1, 2, 3.
      Trust check: if bar for titer=3 is NOT the highest, the data defies biology — flag it.
    file: titer_gradient_vs_typhoid.png

  - id: titer_gradient_vs_rickettsia
    type: grouped bar
    description: >
      Same as above but for OX2, OXK, OX9 vs Rickettsia_Suspect=Yes.
      Only uses the 709 patients who received the Rickettsia panel.
    file: titer_gradient_vs_rickettsia.png

  - id: titer_correlation_heatmap
    type: heatmap (Spearman)
    description: >
      9×9 Spearman correlation matrix for all titer columns.
      Annotate coefficients. Expected: TO/TH should correlate (both typhoid antigens).
      OX2/OXK/OX9 should correlate (all Rickettsia antigens).
    file: titer_correlation_heatmap.png

  - id: age_vs_typhoid_boxplot
    type: box plot
    description: >
      Age distribution split by Typhoid outcome (Negative / Minimal / Positive).
      Shows whether Positive cases cluster in a specific age group.
    file: age_vs_typhoid_boxplot.png

  - id: gender_vs_outcomes
    type: stacked bar
    description: >
      For each of the 5 diagnosis targets: show Male/Female breakdown of 
      Positive vs Negative (or equivalent). One row per target.
    file: gender_vs_outcomes.png

  - id: diagnosis_cooccurrence
    type: heatmap (counts)
    description: >
      5×5 co-occurrence matrix of all diagnosis targets.
      Reveals: can a patient be Typhoid Positive AND Rickettsia Suspect simultaneously?
      This "co-infection" signal is important in endemic regions.
    file: diagnosis_cooccurrence.png

  - id: titer_outcome_heatmap
    type: heatmap (mean titer value)
    description: >
      9 titer columns (rows) × Typhoid outcome 0/1/2 (columns).
      Cell value = mean titer for that antigen in that outcome group.
      The single clearest view of which antigens drive each diagnosis.
    file: titer_outcome_heatmap.png

  - id: age_group_vs_diagnosis_rate
    type: grouped bar
    description: >
      Bin patients into: Infant (<2y), Child (2–14y), Adult (15–59y), Elderly (60y+).
      Plot Typhoid Positive rate per group. Reveals age-based risk patterns.
    file: age_group_vs_diagnosis_rate.png

  - id: pca_scatter
    type: 2D scatter (PCA)
    description: >
      PCA on all 9 titer features. Plot first 2 PCs. 
      Colour points by Typhoid outcome (0/1/2).
      Shows whether patient clusters align with clinical diagnosis.
    file: pca_scatter.png

  - id: parallel_coordinates
    type: parallel coordinates (plotly)
    description: >
      Each line = one patient. Axes = all 9 titer features.
      Colour by Typhoid outcome. Interactive HTML + static PNG.
      Shows if Positive patients have consistently elevated profiles.
    files:
      - parallel_coordinates.html
      - parallel_coordinates.png
```

---

### Agent 3 — `ModelAgent`
**Role:** Feature engineering, model training, evaluation.  
**Owns:** All model files, metrics, and calibration plots.

```yaml
agent: ModelAgent
description: >
  Trains an explainable, clinically trustworthy model for Typhoid detection.
  Uses cost-sensitive learning to handle class imbalance.
  Applies monotonicity constraints so the model cannot violate clinical logic.

inputs:
  - data/processed/cleaned_dataset.csv

outputs:
  - outputs/models/typhoid_model.pkl
  - outputs/models/feature_columns.json
  - outputs/plots/class_weight_comparison.png
  - outputs/plots/pr_curve.png
  - outputs/plots/roc_curve.png
  - outputs/plots/reliability_diagram.png
  - outputs/plots/pdp_age.png
  - outputs/plots/pdp_gender.png
  - outputs/reports/model_metrics.md

dependencies:
  - DataEngineer
  - EDAAgent   # run EDA first so human can verify data before modelling

libraries_required:
  - pandas, numpy, scikit-learn, xgboost, catboost, shap, matplotlib, joblib

tasks:

  - id: feature_target_split
    description: >
      Primary target: Typhoid (3-class: 0=Negative, 1=Minimal, 2=Positive).
      Features: Age_years, Gender, rickettsia_panel_conducted, 
                TO, TH, AH, BH, OX2, OXK, OX9, A, M.
      Drop all other diagnosis columns from features.
      Train/test split: 80/20, stratified on Typhoid, random_state=42.

  - id: train_model
    description: >
      Train an XGBoost classifier with:
        - objective: multi:softprob (3 classes)
        - scale_pos_weight: computed per class (inverse of class frequency)
        - monotone_constraints: all titer features constrained to +1 
          (increasing titer must not decrease disease probability)
        - n_estimators: 500, early_stopping_rounds=30
        - Use StratifiedKFold(n_splits=5) for cross-validation
      
      Monotonicity constraint vector (one value per feature, same order as features list):
        Age_years: 0, Gender: 0, rickettsia_panel_conducted: 0,
        TO: 1, TH: 1, AH: 1, BH: 1, OX2: 1, OXK: 1, OX9: 1, A: 1, M: 1
    
    verify: >
      After training, check monotonicity manually:
      Create synthetic patients with titer escalating from 1→2→3 while holding 
      all other features constant. Assert that P(Typhoid=Positive) is non-decreasing.

  - id: evaluate_model
    description: >
      Compute the full clinical scorecard on the held-out test set:
        - Sensitivity (Recall) for Positive class
        - Specificity for Positive class
        - PPV (Precision) for Positive class
        - NPV for Positive class
        - F1 (macro-averaged across all 3 classes)
        - AUC-ROC (one-vs-rest for each class)
        - AUC-PR (Precision-Recall curve for Positive class)
      Save all metrics to outputs/reports/model_metrics.md.
      Format: "The model correctly identified X% of all Typhoid Positive cases."

  - id: plot_pr_and_roc_curves
    description: >
      Generate Precision-Recall curve for the Positive class.
      Generate ROC curve (one-vs-rest) for all three classes.
      Mark the operating threshold that maximises F1 for the Positive class.
      Save both plots. Include a note: PR-Curve is preferred over ROC when 
      classes are imbalanced (7% Positive).

  - id: reliability_diagram
    description: >
      Calibration plot (reliability diagram) comparing predicted probability 
      to actual positive rate across 10 bins.
      If the model is overconfident, apply CalibratedClassifierCV with method='isotonic'.
      A calibrated model is required — doctors need to trust the probability score.

  - id: pdp_bias_check
    description: >
      Generate Partial Dependence Plots for Age_years and Gender.
      Purpose: confirm the model is learning titer signals, not demographic shortcuts.
      If PDP for Gender shows a large effect, flag it and investigate.
      Save outputs/plots/pdp_age.png and outputs/plots/pdp_gender.png.

  - id: save_model
    description: >
      Save the trained, calibrated model to outputs/models/typhoid_model.pkl using joblib.
      Save the ordered feature column list to outputs/models/feature_columns.json
      so the inference step always uses features in the correct order.
```

---

### Agent 4 — `ExplainabilityAgent`
**Role:** Generate all SHAP explanations and the final clinical report.  
**Owns:** SHAP plots, feature importance, clinical summary document.

```yaml
agent: ExplainabilityAgent
description: >
  Generates patient-level and global explanations for every model prediction.
  Produces a doctor-readable Clinical Summary Report.
  A clinician should be able to understand every output without reading code.

inputs:
  - data/processed/cleaned_dataset.csv
  - outputs/models/typhoid_model.pkl
  - outputs/models/feature_columns.json

outputs:
  - outputs/plots/shap_summary_beeswarm.png
  - outputs/plots/shap_waterfall_positive_patient.png
  - outputs/plots/shap_waterfall_negative_patient.png
  - outputs/plots/shap_waterfall_minimal_patient.png
  - outputs/plots/shap_feature_importance_bar.png
  - outputs/reports/clinical_summary.md

dependencies:
  - ModelAgent

libraries_required:
  - shap, matplotlib, pandas, joblib, numpy

tasks:

  - id: shap_global_summary
    description: >
      Compute SHAP values for the entire test set.
      Generate a beeswarm summary plot showing global feature importance.
      X-axis: SHAP value (impact on prediction). Y-axis: features ranked by importance.
      Colour: feature value (red=high, blue=low).
      Expected result: TO, TH titer levels should rank among the top features.
    file: shap_summary_beeswarm.png

  - id: shap_feature_importance_bar
    description: >
      Bar plot of mean |SHAP| values per feature (global importance).
      This is the version to show to a clinician — simple, ranked, unambiguous.
    file: shap_feature_importance_bar.png

  - id: shap_waterfall_three_patients
    description: >
      Select 3 representative patients from the test set:
        - One correctly predicted as Positive (true positive)
        - One correctly predicted as Negative (true negative)
        - One predicted as Minimal (the ambiguous middle class)
      Use the shap_report skill to generate waterfall plots for each.
      Each plot must show:
        - The base rate (average prediction across all patients)
        - The contribution of each feature that pushed the prediction up or down
        - The final predicted probability
      Label each plot with the actual diagnosis and the predicted diagnosis.
    skill: shap_report.py

  - id: generate_clinical_summary
    description: >
      Write outputs/reports/clinical_summary.md — a plain-English document structured as:

      # Clinical Performance Summary

      ## What the model does
      [1 paragraph — plain language, no jargon]

      ## Performance on test set (N=220 patients)
      | Metric | Value |
      |--------|-------|
      | Sensitivity (Typhoid Positive) | X% |
      | Specificity | X% |
      | Positive Predictive Value | X% |
      | Negative Predictive Value | X% |
      | AUC-PR (Positive class) | X.XX |

      ## What the model looks at (top 5 features)
      [Insert shap_feature_importance_bar.png]
      [2 sentences explaining what the top features mean clinically]

      ## Example: How the model explains a Positive case
      [Insert shap_waterfall_positive_patient.png]
      "This patient was predicted Positive (Prob=0.86) because TO titer was 1:320 (+0.42), 
      TH titer was 1:320 (+0.31), and Age was 5 years (+0.12). 
      The BH titer of 1:80 slightly lowered the prediction (−0.06)."

      ## Limitations & Caveats
      - The model was trained on data where only 7% of patients were Positive. 
        It may underperform on populations with different endemic rates.
      - The Rickettsia panel was not available for 36% of patients. 
        Predictions for untested patients rely only on typhoid antigens.
      - Monotonicity is enforced: higher titers always increase predicted disease probability.
        However, the model cannot distinguish between true infection and cross-reactive titers.

      ## Bias Check
      [Insert pdp_age.png and pdp_gender.png]
      "The model's predictions are driven by antigen titer levels, not by age or gender alone.
       Partial Dependence Plots confirm no significant demographic shortcut is being used."
```

---

## Execution Order

```
DataEngineer  →  EDAAgent  →  (human review)  →  ModelAgent  →  ExplainabilityAgent
```

> **Human Review Checkpoint** (between EDAAgent and ModelAgent):  
> Before training any model, a human must confirm:  
> 1. The titer gradient plot shows increasing Positive rate with higher titers.  
> 2. The missing data matrix shows a clean block (not scattered) for Rickettsia columns.  
> 3. The class imbalance donut shows Positive ≈ 7% as expected.  
> If any of these fail, DataEngineer must re-run. Do not proceed to ModelAgent.

---

## Requirements

```
# requirements.txt — auto-generate this file in the project root

pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
catboost>=1.2
shap>=0.44
matplotlib>=3.7
seaborn>=0.13
missingno>=0.5
plotly>=5.15
joblib>=1.3
scipy>=1.11
ipykernel>=6.25
notebook>=7.0
```

---

## Antigravity Commands Reference

| Command | When to use |
|---------|-------------|
| `/plan` | Before starting any phase — review the agent's intended steps |
| `/verify` | After DataEngineer completes — check data shapes and NaN counts |
| `@DataEngineer` | To re-run a specific cleaning task without restarting |
| `@EDAAgent` | To regenerate a specific plot with different parameters |
| `@ModelAgent` | To retrain with adjusted hyperparameters |
| `@ExplainabilityAgent` | To generate a SHAP plot for a specific patient ID |
| `/artifact preview` | To view any generated plot or report inline |

---

*Generated for: Widal Test Typhoid Detection Project*  
*Dataset: Original_Dataset.csv — 1,106 patients, 17 columns*  
*Clinical priority: Sensitivity over Accuracy — never miss a Typhoid case*
