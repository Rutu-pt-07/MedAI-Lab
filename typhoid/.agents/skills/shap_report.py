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
    plt.title(f"SHAP Explanation - Patient #{patient_idx} (Actual: {label})", fontsize=13)
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, f"shap_waterfall_patient_{patient_idx}_{label}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[shap_report] Saved: {out_path}")
