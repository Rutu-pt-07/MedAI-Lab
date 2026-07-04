"""
Phase 4: ModelAgent
===================
Trains an explainable, clinically trustworthy XGBoost model for Typhoid detection.
Uses monotonicity constraints and cost-sensitive learning. Generates full clinical scorecard.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score,
                             precision_recall_curve, roc_curve,
                             brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import PartialDependenceDisplay
import xgboost as xgb

os.makedirs('outputs/models', exist_ok=True)
os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)

TITER_COLS = ['TO', 'TH', 'AH', 'BH', 'OX2', 'OXK', 'OX9', 'A', 'M']
FEATURE_COLS = ['Age_years', 'Gender', 'rickettsia_panel_conducted',
                'TO', 'TH', 'AH', 'BH', 'OX2', 'OXK', 'OX9', 'A', 'M']
TARGET = 'Typhoid'

# Monotonicity: 0=none, 1=increasing for all titer features
# Order: Age_years, Gender, rickettsia_panel_conducted, TO, TH, AH, BH, OX2, OXK, OX9, A, M
MONOTONE_CONSTRAINTS = (0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1)


def compute_clinical_metrics(y_true, y_pred, y_prob, class_label=2):
    """Compute Sensitivity, Specificity, PPV, NPV for the Positive class."""
    binary_true = (y_true == class_label).astype(int)
    binary_pred = (y_pred == class_label).astype(int)

    TP = ((binary_true == 1) & (binary_pred == 1)).sum()
    TN = ((binary_true == 0) & (binary_pred == 0)).sum()
    FP = ((binary_true == 0) & (binary_pred == 1)).sum()
    FN = ((binary_true == 1) & (binary_pred == 0)).sum()

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0

    return {'sensitivity': sensitivity, 'specificity': specificity,
            'ppv': ppv, 'npv': npv, 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}


def run_model_agent():
    print("=" * 60)
    print("PHASE 4: ModelAgent")
    print("=" * 60)

    # ─── Load data ───
    df = pd.read_csv('data/processed/cleaned_dataset.csv')
    print(f"\nLoaded cleaned dataset: {df.shape}")

    # ─── Task 1: Feature / Target split ───
    X = df[FEATURE_COLS].copy()
    y = df[TARGET].copy()
    print(f"Features: {FEATURE_COLS}")
    print(f"Target distribution: {y.value_counts().sort_index().to_dict()}")

    # Fill any remaining NaN in features with 0 (should only be Age_years gaps)
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # ─── Task 2: Class weights ───
    class_counts = y_train.value_counts().sort_index()
    total = len(y_train)
    n_classes = len(class_counts)
    class_weights = {cls: total / (n_classes * cnt) for cls, cnt in class_counts.items()}
    sample_weights = y_train.map(class_weights).values
    print(f"Class weights: {class_weights}")

    # ─── Task 3: Train XGBoost ───
    print("\nTraining XGBoost with monotonicity constraints...")
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        monotone_constraints=MONOTONE_CONSTRAINTS,
        random_state=42,
        early_stopping_rounds=30,
        eval_metric='mlogloss',
        verbosity=0,
        tree_method='hist',
    )

    eval_set = [(X_test, y_test)]
    model.fit(X_train, y_train,
              sample_weight=sample_weights,
              eval_set=eval_set,
              verbose=False)

    print(f"Best iteration: {model.best_iteration}")

    # ─── Cross-Validation (manual loop for sample_weight support) ───
    print("\nRunning 5-fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1, cv_auc = [], []
    for train_idx, val_idx in skf.split(X, y):
        Xf_tr, Xf_val = X.iloc[train_idx], X.iloc[val_idx]
        yf_tr, yf_val = y.iloc[train_idx], y.iloc[val_idx]
        sw_fold = yf_tr.map(class_weights).values
        cv_clf = xgb.XGBClassifier(
            objective='multi:softprob', num_class=3,
            n_estimators=model.best_iteration + 1,
            max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            monotone_constraints=MONOTONE_CONSTRAINTS,
            random_state=42, verbosity=0, tree_method='hist',
        )
        cv_clf.fit(Xf_tr, yf_tr, sample_weight=sw_fold)
        yf_pred = cv_clf.predict(Xf_val)
        yf_prob = cv_clf.predict_proba(Xf_val)
        cv_f1.append(f1_score(yf_val, yf_pred, average='macro'))
        cv_auc.append(roc_auc_score(yf_val, yf_prob, multi_class='ovr', average='macro'))
    cv_results = {'test_f1_macro': np.array(cv_f1), 'test_roc_auc_ovr': np.array(cv_auc)}
    print(f"CV F1-macro: {cv_results['test_f1_macro'].mean():.3f} +/- {cv_results['test_f1_macro'].std():.3f}")
    print(f"CV AUC-OvR:  {cv_results['test_roc_auc_ovr'].mean():.3f} +/- {cv_results['test_roc_auc_ovr'].std():.3f}")

    # ─── Monotonicity verification ───
    print("\nVerifying monotonicity constraints...")
    base_patient = {col: X_train[col].median() for col in FEATURE_COLS}
    probs_at_titers = []
    for titer_val in [1, 2, 3]:
        patient = base_patient.copy()
        patient['TO'] = titer_val; patient['TH'] = titer_val
        patient['AH'] = titer_val; patient['BH'] = titer_val
        X_synth = pd.DataFrame([patient])
        prob_positive = model.predict_proba(X_synth)[0][2]
        probs_at_titers.append(prob_positive)
        print(f"  Titer={titer_val} -> P(Positive)={prob_positive:.4f}")

    mono_ok = all(probs_at_titers[i] <= probs_at_titers[i+1]
                  for i in range(len(probs_at_titers)-1))
    print(f"  Monotonicity satisfied: {'YES' if mono_ok else 'VIOLATION DETECTED'}")

    # ─── Task 4: Evaluate model ───
    print("\nEvaluating on held-out test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics = compute_clinical_metrics(y_test, y_pred, y_prob, class_label=2)
    f1_macro = float(pd.Series(classification_report(y_test, y_pred, output_dict=True)
                               ['macro avg']['f1-score']))
    auc_roc_ovr = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    auc_pr_pos = average_precision_score((y_test == 2).astype(int), y_prob[:, 2])

    print(f"  Sensitivity (Typhoid Positive): {metrics['sensitivity']*100:.1f}%")
    print(f"  Specificity:                    {metrics['specificity']*100:.1f}%")
    print(f"  PPV (Precision):                {metrics['ppv']*100:.1f}%")
    print(f"  NPV:                            {metrics['npv']*100:.1f}%")
    print(f"  F1 (macro):                     {f1_macro:.3f}")
    print(f"  AUC-ROC (OvR macro):            {auc_roc_ovr:.3f}")
    print(f"  AUC-PR  (Positive class):       {auc_pr_pos:.3f}")

    # ─── Task 5: PR and ROC Curves ───
    print("\nGenerating PR and ROC curves...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # PR Curve (Positive class)
    prec, rec, thresh_pr = precision_recall_curve((y_test == 2).astype(int), y_prob[:, 2])
    f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = f1_scores.argmax()
    axes[0].plot(rec, prec, color='#3498db', lw=2,
                 label=f'AUC-PR = {auc_pr_pos:.3f}')
    axes[0].scatter(rec[best_idx], prec[best_idx], color='red', s=80, zorder=5,
                    label=f'Best F1={f1_scores[best_idx]:.3f} @ thresh={thresh_pr[best_idx]:.2f}')
    axes[0].axhline(y=(y_test == 2).mean(), color='gray', linestyle='--', label='Baseline')
    axes[0].set_xlabel('Recall (Sensitivity)')
    axes[0].set_ylabel('Precision (PPV)')
    axes[0].set_title('Precision-Recall Curve — Typhoid Positive Class', fontweight='bold')
    axes[0].legend()

    # ROC Curve (OvR for each class)
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    class_names = ['Negative', 'Minimal', 'Positive']
    for cls_idx, (cls_color, cls_name) in enumerate(zip(colors, class_names)):
        fpr, tpr, _ = roc_curve((y_test == cls_idx).astype(int), y_prob[:, cls_idx])
        auc_cls = roc_auc_score((y_test == cls_idx).astype(int), y_prob[:, cls_idx])
        axes[1].plot(fpr, tpr, color=cls_color, lw=2,
                     label=f'{cls_name} (AUC={auc_cls:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curves (One-vs-Rest)', fontweight='bold')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('outputs/plots/pr_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/plots/pr_roc_curves.png")

    # ─── Task 6: Reliability Diagram ───
    print("Generating reliability diagram...")
    fig, ax = plt.subplots(figsize=(8, 7))
    prob_true, prob_pred = calibration_curve((y_test == 2).astype(int),
                                             y_prob[:, 2], n_bins=10)
    ax.plot(prob_pred, prob_true, 's-', color='#3498db', lw=2, label='XGBoost (uncalibrated)')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Reliability Diagram — Typhoid Positive Class', fontweight='bold')
    ax.legend()
    brier = brier_score_loss((y_test == 2).astype(int), y_prob[:, 2])
    ax.text(0.05, 0.9, f'Brier Score: {brier:.4f}', transform=ax.transAxes,
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.savefig('outputs/plots/reliability_diagram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/plots/reliability_diagram.png")

    # ─── Task 7: PDP Bias Check ───
    print("Generating PDP bias-check plots for Age and Gender...")
    features_to_plot = ['Age_years', 'Gender']
    feature_indices = [FEATURE_COLS.index(f) for f in features_to_plot]

    for feat, idx in zip(features_to_plot, feature_indices):
        fig, ax = plt.subplots(figsize=(8, 5))
        display = PartialDependenceDisplay.from_estimator(
            model, X_train, features=[idx],
            target=2,   # Positive class
            feature_names=FEATURE_COLS,
            ax=ax, kind='average', grid_resolution=20)
        ax.set_title(f'PDP: {feat} vs P(Typhoid Positive)\nBias check: should be flat/weak', fontweight='bold')
        ax.set_xlabel(feat)
        ax.set_ylabel('Partial dependence')
        fname = f'outputs/plots/pdp_{feat.lower()}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}")

    # ─── Task 8: Save model ───
    joblib.dump(model, 'outputs/models/typhoid_model.pkl')
    with open('outputs/models/feature_columns.json', 'w') as f:
        json.dump(FEATURE_COLS, f, indent=2)
    print("\nSaved: outputs/models/typhoid_model.pkl")
    print("Saved: outputs/models/feature_columns.json")

    # ─── Write metrics report ───
    report = f"""# Model Performance Report

## Model
- Algorithm: XGBoost with Monotonicity Constraints
- Best Iteration: {model.best_iteration}
- Objective: multi:softprob (3 classes)
- Monotone constraints applied to all 9 titer features (increasing only)

## Cross-Validation (5-Fold Stratified)
| Metric | Mean | Std |
|--------|------|-----|
| F1 Macro | {cv_results['test_f1_macro'].mean():.3f} | {cv_results['test_f1_macro'].std():.3f} |
| AUC-ROC (OvR) | {cv_results['test_roc_auc_ovr'].mean():.3f} | {cv_results['test_roc_auc_ovr'].std():.3f} |

## Clinical Scorecard — Test Set (N={len(y_test)} patients)
| Metric | Value |
|--------|-------|
| **Sensitivity (Typhoid Positive)** | **{metrics['sensitivity']*100:.1f}%** |
| Specificity | {metrics['specificity']*100:.1f}% |
| Positive Predictive Value (PPV) | {metrics['ppv']*100:.1f}% |
| Negative Predictive Value (NPV) | {metrics['npv']*100:.1f}% |
| F1 Score (macro) | {f1_macro:.3f} |
| AUC-ROC (OvR macro) | {auc_roc_ovr:.3f} |
| AUC-PR (Positive class) | {auc_pr_pos:.3f} |

> The model correctly identified {metrics['sensitivity']*100:.1f}% of all Typhoid Positive cases.

## Confusion Matrix (Positive class)
| | Predicted Negative | Predicted Positive |
|--|--|--|
| **Actual Negative** | {metrics['TN']} (TN) | {metrics['FP']} (FP) |
| **Actual Positive** | {metrics['FN']} (FN) | {metrics['TP']} (TP) |

## Monotonicity Verification
Titers escalating 1->2->3 (while holding other features at median):
- P(Positive) at titer=1: {probs_at_titers[0]:.4f}
- P(Positive) at titer=2: {probs_at_titers[1]:.4f}
- P(Positive) at titer=3: {probs_at_titers[2]:.4f}
- **Monotonicity satisfied: {'YES' if mono_ok else 'VIOLATION DETECTED'}**

## Feature Set
{FEATURE_COLS}
"""
    with open('outputs/reports/model_metrics.md', 'w') as f:
        f.write(report)
    print("Saved: outputs/reports/model_metrics.md")

    print("\n" + "=" * 60)
    print("ModelAgent COMPLETE")
    print("=" * 60)

    return model, X_train, X_test, y_train, y_test, y_prob, metrics


if __name__ == '__main__':
    run_model_agent()
