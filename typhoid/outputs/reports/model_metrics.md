# Model Performance Report

## Model
- Algorithm: XGBoost with Monotonicity Constraints
- Best Iteration: 499
- Objective: multi:softprob (3 classes)
- Monotone constraints applied to all 9 titer features (increasing only)

## Cross-Validation (5-Fold Stratified)
| Metric | Mean | Std |
|--------|------|-----|
| F1 Macro | 0.989 | 0.009 |
| AUC-ROC (OvR) | 0.998 | 0.002 |

## Clinical Scorecard — Test Set (N=220 patients)
| Metric | Value |
|--------|-------|
| **Sensitivity (Typhoid Positive)** | **93.8%** |
| Specificity | 100.0% |
| Positive Predictive Value (PPV) | 100.0% |
| Negative Predictive Value (NPV) | 99.5% |
| F1 Score (macro) | 0.988 |
| AUC-ROC (OvR macro) | 0.998 |
| AUC-PR (Positive class) | 0.975 |

> The model correctly identified 93.8% of all Typhoid Positive cases.

## Confusion Matrix (Positive class)
| | Predicted Negative | Predicted Positive |
|--|--|--|
| **Actual Negative** | 204 (TN) | 0 (FP) |
| **Actual Positive** | 1 (FN) | 15 (TP) |

## Monotonicity Verification
Titers escalating 1->2->3 (while holding other features at median):
- P(Positive) at titer=1: 0.0015
- P(Positive) at titer=2: 0.0160
- P(Positive) at titer=3: 0.9998
- **Monotonicity satisfied: YES**

## Feature Set
['Age_years', 'Gender', 'rickettsia_panel_conducted', 'TO', 'TH', 'AH', 'BH', 'OX2', 'OXK', 'OX9', 'A', 'M']
