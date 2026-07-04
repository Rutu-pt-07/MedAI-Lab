"""
Learning Curve Generator
========================
Generates Training vs Validation Loss and Accuracy graphs
for the trained XGBoost Typhoid detection model.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
import xgboost as xgb

os.makedirs('outputs/plots', exist_ok=True)

FEATURE_COLS = ['Age_years', 'Gender', 'rickettsia_panel_conducted',
                'TO', 'TH', 'AH', 'BH', 'OX2', 'OXK', 'OX9', 'A', 'M']
TARGET = 'Typhoid'
MONOTONE_CONSTRAINTS = (0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1)

def run():
    print("=" * 60)
    print("LEARNING CURVE GENERATOR")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────
    df = pd.read_csv('data/processed/cleaned_dataset.csv')
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)

    # ── Class weights (same as phase 4) ───────────────────────────
    class_counts = y_train.value_counts().sort_index()
    total = len(y_train)
    n_classes = len(class_counts)
    class_weights = {cls: total / (n_classes * cnt) for cls, cnt in class_counts.items()}
    sample_weights = y_train.map(class_weights).values

    # ── Train with eval logging on BOTH train and test sets ───────
    print("\nTraining XGBoost (capturing eval history)...")
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
        eval_metric=['mlogloss', 'merror'],
        verbosity=0,
        tree_method='hist',
    )

    eval_set = [
        (X_train, y_train),   # index 0 → training
        (X_test,  y_test),    # index 1 → validation
    ]
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=eval_set,
        verbose=False,
    )

    evals = model.evals_result()
    # Keys: 'validation_0' (train), 'validation_1' (val)
    train_loss = evals['validation_0']['mlogloss']
    val_loss   = evals['validation_1']['mlogloss']
    train_err  = evals['validation_0']['merror']
    val_err    = evals['validation_1']['merror']

    train_acc = [1 - e for e in train_err]
    val_acc   = [1 - e for e in val_err]

    best_iter = model.best_iteration
    rounds = list(range(1, len(train_loss) + 1))

    print(f"Best iteration (early stopping): {best_iter}")
    print(f"Final Train Loss: {train_loss[-1]:.4f}  |  Val Loss: {val_loss[-1]:.4f}")
    print(f"Final Train Acc : {train_acc[-1]*100:.2f}%  |  Val Acc : {val_acc[-1]*100:.2f}%")

    # ── Plot ───────────────────────────────────────────────────────
    plt.style.use('seaborn-v0_8-darkgrid')

    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)

    TRAIN_COLOR = '#4A90D9'   # steel-blue
    VAL_COLOR   = '#E74C3C'   # vivid red
    BEST_COLOR  = '#2ECC71'   # emerald

    # ── Panel 1: Loss ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(rounds, train_loss, color=TRAIN_COLOR, lw=2,
             label='Training Loss', alpha=0.9)
    ax1.plot(rounds, val_loss, color=VAL_COLOR, lw=2,
             label='Validation Loss', alpha=0.9)
    ax1.fill_between(rounds, train_loss, val_loss,
                     where=[v >= t for v, t in zip(val_loss, train_loss)],
                     alpha=0.08, color=VAL_COLOR, label='Generalisation gap')
    ax1.axvline(x=best_iter + 1, color=BEST_COLOR, linestyle='--', lw=1.5,
                label=f'Best iteration ({best_iter + 1})')
    ax1.scatter(best_iter + 1, val_loss[best_iter], color=BEST_COLOR,
                s=80, zorder=5)
    ax1.set_xlabel('Boosting Round', fontsize=12)
    ax1.set_ylabel('Multi-class Log Loss (mlogloss)', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold', pad=14)
    ax1.legend(fontsize=10)
    ax1.set_xlim(1, len(rounds))

    # annotate final val loss
    ax1.annotate(f'Val loss = {val_loss[best_iter]:.4f}',
                 xy=(best_iter + 1, val_loss[best_iter]),
                 xytext=(best_iter + 20, val_loss[best_iter] + 0.05),
                 fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='black', lw=1),
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

    # ── Panel 2: Accuracy ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(rounds, [a * 100 for a in train_acc], color=TRAIN_COLOR, lw=2,
             label='Training Accuracy', alpha=0.9)
    ax2.plot(rounds, [a * 100 for a in val_acc], color=VAL_COLOR, lw=2,
             label='Validation Accuracy', alpha=0.9)
    ax2.axvline(x=best_iter + 1, color=BEST_COLOR, linestyle='--', lw=1.5,
                label=f'Best iteration ({best_iter + 1})')
    ax2.scatter(best_iter + 1, val_acc[best_iter] * 100, color=BEST_COLOR,
                s=80, zorder=5)
    ax2.set_xlabel('Boosting Round', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold', pad=14)
    ax2.legend(fontsize=10)
    ax2.set_xlim(1, len(rounds))
    ax2.set_ylim(
        max(0, min(min(train_acc), min(val_acc)) * 100 - 5),
        min(100, max(max(train_acc), max(val_acc)) * 100 + 3)
    )

    # annotate best val accuracy
    ax2.annotate(f'Val acc = {val_acc[best_iter]*100:.2f}%',
                 xy=(best_iter + 1, val_acc[best_iter] * 100),
                 xytext=(best_iter + 20, val_acc[best_iter] * 100 - 3),
                 fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='black', lw=1),
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

    # ── Super-title ────────────────────────────────────────────────
    fig.suptitle(
        'XGBoost Learning Curves — Typhoid Detection Model\n'
        '(with Monotonicity Constraints & Cost-Sensitive Weighting)',
        fontsize=15, fontweight='bold', y=1.02
    )

    plt.savefig('outputs/plots/learning_curves.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: outputs/plots/learning_curves.png")

    # ── Zoomed-in view (first 200 rounds) ─────────────────────────
    zoom_n = min(200, len(rounds))
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle('Learning Curves (First 200 Rounds — Early Phase Detail)',
                  fontsize=14, fontweight='bold')

    ax3.plot(rounds[:zoom_n], train_loss[:zoom_n], color=TRAIN_COLOR, lw=2, label='Train Loss')
    ax3.plot(rounds[:zoom_n], val_loss[:zoom_n],   color=VAL_COLOR,   lw=2, label='Val Loss')
    if best_iter < zoom_n:
        ax3.axvline(x=best_iter + 1, color=BEST_COLOR, linestyle='--', lw=1.5,
                    label=f'Best iter ({best_iter+1})')
    ax3.set_xlabel('Boosting Round'); ax3.set_ylabel('mlogloss')
    ax3.set_title('Loss (Zoomed)', fontweight='bold')
    ax3.legend()

    ax4.plot(rounds[:zoom_n], [a*100 for a in train_acc[:zoom_n]], color=TRAIN_COLOR, lw=2, label='Train Acc')
    ax4.plot(rounds[:zoom_n], [a*100 for a in val_acc[:zoom_n]],   color=VAL_COLOR,   lw=2, label='Val Acc')
    if best_iter < zoom_n:
        ax4.axvline(x=best_iter + 1, color=BEST_COLOR, linestyle='--', lw=1.5,
                    label=f'Best iter ({best_iter+1})')
    ax4.set_xlabel('Boosting Round'); ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Accuracy (Zoomed)', fontweight='bold')
    ax4.legend()

    plt.tight_layout()
    plt.savefig('outputs/plots/learning_curves_zoom.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/plots/learning_curves_zoom.png")

    print("\n" + "=" * 60)
    print("LEARNING CURVES COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    run()
