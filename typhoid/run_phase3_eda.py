"""
Phase 3: EDAAgent
=================
Produces a complete visual EDA of the cleaned dataset.
Every plot is saved as a high-resolution PNG to outputs/plots/.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Style setup ───
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
})
sns.set_style("whitegrid")
PLOT_DIR = 'outputs/plots'
os.makedirs(PLOT_DIR, exist_ok=True)

TITER_COLS = ['TO', 'TH', 'AH', 'BH', 'OX2', 'OXK', 'OX9', 'A', 'M']
TYPHOID_LABELS = {0: 'Negative', 1: 'Minimal', 2: 'Positive'}
TYPHOID_COLORS = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}


def save_plot(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


def run_eda():
    print("=" * 60)
    print("PHASE 3: EDAAgent")
    print("=" * 60)
    
    df = pd.read_csv('data/processed/cleaned_dataset.csv')
    # Also load raw for missing data viz (before encoding)
    df_raw = pd.read_csv('data/raw/Original_Dataset.csv')
    df_raw.columns = df_raw.columns.str.strip()
    
    print(f"Loaded cleaned dataset: {df.shape}")
    
    # ─── 1. Missing Data Matrix ───
    print("\n[1/14] Missing data matrix...")
    fig, ax = plt.subplots(figsize=(14, 8))
    # Use raw data (before imputation) to show the systematic NaN block
    cols_for_missing = ['TO', 'TH', 'AH', 'BH', 'OX2', 'OXK', 'OX9', 'A', 'M', 
                        'Rickettsia_Suspect', 'Acute_typhoid', 'Paratyphoid_A', 'Paratyphoid_B', 'Typhoid']
    msno.matrix(df_raw[cols_for_missing], ax=ax, fontsize=10, sparkline=False,
                color=(0.27, 0.52, 0.96))
    ax.set_title('Missing Data Matrix — Systematic Rickettsia Panel Absence', fontsize=14, fontweight='bold')
    save_plot(fig, 'missing_data_matrix.png')
    
    # ─── 2. Age Distribution ───
    print("[2/14] Age distribution...")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(df['Age_years'].dropna(), bins=range(0, 100, 5), color='#3498db', 
            edgecolor='white', alpha=0.8, label='Count')
    ax2 = ax.twinx()
    df['Age_years'].dropna().plot.kde(ax=ax2, color='#e74c3c', linewidth=2, label='KDE')
    
    # Age bands
    ax.axvspan(0, 15, alpha=0.08, color='blue', label='Pediatric (<15y)')
    ax.axvspan(15, 60, alpha=0.05, color='green', label='Adult (15-59y)')
    ax.axvspan(60, 100, alpha=0.08, color='red', label='Elderly (60y+)')
    
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Count')
    ax2.set_ylabel('Density')
    ax.set_title('Age Distribution of 1,100 Patients', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    save_plot(fig, 'age_distribution.png')
    
    # ─── 3. Gender Balance ───
    print("[3/14] Gender balance...")
    fig, ax = plt.subplots(figsize=(8, 5))
    gender_counts = df['Gender'].value_counts()
    bars = ax.bar(['Male', 'Female'], [gender_counts.get(1, 0), gender_counts.get(0, 0)],
                  color=['#3498db', '#e91e8a'], edgecolor='white', width=0.5)
    for bar, count in zip(bars, [gender_counts.get(1, 0), gender_counts.get(0, 0)]):
        pct = 100 * count / len(df)
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_title('Gender Distribution', fontsize=14, fontweight='bold')
    save_plot(fig, 'gender_balance.png')
    
    # ─── 4. Typhoid Class Distribution (Donut) ───
    print("[4/14] Typhoid class distribution (donut)...")
    fig, ax = plt.subplots(figsize=(8, 8))
    typhoid_counts = df['Typhoid'].value_counts().sort_index()
    colors = [TYPHOID_COLORS[i] for i in typhoid_counts.index]
    labels = [f"{TYPHOID_LABELS[i]}\n{cnt} ({100*cnt/len(df):.1f}%)" 
              for i, cnt in typhoid_counts.items()]
    wedges, texts = ax.pie(typhoid_counts.values, labels=labels, colors=colors,
                           startangle=90, pctdistance=0.85, 
                           wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))
    for t in texts:
        t.set_fontsize(11)
        t.set_fontweight('bold')
    ax.set_title('Typhoid Outcome Distribution', fontsize=14, fontweight='bold', pad=20)
    ax.text(0, 0, f'N={len(df)}', ha='center', va='center', fontsize=16, fontweight='bold')
    # Callout
    fig.text(0.5, 0.02, 
             "⚠ A naive model guessing Negative gets ~42% accuracy — Sensitivity matters more than Accuracy.",
             ha='center', fontsize=10, style='italic', color='#666')
    save_plot(fig, 'typhoid_class_distribution.png')
    
    # ─── 5. All Targets Class Distribution ───
    print("[5/14] All targets class distribution...")
    fig, ax = plt.subplots(figsize=(12, 7))
    targets = ['Typhoid', 'Acute_typhoid', 'Paratyphoid_A', 'Paratyphoid_B', 'Rickettsia_Suspect']
    target_data = []
    for t in targets:
        vc = df[t].value_counts().sort_index()
        for val, cnt in vc.items():
            if pd.notna(val):
                target_data.append({'Target': t, 'Class': int(val), 'Count': cnt})
    td = pd.DataFrame(target_data)
    
    y_pos = np.arange(len(targets))
    bar_height = 0.25
    class_colors = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
    
    for cls in sorted(td['Class'].unique()):
        subset = td[td['Class'] == cls]
        counts = []
        for t in targets:
            match = subset[subset['Target'] == t]
            counts.append(match['Count'].values[0] if len(match) > 0 else 0)
        offset = cls * bar_height
        label_map = {0: 'Negative/No', 1: 'Minimal/Yes', 2: 'Positive'}
        ax.barh(y_pos + offset, counts, bar_height, 
                label=label_map.get(cls, str(cls)),
                color=class_colors.get(cls, '#999'), edgecolor='white')
    
    ax.set_yticks(y_pos + bar_height)
    ax.set_yticklabels(targets)
    ax.set_xlabel('Count')
    ax.set_title('Class Distribution Across All 5 Diagnosis Targets', fontsize=14, fontweight='bold')
    ax.legend()
    ax.invert_yaxis()
    save_plot(fig, 'all_targets_class_distribution.png')
    
    # ─── 6. Titer Gradient vs Typhoid ───
    print("[6/14] Titer gradient vs Typhoid...")
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    for ax_i, col in zip(axes, ['TO', 'TH', 'AH', 'BH']):
        pos_mask = df['Typhoid'] == 2
        for titer_val, color, label in [(1, '#2ecc71', '1:80'), (2, '#f39c12', '1:160'), (3, '#e74c3c', '1:320')]:
            total_at_titer = (df[col] == titer_val).sum()
            pos_at_titer = (df[col][pos_mask] == titer_val).sum()
            pct = 100 * pos_at_titer / total_at_titer if total_at_titer > 0 else 0
            ax_i.bar(label, pct, color=color, edgecolor='white', width=0.6)
            ax_i.text(label, pct + 0.5, f'{pct:.1f}%', ha='center', fontsize=9)
        ax_i.set_title(f'{col}', fontweight='bold')
        ax_i.set_ylabel('% Typhoid Positive' if col == 'TO' else '')
        ax_i.set_xlabel('Titer Level')
    fig.suptitle('Titer Gradient vs Typhoid Positive Rate — Biology Check', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, 'titer_gradient_vs_typhoid.png')
    
    # ─── 7. Titer Gradient vs Rickettsia ───
    print("[7/14] Titer gradient vs Rickettsia...")
    df_rick = df[df['rickettsia_panel_conducted'] == 1].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for ax_i, col in zip(axes, ['OX2', 'OXK', 'OX9']):
        sus_mask = df_rick['Rickettsia_Suspect'] == 1
        for titer_val, color, label in [(1, '#2ecc71', '1:80'), (2, '#f39c12', '1:160'), (3, '#e74c3c', '1:320')]:
            total = (df_rick[col] == titer_val).sum()
            pos = (df_rick[col][sus_mask] == titer_val).sum()
            pct = 100 * pos / total if total > 0 else 0
            ax_i.bar(label, pct, color=color, edgecolor='white', width=0.6)
            ax_i.text(label, pct + 0.5, f'{pct:.1f}%', ha='center', fontsize=9)
        ax_i.set_title(f'{col}', fontweight='bold')
        ax_i.set_ylabel('% Rickettsia Suspect' if col == 'OX2' else '')
    fig.suptitle(f'Titer Gradient vs Rickettsia Suspect (N={len(df_rick)} tested patients)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, 'titer_gradient_vs_rickettsia.png')
    
    # ─── 8. Titer Correlation Heatmap ───
    print("[8/14] Titer correlation heatmap (Spearman)...")
    fig, ax = plt.subplots(figsize=(10, 8))
    # Use only rows where all titers are available (tested patients)
    titer_data = df[TITER_COLS].dropna()
    corr = titer_data.corr(method='spearman')
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                mask=mask, ax=ax, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Spearman Correlation — 9 Titer Columns', fontsize=14, fontweight='bold')
    save_plot(fig, 'titer_correlation_heatmap.png')
    
    # ─── 9. Age vs Typhoid Boxplot ───
    print("[9/14] Age vs Typhoid boxplot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    data_for_box = [df[df['Typhoid'] == i]['Age_years'].dropna() for i in [0, 1, 2]]
    bp = ax.boxplot(data_for_box, labels=['Negative', 'Minimal', 'Positive'],
                    patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], [TYPHOID_COLORS[0], TYPHOID_COLORS[1], TYPHOID_COLORS[2]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel('Typhoid Outcome')
    ax.set_ylabel('Age (years)')
    ax.set_title('Age Distribution by Typhoid Outcome', fontsize=14, fontweight='bold')
    save_plot(fig, 'age_vs_typhoid_boxplot.png')
    
    # ─── 10. Gender vs Outcomes ───
    print("[10/14] Gender vs outcomes...")
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for ax_i, target in zip(axes, targets):
        ct = pd.crosstab(df['Gender'], df[target], normalize='index') * 100
        ct.index = ['Female', 'Male']
        ct.plot(kind='bar', stacked=True, ax=ax_i, color=['#2ecc71', '#f39c12', '#e74c3c'][:len(ct.columns)],
                edgecolor='white', width=0.5)
        ax_i.set_title(target, fontweight='bold', fontsize=10)
        ax_i.set_ylabel('Percentage')
        ax_i.set_xticklabels(ax_i.get_xticklabels(), rotation=0)
        ax_i.legend(fontsize=7, title='Class')
    fig.suptitle('Gender Breakdown by Diagnosis Outcome', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, 'gender_vs_outcomes.png')
    
    # ─── 11. Diagnosis Co-occurrence ───
    print("[11/14] Diagnosis co-occurrence heatmap...")
    fig, ax = plt.subplots(figsize=(8, 7))
    binary_targets = ['Acute_typhoid', 'Paratyphoid_A', 'Paratyphoid_B', 'Rickettsia_Suspect']
    typhoid_pos = (df['Typhoid'] == 2).astype(int)
    cooc_df = df[binary_targets].copy()
    cooc_df['Typhoid_Positive'] = typhoid_pos
    cooc_cols = ['Typhoid_Positive'] + binary_targets
    cooc = pd.DataFrame(np.zeros((len(cooc_cols), len(cooc_cols))), 
                        index=cooc_cols, columns=cooc_cols)
    for i, c1 in enumerate(cooc_cols):
        for j, c2 in enumerate(cooc_cols):
            mask_both = (cooc_df[c1] == 1) & (cooc_df[c2] == 1)
            cooc.iloc[i, j] = mask_both.sum()
    
    sns.heatmap(cooc.astype(int), annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                linewidths=0.5, square=True)
    ax.set_title('Diagnosis Co-occurrence Matrix', fontsize=14, fontweight='bold')
    save_plot(fig, 'diagnosis_cooccurrence.png')
    
    # ─── 12. Titer-Outcome Heatmap ───
    print("[12/14] Titer-outcome heatmap...")
    fig, ax = plt.subplots(figsize=(8, 8))
    heatmap_data = df.groupby('Typhoid')[TITER_COLS].mean().T
    heatmap_data.columns = ['Negative (0)', 'Minimal (1)', 'Positive (2)']
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, cbar_kws={'label': 'Mean Titer Value'})
    ax.set_title('Mean Titer Value by Typhoid Outcome', fontsize=14, fontweight='bold')
    ax.set_ylabel('Antigen')
    save_plot(fig, 'titer_outcome_heatmap.png')
    
    # ─── 13. Age Group vs Diagnosis Rate ───
    print("[13/14] Age group vs diagnosis rate...")
    bins = [0, 2, 14, 59, 120]
    labels_age = ['Infant (<2y)', 'Child (2-14y)', 'Adult (15-59y)', 'Elderly (60y+)']
    df['Age_group'] = pd.cut(df['Age_years'], bins=bins, labels=labels_age, right=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rates = df.groupby('Age_group', observed=False).apply(
        lambda x: (x['Typhoid'] == 2).mean() * 100
    )
    bars = ax.bar(range(len(rates)), rates.values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
                  edgecolor='white', width=0.6)
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(labels_age)
    ax.set_ylabel('Typhoid Positive Rate (%)')
    ax.set_title('Typhoid Positive Rate by Age Group', fontsize=14, fontweight='bold')
    for bar, rate in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{rate:.1f}%', ha='center', fontweight='bold')
    df.drop(columns=['Age_group'], inplace=True)
    save_plot(fig, 'age_group_vs_diagnosis_rate.png')
    
    # ─── 14. PCA Scatter ───
    print("[14/14] PCA scatter...")
    fig, ax = plt.subplots(figsize=(10, 8))
    titer_data_pca = df[TITER_COLS].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(titer_data_pca)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    for typhoid_val, color in TYPHOID_COLORS.items():
        mask = df['Typhoid'] == typhoid_val
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, 
                   label=TYPHOID_LABELS[typhoid_val], alpha=0.6, s=30, edgecolors='white', linewidth=0.3)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    ax.set_title('PCA of 9 Titer Features — Colored by Typhoid Outcome', fontsize=14, fontweight='bold')
    ax.legend()
    save_plot(fig, 'pca_scatter.png')
    
    print("\n" + "=" * 60)
    print("EDAAgent COMPLETE — All 14 plots saved to outputs/plots/")
    print("=" * 60)
    
    # Print human review checklist
    print("\n📋 HUMAN REVIEW CHECKPOINT:")
    print("  1. Check titer_gradient_vs_typhoid.png — higher titers should show higher Positive %")
    print("  2. Check missing_data_matrix.png — Rickettsia columns should have clean NaN block")
    print("  3. Check typhoid_class_distribution.png — Positive should be ~7%")
    print("  → If all pass, proceed to Phase 4 (ModelAgent)")


if __name__ == '__main__':
    run_eda()
