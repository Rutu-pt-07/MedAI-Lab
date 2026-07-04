"""
Phase 2: DataEngineer Agent
===========================
Transforms raw clinical dataset into clean, analysis-ready DataFrame.
Applies all biological logic constraints from medical_logic.md.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add skills to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.agents', 'skills'))
from titer_converter import convert_titers, TITER_COLS
from age_parser import parse_age_to_years

def run_data_engineer():
    print("=" * 60)
    print("PHASE 2: DataEngineer Agent")
    print("=" * 60)
    
    # ─── Load raw data ───
    df = pd.read_csv('data/raw/Original_Dataset.csv')
    initial_rows = len(df)
    print(f"\n[1] Loaded raw dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # ─── Task 1: Strip column names ───
    df.columns = df.columns.str.strip()
    print(f"[2] Stripped column names. Columns: {list(df.columns)}")
    
    # ─── Task 2: Drop Name column ───
    df.drop(columns=['Name'], inplace=True)
    print(f"[3] Dropped encrypted Name column. Remaining: {df.shape[1]} columns")
    
    # ─── Task 3: Parse Age ───
    df['Age_years'] = parse_age_to_years(df['Age'])
    df.drop(columns=['Age'], inplace=True)
    
    # Validate age range
    valid_ages = df['Age_years'].dropna()
    assert valid_ages.between(0, 120).all(), "Age out of plausible human range"
    print(f"[4] Parsed Age -> Age_years. Range: {valid_ages.min():.3f}y - {valid_ages.max():.1f}y")
    
    # ─── Task 4: Rickettsia flag (BEFORE any NaN handling) ───
    df['rickettsia_panel_conducted'] = df['OX2'].notna().astype(int)
    n_tested = df['rickettsia_panel_conducted'].sum()
    n_not_tested = len(df) - n_tested
    assert df['rickettsia_panel_conducted'].isin([0, 1]).all()
    print(f"[5] Created rickettsia_panel_conducted flag: {n_tested} tested / {n_not_tested} not tested")
    
    # ─── Task 5: Convert titers ───
    df = convert_titers(df)
    
    # ─── Task 6: Fill Rickettsia NaNs for untested patients ───
    rickettsia_titers = ['OX2', 'OXK', 'OX9', 'A', 'M']
    mask_not_tested = df['rickettsia_panel_conducted'] == 0
    df.loc[mask_not_tested, rickettsia_titers] = df.loc[mask_not_tested, rickettsia_titers].fillna(0)
    print(f"[6] Filled Rickettsia NaNs for {mask_not_tested.sum()} untested patients with 0")
    
    # ─── Task 7: Encode Gender ───
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    print(f"[7] Encoded Gender: Male=1, Female=0. Distribution: {df['Gender'].value_counts().to_dict()}")
    
    # ─── Task 8: Handle corrupt rows ───
    # Rows with NaN across TO/TH and all diagnosis columns
    widal_cols = ['TO', 'TH']
    diag_cols = ['Typhoid', 'Acute_typhoid', 'Paratyphoid_A', 'Paratyphoid_B', 'Rickettsia_Suspect']
    
    # Find rows where TO and TH are NaN (no Widal test at all) AND diagnosis columns are NaN
    corrupt_mask = df[['TO', 'TH', 'Typhoid']].isna().all(axis=1)
    n_corrupt = corrupt_mask.sum()
    
    os.makedirs('data/processed', exist_ok=True)
    df[corrupt_mask].to_csv('data/processed/flagged_corrupt_rows.csv', index=True)
    df = df[~corrupt_mask].reset_index(drop=True)
    print(f"[8] Isolated {n_corrupt} corrupt rows → flagged_corrupt_rows.csv. Remaining: {len(df)} rows")
    
    # ─── Task 9: Encode targets ───
    df['Typhoid'] = df['Typhoid'].map({'Negative': 0, 'Minimal': 1, 'Positive': 2})
    for col in ['Acute_typhoid', 'Paratyphoid_A', 'Paratyphoid_B', 'Rickettsia_Suspect']:
        df[col] = df[col].map({'No': 0, 'Yes': 1})
    print(f"[9] Encoded targets. Typhoid distribution: {df['Typhoid'].value_counts().sort_index().to_dict()}")
    
    # ─── Task 10: Save cleaned data ───
    df.to_csv('data/processed/cleaned_dataset.csv', index=False)
    print(f"[10] Saved cleaned dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # ─── Task 11: Write preprocessing report ───
    write_preprocessing_report(df, initial_rows, n_corrupt, n_tested, n_not_tested)
    
    print("\n" + "=" * 60)
    print("DataEngineer Agent COMPLETE")
    print("=" * 60)
    return df


def write_preprocessing_report(df, initial_rows, n_corrupt, n_tested, n_not_tested):
    """Generate a detailed preprocessing report."""
    report = f"""# Preprocessing Report

## Row Counts
| Stage | Count |
|-------|-------|
| Raw dataset | {initial_rows} |
| Corrupt rows removed | {n_corrupt} |
| Final cleaned dataset | {len(df)} |

## Column Data Types
| Column | Dtype |
|--------|-------|
"""
    for col in df.columns:
        report += f"| {col} | {df[col].dtype} |\n"
    
    report += f"""
## NaN Counts (Post-Cleaning)
| Column | NaN Count | % Missing |
|--------|-----------|-----------|
"""
    for col in df.columns:
        n_nan = df[col].isna().sum()
        pct = 100 * n_nan / len(df)
        report += f"| {col} | {n_nan} | {pct:.1f}% |\n"
    
    report += f"""
## Rickettsia Panel Breakdown
| Status | Count |
|--------|-------|
| Panel conducted | {n_tested} |
| Panel NOT conducted | {n_not_tested} |

## Age Statistics
| Stat | Value |
|------|-------|
| Min | {df['Age_years'].min():.3f} years |
| Max | {df['Age_years'].max():.1f} years |
| Mean | {df['Age_years'].mean():.1f} years |
| Median | {df['Age_years'].median():.1f} years |

## Titer Value Distributions
"""
    titer_cols = ['TO', 'TH', 'AH', 'BH', 'OX2', 'OXK', 'OX9', 'A', 'M']
    for col in titer_cols:
        if col in df.columns:
            vc = df[col].value_counts().sort_index().to_dict()
            report += f"\n### {col}\n| Value | Count |\n|-------|-------|\n"
            for val, cnt in vc.items():
                label = {0: '0 (not tested)', 1: '1:80', 2: '1:160', 3: '1:320'}.get(val, str(val))
                report += f"| {label} | {cnt} |\n"
    
    report += f"""
## Target Variable Distributions

### Typhoid (Primary Target)
| Class | Count | % |
|-------|-------|---|
"""
    for val, label in [(0, 'Negative'), (1, 'Minimal'), (2, 'Positive')]:
        cnt = (df['Typhoid'] == val).sum()
        pct = 100 * cnt / len(df)
        report += f"| {label} ({val}) | {cnt} | {pct:.1f}% |\n"
    
    for target in ['Acute_typhoid', 'Paratyphoid_A', 'Paratyphoid_B', 'Rickettsia_Suspect']:
        report += f"\n### {target}\n| Class | Count |\n|-------|-------|\n"
        vc = df[target].value_counts().sort_index()
        for val, cnt in vc.items():
            if pd.notna(val):
                report += f"| {int(val)} | {cnt} |\n"
        n_nan = df[target].isna().sum()
        if n_nan > 0:
            report += f"| NaN | {n_nan} |\n"
    
    with open('data/processed/preprocessing_report.md', 'w') as f:
        f.write(report)
    print(f"[11] Saved preprocessing report → data/processed/preprocessing_report.md")


if __name__ == '__main__':
    run_data_engineer()
