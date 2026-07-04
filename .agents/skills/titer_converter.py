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
