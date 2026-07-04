# .agents/skills/age_parser.py

import pandas as pd
import re

def parse_age_to_years(age_series: pd.Series) -> pd.Series:
    """
    Parses mixed-format age strings to decimal years.
    
    Handles: years (y), months (m), days (d), hours (h)
    Strips: leading/trailing spaces, internal spaces between number and unit.
    
    Examples:
        '5y'    -> 5.0
        '3.5y'  -> 3.5
        '10m'   -> 0.833
        '25d'   -> 0.068
        '9h'    -> 0.001
        ' 25y'  -> 25.0   (leading space handled)
        '52 y'  -> 52.0   (internal space handled)
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
    
    print(f"[age_parser] Age range after parsing: {parsed.min():.3f}y - {parsed.max():.1f}y")
    return parsed
