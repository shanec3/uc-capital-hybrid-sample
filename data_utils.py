import pandas as pd
import numpy as np

def _parse_cols(s: str) -> list[int]:
    """Parse a comma-separated string of integers into a list."""
    return [] if pd.isna(s) or s in ("", "None") else list(map(int, s.split(",")))

def _to_num(col: pd.Series) -> pd.Series:
    """Convert a pandas Series to numeric, stripping non-numeric characters."""
    return pd.to_numeric(col.astype(str).str.replace(r"[^\d\.\-+]", "", regex=True), errors="coerce") 