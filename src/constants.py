"""
constants.py
Single source of truth for all schema-level names and mappings.

WHY THIS FILE EXISTS
--------------------
The original project was designed around synthetic sector-index columns
(OIL_GAS, AUTO, FMCG ...).  The live data pipeline delivers stock-level
columns (RELIANCE, TATAMOTORS ...).  Every module that needs sector-level
aggregation imports SECTOR_MAP from here so the mapping is never duplicated
or silently inconsistent.

Column name constants (MARKET_COL, OIL_COL, FX_COL) are also defined once
here to prevent NIFTY / NIFTY50 and similar mismatches cascading.
"""

from typing import Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN NAME CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MARKET_COL: str = "NIFTY"    # broad-market benchmark in returns df
OIL_COL:    str = "BRENT"    # Brent crude column
FX_COL:     str = "USDINR"   # INR/USD  (higher = weaker INR)

# ─────────────────────────────────────────────────────────────────────────────
# SECTOR → STOCK MAPPING
# ─────────────────────────────────────────────────────────────────────────────

SECTOR_MAP: Dict[str, List[str]] = {
    "OIL_GAS": ["RELIANCE", "ONGC", "IOC", "BPCL", "HPCL"],
    "AUTO":    ["TATAMOTORS", "MARUTI"],
    "FMCG":    ["HINDUNILVR", "ITC"],
    "IT":      ["TCS", "INFY"],
    "PHARMA":  ["CIPLA"],
}

SECTORS: List[str] = ["OIL_GAS", "AUTO", "FMCG", "IT", "PHARMA"]

# ─────────────────────────────────────────────────────────────────────────────
# STOCK UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────

STOCK_UNIVERSE: List[str] = [
    "RELIANCE", "ONGC", "IOC", "BPCL", "INDIGO", "HPCL",
    "ADANIPORTS", "TATAMOTORS", "MARUTI", "ASIANPAINT",
    "HINDUNILVR", "ITC", "TCS", "INFY", "CIPLA",
]

NON_STOCK_COLS: List[str] = [MARKET_COL, OIL_COL, FX_COL]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def available_stocks(columns) -> List[str]:
    """Return STOCK_UNIVERSE members present in `columns`."""
    col_set = set(columns)
    return [s for s in STOCK_UNIVERSE if s in col_set]


def build_sector_returns(returns, sector_map: Dict[str, List[str]] = None):
    """
    Compute equal-weight sector returns from a stock-level returns DataFrame.

    Returns a DataFrame with one column per sector plus macro passthrough
    columns (MARKET_COL, OIL_COL, FX_COL).

    Raises KeyError with a clear message if any sector has no available stocks.
    """
    import pandas as pd

    sm = sector_map or SECTOR_MAP
    out = {}

    for sector, stocks in sm.items():
        avail = [s for s in stocks if s in returns.columns]
        if not avail:
            raise KeyError(
                f"Sector '{sector}': none of {stocks} found in returns "
                f"(available: {list(returns.columns)})"
            )
        out[sector] = returns[avail].mean(axis=1)

    for col in [MARKET_COL, OIL_COL, FX_COL]:
        if col in returns.columns:
            out[col] = returns[col]

    return pd.DataFrame(out, index=returns.index)


def validate_returns(returns, required_cols=None, context=""):
    """
    Raise an informative ValueError if required columns are missing.

    Parameters
    ----------
    required_cols : list, optional
        Defaults to [MARKET_COL, OIL_COL, FX_COL].
    context : str
        Module/function name for the error message.
    """
    if required_cols is None:
        required_cols = [MARKET_COL, OIL_COL, FX_COL]

    missing = [c for c in required_cols if c not in returns.columns]
    if missing:
        raise ValueError(
            f"[{context}] Missing required columns {missing}. "
            f"Available: {list(returns.columns)}"
        )
