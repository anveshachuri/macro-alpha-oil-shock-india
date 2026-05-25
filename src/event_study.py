"""
event_study.py
Core implementation of the sectoral event study framework.

Schema note
-----------
Returns DataFrame contains stock-level columns (RELIANCE, ONGC, ...) plus
NIFTY, BRENT, USDINR.  The event study operates on *sector-level* aggregated
returns.  Call `get_sector_returns(returns)` once before passing to the study
functions, or pass the raw returns — all public functions handle both cases
transparently by checking for sector columns.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional

from constants import (
    MARKET_COL, OIL_COL, FX_COL,
    SECTOR_MAP, SECTORS,
    build_sector_returns, validate_returns,
)


# ─────────────────────────────────────────────
# EVENT CATALOGUE
# ─────────────────────────────────────────────

# HISTORICAL EVENTS (2010-2018) — catalogued for completeness.
# Require pre-2019 NSE data. Extend DATA_START in data.py to activate.
# {"date":"2010-12-01","label":"OPEC Freeze","type":"supply_cut","direction":"up"},
# {"date":"2011-02-21","label":"Libya Civil War","type":"supply_cut","direction":"up"},
# {"date":"2011-09-01","label":"Libya Exports Resume","type":"supply_surge","direction":"down"},
# {"date":"2012-07-01","label":"EU Iran Embargo","type":"supply_cut","direction":"up"},
# {"date":"2013-08-29","label":"Syria Strike Threat","type":"supply_cut","direction":"up"},
# {"date":"2014-06-12","label":"ISIS Mosul","type":"supply_cut","direction":"up"},
# {"date":"2014-11-27","label":"OPEC No-Cut","type":"supply_surge","direction":"down"},
# {"date":"2016-01-16","label":"Iran Sanctions Lifted","type":"supply_surge","direction":"down"},
# {"date":"2016-11-30","label":"OPEC Production-Cut Deal","type":"supply_cut","direction":"up"},
# {"date":"2018-05-08","label":"US Exits Iran Deal","type":"supply_cut","direction":"up"},
# {"date":"2018-11-05","label":"US Iran Sanctions Reimposed","type":"supply_surge","direction":"down"},

# ACTIVE EVENT CATALOGUE (2019-2024, covered by current dataset)
OIL_SHOCK_EVENTS = [
    {"date": "2019-09-14", "label": "Saudi Aramco Drone Attacks",        "type": "supply_cut",   "direction": "up"},
    {"date": "2020-03-09", "label": "Saudi-Russia Price War Erupts",     "type": "supply_surge", "direction": "down"},
    {"date": "2020-04-20", "label": "WTI Negative Price Crash",          "type": "demand_crash", "direction": "down"},
    {"date": "2021-03-08", "label": "Suez Canal Blockage",               "type": "supply_cut",   "direction": "up"},
    {"date": "2022-02-24", "label": "Russia-Ukraine Invasion",           "type": "supply_cut",   "direction": "up"},
    {"date": "2022-06-02", "label": "OPEC+ Production Hike",             "type": "supply_surge", "direction": "down"},
    {"date": "2023-04-02", "label": "OPEC+ Surprise Cut",                "type": "supply_cut",   "direction": "up"},
    {"date": "2023-10-07", "label": "Hamas-Israel Conflict Start",       "type": "supply_cut",   "direction": "up"},
    {"date": "2024-01-12", "label": "Houthi Red Sea Attacks Escalate",   "type": "supply_cut",   "direction": "up"},
    {"date": "2024-04-14", "label": "Iran-Israel Direct Strike",         "type": "supply_cut",   "direction": "up"},
]


# ─────────────────────────────────────────────
# SECTOR AGGREGATION
# ─────────────────────────────────────────────

def get_sector_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Convert stock-level returns into sector-level returns (equal-weight).

    If the DataFrame already has sector columns (OIL_GAS, AUTO …), returns it
    unchanged.  This makes all downstream functions schema-agnostic.

    Returns
    -------
    DataFrame with columns: OIL_GAS, AUTO, FMCG, IT, PHARMA, NIFTY, BRENT [, USDINR]
    """
    # Already sector-level?
    if "OIL_GAS" in returns.columns:
        return returns

    # Validate that we have the macro columns we need
    validate_returns(returns, context="get_sector_returns")

    return build_sector_returns(returns)


# ─────────────────────────────────────────────
# RETURN CALCULATIONS
# ─────────────────────────────────────────────

def log_returns(prices: pd.Series) -> pd.Series:
    """Compute daily log returns from a price series."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_all_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns for all columns in price_df."""
    return price_df.apply(log_returns).dropna()


# ─────────────────────────────────────────────
# EVENT WINDOW HELPERS
# ─────────────────────────────────────────────

def get_event_window(
    returns: pd.DataFrame,
    event_date: str,
    pre: int = 5,
    post: int = 5,
) -> Optional[pd.DataFrame]:
    """
    Slice returns around an event date (relative index −pre … +post).
    Returns None if the window falls outside the data range.
    """
    event_dt = pd.Timestamp(event_date)
    idx = returns.index.searchsorted(event_dt)

    if idx < pre or idx + post + 1 > len(returns):
        return None

    window = returns.iloc[idx - pre : idx + post + 1].copy()
    window.index = range(-pre, post + 1)
    return window


# ─────────────────────────────────────────────
# ABNORMAL RETURNS & CAR
# ─────────────────────────────────────────────

def abnormal_returns(
    window: pd.DataFrame,
    market_col: str = MARKET_COL,   # uses constant — never hard-coded
) -> pd.DataFrame:
    """
    AR_it = R_it − R_market_t  (market-adjusted model, β = 1).
    Excludes the market col, BRENT, and USDINR from AR computation.
    """
    exclude = {market_col, OIL_COL, FX_COL}
    cols = [c for c in window.columns if c not in exclude]

    if market_col not in window.columns:
        raise KeyError(
            f"market_col='{market_col}' not found in window columns {list(window.columns)}. "
            f"Did you call get_sector_returns() first?"
        )

    ar = window[cols].subtract(window[market_col], axis=0)
    return ar


def cumulative_ar(ar: pd.DataFrame) -> pd.Series:
    """CAR = sum of ARs over the full event window."""
    return ar.sum(axis=0)


def car_by_day(ar: pd.DataFrame) -> pd.DataFrame:
    """Running cumulative AR for each column across the event window."""
    return ar.cumsum()


# ─────────────────────────────────────────────
# WINSORIZATION
# ─────────────────────────────────────────────

def winsorize_returns(returns: pd.DataFrame, clip_pct: float = 0.10) -> pd.DataFrame:
    """
    Clip extreme Brent (and other) returns at ±clip_pct.

    Removes nonsensical data artefacts (e.g. +152% single-day Brent moves from
    Yahoo Finance roll-date errors) without distorting the distribution.
    Applied only to OIL_COL; other columns clipped at ±50% as a safety net.
    """
    out = returns.copy()
    if OIL_COL in out.columns:
        out[OIL_COL] = out[OIL_COL].clip(lower=-clip_pct, upper=clip_pct)
    # Safety net for all other numeric columns
    other = [c for c in out.columns if c != OIL_COL]
    out[other] = out[other].clip(lower=-0.50, upper=0.50)
    return out


# ─────────────────────────────────────────────
# FULL EVENT LOOP
# ─────────────────────────────────────────────

def run_event_study(
    returns: pd.DataFrame,
    events: List[Dict],
    pre: int = 5,
    post: int = 5,
    market_col: str = MARKET_COL,
    winsorize: bool = True,
) -> Dict:
    """
    Run the full event study on sector-level returns.

    Automatically converts stock-level returns to sector-level via
    get_sector_returns() before processing — callers do not need to pre-aggregate.

    Returns
    -------
    dict with keys:
        ar_all   : list of (event_dict, AR DataFrame)
        car_all  : list of (event_dict, CAR Series)
        avg_ar   : average AR per day across events
        avg_car  : average CAR across events
    """
    # Aggregate to sectors (idempotent if already sector-level)
    sec_returns = get_sector_returns(returns)

    if winsorize:
        sec_returns = winsorize_returns(sec_returns)

    ar_all, car_all = [], []

    for ev in events:
        window = get_event_window(sec_returns, ev["date"], pre, post)
        if window is None:
            continue

        ar  = abnormal_returns(window, market_col)
        car = cumulative_ar(ar)

        if OIL_COL in window.columns:
            ev = {**ev, "brent_window_ret": round(window[OIL_COL].sum() * 100, 2)}

        ar_all.append((ev, ar))
        car_all.append((ev, car))

    if not ar_all:
        raise RuntimeError(
            "No events found within the data range. "
            f"Data: {returns.index.min().date()} – {returns.index.max().date()}. "
            f"Events: {[e['date'] for e in events]}"
        )

    all_ar_dfs = [ar for _, ar in ar_all]
    avg_ar = pd.concat(all_ar_dfs).groupby(level=0).mean()

    car_df  = pd.DataFrame([car for _, car in car_all])
    avg_car = car_df.mean()

    return {
        "ar_all":  ar_all,
        "car_all": car_all,
        "avg_ar":  avg_ar,
        "avg_car": avg_car,
    }


# ─────────────────────────────────────────────
# STATISTICAL TESTS
# ─────────────────────────────────────────────

def t_test_car(car_list: List[Tuple], sector: str) -> Dict:
    """One-sample t-test: H0: CAR = 0 for a given sector across events."""
    values = [car[sector] for _, car in car_list if sector in car.index]
    if len(values) < 2:
        return {"sector": sector, "n_events": len(values), "error": "insufficient data"}

    t_stat, p_val = stats.ttest_1samp(values, popmean=0)
    return {
        "sector":      sector,
        "n_events":    len(values),
        "mean_car":    np.mean(values),
        "mean_car_pct": round(np.mean(values) * 100, 3),
        "std_car":     np.std(values),
        "t_stat":      round(t_stat, 3),
        "p_value":     round(p_val, 4),
        "significant": p_val < 0.05,
    }


def asymmetry_test(
    car_list: List[Tuple[Dict, pd.Series]],
    sector: str,
) -> Dict:
    """Welch's two-sample t-test: up-shock CAR vs down-shock CAR."""
    up   = [car[sector] for ev, car in car_list if ev.get("direction") == "up"   and sector in car.index]
    down = [car[sector] for ev, car in car_list if ev.get("direction") == "down" and sector in car.index]

    if len(up) < 2 or len(down) < 2:
        return {"sector": sector, "error": "insufficient data for asymmetry test",
                "n_up": len(up), "n_down": len(down)}

    t_stat, p_val = stats.ttest_ind(up, down, equal_var=False)
    return {
        "sector":           sector,
        "n_up":             len(up),
        "n_down":           len(down),
        "mean_car_up":      round(np.mean(up),   4),
        "mean_car_up_pct":  round(np.mean(up)   * 100, 3),
        "mean_car_down":    round(np.mean(down), 4),
        "mean_car_down_pct":round(np.mean(down) * 100, 3),
        "t_stat":           round(t_stat, 3),
        "p_value":          round(p_val,  4),
        "asymmetric":       p_val < 0.10,
    }


# ─────────────────────────────────────────────
# OIL SENSITIVITY
# ─────────────────────────────────────────────

def oil_sensitivity(returns: pd.DataFrame, lag: int = 0) -> pd.DataFrame:
    """
    Pearson correlation of Brent returns (possibly lagged) with each column.

    Automatically aggregates to sectors first; excludes NIFTY, BRENT, USDINR
    from the output (they are not sectors).
    """
    if OIL_COL not in returns.columns:
        raise ValueError(f"'{OIL_COL}' not found in returns.")

    sec_returns = get_sector_returns(returns)
    brent = sec_returns[OIL_COL].shift(lag)

    exclude = {MARKET_COL, OIL_COL, FX_COL}
    sectors = [c for c in sec_returns.columns if c not in exclude]

    rows = []
    for sec in sectors:
        valid = pd.concat([brent, sec_returns[sec]], axis=1).dropna()
        r, p  = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
        rows.append({
            "sector":      sec,
            "lag":         lag,
            "corr":        round(r, 4),
            "p_value":     round(p, 4),
            "significant": p < 0.05,
        })

    return pd.DataFrame(rows).set_index("sector")
