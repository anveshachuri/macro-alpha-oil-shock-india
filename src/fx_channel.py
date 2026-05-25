"""
fx_channel.py
Cross-asset macro signal extraction — INR/USD transmission channel.

Theory
------
India imports ~85% of its crude oil.  A Brent supply shock therefore widens
the current-account deficit almost mechanically, putting depreciation pressure
on the INR against the USD.  This creates a *second* cross-sectoral signal on
top of the direct cost shock:

    Oil supply shock → Brent ↑ → CAD widens → INR depreciates (USDINR ↑)

Sector FX sensitivity (USD-revenue vs INR-cost structure):

    IT  (TCS, INFY):          ~65-70% USD revenues, INR cost base
                               → INR depreciation = revenue TAILWIND  (fx_beta > 0)
    OIL_GAS (ONGC, IOC):      crude priced in USD, domestic sales INR
                               → partial hedge; reserve revaluation dominant (fx_beta ≈ 0..+0.2)
    AUTO (MARUTI, TATAMOTORS): imported steel / components priced in USD
                               → INR depreciation = input cost HEADWIND  (fx_beta < 0)
    FMCG (HINDUNILVR, ITC):    palm oil, packaging imports priced in USD
                               → INR depreciation = input cost HEADWIND  (fx_beta < 0)

Key output: dual-channel decomposition
    sector_AR ≈ β_oil · Δoil  +  β_fx · Δusdinr  +  ε
    → β_fx quantifies how much of the post-shock sector move is *FX-driven*
    → IT has positive β_fx: oil-neutral + FX tailwind = compounding long signal
    → AUTO/FMCG have negative β_fx: cost shock + FX headwind = compounding short signal

Public API
----------
fx_event_study(returns, events, pre, post)
    → Event-window average of USDINR returns around oil shocks.

decompose_sector_returns(returns, stock_cols, oil_col, fx_col, window)
    → Rolling OLS: AR ~ β_oil · oil_ret + β_fx · usdinr_ret
      Returns dict {stock: DataFrame of daily betas}.

fx_channel_summary(returns, events, stock_cols)
    → Summary DataFrame: mean β_oil, mean β_fx, FX-channel CAR per stock.

dual_channel_signal(returns, events, stock_cols)
    → Composite score: oil_channel + fx_amplification per event.
      Positive = long candidate, negative = short candidate.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────

OIL_COL   = "BRENT"
FX_COL    = "USDINR"   # Yahoo Finance "INR=X": higher = INR weaker = depreciation

# Stocks grouped by expected FX sensitivity
FX_BETA_PRIORS = {
    # IT — strong USD revenue, weak INR cost → expect positive fx_beta
    "TCS":        +0.8,
    "INFY":       +0.8,
    # OIL_GAS — USD crude, domestic pricing → near-zero to mild positive
    "RELIANCE":   +0.2,
    "ONGC":       +0.1,
    "IOC":        +0.0,
    "BPCL":       +0.0,
    "HPCL":       +0.0,
    # Airlines — jet fuel in USD, domestic revenues in INR → negative
    "INDIGO":     -0.5,
    # Auto — imported components, discretionary demand → negative
    "TATAMOTORS": -0.3,
    "MARUTI":     -0.2,
    # FMCG — palm oil / packaging imports → mildly negative
    "ASIANPAINT": -0.2,
    "HINDUNILVR": -0.2,
    "ITC":        -0.1,
    # Pharma — mixed (API imports USD, formulation exports USD too)
    "CIPLA":      +0.1,
    # Ports — cargo volumes, global trade → mild positive
    "ADANIPORTS": +0.1,
}

STOCK_COLS = list(FX_BETA_PRIORS.keys())


# ─────────────────────────────────────────────
# 1.  FX EVENT STUDY
# ─────────────────────────────────────────────

def fx_event_study(
    returns: pd.DataFrame,
    events: List[Dict],
    pre:  int = 5,
    post: int = 5,
    fx_col: str = FX_COL,
) -> pd.DataFrame:
    """
    Run an event study on the USDINR return series around oil shocks.

    Returns
    -------
    avg_fx_ar : pd.DataFrame
        Columns: ['mean', 'std', 'n', 't_stat', 'p_value', 'cum_mean']
        Index  : relative day (-pre … +post)
    """
    if fx_col not in returns.columns:
        log.warning(f"'{fx_col}' not in returns; FX event study skipped.")
        return pd.DataFrame()

    windows = []
    for ev in events:
        dt  = pd.Timestamp(ev["date"])
        idx = returns.index.searchsorted(dt)
        if idx < pre or idx + post + 1 > len(returns):
            continue
        win = returns[fx_col].iloc[idx - pre : idx + post + 1].values
        if len(win) == pre + post + 1:
            windows.append(win)

    if not windows:
        return pd.DataFrame()

    mat  = np.array(windows)          # shape (n_events, window_len)
    days = range(-pre, post + 1)

    rows = []
    for i, d in enumerate(days):
        col_vals = mat[:, i]
        n  = len(col_vals)
        m  = col_vals.mean()
        sd = col_vals.std(ddof=1) if n > 1 else 0.0
        t, p = stats.ttest_1samp(col_vals, popmean=0)
        rows.append({
            "day":      d,
            "mean":     round(m * 100, 4),   # in pct
            "std":      round(sd * 100, 4),
            "n_events": n,
            "t_stat":   round(t, 3),
            "p_value":  round(p, 4),
        })

    df = pd.DataFrame(rows).set_index("day")
    df["cum_mean"] = df["mean"].cumsum().round(4)
    return df


# ─────────────────────────────────────────────
# 2.  ROLLING OLS DECOMPOSITION
# ─────────────────────────────────────────────

def decompose_sector_returns(
    returns:   pd.DataFrame,
    stock_cols: List[str] = None,
    oil_col:   str = OIL_COL,
    fx_col:    str = FX_COL,
    window:    int = 60,
) -> Dict[str, pd.DataFrame]:
    """
    Estimate rolling 2-factor model for each stock:
        R_stock,t = α + β_oil · R_oil,t  +  β_fx · R_usdinr,t  +  ε_t

    Uses an expanding / rolling OLS over `window` trading days.

    Returns
    -------
    betas : dict {stock_name: DataFrame(date, alpha, beta_oil, beta_fx, r2)}
    """
    if stock_cols is None:
        stock_cols = [c for c in STOCK_COLS if c in returns.columns]

    if oil_col not in returns.columns or fx_col not in returns.columns:
        log.warning("BRENT or USDINR missing — decomposition skipped.")
        return {}

    betas = {}
    for stock in stock_cols:
        if stock not in returns.columns:
            continue

        rows = []
        for i in range(window, len(returns)):
            sl = returns.iloc[i - window : i]
            y  = sl[stock].values
            X  = np.column_stack([
                np.ones(len(sl)),
                sl[oil_col].values,
                sl[fx_col].values,
            ])
            # OLS: β = (X'X)^{-1} X'y
            try:
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                continue

            y_hat = X @ coef
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            rows.append({
                "date":     returns.index[i],
                "alpha":    round(coef[0] * 252, 6),   # annualised alpha
                "beta_oil": round(coef[1], 4),
                "beta_fx":  round(coef[2], 4),
                "r2":       round(r2, 4),
            })

        if rows:
            betas[stock] = pd.DataFrame(rows).set_index("date")

    return betas


# ─────────────────────────────────────────────
# 3.  FX CHANNEL SUMMARY
# ─────────────────────────────────────────────

def fx_channel_summary(
    returns:    pd.DataFrame,
    events:     List[Dict],
    stock_cols: List[str] = None,
    oil_col:    str = OIL_COL,
    fx_col:     str = FX_COL,
    window:     int = 60,
    post:       int = 5,
) -> pd.DataFrame:
    """
    For each stock, compute:
        - Mean β_oil   (direct oil sensitivity, OLS from 60-day rolling window)
        - Mean β_fx    (FX sensitivity — positive = USD-revenue tailwind)
        - FX-channel CAR: β_fx_mean × mean USDINR move post-shock (%)
        - Dual-channel direction: combined oil + FX signal

    Returns a tidy DataFrame indexed by stock, suitable for direct LaTeX/markdown output.
    """
    if stock_cols is None:
        stock_cols = [c for c in STOCK_COLS if c in returns.columns]

    # 1. Rolling betas
    betas = decompose_sector_returns(returns, stock_cols, oil_col, fx_col, window)

    # 2. Average USDINR move post-shock
    fx_study = fx_event_study(returns, events, pre=0, post=post, fx_col=fx_col)
    mean_fx_move = fx_study["mean"].iloc[1:].sum() if not fx_study.empty else 0.0  # post-event window sum

    rows = []
    for stock in stock_cols:
        if stock not in betas:
            continue

        bt = betas[stock]
        # Average betas over all periods (could also filter to event periods)
        b_oil = bt["beta_oil"].mean()
        b_fx  = bt["beta_fx"].mean()

        # FX-channel CAR: how much of sector move is FX-driven?
        fx_car = b_fx * mean_fx_move   # %, fraction of %

        # Prior hypothesis
        prior_dir_fx = "+" if FX_BETA_PRIORS.get(stock, 0) > 0 else (
                       "-" if FX_BETA_PRIORS.get(stock, 0) < 0 else "≈0"
        )

        rows.append({
            "stock":          stock,
            "beta_oil":       round(b_oil, 3),
            "beta_fx":        round(b_fx, 3),
            "fx_car_pct":     round(fx_car, 3),
            "prior_fx_dir":   prior_dir_fx,
            "fx_confirmed":   (b_fx > 0.05 and FX_BETA_PRIORS.get(stock, 0) > 0) or
                              (b_fx < -0.05 and FX_BETA_PRIORS.get(stock, 0) < 0),
        })

    summary = pd.DataFrame(rows).set_index("stock")
    summary = summary.sort_values("beta_fx", ascending=False)
    return summary


# ─────────────────────────────────────────────
# 4.  DUAL-CHANNEL COMPOSITE SIGNAL
# ─────────────────────────────────────────────

def dual_channel_signal(
    returns:    pd.DataFrame,
    events:     List[Dict],
    stock_cols: List[str] = None,
    oil_col:    str = OIL_COL,
    fx_col:     str = FX_COL,
    window:     int = 60,
    post:       int = 3,
) -> pd.DataFrame:
    """
    Combine the oil channel and FX channel into a single composite signal score
    for each event × stock pair.

        score = β_oil × oil_shock_magnitude + β_fx × usdinr_shock_magnitude

    A strongly positive score = long candidate (e.g. IT: low β_oil, high β_fx).
    A strongly negative score = short candidate (e.g. AUTO: negative β_oil, β_fx).

    Returns
    -------
    pd.DataFrame with columns [event_date, stock, beta_oil, beta_fx,
                                oil_shock, fx_shock, oil_score, fx_score,
                                composite_score, signal]
    """
    if stock_cols is None:
        stock_cols = [c for c in STOCK_COLS if c in returns.columns]

    betas = decompose_sector_returns(returns, stock_cols, oil_col, fx_col, window)

    rows = []
    for ev in events:
        dt  = pd.Timestamp(ev["date"])
        idx = returns.index.searchsorted(dt)
        if idx < window or idx + post >= len(returns):
            continue

        # Shock magnitudes at event date
        oil_shock = returns[oil_col].rolling(2).sum().iloc[idx] if oil_col in returns.columns else 0.0
        fx_shock  = returns[fx_col].rolling(2).sum().iloc[idx]  if fx_col  in returns.columns else 0.0

        for stock in stock_cols:
            if stock not in betas:
                continue

            # Get beta as of the event date (most recent estimate ≤ event date)
            bt = betas[stock]
            bt_before = bt.loc[bt.index <= dt]
            if bt_before.empty:
                continue

            b_oil = bt_before["beta_oil"].iloc[-1]
            b_fx  = bt_before["beta_fx"].iloc[-1]

            oil_score = b_oil * oil_shock * 100
            fx_score  = b_fx  * fx_shock  * 100
            comp      = oil_score + fx_score

            rows.append({
                "event_date":      dt.date(),
                "event_label":     ev.get("label", ""),
                "stock":           stock,
                "beta_oil":        round(b_oil, 3),
                "beta_fx":         round(b_fx,  3),
                "oil_shock_pct":   round(oil_shock * 100, 2),
                "fx_shock_pct":    round(fx_shock  * 100, 2),
                "oil_score":       round(oil_score, 3),
                "fx_score":        round(fx_score,  3),
                "composite_score": round(comp,       3),
                "signal":          int(np.sign(comp)) if abs(comp) > 0.01 else 0,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values(["event_date", "composite_score"], ascending=[True, False])


# ─────────────────────────────────────────────
# 5.  REGIME: OIL-FX COUPLING STRENGTH
# ─────────────────────────────────────────────

def oil_fx_regime(
    returns: pd.DataFrame,
    oil_col: str = OIL_COL,
    fx_col:  str = FX_COL,
    window:  int = 60,
) -> pd.DataFrame:
    """
    Compute rolling correlation between Brent and USDINR.

    When the oil-FX correlation is high (tight coupling), the FX channel is
    "live" and the dual-channel signal should be trusted more.
    When the correlation is near zero, oil shocks are being absorbed by
    monetary policy or FX reserves rather than transmitted to currency markets.

    Returns
    -------
    pd.DataFrame with columns [oil_fx_corr_60d, coupling_regime]
        coupling_regime: 'tight' (|corr| > 0.4), 'loose' (|corr| < 0.2), 'moderate'
    """
    if oil_col not in returns.columns or fx_col not in returns.columns:
        return pd.DataFrame()

    corr = returns[oil_col].rolling(window).corr(returns[fx_col])
    df = pd.DataFrame({"oil_fx_corr": corr.round(4)})

    def _regime(c):
        if pd.isna(c):   return "unknown"
        if abs(c) > 0.4: return "tight"
        if abs(c) < 0.2: return "loose"
        return "moderate"

    df["coupling_regime"] = df["oil_fx_corr"].apply(_regime)
    return df.dropna()


# ─────────────────────────────────────────────
# 6.  CONVENIENCE RUNNER
# ─────────────────────────────────────────────

def run_fx_analysis(
    returns:    pd.DataFrame,
    events:     List[Dict],
    stock_cols: List[str] = None,
    output_dir: Optional["Path"] = None,  # type: ignore[name-defined]
) -> Dict:
    """
    End-to-end FX channel analysis.  Returns dict of results DataFrames.
    Optionally saves CSVs to output_dir/tables/.
    """
    from pathlib import Path

    log.info("Running FX channel analysis …")

    fx_study = fx_event_study(returns, events)
    log.info(f"  USDINR event study: {len(fx_study)} day window")

    summary  = fx_channel_summary(returns, events, stock_cols)
    log.info(f"  Beta decomposition: {len(summary)} stocks")

    signals  = dual_channel_signal(returns, events, stock_cols)
    log.info(f"  Dual-channel signals: {len(signals)} rows")

    regime   = oil_fx_regime(returns)
    log.info(f"  Oil-FX coupling regime computed")

    results = {
        "fx_event_study":     fx_study,
        "beta_summary":       summary,
        "dual_channel_signals": signals,
        "oil_fx_regime":      regime,
    }

    if output_dir is not None:
        tables = Path(output_dir) / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        for name, df in results.items():
            if not df.empty:
                df.to_csv(tables / f"fx_{name}.csv")
                log.info(f"  Saved {name} → {tables / f'fx_{name}.csv'}")

    return results
