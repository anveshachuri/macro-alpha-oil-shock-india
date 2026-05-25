"""
strategy.py
Cross-sectional portfolio construction.
Long top-K% predicted return stocks, short bottom-K% — dollar-neutral.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────

TOP_K          = 0.50   # long top 30% of predicted returns
BOTTOM_K       = 0.50   # short bottom 30%
REBALANCE_DAYS = 1      # rebalance every N days
MAX_SINGLE_POS = 0.40   # max weight per stock (concentration cap)
TRANSACTION_COST = 0.0015  # 15bps round-trip per leg


# ─────────────────────────────────────────────
# SIGNAL → WEIGHTS
# ─────────────────────────────────────────────

def rank_signals(
    predicted_returns: pd.Series,
    top_k: float = TOP_K,
    bottom_k: float = BOTTOM_K,
) -> pd.Series:
    """
    Convert predicted returns into {-1, 0, +1} position signals.
    Ranks are computed within each date's cross-section.

    Parameters
    ----------
    predicted_returns : pd.Series with MultiIndex (date, stock)

    Returns
    -------
    pd.Series with same index, values in {-1, 0, +1}
    """
    signals = pd.Series(0.0, index=predicted_returns.index)

    for date, group in predicted_returns.groupby(level="date"):
        n = len(group)
        if n < 3:
            continue

        ranked = group.rank(method="first", pct=True)  # 👈 FIX 1 (important)

        long_idx  = ranked[ranked >= (1 - top_k)].index
        short_idx = ranked[ranked <= bottom_k].index

        if len(long_idx) == 0 or len(short_idx) == 0:
            continue

        signals.loc[long_idx]  = +1.0
        signals.loc[short_idx] = -1.0

    
    return signals


def signal_weights(
    signals: pd.Series,
    max_pos: float = MAX_SINGLE_POS,
) -> pd.Series:
    """
    Convert {-1, 0, +1} signals into dollar-neutral portfolio weights.
    Each leg (long/short) sums to ±1 in aggregate.
    Weights are capped at `max_pos` per stock.
    """
    weights = pd.Series(0.0, index=signals.index)

    for date, group in signals.groupby(level="date"):
        longs  = group[group > 0]
        shorts = group[group < 0]

        # Skip invalid cases safely
        if len(longs) == 0 or len(shorts) == 0:
            continue

        # Equal weight within each leg, then cap
        if len(longs) > 0:
            long_w = pd.Series(1.0 / len(longs), index=longs.index)
            weights.loc[long_w.index] = long_w

        if len(shorts) > 0:
            short_w = pd.Series(-1.0 / len(shorts), index=shorts.index)
            weights.loc[short_w.index] = short_w

        weights.loc[long_w.index]  = long_w
        weights.loc[short_w.index] = short_w

    return weights


# ─────────────────────────────────────────────
# PORTFOLIO CONSTRUCTION
# ─────────────────────────────────────────────

def build_portfolio(
    predicted_returns: pd.Series,
    actual_returns: pd.DataFrame,
    top_k: float   = TOP_K,
    bottom_k: float = BOTTOM_K,
    hold_days: int = 3,
    tc: float      = TRANSACTION_COST,
) -> pd.DataFrame:
    """
    For each event date's cross-section:
      1. Rank predicted returns → long/short signals
      2. Compute equal-weight portfolio PnL over hold_days
      3. Deduct transaction costs

    Returns a DataFrame with one row per (date, stock) position.
    """
    signals = rank_signals(predicted_returns, top_k, bottom_k)
    weights = signal_weights(signals)

    rows = []
    for date, wgroup in weights.groupby(level="date"):
        date_idx = actual_returns.index.searchsorted(date)
        if date_idx + hold_days >= len(actual_returns):
            continue

        hold_window = actual_returns.iloc[date_idx + 1 : date_idx + 1 + hold_days]

        for (d, stock), weight in wgroup.items():
            if weight == 0 or stock not in actual_returns.columns:
                continue

            gross_ret = hold_window[stock].sum()     # sum of log returns
            net_ret   = gross_ret * weight - abs(weight) * tc

            pred_ret  = predicted_returns.loc[(d, stock)] if (d, stock) in predicted_returns.index else 0.0

            rows.append({
                "date":             date,
                "stock":            stock,
                "signal":           int(np.sign(weight)),
                "weight":           round(weight, 4),
                "predicted_return": round(pred_ret * 100, 4),
                "gross_return":     round(gross_ret * 100, 4),
                "net_return":       round(net_ret  * 100, 4),
                "hit":              (gross_ret * weight) > 0,
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# PORTFOLIO-LEVEL AGGREGATION
# ─────────────────────────────────────────────

def daily_pnl(positions: pd.DataFrame) -> pd.Series:
    """
    Aggregate to daily portfolio PnL (sum of net_return across all positions).
    Equal-weight long/short, so sum is approximately dollar-neutral return.
    """
    return (
        positions
        .groupby("date")["net_return"]
        .sum()
        .rename("portfolio_pnl")
    )


def long_short_attribution(positions: pd.DataFrame) -> pd.DataFrame:
    """Split PnL into long leg and short leg contributions."""
    positions = positions.copy()
    positions["leg"] = positions["signal"].map({1: "long", -1: "short"})
    attr = (
        positions
        .groupby(["date", "leg"])["net_return"]
        .sum()
        .unstack("leg")
        .fillna(0)
    )
    attr["total"] = attr.get("long", 0) + attr.get("short", 0)
    return attr


def portfolio_summary(positions: pd.DataFrame) -> Dict:
    """Quick scalar summary of the full backtest."""
    pnl_series = daily_pnl(positions)
    pnls = pnl_series.values / 100  # back to decimals

    n = len(pnls)
    if n == 0:
        return {}

    mean_ret = pnls.mean()
    std_ret  = pnls.std(ddof=1) if n > 1 else 1e-9
    sharpe   = mean_ret / std_ret if std_ret > 0 else 0.0

    equity   = np.cumprod(1 + pnls)
    max_dd   = ((equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity)).min()

    return {
        "n_event_dates":    int(positions["date"].nunique()),
        "n_positions":      int(len(positions)),
        "hit_ratio":        round(float(positions["hit"].mean()), 4),
        "mean_daily_pnl":   round(mean_ret * 100, 3),
        "sharpe_daily":     round(sharpe, 3),
        "sharpe_ann":       round(sharpe * np.sqrt(252), 3),
        "max_drawdown":     round(max_dd * 100, 3),
        "total_return":     round((equity[-1] - 1) * 100, 2),
        "long_hit_ratio":   round(float(positions[positions["signal"]==1]["hit"].mean()), 4),
        "short_hit_ratio":  round(float(positions[positions["signal"]==-1]["hit"].mean()), 4),
        "top_long_stock":   positions[positions["signal"]==1].groupby("stock")["net_return"].mean().idxmax() if len(positions[positions["signal"]==1]) else "N/A",
        "top_short_stock":  positions[positions["signal"]==-1].groupby("stock")["net_return"].mean().idxmin() if len(positions[positions["signal"]==-1]) else "N/A",
    }
