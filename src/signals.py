"""
signals.py
Converts event-study findings into concrete trading signals.

Signal logic:
  - TRIGGER : Brent log-return > SHOCK_THRESHOLD over DETECTION_WINDOW days
  - ENTRY   : Close of Day 0 (same-day trigger) OR open of Day +1
  - EXIT    : Close of Day +EXIT_HOLD
  - LONGS   : OIL_GAS  (positive CAR asymmetry confirmed)
  - SHORTS  : AUTO, FMCG (negative CAR, statistically consistent)

Schema note
-----------
Sector names (OIL_GAS, AUTO, FMCG) are resolved to stock lists via SECTOR_MAP
imported from constants.py.  The returns DataFrame is expected to contain stock-
level columns; backtest_signals() handles the expansion automatically.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from constants import SECTOR_MAP, OIL_COL
# ─────────────────────────────────────────────
# STRATEGY PARAMETERS  (tune here)
# ─────────────────────────────────────────────

SHOCK_THRESHOLD  = 0.03   # Brent must move > 3% over detection window
DETECTION_WINDOW = 2      # rolling days for shock detection
ENTRY_DAY        = 1      # 0 = same-day close, 1 = next-day open proxy
EXIT_HOLD        = 3      # hold for N days after entry
# Long OIL_GAS on up-shocks (reserve revaluation + revenue pass-through).
# IT excluded: FX tailwind theory is economically sound but not significant
# in the 2019-2024 sample (7 up-shock events, low statistical power).
# Re-test with pre-2019 data before adding IT to live signals.
LONG_SECTORS     = ["OIL_GAS"]
SHORT_SECTORS    = ["AUTO", "FMCG"]
TRANSACTION_COST = 0.0010  # 10bps round-trip per leg


# ─────────────────────────────────────────────
# SIGNAL DATACLASS
# ─────────────────────────────────────────────

@dataclass
class Signal:
    trigger_date: pd.Timestamp
    entry_date:   pd.Timestamp
    exit_date:    pd.Timestamp
    brent_move:   float          # % move that triggered the signal
    direction:    str            # "up" | "down"
    longs:        List[str]
    shorts:       List[str]
    # filled after backtest
    pnl:          float = 0.0
    pnl_long:     float = 0.0
    pnl_short:    float = 0.0
    hit:          bool  = False
    details:      Dict  = field(default_factory=dict)


# ─────────────────────────────────────────────
# SHOCK DETECTOR
# ─────────────────────────────────────────────

def detect_shocks(
    returns: pd.DataFrame,
    threshold: float = SHOCK_THRESHOLD,
    window: int      = DETECTION_WINDOW,
    col: str         = "BRENT",
    cooldown: int    = 10,          # min trading days between signals
) -> List[Signal]:
    """
    Scan returns for Brent moves exceeding `threshold` over `window` days.
    Returns a list of Signal objects (un-filled, no PnL yet).

    Parameters
    ----------
    cooldown : int
        Minimum days between consecutive signals (prevents clustering).
    """
    if col not in returns.columns:
        raise ValueError(f"Column '{col}' not found. Available: {list(returns.columns)}")

    brent_cum = returns[col].rolling(window).sum()  # cumulative log-return
    signals   = []
    last_sig  = -cooldown  # day index of last signal

    for i, (dt, cum_ret) in enumerate(brent_cum.items()):
        if pd.isna(cum_ret):
            continue
        if abs(cum_ret) < threshold:
            continue
        if i - last_sig < cooldown:
            continue

        direction = "up" if cum_ret > 0 else "down"

        # Entry: entry_day trading days after trigger
        entry_idx = min(i + ENTRY_DAY, len(returns) - 1)
        exit_idx  = min(i + ENTRY_DAY + EXIT_HOLD, len(returns) - 1)

        sig = Signal(
            trigger_date = dt,
            entry_date   = returns.index[entry_idx],
            exit_date    = returns.index[exit_idx],
            brent_move   = round(cum_ret * 100, 2),
            direction    = direction,
            longs        = LONG_SECTORS  if direction == "up" else SHORT_SECTORS,
            shorts       = SHORT_SECTORS if direction == "up" else LONG_SECTORS,
        )
        signals.append(sig)
        last_sig = i

    return signals


# ─────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────

def backtest_signals(
    signals: List[Signal],
    returns: pd.DataFrame,
    tc: float = TRANSACTION_COST,
) -> List[Signal]:
    """
    For each signal, compute PnL over the hold window.

    PnL formula (equal-weight long/short book):
        long_leg  = mean(sector returns over hold window) for longs
        short_leg = -mean(sector returns over hold window) for shorts
        pnl = 0.5 * long_leg + 0.5 * short_leg - tc
    """
    filled = []
    for sig in signals:
        entry_idx = returns.index.searchsorted(sig.entry_date)
        exit_idx  = returns.index.searchsorted(sig.exit_date)

        if exit_idx <= entry_idx or exit_idx >= len(returns):
            continue

        window = returns.iloc[entry_idx:exit_idx + 1]

        # Expand sector names into actual stock columns

        long_cols = []
        for sector in sig.longs:
            if sector in SECTOR_MAP:
                long_cols.extend(SECTOR_MAP[sector])
            elif sector in window.columns:
                long_cols.append(sector)

        short_cols = []
        for sector in sig.shorts:
            if sector in SECTOR_MAP:
                short_cols.extend(SECTOR_MAP[sector])
            elif sector in window.columns:
                short_cols.append(sector)

        # Keep only columns actually present
        long_cols = [c for c in long_cols if c in window.columns]
        short_cols = [c for c in short_cols if c in window.columns]

        if not long_cols or not short_cols:
            continue

        pnl_long  = window[long_cols].sum(axis=0).mean()   # sum over days, mean over sectors
        pnl_short = -window[short_cols].sum(axis=0).mean()

        gross_pnl = 0.5 * pnl_long + 0.5 * pnl_short
        net_pnl   = gross_pnl - tc

        sig.pnl       = round(net_pnl   * 100, 4)
        sig.pnl_long  = round(pnl_long  * 100, 4)
        sig.pnl_short = round(pnl_short * 100, 4)
        sig.hit       = net_pnl > 0
        sig.details   = {
            "entry_idx": entry_idx,
            "exit_idx":  exit_idx,
            "hold_days": exit_idx - entry_idx,
        }
        filled.append(sig)

    return filled


# ─────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────

def performance_metrics(signals: List[Signal]) -> Dict:
    """Compute Sharpe, hit ratio, max drawdown, CAGR-equivalent."""
    if not signals:
        return {}

    pnls = np.array([s.pnl / 100 for s in signals])   # back to decimals

    n          = len(pnls)
    mean_pnl   = pnls.mean()
    std_pnl    = pnls.std(ddof=1) if n > 1 else 1e-9
    hit_ratio  = (pnls > 0).mean()

    # Trade-level Sharpe (not annualised — valid for event strategies)
    sharpe_trade = mean_pnl / std_pnl if std_pnl > 0 else 0.0

    # Annualised Sharpe: assume ~12 trades/year (rough), but compute from signal dates
    days_total  = (signals[-1].exit_date - signals[0].entry_date).days
    trades_py   = n / max(days_total / 365.25, 1)
    sharpe_ann  = sharpe_trade * np.sqrt(trades_py)

    # Equity curve & drawdown
    equity  = np.cumprod(1 + pnls)
    running_max = np.maximum.accumulate(equity)
    drawdown    = (equity - running_max) / running_max
    max_dd      = drawdown.min()

    # Total return and CAGR
    total_ret = equity[-1] - 1
    years     = days_total / 365.25
    cagr      = (equity[-1] ** (1 / max(years, 1))) - 1

    return {
        "n_trades":       n,
        "hit_ratio":      round(hit_ratio,   4),
        "mean_pnl_pct":   round(mean_pnl * 100, 3),
        "std_pnl_pct":    round(std_pnl  * 100, 3),
        "sharpe_trade":   round(sharpe_trade, 3),
        "sharpe_ann":     round(sharpe_ann,   3),
        "max_drawdown":   round(max_dd * 100, 3),
        "total_return":   round(total_ret * 100, 2),
        "cagr":           round(cagr * 100, 2),
        "best_trade":     round(pnls.max()  * 100, 3),
        "worst_trade":    round(pnls.min()  * 100, 3),
        "profit_factor":  round(pnls[pnls > 0].sum() / abs(pnls[pnls < 0].sum()), 3) if (pnls < 0).any() else float("inf"),
    }


def signals_to_df(signals: List[Signal]) -> pd.DataFrame:
    """Convert list of Signal objects to a tidy DataFrame."""
    rows = []
    for s in signals:
        rows.append({
            "trigger_date": s.trigger_date,
            "entry_date":   s.entry_date,
            "exit_date":    s.exit_date,
            "brent_move":   s.brent_move,
            "direction":    s.direction,
            "longs":        "+".join(s.longs),
            "shorts":       "-".join(s.shorts),
            "pnl_pct":      s.pnl,
            "pnl_long":     s.pnl_long,
            "pnl_short":    s.pnl_short,
            "hit":          s.hit,
        })
    return pd.DataFrame(rows)
