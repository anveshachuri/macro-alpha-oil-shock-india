from constants import MARKET_COL, OIL_COL
"""
backtest.py
Realistic backtest engine with:
  - Strict time-based train/test split
  - Slippage model
  - Walk-forward expanding window
  - Full metrics: Sharpe, CAGR, max drawdown, hit ratio, profit factor, IC
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

log = logging.getLogger(__name__)

SLIPPAGE_BPS = 5     # 5bps slippage per trade (market impact)


# ─────────────────────────────────────────────
# SLIPPAGE MODEL
# ─────────────────────────────────────────────

def apply_slippage(gross_return: float, vol: float, slippage_bps: float = SLIPPAGE_BPS) -> float:
    """
    Simple slippage: fixed component + vol-scaled component.
    Higher vol stocks cost more to trade (wider spreads).
    """
    fixed_slip  = slippage_bps / 10_000
    vol_slip    = 0.5 * vol * (1 / 252**0.5)  # ~half daily vol
    total_slip  = fixed_slip + vol_slip
    return gross_return - total_slip


# ─────────────────────────────────────────────
# FULL EVALUATION METRICS
# ─────────────────────────────────────────────

def compute_metrics(pnl_series: pd.Series) -> Dict:
    """
    Full performance metrics from a daily PnL series.
    """
    pnls = pnl_series.dropna().values / 100   # convert pct to decimal

    if len(pnls) < 5:
        return {"error": "insufficient data"}

    n         = len(pnls)
    mean_ret  = pnls.mean()
    std_ret   = pnls.std(ddof=1)
    sharpe    = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0.0

    equity    = np.cumprod(1 + pnls)
    run_max   = np.maximum.accumulate(equity)
    dd        = (equity - run_max) / run_max
    max_dd    = dd.min()

    # CAGR — approximate from total return and duration
    total_ret = equity[-1] - 1
    years     = n / 252
    cagr      = (equity[-1] ** (1 / max(years, 0.1))) - 1

    # Hit ratio
    hits      = (pnls > 0).mean()

    # Profit factor
    wins  = pnls[pnls > 0].sum()
    loses = abs(pnls[pnls < 0].sum())
    pf    = wins / loses if loses > 0 else float("inf")

    # Sortino (downside deviation — root-mean-square of sub-zero returns,
    # NOT std of negative returns only; zeros count in the denominator).
    # Formula mirrors Sharpe: ann_return / ann_downside_dev.
    #   daily_downside_dev = sqrt( mean( min(r, 0)^2 ) )
    #   ann_downside_dev   = daily_downside_dev * sqrt(252)
    #   sortino_ann        = mean_ret * 252 / ann_downside_dev
    #                      = mean_ret / daily_downside_dev * sqrt(252)
    daily_dd = np.sqrt(np.mean(np.minimum(pnls, 0.0) ** 2))
    sortino  = mean_ret / daily_dd * np.sqrt(252) if daily_dd > 0 else 0.0

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "n_periods":     n,
        "hit_ratio":     round(hits, 4),
        "mean_pnl_pct":  round(mean_ret * 100, 4),
        "sharpe_ann":    round(sharpe, 3),
        "sortino_ann":   round(sortino, 3),
        "calmar":        round(calmar, 3),
        "max_drawdown":  round(max_dd * 100, 3),
        "total_return":  round(total_ret * 100, 2),
        "cagr":          round(cagr * 100, 2),
        "profit_factor": round(pf, 3),
        "best_day":      round(pnls.max() * 100, 3),
        "worst_day":     round(pnls.min() * 100, 3),
    }


# ─────────────────────────────────────────────
# INFORMATION COEFFICIENT
# ─────────────────────────────────────────────

def compute_ic(
    predictions: pd.Series,
    actuals: pd.Series,
) -> pd.Series:
    """
    Compute daily IC (Spearman rank correlation of predicted vs actual returns).
    Returns a Series indexed by date.
    """
    ic_rows = []
    dates   = predictions.index.get_level_values("date").unique()

    for date in dates:
        pred = predictions.xs(date, level="date") if date in predictions.index.get_level_values("date") else pd.Series(dtype=float)
        act  = actuals.xs(date, level="date")     if date in actuals.index.get_level_values("date")  else pd.Series(dtype=float)

        common = pred.index.intersection(act.index)
        if len(common) < 3:
            continue

        ic, _ = spearmanr(pred.loc[common], act.loc[common])
        ic_rows.append({"date": date, "ic": ic})

    ic_series = pd.DataFrame(ic_rows).set_index("date")["ic"]
    return ic_series


# ─────────────────────────────────────────────
# ROLLING WALK-FORWARD BACKTEST
# ─────────────────────────────────────────────

def walk_forward_backtest(
    panel: pd.DataFrame,
    returns: pd.DataFrame,
    model_class,
    train_window_months: int = 18,
    test_window_months:  int = 3,
    model_type: str = "ridge",
    top_k: float    = 0.30,
    bottom_k: float = 0.30,
    hold_days: int  = 3,
    tc: float       = 0.0015,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Walk-forward expanding window backtest.

    At each step:
      - Train on all data up to train_end
      - Test on next test_window_months
      - Advance by test_window_months

    Returns (positions_df, overall_metrics)
    """
    from strategy import build_portfolio, daily_pnl, portfolio_summary

    dates = panel.index.get_level_values("date").unique().sort_values()
    all_positions = []

    step_days  = test_window_months  * 21
    train_days = train_window_months * 21

    cursor = train_days
    while cursor < len(dates):
        train_end  = dates[min(cursor - 1, len(dates) - 1)]
        test_start = dates[cursor]
        test_end   = dates[min(cursor + step_days - 1, len(dates) - 1)]

        train_panel = panel.loc[panel.index.get_level_values("date") <= train_end]
        test_panel  = panel.loc[
            (panel.index.get_level_values("date") >= test_start) &
            (panel.index.get_level_values("date") <= test_end)
        ]

        if len(train_panel) < 50 or len(test_panel) < 5:
            cursor += step_days
            continue

        log.info(f"  WF fold: train→{train_end.date()} | test {test_start.date()}→{test_end.date()}")

        model = model_class(model_type=model_type)
        model.fit(train_panel)
        preds = model.predict(test_panel)

        pos = build_portfolio(preds, returns, top_k=top_k, bottom_k=bottom_k,
                              hold_days=hold_days, tc=tc)
        all_positions.append(pos)
        cursor += step_days

    if not all_positions:
        return pd.DataFrame(), {}

    positions = pd.concat(all_positions, ignore_index=True)
    pnl       = daily_pnl(positions)
    metrics   = compute_metrics(pnl)
    metrics.update(portfolio_summary(positions))
    return positions, metrics


# ─────────────────────────────────────────────
# BENCHMARKS
# ─────────────────────────────────────────────

def nifty_benchmark(returns: pd.DataFrame, event_dates: pd.Index) -> Dict:
    """Buy-and-hold NIFTY on event days — the simplest benchmark."""
    if MARKET_COL not in returns.columns:
        return {}
    nifty_event = returns.loc[returns.index.isin(event_dates), MARKET_COL]
    return compute_metrics(nifty_event * 100)


def random_baseline(
    positions: pd.DataFrame,
    n_trials: int = 1000,
    seed: int     = 42,
) -> Dict:
    """
    Random portfolio baseline: randomly assign long/short weights,
    compute average Sharpe over many trials.
    Used to compute strategy's information ratio vs chance.
    """
    np.random.seed(seed)
    sharpes = []
    stocks  = positions["stock"].unique()
    dates   = positions["date"].unique()

    for _ in range(n_trials):
        pnls = []
        for date in dates:
            pool = positions[positions["date"] == date]
            if len(pool) < 2:
                continue
            random_signs = np.random.choice([-1, 1], size=len(pool))
            pnl = (pool["gross_return"].values * random_signs).mean()
            pnls.append(pnl)

        if len(pnls) > 5:
            pnl_arr = np.array(pnls) / 100
            s = pnl_arr.mean() / (pnl_arr.std() + 1e-9) * np.sqrt(252)
            sharpes.append(s)

    return {
        "random_mean_sharpe": round(np.mean(sharpes), 3),
        "random_p95_sharpe":  round(np.percentile(sharpes, 95), 3),
    }
