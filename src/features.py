"""
features.py
Cross-sectional feature engineering: one row per (date × stock).
All features are computed with no future information.
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

STOCK_COLS  = [
    "RELIANCE","ONGC","IOC","BPCL","INDIGO","HPCL",
    "ADANIPORTS","TATAMOTORS","MARUTI","ASIANPAINT",
    "HINDUNILVR","ITC","TCS","INFY","CIPLA",
]
OIL_COL    = "BRENT"
NIFTY_COL  = "NIFTY"


# ─────────────────────────────────────────────
# PER-STOCK ROLLING FEATURES
# ─────────────────────────────────────────────

def stock_rolling_features(
    returns: pd.DataFrame,
    stock: str,
    oil_col: str   = OIL_COL,
    nifty_col: str = NIFTY_COL,
) -> pd.DataFrame:
    """
    Compute rolling features for a single stock (no lookahead).

    Features:
      stock_ret_1d          : yesterday's return
      stock_mom_5d          : 5-day momentum
      stock_mom_20d         : 20-day momentum
      stock_vol_20d         : annualised realised vol
      oil_stock_corr_60d    : rolling 60d Brent-stock correlation
      stock_beta_nifty_60d  : rolling 60d NIFTY beta
      stock_idio_vol_20d    : idiosyncratic vol (residual from market)
    """
    if stock not in returns.columns:
        return pd.DataFrame()

    s     = returns[stock]
    oil   = returns[oil_col]   if oil_col   in returns.columns else pd.Series(0.0, index=returns.index)
    nifty = returns[nifty_col] if nifty_col in returns.columns else pd.Series(0.0, index=returns.index)

    feat = pd.DataFrame(index=returns.index)
    feat["relative_return"] = s - nifty
    feat["stock_ret_1d"]       = s
    feat["stock_mom_5d"]       = s.rolling(5).sum()
    feat["stock_mom_20d"]      = s.rolling(20).sum()
    feat["stock_vol_20d"]      = s.rolling(20).std() * np.sqrt(252)

    # Rolling Brent-stock correlation
    feat["oil_stock_corr_60d"] = (
        s.rolling(60).corr(oil).fillna(0.0)
    )

    # Rolling NIFTY beta
    def rolling_beta(y, x, w=60):
        cov = y.rolling(w).cov(x)
        var = x.rolling(w).var().clip(lower=1e-10)
        return cov / var

    feat["stock_beta_nifty_60d"] = rolling_beta(s, nifty)

    # Idiosyncratic vol: residual after removing NIFTY component
    beta     = feat["stock_beta_nifty_60d"].fillna(1.0)
    residual = s - beta * nifty
    feat["stock_idio_vol_20d"] = residual.rolling(20).std() * np.sqrt(252)

    feat["stock"] = stock
    return feat.dropna()


# ─────────────────────────────────────────────
# MACRO CONTEXT FEATURES (shared across stocks)
# ─────────────────────────────────────────────

def macro_features(
    returns: pd.DataFrame,
    sentiment: pd.Series,
    oil_col: str   = OIL_COL,
    nifty_col: str = NIFTY_COL,
    fx_col: str    = "USDINR",
) -> pd.DataFrame:
    """
    Features that are the same for every stock on a given day.
    Built once and then merged into the cross-sectional frame.
    Includes USDINR FX channel features (oil → INR depreciation → sector divergence).
    """
    oil   = returns[oil_col]   if oil_col   in returns.columns else pd.Series(0.0, index=returns.index)
    nifty = returns[nifty_col] if nifty_col in returns.columns else pd.Series(0.0, index=returns.index)
    sent  = sentiment.reindex(returns.index).fillna(0.0)
    # USDINR: higher = INR weaker (depreciation). Positive shock = INR devalued.
    fx    = returns[fx_col]    if fx_col    in returns.columns else pd.Series(0.0, index=returns.index)

    macro = pd.DataFrame(index=returns.index)
    macro["oil_return_1d"]     = oil
    macro["oil_return_2d"]     = oil.rolling(2).sum()
    macro["oil_return_5d"]     = oil.rolling(5).sum()
    macro["oil_vol_20d"]       = oil.rolling(20).std() * np.sqrt(252)
    macro["market_ret_1d"]     = nifty
    macro["market_vol_20d"]    = nifty.rolling(20).std() * np.sqrt(252)
    macro["sentiment_1d"]      = sent
    macro["sentiment_5d"]      = sent.rolling(5).mean()

    # KEY: interaction term — captures "bad news + oil spike" jointly
    macro["oil_sent_interact"] = macro["oil_return_2d"] * sent
    macro["oil_direction"]     = np.sign(macro["oil_return_2d"])

    # Regime flag (high vol environment)
    macro["vol_regime"]        = (macro["market_vol_20d"] > 0.25).astype(int)

    # ── FX channel features ──────────────────────────────────────────────────
    # usdinr_ret_1d: positive = INR depreciation (tailwind for USD-revenue stocks)
    macro["usdinr_ret_1d"]     = fx
    macro["usdinr_ret_3d"]     = fx.rolling(3).sum()
    macro["usdinr_ret_5d"]     = fx.rolling(5).sum()
    macro["usdinr_vol_20d"]    = fx.rolling(20).std() * np.sqrt(252)
    # Oil-FX co-movement: captures "oil spike + INR sell-off" regime
    macro["oil_fx_interact"]   = macro["oil_return_2d"] * macro["usdinr_ret_3d"]
    # Rolling oil-FX correlation
    macro["oil_fx_corr_20d"]   = oil.rolling(20).corr(fx).fillna(0.0)

    return macro.dropna()


# ─────────────────────────────────────────────
# CROSS-SECTIONAL PANEL BUILDER
# ─────────────────────────────────────────────

def build_panel(
    returns: pd.DataFrame,
    sentiment: pd.Series,
    events: pd.Series,
    target_horizon: int = 1,
    stocks: list        = None,
) -> pd.DataFrame:
    """
    Build the full cross-sectional feature panel:
        index : pd.MultiIndex (date, stock)
        columns : all features + target

    target = forward log return over `target_horizon` days
    (computed without look-ahead: uses t+1 to t+horizon)

    Only includes rows where `events[date] == True`.
    """
    if stocks is None:
        stocks = [c for c in STOCK_COLS if c in returns.columns]

    log.info(f"Building panel for {len(stocks)} stocks, {events.sum()} event dates …")

    macro = macro_features(returns, sentiment)
    rows  = []

    for stock in stocks:
        stock_feat = stock_rolling_features(returns, stock)
        if stock_feat.empty:
            continue

        # Merge macro context
        merged = stock_feat.join(macro, how="inner")

        # Compute forward return (target) — NO lookahead: shift by 1 then sum 3 days
        fwd_ret = returns[stock].shift(-3).rolling(3).sum()
        merged["target"] = fwd_ret

        merged["stock"]  = stock
        merged.index.name = "date"
        rows.append(merged)

    if not rows:
        raise RuntimeError("No features built. Check returns columns.")

    panel = pd.concat(rows).reset_index()
    panel = panel.set_index(["date", "stock"])

    # Keep only event dates to focus model on shock-period behaviour
    event_dates = events[events].index
    panel = panel.loc[panel.index.get_level_values("date").isin(event_dates)]

    panel = panel.dropna(subset=["target"])
    log.info(f"Panel shape: {panel.shape}  ({panel.index.get_level_values('date').nunique()} event dates)")
    return panel


# ─────────────────────────────────────────────
# FULL DAILY FEATURES (for live prediction)
# ─────────────────────────────────────────────

def build_live_features(
    returns: pd.DataFrame,
    sentiment: pd.Series,
    stocks: list = None,
    as_of: str   = None,
) -> pd.DataFrame:
    """
    Build features for today (or `as_of` date) across all stocks.
    No target — for inference only.
    """
    if stocks is None:
        stocks = [c for c in STOCK_COLS if c in returns.columns]

    if as_of:
        returns   = returns.loc[:as_of]
        sentiment = sentiment.loc[:as_of]

    macro = macro_features(returns, sentiment)

    rows = []
    for stock in stocks:
        sf = stock_rolling_features(returns, stock)
        if sf.empty:
            continue
        merged = sf.join(macro, how="inner")
        merged["stock"] = stock
        rows.append(merged.iloc[[-1]])  # only most recent row

    if not rows:
        return pd.DataFrame()

    live = pd.concat(rows)
    live.index.name = "date"
    live = live.reset_index().set_index(["date","stock"])
    return live


FEATURE_COLS = [
    # Macro / oil
    "oil_return_1d", "oil_return_2d", "oil_return_5d", "oil_vol_20d",
    "market_ret_1d", "market_vol_20d",
    # Sentiment
    "sentiment_1d", "sentiment_5d",
    # Interaction
    "oil_sent_interact", "oil_direction",
    # Regime
    "vol_regime",
    # FX channel (INR/USD transmission)
    "usdinr_ret_1d", "usdinr_ret_3d", "usdinr_ret_5d", "usdinr_vol_20d",
    "oil_fx_interact", "oil_fx_corr_20d",
    # Stock-level
    "relative_return",
    "stock_ret_1d", "stock_mom_5d", "stock_mom_20d", "stock_vol_20d",
    "oil_stock_corr_60d", "stock_beta_nifty_60d", "stock_idio_vol_20d",
]
