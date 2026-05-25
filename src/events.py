"""
events.py
Fully algorithmic oil shock event detection.
Zero hardcoded dates. All events are derived from real-time data.

Event logic (configurable):
    oil_shock    = oil_return_Nd > OIL_THRESHOLD   (Brent N-day log return)
    sentiment_spike = sentiment_score < SENT_THRESHOLD
    event        = oil_shock OR (oil_shock AND sentiment_spike)  [configurable]

Outputs: pd.Series of boolean flags indexed by trading date.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# DETECTION PARAMETERS
# ─────────────────────────────────────────────

OIL_THRESHOLD   = 0.03   # 4% Brent 2-day log return
SENT_THRESHOLD  = -0.20  # sentiment score < -0.5 confirms supply concern
OIL_WINDOW      = 2      # rolling days for Brent cumulation
COOLDOWN        = 5      # minimum days between events (prevents clustering)


# ─────────────────────────────────────────────
# CORE DETECTOR
# ─────────────────────────────────────────────

def detect_events(
    returns: pd.DataFrame,
    sentiment: pd.Series,
    oil_col: str       = "BRENT",
    threshold: float   = OIL_THRESHOLD,
    window: int        = OIL_WINDOW,
    sent_thresh: float = SENT_THRESHOLD,
    mode: Literal["oil_only", "oil_and_sentiment", "oil_or_sentiment"] = "oil_and_sentiment",
    cooldown: int      = COOLDOWN,
) -> pd.Series:
    """
    Detect oil supply shock events algorithmically.

    Parameters
    ----------
    returns     : DataFrame of daily log returns (must include `oil_col`)
    sentiment   : daily sentiment series (aligned to returns.index)
    mode        : 
        'oil_only'          → trigger on oil move alone
        'oil_and_sentiment' → require both oil move AND negative sentiment
        'oil_or_sentiment'  → either condition triggers

    Returns
    -------
    pd.Series[bool] indexed by date — True on event days.
    """
    if oil_col not in returns.columns:
        raise ValueError(f"'{oil_col}' not in returns. Available: {list(returns.columns)}")

    oil_cum = returns[oil_col].rolling(window).sum()
    oil_shock = oil_cum.abs() >= threshold
    # Align sentiment to returns index
    sent_aligned = sentiment.reindex(returns.index).fillna(0.0)
    sent_spike   = sent_aligned <= sent_thresh

    if mode == "oil_only":
        raw_events = oil_shock
    elif mode == "oil_and_sentiment":
        raw_events = oil_shock & sent_spike
    else:  # oil_or_sentiment
        raw_events = oil_shock | sent_spike

    # Apply cooldown to prevent event clustering
    events = pd.Series(False, index=returns.index)
    last_event_idx = -cooldown
    for i, (dt, is_event) in enumerate(raw_events.items()):
        if is_event and (i - last_event_idx) >= cooldown:
            events[dt]    = True
            last_event_idx = i

    n = events.sum()
    log.info(f"Events detected: {n}  (mode={mode}, threshold={threshold:.0%})")
    return events


# ─────────────────────────────────────────────
# EVENT METADATA
# ─────────────────────────────────────────────

def event_metadata(
    events: pd.Series,
    returns: pd.DataFrame,
    sentiment: pd.Series,
    oil_col: str = "BRENT",
    window: int  = OIL_WINDOW,
) -> pd.DataFrame:
    """
    Build a metadata table for all detected events.
    Useful for inspection and reporting.
    """
    event_dates = events[events].index
    rows = []
    oil_cum = returns[oil_col].rolling(window).sum()
    sent_aligned = sentiment.reindex(returns.index).fillna(0.0)

    for dt in event_dates:
        rows.append({
            "date":        dt,
            "oil_cum_ret": round(oil_cum.get(dt, 0.0) * 100, 2),
            "direction":   "up" if oil_cum.get(dt, 0) > 0 else "down",
            "sentiment":   round(sent_aligned.get(dt, 0.0), 3),
            "nifty_ret":   round(returns.get("NIFTY", pd.Series(dtype=float)).get(dt, 0.0) * 100, 2),
        })

    return pd.DataFrame(rows).set_index("date")


# ─────────────────────────────────────────────
# ROLLING EVENT FEATURES
# ─────────────────────────────────────────────

def event_context_features(
    returns: pd.DataFrame,
    sentiment: pd.Series,
    oil_col: str  = "BRENT",
    nifty_col: str = "NIFTY",
) -> pd.DataFrame:
    """
    Compute rolling features that describe the *context* of any event.
    These are available at any point in time — no future leakage.

    Features:
      oil_return_2d   : Brent 2-day log return
      oil_return_5d   : Brent 5-day log return
      oil_vol_20d     : Brent annualised realised vol (20d)
      market_vol_20d  : NIFTY annualised realised vol (20d)
      sentiment_1d    : current day sentiment
      oil_sent_interact: oil_return_2d × sentiment (interaction term)
      vol_regime      : 1 if NIFTY vol > 25% ann., else 0
    """
    oil   = returns[oil_col] if oil_col in returns.columns else pd.Series(0.0, index=returns.index)
    nifty = returns[nifty_col] if nifty_col in returns.columns else pd.Series(0.0, index=returns.index)
    sent  = sentiment.reindex(returns.index).fillna(0.0)

    feat = pd.DataFrame(index=returns.index)
    feat["oil_return_2d"]      = oil.rolling(2).sum()
    feat["oil_return_5d"]      = oil.rolling(5).sum()
    feat["oil_vol_20d"]        = oil.rolling(20).std() * np.sqrt(252)
    feat["market_vol_20d"]     = nifty.rolling(20).std() * np.sqrt(252)
    feat["sentiment_1d"]       = sent
    feat["oil_sent_interact"]  = 2 * feat["oil_return_2d"] * sent   # KEY interaction
    feat["vol_regime"]         = (feat["market_vol_20d"] > 0.25).astype(int)
    feat["oil_direction"] = np.sign(feat["oil_return_2d"])

    return feat.dropna()
