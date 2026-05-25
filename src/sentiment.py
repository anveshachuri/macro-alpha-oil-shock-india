"""
sentiment.py
Converts raw news headlines into daily sentiment scores.
Uses VADER (fast, no GPU) with optional FinBERT fallback.
Outputs a daily sentiment time series that feeds into event detection.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT            = Path(__file__).resolve().parent.parent
SENTIMENT_CACHE = ROOT / "data" / "processed" / "sentiment.parquet"


# ─────────────────────────────────────────────
# VADER SCORING
# ─────────────────────────────────────────────

def score_headlines_vader(headlines: pd.Series) -> pd.Series:
    """
    Score each headline with VADER. Returns compound score [-1, +1].
    Positive = bullish oil sentiment / supply comfortable
    Negative = bearish / supply disruption / geopolitical risk
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        return headlines.apply(lambda t: sia.polarity_scores(str(t))["compound"])
    except ImportError:
        log.warning("vaderSentiment not installed. pip install vaderSentiment. Returning zeros.")
        return pd.Series(0.0, index=headlines.index)


def score_headlines_finbert(headlines: pd.Series) -> pd.Series:
    """
    FinBERT scoring (financial domain, more accurate for price-sensitive text).
    Requires: pip install transformers torch
    Falls back to VADER if unavailable.
    """
    try:
        from transformers import pipeline
        pipe = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            return_all_scores=False,
            truncation=True,
        )
        label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        scores = []
        for text in headlines:
            try:
                result = pipe(str(text)[:512])[0]
                scores.append(label_map.get(result["label"], 0.0) * result["score"])
            except Exception:
                scores.append(0.0)
        return pd.Series(scores, index=headlines.index)
    except ImportError:
        log.warning("transformers not installed. Falling back to VADER.")
        return score_headlines_vader(headlines)


# ─────────────────────────────────────────────
# DAILY AGGREGATION
# ─────────────────────────────────────────────

def daily_sentiment(
    news_df: pd.DataFrame,
    method: str = "vader",
    agg: str = "mean",
) -> pd.Series:
    """
    Compute daily aggregated sentiment from news DataFrame.

    Parameters
    ----------
    news_df : DataFrame with columns ['date', 'text']
    method  : 'vader' | 'finbert'
    agg     : aggregation function — 'mean' | 'min' | 'sum'

    Returns
    -------
    pd.Series indexed by date (daily sentiment score)
    """
    if news_df.empty:
        return pd.Series(dtype=float)

    news_df = news_df.copy()
    news_df["date"] = pd.to_datetime(news_df["date"])

    scorer = score_headlines_vader if method == "vader" else score_headlines_finbert
    news_df["score"] = scorer(news_df["text"])

    daily = news_df.groupby("date")["score"].agg(agg)
    daily.name = "sentiment"
    return daily


# ─────────────────────────────────────────────
# FILL MISSING DAYS
# ─────────────────────────────────────────────

def reindex_sentiment(
    sentiment: pd.Series,
    dates: pd.DatetimeIndex,
    decay: float = 0.85,
) -> pd.Series:
    """
    Align sentiment to trading calendar with exponential decay fill.
    Days with no news carry forward at `decay` rate per day, rather than
    snapping to zero. This preserves signal between news events.
    """
    s = sentiment.reindex(dates)
    last_val   = 0.0
    days_since = 0
    filled     = []
    for dt in dates:
        raw = s.get(dt, np.nan) if hasattr(s, "get") else s.loc[dt] if dt in s.index else np.nan
        if not pd.isna(raw):
            last_val   = float(raw)
            days_since = 0
            filled.append(last_val)
        else:
            days_since += 1
            filled.append(last_val * (decay ** days_since))
    return pd.Series(filled, index=dates, name="sentiment")


# ─────────────────────────────────────────────
# ROLLING SENTIMENT FEATURES
# ─────────────────────────────────────────────

def sentiment_features(daily: pd.Series) -> pd.DataFrame:
    """
    Derive secondary sentiment features for the model:
      - sentiment_1d   : raw daily score
      - sentiment_5d   : 5-day rolling mean
      - sentiment_vol  : 5-day rolling std (uncertainty proxy)
      - sentiment_spike: 1 if |score| > 1 std from 20d mean
    """
    df = pd.DataFrame({"sentiment_1d": daily})
    df["sentiment_5d"]  = daily.rolling(5).mean()
    df["sentiment_vol"] = daily.rolling(5).std()
    rolling_mean        = daily.rolling(20).mean()
    rolling_std         = daily.rolling(20).std().clip(lower=1e-6)
    df["sentiment_z"]   = (daily - rolling_mean) / rolling_std
    df["sentiment_spike"] = (df["sentiment_z"].abs() > 1.5).astype(int)
    return df


# ─────────────────────────────────────────────
# CACHE MANAGEMENT
# ─────────────────────────────────────────────

def load_or_compute_sentiment(
    news_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    method: str = "vader",
) -> pd.DataFrame:
    """
    Load from cache if fresh; otherwise recompute and save.
    Returns a DataFrame of sentiment features aligned to trading_dates.
    """
    daily = daily_sentiment(news_df, method=method)
    filled = reindex_sentiment(daily, trading_dates)
    feat   = sentiment_features(filled)

    SENTIMENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(SENTIMENT_CACHE)
    return feat


def simulate_sentiment(
    dates: pd.DatetimeIndex,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate realistic sentiment for offline development.
    Includes occasional negative spikes correlated with oil shock dates.
    """
    np.random.seed(seed)
    n     = len(dates)
    base  = np.random.normal(0, 0.2, n)

    # Add autocorrelation (news clusters)
    for i in range(1, n):
        base[i] = 0.6 * base[i-1] + 0.4 * base[i]

    # Inject negative spikes at known shock dates
    shock_dates = pd.to_datetime([
        "2019-09-14","2020-03-09","2020-04-20","2021-03-08",
        "2022-02-24","2022-06-02","2023-04-02","2023-10-07",
        "2024-01-12","2024-04-14",
    ])
    s = pd.Series(base, index=dates, name="sentiment")
    for sd in shock_dates:
        idx = dates.searchsorted(sd)
        for d in range(3):
            if idx + d < n:
                s.iloc[idx+d] -= 0.6 * (1 - d * 0.3)

    s = s.clip(-1, 1)
    return sentiment_features(s)
