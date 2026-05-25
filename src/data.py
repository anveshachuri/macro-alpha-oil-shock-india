from constants import OIL_COL
"""
data.py
Incremental data ingestion: prices (yfinance) + news (NewsAPI / RSS).
All writes are append-only — run daily and the dataset grows into 2026+.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PATHS & UNIVERSE
# ─────────────────────────────────────────────

ROOT      = Path(__file__).resolve().parent.parent
DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
for p in [DATA_RAW, DATA_PROC]:
    p.mkdir(parents=True, exist_ok=True)

# Stock universe — Yahoo Finance tickers for NSE-listed stocks
STOCK_UNIVERSE = {
    "RELIANCE":  "RELIANCE.NS",
    "ONGC":      "ONGC.NS",
    "IOC":       "IOC.NS",
    "BPCL":      "BPCL.NS",
    "INDIGO":    "INDIGO.NS",
    "HPCL":      "HPCL.NS",
    "ADANIPORTS":"ADANIPORTS.NS",
    "TATAMOTORS":"TATAMOTORS.NS",
    "MARUTI":    "MARUTI.NS",
    "ASIANPAINT":"ASIANPAINT.NS",
    "HINDUNILVR":"HINDUNILVR.NS",
    "ITC":       "ITC.NS",
    "TCS":       "TCS.NS",
    "INFY":      "INFY.NS",
    "CIPLA":     "CIPLA.NS",
}

MACRO_TICKERS = {
    "BRENT":  "BZ=F",
    "NIFTY":  "^NSEI",
    "USDINR": "INR=X",
}

NEWS_QUERIES = [
    "oil OR crude oil OR oil price",
    "Brent OR WTI crude",
    "OPEC OR oil production OR oil supply",
    "energy crisis OR fuel prices OR gasoline",
    "middle east OR geopolitics oil OR war oil",
]


# ─────────────────────────────────────────────
# PRICE DATA
# ─────────────────────────────────────────────

def _yf_download(tickers: dict, start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices; return DataFrame keyed by label."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    frames = {}
    for label, ticker in tickers.items():
        success = False

        for attempt in range(3):
            try:
                raw = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=True,
                    threads=False   # IMPORTANT: disable threading
                )

                if not raw.empty:
                    frames[label] = raw["Close"].squeeze()
                    success = True
                    break

            except Exception as e:
                time.sleep(1)

        if not success:
            log.warning(f"Skipping {ticker} (failed after retries)")
    
    df = pd.DataFrame(frames)

    # Drop columns with too many NaNs
    df = df.dropna(axis=1, thresh=int(0.8 * len(df)))

    return df


def load_prices(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load prices from cache, extending with any new data since last run.
    Returns DataFrame indexed by date, columns = stock + macro labels.
    """
    cache = DATA_RAW / "prices.parquet"
    start = "2019-01-01"  # extend to 2010 manually if pre-2019 data needed

    if cache.exists() and not force_refresh:
        existing  = pd.read_parquet(cache)
        last_date = existing.index[-1]
        fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        today       = datetime.utcnow().date()

        if last_date.date() >= today:
            log.info("Prices already current.")
            return existing

        log.info(f"Extending prices from {fetch_start} …")
        all_tickers = {**STOCK_UNIVERSE, **MACRO_TICKERS}
        new_df = _yf_download(all_tickers, fetch_start,
                              (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d"))

        if new_df.empty:
            return existing

        new_df.index = pd.to_datetime(new_df.index)
        combined = pd.concat([existing, new_df[~new_df.index.isin(existing.index)]])
        combined = combined.sort_index().ffill().dropna(how="all")
        combined.to_parquet(cache)
        return combined

    log.info(f"Full download from {start} …")
    all_tickers = {**STOCK_UNIVERSE, **MACRO_TICKERS}
    df = _yf_download(all_tickers, start,
                      (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d"))
    if df.empty:
        raise RuntimeError("Price download returned empty. Check network / tickers.")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().ffill().dropna(how="all")
    df.to_parquet(cache)
    return df


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns for all columns."""
    return np.log(prices / prices.shift(1)).dropna()


def save_returns(returns: pd.DataFrame):
    path = DATA_PROC / "returns.parquet"
    returns.to_parquet(path)
    log.info(f"Returns saved → {path}  {returns.shape}")
    return returns


def load_returns() -> pd.DataFrame:
    path = DATA_PROC / "returns.parquet"
    if not path.exists():
        raise FileNotFoundError("Returns not found. Run data pipeline first.")
    return pd.read_parquet(path)


# ─────────────────────────────────────────────
# NEWS DATA
# ─────────────────────────────────────────────

NEWS_CACHE = DATA_RAW / "news.parquet"

import requests

NEWS_API_KEY = "8091358d08514d93946c3bd4706dfbac"

def fetch_news_newsapi(days_back: int = 5):
    import requests
    
    to_date = datetime.utcnow().date()
    from_date = to_date - timedelta(days=days_back)

    all_articles = []

    for q in NEWS_QUERIES:
        for page in range(1, 6):   # pagination = BIG upgrade
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": q,
                "from": str(from_date),
                "to": str(to_date),
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 100,
                "page": page,
                "apiKey": NEWS_API_KEY,
            }

            try:
                response = requests.get(url, params=params, timeout=5)
                data = response.json()
            except:
                continue

            if "articles" not in data:
                continue

            for art in data["articles"]:
                all_articles.append({
                    "date": art.get("publishedAt", "")[:10] if art.get("publishedAt") else None,
                    "text": (art["title"] or "") + " " + (art.get("description") or "")
                })

    if not all_articles:
        log.warning("NewsAPI returned 0 articles for all queries.")
        return pd.DataFrame(columns=["date", "text"])

    df = pd.DataFrame(all_articles)
    df = df.dropna(subset=["date"])            # drop rows with missing dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])            # drop rows where coerce produced NaT
    return df.drop_duplicates(subset=["text"])

def fetch_news_rss(days_back: int = 5) -> pd.DataFrame:
    """Fallback: pull from Reuters / OilPrice RSS."""
    feeds = [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.oilprice.com/rss/main",
    ]
    rows = []
    try:
        import feedparser
        keywords = ["oil", "crude", "opec", "brent", "energy", "middle east"]
        for url in feeds:
            try:
                feed = feedparser.parse(url)
                for e in feed.entries:
                    title = e.get("title", "")
                    if any(k in title.lower() for k in keywords):
                        rows.append({"date": datetime.utcnow().strftime("%Y-%m-%d"),
                                     "text": title + " " + e.get("summary", ""), "source": url})
            except Exception:
                pass
    except ImportError:
        log.warning("feedparser not installed. pip install feedparser.")
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date","title","source"])


def load_news(api_key: str = "", days_back: int = 5) -> pd.DataFrame:
    """Load news, extending cache incrementally."""
    new_rows = fetch_news_newsapi(days_back) if api_key else fetch_news_rss(days_back)

    if NEWS_CACHE.exists():
        try:
            cached = pd.read_parquet(NEWS_CACHE)
            # Normalise column name: old code used 'title', new code uses 'text'
            if "title" in cached.columns and "text" not in cached.columns:
                cached = cached.rename(columns={"title": "text"})
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
            cached = cached.dropna(subset=["date"])
        except Exception as e:
            log.warning(f"Cache unreadable ({e}). Starting fresh.")
            cached = pd.DataFrame(columns=["date", "text"])

        if new_rows.empty:
            return cached

        combined = pd.concat([cached, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=["text"])
        combined.to_parquet(NEWS_CACHE)
        return combined

    if new_rows.empty:
        return pd.DataFrame(columns=["date", "text"])

    new_rows.to_parquet(NEWS_CACHE)
    return new_rows


# ─────────────────────────────────────────────
# SIMULATION FALLBACK (for offline dev / CI)
# ─────────────────────────────────────────────

def simulate_prices(start: str = "2010-01-01", end: str = "2024-12-31",
                    seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic simulated prices when live data is unavailable.
    Injects cross-sectional oil-shock responses to preserve model signal.
    """
    np.random.seed(seed)
    dates = pd.bdate_range(start, end)
    n     = len(dates)
    dt    = 1 / 252

    def gbm(s0, mu, sigma):
        lr = np.random.normal((mu - 0.5*sigma**2)*dt, sigma*np.sqrt(dt), n)
        return s0 * np.exp(np.cumsum(lr))

    # Base parameters per stock (mu, sigma) — loosely calibrated to NSE reality
    params = {
        "RELIANCE":   (15000, 0.12, 0.20),
        "ONGC":       (160,   0.08, 0.25),
        "IOC":        (100,   0.09, 0.24),
        "BPCL":       (430,   0.10, 0.26),
        "INDIGO":     (2200,  0.11, 0.30),
        "HPCL":       (280,   0.09, 0.26),
        "ADANIPORTS": (750,   0.16, 0.28),
        "TATAMOTORS": (450,   0.12, 0.32),
        "MARUTI":     (9000,  0.10, 0.22),
        "ASIANPAINT": (3200,  0.13, 0.18),
        "HINDUNILVR": (2600,  0.11, 0.16),
        "ITC":        (350,   0.08, 0.18),
        "TCS":        (3500,  0.15, 0.19),
        "INFY":       (1600,  0.14, 0.21),
        "CIPLA":      (1100,  0.12, 0.20),
        "BRENT":      (70,    0.04, 0.35),
        "NIFTY":      (11000, 0.12, 0.18),
        "USDINR":     (72,    0.03, 0.05),
    }

    df = pd.DataFrame(
        {k: gbm(s0, mu, sigma) for k, (s0, mu, sigma) in params.items()},
        index=dates,
    )

    # Inject cross-sectional oil-shock responses
    oil_shocks = [
        ("2019-09-14", +0.08), ("2020-03-09", -0.15), ("2020-04-20", -0.12),
        ("2021-03-08", +0.05), ("2022-02-24", +0.12), ("2022-06-02", -0.07),
        ("2023-04-02", +0.06), ("2023-10-07", +0.09), ("2024-01-12", +0.05),
        ("2024-04-14", +0.07),
    ]
    # Stock-level oil sensitivities (beta to Brent)
    oil_beta = {
        "RELIANCE":+0.6,"ONGC":+1.2,"IOC":+0.8,"BPCL":+0.9,"INDIGO":-1.4,
        "HPCL":+0.7,"ADANIPORTS":-0.3,"TATAMOTORS":-0.5,"MARUTI":-0.4,
        "ASIANPAINT":-0.2,"HINDUNILVR":-0.2,"ITC":-0.1,"TCS":0.0,
        "INFY":0.0,"CIPLA":0.1,
    }

    for date_str, shock in oil_shocks:
        dt_event = pd.Timestamp(date_str)
        idx = df.index.searchsorted(dt_event)
        if idx >= len(df): continue
        df.iloc[idx, df.columns.get_loc("BRENT")] *= (1 + shock)
        for stock, beta in oil_beta.items():
            for d in range(4):
                if idx + d >= len(df): break
                decay = (1 - d * 0.25)
                df.iloc[idx+d, df.columns.get_loc(stock)] *= (1 + beta * shock * 0.015 * decay)

    return df


def get_prices(use_live: bool = True, api_key: str = "",
               force_refresh: bool = False) -> pd.DataFrame:
    """
    Entry point: tries live yfinance first; falls back to simulation.
    """
    if use_live:
        try:
            return load_prices(force_refresh=force_refresh)
        except Exception as e:
            log.warning(f"Live data unavailable ({e}). Using simulation.")
    cache = DATA_RAW / "prices.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    sim = simulate_prices()
    sim.to_parquet(cache)
    return sim
