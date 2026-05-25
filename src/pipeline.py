"""
pipeline.py  —  Daily execution orchestrator.
Cron: 0 18 * * 1-5  python -m src.pipeline --run

Steps:
  1. Fetch prices (incremental yfinance)
  2. Fetch news headlines
  3. Compute daily sentiment
  4. Update returns parquet
  5. Detect events algorithmically
  6. Build live features
  7. Load / retrain alpha model
  8. Generate cross-sectional signals
  9. Append to outputs/signal_log.csv
"""

import argparse, logging, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
SIGNAL_LOG  = ROOT / "outputs" / "signal_log.csv"
MODELS_DIR  = ROOT / "outputs" / "models"
MODEL_STALE = 30  # retrain if model older than N days


def step_prices(use_sim=False):
    from data import get_prices, log_returns, save_returns
    log.info("Step 1 — Fetching prices …")
    prices  = get_prices(use_live=not use_sim)
    returns = log_returns(prices)
    save_returns(returns)
    log.info(f"  {prices.shape[0]} dates × {prices.shape[1]} series")
    return returns


def step_news(api_key=""):
    from data import load_news
    log.info("Step 2 — Fetching news …")
    try:
        news = load_news(api_key=api_key)
        log.info(f"  {len(news)} headlines")
        return news
    except Exception as e:
        log.warning(f"  News unavailable ({e}). Proceeding without.")
        return pd.DataFrame(columns=["date","title","source"])


def step_sentiment(news_df, trading_dates):
    from sentiment import load_or_compute_sentiment, simulate_sentiment
    log.info("Step 3 — Computing sentiment …")

    if news_df.empty:
        return simulate_sentiment(trading_dates)

    return load_or_compute_sentiment(news_df, trading_dates)


def step_events(returns, sentiment_df):
    from events import detect_events
    log.info("Step 4 — Detecting events …")
    sent = sentiment_df["sentiment_1d"] if "sentiment_1d" in sentiment_df else sentiment_df.iloc[:,0]
    events = detect_events(returns, sent, mode="oil_or_sentiment")
    log.info(f"  {events.sum()} events detected")
    return events


def step_features(returns, sentiment_df, events):
    from features import build_panel
    log.info("Step 5 — Building feature panel …")
    sent = sentiment_df["sentiment_1d"] if "sentiment_1d" in sentiment_df else sentiment_df.iloc[:,0]
    panel = build_panel(returns, sent, events)
    log.info(f"  Panel: {panel.shape}")
    return panel


def step_model(panel, force_retrain=False):
    from model import AlphaModel, time_split
    log.info("Step 6 — Model …")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mpath = MODELS_DIR / "alpha_model.pkl"
    stale = force_retrain or not mpath.exists()
    if not stale:
        age = (datetime.utcnow() - datetime.fromtimestamp(mpath.stat().st_mtime)).days
        stale = age > MODEL_STALE
        if stale: log.info(f"  Model stale ({age}d). Retraining …")

    if stale:
        train, _ = time_split(panel, 0.70)
        m = AlphaModel(model_type="lgbm")
        m.fit(train)
        m.save()
        log.info(f"  Trained. CV IC={m.cv_metrics_.get('mean_ic')}")
    else:
        m = AlphaModel.load()
        log.info(f"  Loaded. CV IC={m.cv_metrics_.get('mean_ic')}")
    return m


def step_signals(model, returns, sentiment_df, events):
    from features import build_live_features
    from strategy import rank_signals, signal_weights
    log.info("Step 7 — Signals …")
    today = returns.index[-1]
    if not events.get(today, False):
        log.info(f"  No event today ({today.date()}).")
        return pd.DataFrame()

    sent = sentiment_df["sentiment_1d"] if "sentiment_1d" in sentiment_df else sentiment_df.iloc[:,0]
    live = build_live_features(returns, sent)
    if live.empty:
        return pd.DataFrame()

    preds   = model.predict(live)
    signals = rank_signals(preds)
    weights = signal_weights(signals)

    df = pd.DataFrame({
        "date":             today,
        "stock":            preds.index.get_level_values("stock"),
        "signal":           signals.values.astype(int),
        "predicted_return": (preds.values * 100).round(4),
        "weight":           weights.values.round(4),
        "actual_return":    np.nan,
    })
    df = df[df["signal"] != 0].reset_index(drop=True)
    log.info(f"  {len(df)} positions ({(df.signal==1).sum()} L / {(df.signal==-1).sum()} S)")
    return df


def step_log(sig):
    log.info("Step 8 — Logging …")
    if sig.empty:
        log.info("  Nothing to log.")
        return
    SIGNAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    if SIGNAL_LOG.exists():
        old = pd.read_csv(SIGNAL_LOG, parse_dates=["date"])
        out = pd.concat([old, sig], ignore_index=True)
    else:
        out = sig
    out.to_csv(SIGNAL_LOG, index=False)
    log.info(f"  {len(out)} rows → {SIGNAL_LOG}")


def run(api_key="", use_sim=False, force_retrain=False):
    print("\n" + "="*58)
    print(f"  MACRO ALPHA PIPELINE | {datetime.utcnow():%Y-%m-%d %H:%M} UTC")
    print("="*58)
    returns      = step_prices(use_sim=use_sim)
    news         = step_news(api_key=api_key)
    sentiment_df = step_sentiment(news, returns.index)
    events       = step_events(returns, sentiment_df)
    panel        = step_features(returns, sentiment_df, events)
    model        = step_model(panel, force_retrain=force_retrain)
    sigs         = step_signals(model, returns, sentiment_df, events)
    step_log(sigs)
    print("="*58 + "\n")


def status():
    from data import DATA_RAW
    p = DATA_RAW / "prices.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        print(f"Prices  : {df.index[-1].date()}  ({len(df)} rows)")
    if SIGNAL_LOG.exists():
        sl = pd.read_csv(SIGNAL_LOG)
        print(f"Signals : {len(sl)} rows  |  last date: {sl['date'].max()}")
        print(sl.tail(5).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run",     action="store_true")
    p.add_argument("--status",  action="store_true")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--sim",     action="store_true")
    p.add_argument("--newsapi", default="")
    args = p.parse_args()
    if   args.run:    run(api_key=args.newsapi, use_sim=args.sim, force_retrain=args.retrain)
    elif args.status: status()
    else:             p.print_help()
