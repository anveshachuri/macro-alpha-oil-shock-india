"""
utils.py
Data loading, preprocessing, and plotting helpers.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parent.parent
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
OUTPUTS     = ROOT / "outputs"
PLOTS_DIR   = OUTPUTS / "plots"
TABLES_DIR  = OUTPUTS / "tables"

for d in [DATA_RAW, DATA_PROC, PLOTS_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def download_data(tickers: dict, start: str = "2019-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """
    Download adjusted close prices using yfinance.

    Parameters
    ----------
    tickers : dict  {label: yahoo_ticker}
    start, end : date strings

    Returns
    -------
    price_df : DataFrame with columns = labels
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Run: pip install yfinance")

    dfs = {}
    for label, ticker in tickers.items():
        print(f"  Downloading {label} ({ticker})...")
        try:
            raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            dfs[label] = raw["Close"].squeeze()
        except Exception as e:
            print(f"    ⚠️  Failed for {ticker}: {e}")

    df = pd.DataFrame(dfs).dropna(how="all")
    df.index = pd.to_datetime(df.index)
    df = df.ffill().dropna()
    return df


def load_or_download(tickers: dict, start: str = "2019-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Cache prices to parquet; reload if already downloaded."""
    cache = DATA_RAW / "prices.parquet"
    if cache.exists():
        print("Loading cached prices...")
        return pd.read_parquet(cache)
    print("Downloading prices...")
    df = download_data(tickers, start, end)
    df.to_parquet(cache)
    print(f"Saved to {cache}")
    return df


# ─────────────────────────────────────────────
# PLOTTING THEME
# ─────────────────────────────────────────────

PALETTE = {
    "OIL_GAS":  "#FF6B35",
    "FMCG":     "#4ECDC4",
    "AUTO":     "#45B7D1",
    "IT":       "#96CEB4",
    "PHARMA":   "#FFEAA7",
    "NIFTY50":  "#DDA0DD",
}

def set_theme():
    plt.rcParams.update({
        "figure.facecolor":  "#0D1117",
        "axes.facecolor":    "#161B22",
        "axes.edgecolor":    "#30363D",
        "axes.labelcolor":   "#C9D1D9",
        "xtick.color":       "#8B949E",
        "ytick.color":       "#8B949E",
        "text.color":        "#C9D1D9",
        "grid.color":        "#21262D",
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "font.family":       "monospace",
        "axes.titlesize":    13,
        "axes.labelsize":    11,
    })


# ─────────────────────────────────────────────
# PLOT: AVERAGE CAR ACROSS EVENTS
# ─────────────────────────────────────────────

def plot_avg_car(avg_ar: pd.DataFrame, title: str = "Average Cumulative Abnormal Returns Around Oil Shocks"):
    """
    Plot cumulative AR path for each sector across the [-5, +5] window.
    avg_ar : DataFrame indexed by relative day (-5..+5), columns = sectors.
    """
    set_theme()
    sectors = [c for c in avg_ar.columns if c != "NIFTY50"]
    car_df  = avg_ar[sectors].cumsum()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axvline(0, color="#F85149", lw=1.5, ls="--", label="Event Day", alpha=0.8)
    ax.axhline(0, color="#8B949E", lw=0.8, alpha=0.5)

    for sec in sectors:
        color = PALETTE.get(sec, "#FFFFFF")
        ax.plot(car_df.index, car_df[sec] * 100, label=sec, color=color, lw=2.2, marker="o", ms=4)

    ax.set_title(title, color="#E6EDF3", fontsize=14, pad=14)
    ax.set_xlabel("Days Relative to Event")
    ax.set_ylabel("Avg. CAR (%)")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f%%"))
    ax.legend(framealpha=0.15, edgecolor="#30363D", fontsize=9)
    ax.grid(True)

    plt.tight_layout()
    path = PLOTS_DIR / "avg_car.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {path}")


# ─────────────────────────────────────────────
# PLOT: CAR HEATMAP (event × sector)
# ─────────────────────────────────────────────

def plot_car_heatmap(car_list):
    """
    Heatmap: rows = events, columns = sectors, values = CAR (%).
    """
    set_theme()
    sectors = [c for c in car_list[0][1].index if c not in ("NIFTY50",)]
    rows = []
    for ev, car in car_list:
        row = {s: round(car.get(s, np.nan) * 100, 2) for s in sectors}
        row["Event"] = ev["label"]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Event")[sectors]

    fig, ax = plt.subplots(figsize=(10, len(rows) * 0.7 + 1.5))
    sns.heatmap(
        df, annot=True, fmt=".2f", center=0,
        cmap="RdYlGn", linewidths=0.5, linecolor="#0D1117",
        ax=ax, cbar_kws={"label": "CAR (%)"},
    )
    ax.set_title("Cumulative Abnormal Returns by Event & Sector (%)", pad=14, color="#E6EDF3")
    plt.tight_layout()
    path = PLOTS_DIR / "car_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {path}")


# ─────────────────────────────────────────────
# PLOT: OIL SENSITIVITY BAR CHART
# ─────────────────────────────────────────────

def plot_oil_sensitivity(sensitivity_df: pd.DataFrame, lag: int = 0):
    set_theme()
    df = sensitivity_df.copy()
    colors = [PALETTE.get(s, "#888") for s in df.index]
    sig_alpha = [1.0 if sig else 0.4 for sig in df["significant"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(df.index, df["corr"], color=colors, alpha=0.85, edgecolor="#0D1117", lw=0.5)
    for bar, alpha in zip(bars, sig_alpha):
        bar.set_alpha(alpha)

    ax.axhline(0, color="#8B949E", lw=0.8)
    ax.set_ylabel("Pearson Correlation with Brent Returns")
    tag = "Same Day" if lag == 0 else f"Lag {lag}d"
    ax.set_title(f"Oil–Sector Return Correlation ({tag})", color="#E6EDF3", pad=12)
    ax.annotate("Faded = not significant at 5%", xy=(0.01, 0.02),
                xycoords="axes fraction", fontsize=8, color="#8B949E")
    plt.tight_layout()
    path = PLOTS_DIR / f"oil_sensitivity_lag{lag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {path}")


# ─────────────────────────────────────────────
# TABLE EXPORT
# ─────────────────────────────────────────────

def save_table(df: pd.DataFrame, name: str):
    path = TABLES_DIR / f"{name}.csv"
    df.to_csv(path)
    print(f"Saved table → {path}")
    return df
