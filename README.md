# Oil Supply Shocks, INR Transmission & Indian Sectoral Equity Mispricing

A fully reproducible quantitative research repository examining how Brent crude supply shocks
propagate through the INR/USD exchange rate into Indian sectoral equity returns.

---

## Quick Start

```bash
# 1. Install dependencies
pip install pandas numpy scipy scikit-learn lightgbm matplotlib seaborn \
            pyarrow jupyter yfinance vaderSentiment

# 2. Launch Jupyter
cd oil-shock-india && jupyter notebook

# 3. Run notebooks in order: 01 → 07
```

No API keys required — all notebooks run fully offline using cached parquet data.

---

## Notebooks (run in order)

| # | Notebook | What it does |
|---|----------|-------------|
| 01 | `01_data_collection.ipynb` | Load cached prices → log returns → descriptive stats |
| 02 | `02_event_study.ipynb` | Abnormal returns, CAR, t-tests, asymmetry tests (10 events) |
| 03 | `03_sector_analysis.ipynb` | Oil sensitivity lags 0/1/2, regime-split correlations |
| 04 | `04_insights.ipynb` | Simple L/S backtest on up-shock events, equity curve |
| 05 | `05_signal_backtest.ipynb` | Rule-based signal engine, threshold sweep, ML predictor |
| 06 | `06_alpha_system.ipynb` | Cross-sectional Ridge/LightGBM, walk-forward CV, OOS portfolio |
| 07 | `07_fx_channel.ipynb` | INR transmission: rolling β decomposition, dual-channel signals |

---

## Repository Structure

```
oil-shock-india/
├── src/
│   ├── constants.py      ← SINGLE SOURCE OF TRUTH: SECTOR_MAP, MARKET_COL, OIL_COL, FX_COL
│   ├── data.py           ← load_prices(), log_returns(), save_returns(), load_returns()
│   ├── event_study.py    ← run_event_study(), get_sector_returns(), oil_sensitivity()
│   ├── signals.py        ← detect_shocks(), backtest_signals(), performance_metrics()
│   ├── predictive.py     ← train_all_sectors(), SectorPredictor (LOO IC)
│   ├── features.py       ← build_panel(), build_live_features()
│   ├── model.py          ← AlphaModel (Ridge/LightGBM), walk_forward_cv(), time_split()
│   ├── strategy.py       ← build_portfolio(), daily_pnl(), rank_signals()
│   ├── backtest.py       ← compute_metrics(), compute_ic(), nifty_benchmark()
│   ├── fx_channel.py     ← fx_event_study(), decompose_sector_returns(), dual_channel_signal()
│   ├── events.py         ← detect_events() — algorithmic, no hard-coded dates
│   ├── sentiment.py      ← simulate_sentiment(), score_headlines_vader()
│   ├── pipeline.py       ← daily orchestrator (cron-ready)
│   └── utils.py          ← DATA_PROC, TABLES_DIR, PLOTS_DIR, set_theme(), save_table()
├── data/
│   ├── raw/prices.parquet         ← 1,899 days × 18 columns (Jan 2019 – Apr 2026)
│   └── processed/returns.parquet  ← log returns, BRENT winsorized at ±25%
├── outputs/
│   ├── plots/   ← all figures saved here by notebooks
│   └── tables/  ← all CSV tables saved here by notebooks
└── notebooks/   ← 01–07, all execute cleanly in order
```

---

## Schema

| Constant | Value | Description |
|----------|-------|-------------|
| `MARKET_COL` | `"NIFTY"` | Broad-market benchmark |
| `OIL_COL` | `"BRENT"` | Brent crude log-returns |
| `FX_COL` | `"USDINR"` | INR/USD (higher = weaker INR) |

**Sector → Stock mapping** (from `constants.SECTOR_MAP`):

| Sector | Stocks |
|--------|--------|
| OIL_GAS | RELIANCE, ONGC, IOC, BPCL, HPCL |
| AUTO | TATAMOTORS, MARUTI |
| FMCG | HINDUNILVR, ITC |
| IT | TCS, INFY |
| PHARMA | CIPLA |

All modules import `SECTOR_MAP` from `constants.py` — never duplicated.

---

## Live Pipeline

```bash
python -m src.pipeline --run          # live yfinance + NewsAPI
python -m src.pipeline --run --sim    # offline / dev mode
python -m src.pipeline --status       # data freshness + last signal

# Cron (IST 18:00, weekdays):
# 0 12 * * 1-5  cd /path/to/oil-shock-india && python -m src.pipeline --run
```

---

## Key Research Findings

**Data:** 1,898 trading days (Jan 2019 – Apr 2026), 10 curated events (7 up-shocks, 3 down-shocks)

**Event study:** No sector reaches p < 0.05 with 10 events — the sample is too small for
significance, not the methodology. Extending to pre-2019 data (~21 total events) is the primary
path to improved statistical power.

**Rule-based strategy (NB05):** 146 detected signals (3% threshold), 57% hit ratio on up-shocks,
Sharpe −0.12 overall. Geopolitical tail events (Russia-Ukraine 2022, Houthi 2024, Iran-Israel 2024)
break the expected sector rotation via broad risk-off selling.

**ML layer (NB05):** LOO IC −0.18 to +0.14 across sectors — noisy at n=146 but framework is
leak-free (walk-forward, no future data in features).

---

## Data Notes

- `BRENT` column winsorized at ±25% to remove a Yahoo Finance roll-date artefact (+152% on 2025-01-02)
- All legitimate large moves preserved (Russia-Ukraine +13%, March 2020 −16%)
- To extend data back to 2010: change `start = "2019-01-01"` → `"2010-01-01"` in `src/data.py`,
  then run `load_prices(force_refresh=True)` in NB01

---

*Research and educational purposes only. Not investment advice.*
