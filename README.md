# Oil Supply Shocks, INR Transmission & Indian Sectoral Equity Mispricing

Research repository examining how Brent crude shocks propagate through USD/INR into Indian sectoral equity returns (2019–2026), using event studies, rule-based and ML signal generation, walk-forward portfolio optimisation, and an FX-channel decomposition.

---

## Quick Start

```bash
# 1. Install dependencies
pip install pandas numpy scipy scikit-learn lightgbm matplotlib seaborn \
            pyarrow jupyter yfinance vaderSentiment

# 2. Launch Jupyter
cd macro-alpha-oil-shock-india && jupyter notebook

# 3. Run notebooks in order: 01 → 07
```

No API keys required — all notebooks run fully offline using the cached parquet data.

---

## Notebooks (run in order)

| # | Notebook | What it does |
|---|----------|-------------|
| 01 | `01_data_collection.ipynb` | Load cached prices → log returns → descriptive stats |
| 02 | `02_event_study.ipynb` | Abnormal returns, CAR [−5,+5], t-tests, up/down asymmetry (10 events) |
| 03 | `03_sector_analysis.ipynb` | Oil sensitivity at lags 0/1/2, regime-split correlations |
| 04 | `04_insights.ipynb` | Simple L/S backtest on up-shock events, equity curve |
| 05 | `05_signal_backtest.ipynb` | Rule-based signal engine, threshold sweep, per-event ML predictor |
| 06 | `06_alpha_system.ipynb` | Cross-sectional Ridge/LightGBM, walk-forward CV, OOS portfolio |
| 07 | `07_fx_channel.ipynb` | INR transmission: rolling β decomposition, dual-channel signals, coupling regimes |

---

## Repository Structure

```
macro-alpha-oil-shock-india/
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
│   ├── raw/prices.parquet         ← daily price history (Jan 2019 – 2026)
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
| `FX_COL` | `"USDINR"` | USD/INR rate (higher = weaker INR) |

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
# 0 12 * * 1-5  cd /path/to/macro-alpha-oil-shock-india && python -m src.pipeline --run
```
## Key Outputs

- Market-adjusted event study around 10 major oil-supply-shock events
- Sector-level abnormal return and CAR analysis
- Brent sensitivity estimation across multiple lag structures
- Threshold-based oil-shock trading signals
- Walk-forward Ridge and LightGBM alpha models
- FX transmission decomposition (Brent → USD/INR → Equities)
- Reproducible research pipeline with exported figures and tables
---

## Key Research Findings

**Dataset**

- Daily data spanning Jan 2019 – 2026
- Brent crude, USD/INR, NIFTY, and representative Indian sector constituents
- 10 curated oil-supply-shock events (7 positive Brent shocks, 3 negative shocks)

### Event Study (NB02)

A market-adjusted event study was conducted around major oil-supply-shock events.

Results:

| Sector | p-value |
|----------|----------|
| OIL_GAS | 0.9278 |
| AUTO | 0.9467 |
| FMCG | 0.9156 |
| IT | 0.8940 |
| PHARMA | 0.3504 |

No sector achieved statistical significance at the 5% level. The limited sample of 10 major oil-shock events substantially reduces statistical power, making it difficult to distinguish persistent effects from noise. Expanding the sample to include earlier oil-shock episodes would substantially improve statistical power.

### Sector Sensitivity Analysis (NB03)

Sector responses to Brent moves were evaluated across multiple lag structures and market regimes.

Key observations:

- Sector responses to Brent shocks are heterogeneous and regime-dependent.
- Correlations vary across market environments and weaken during broad risk-off periods.
- Major geopolitical events can distort or weaken traditional oil-price transmission channels.

### Rule-Based Signal Engine (NB05)

A threshold-based trading system was tested using Brent shock signals.

Results at the 3% Brent-move threshold:

| Metric | Value |
|----------|----------|
| Signals / Trades | 149 |
| Hit Ratio | 48.32% |
| Trade Sharpe | -0.108 |
| Profit Factor | 0.761 |

Threshold sensitivity analysis showed that stronger Brent shocks produced better signal quality:

| Threshold | Trades | Hit Ratio | Trade Sharpe |
|------------|---------|------------|--------------|
| 1% | 186 | 45.16% | -0.082 |
| 2% | 169 | 47.34% | -0.047 |
| 3% | 149 | 48.32% | -0.108 |
| 4% | 119 | 53.78% | 0.011 |
| 5% | 86 | 55.81% | 0.034 |

The strongest shocks generated the most predictive signals, although overall performance remained economically weak.

### Alpha System (NB06)

A leak-free cross-sectional prediction framework was implemented using Ridge Regression and LightGBM with walk-forward evaluation.

Out-of-sample results:

| Metric | Ridge | LightGBM | NIFTY Benchmark |
|----------|----------|----------|----------|
| Annualized Sharpe | -0.115 | -2.773 | 0.934 |
| Hit Ratio | 46.95% | 44.91% | 54.93% |
| Total Return (%) | -2.35 | -20.04 | 3.57 |
| Profit Factor | 0.981 | 0.630 | 1.169 |

Given the limited event count and relatively small cross-sectional universe, neither model generated persistent alpha out of sample.

### FX Transmission Channel (NB07)

The project decomposed sector returns into:

1. Direct oil-price exposure
2. Indirect INR/USD transmission effects

Outputs include:

- Rolling Brent beta estimates
- FX channel decomposition
- Dual-channel signal generation framework

Results suggest that exchange-rate transmission contributes materially to sector-level reactions and should be modeled alongside direct commodity exposure.

### Main Conclusion

Oil-supply shocks appear to influence Indian equity sectors, although the effects are difficult to identify with statistical confidence in a small event sample. The effects are also difficult to exploit using a simple event-driven trading strategy. Statistical significance is limited by sample size, and broad risk-off episodes frequently overwhelm expected sector-rotation effects. The framework provides a reproducible foundation for future work using a longer historical sample, richer cross-sectional coverage, and more sophisticated predictive features.

---

## Data Notes

- `BRENT` column winsorized at ±25% to remove a Yahoo Finance roll-date artefact (+152% on 2025-01-02).
  All legitimate large moves are preserved (Russia–Ukraine +13% Apr 2022, COVID crash −16% Mar 2020).
- Sentiment scores are VADER-simulated from cached headlines; live mode requires a NewsAPI key.
- To extend data back to 2010: change `start = "2019-01-01"` → `"2010-01-01"` in `src/data.py`,
  then call `load_prices(force_refresh=True)` in NB01. This would increase the event catalogue
  and improve event-study statistical power.

---

*Research and educational purposes only. Not investment advice.*
