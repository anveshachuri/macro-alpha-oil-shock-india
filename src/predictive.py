"""
predictive.py
Predictive layer: Ridge/RF models per sector, trained on event-window features.

Schema note
-----------
Sector columns (OIL_GAS, AUTO …) do not exist in the stock-level returns df.
build_features() now accepts either sector names (looked up via SECTOR_MAP)
or stock names directly. NIFTY50 references replaced with MARKET_COL constant.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score
import warnings
warnings.filterwarnings("ignore")

from constants import (
    MARKET_COL, OIL_COL, SECTOR_MAP, SECTORS,
    build_sector_returns,
)


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "brent_shock",
    "brent_5d_ret",
    "brent_vol_20d",
    "nifty_vol_20d",
    "sector_mom_5d",
    "brent_sector_corr_60d",
]


def _resolve_sector_series(returns: pd.DataFrame, name: str) -> pd.Series:
    """
    Return a return series for `name`, which may be:
      - a stock column already in `returns` (e.g. "TCS")
      - a sector key in SECTOR_MAP (e.g. "IT") — returns equal-weight mean

    Raises ValueError with a clear message if neither matches.
    """
    if name in returns.columns:
        return returns[name]

    if name in SECTOR_MAP:
        stocks = [s for s in SECTOR_MAP[name] if s in returns.columns]
        if not stocks:
            raise ValueError(
                f"Sector '{name}': none of {SECTOR_MAP[name]} found in returns "
                f"(available: {list(returns.columns)})"
            )
        return returns[stocks].mean(axis=1)

    raise ValueError(
        f"'{name}' is neither a column in returns nor a key in SECTOR_MAP. "
        f"Available columns: {list(returns.columns)}. "
        f"SECTOR_MAP keys: {list(SECTOR_MAP.keys())}"
    )


def build_features(
    returns: pd.DataFrame,
    event_dates: List[pd.Timestamp],
    sector: str,
    target_hold: int = 3,
) -> pd.DataFrame:
    """
    Build a feature matrix for predicting sector return over `target_hold` days
    after each event date.  `sector` may be a stock name or a SECTOR_MAP key.

    Returns DataFrame with columns: features + 'target' (one row per event).
    """
    brent  = returns.get(OIL_COL)
    nifty  = returns.get(MARKET_COL)

    if brent is None:
        raise ValueError(f"'{OIL_COL}' not found in returns.")
    if nifty is None:
        raise ValueError(
            f"'{MARKET_COL}' not found in returns (columns: {list(returns.columns)}). "
            "Check MARKET_COL in constants.py."
        )

    sec_r = _resolve_sector_series(returns, sector)

    rows = []
    for dt in event_dates:
        idx = returns.index.searchsorted(dt)
        if idx < 60 or idx + target_hold >= len(returns):
            continue

        brent_shock         = float(brent.iloc[idx])
        brent_5d_ret        = float(brent.iloc[idx - 5 : idx].sum())
        brent_vol_20d       = float(brent.iloc[idx - 20 : idx].std() * np.sqrt(252))
        nifty_vol_20d       = float(nifty.iloc[idx - 20 : idx].std() * np.sqrt(252))
        sector_mom_5d       = float(sec_r.iloc[idx - 5 : idx].sum())
        b60, s60            = brent.iloc[idx - 60 : idx], sec_r.iloc[idx - 60 : idx]
        corr_60d            = float(b60.corr(s60))
        target              = float(sec_r.iloc[idx + 1 : idx + 1 + target_hold].sum())

        rows.append({
            "date":                  dt,
            "brent_shock":           brent_shock,
            "brent_5d_ret":          brent_5d_ret,
            "brent_vol_20d":         brent_vol_20d,
            "nifty_vol_20d":         nifty_vol_20d,
            "sector_mom_5d":         sector_mom_5d,
            "brent_sector_corr_60d": corr_60d,
            "target":                target,
        })

    return pd.DataFrame(rows).set_index("date").dropna()


# ─────────────────────────────────────────────
# PREDICTOR CLASS
# ─────────────────────────────────────────────

class SectorPredictor:
    """
    Ridge model per sector.  Designed for small samples (10-30 events),
    so heavily regularised and evaluated with LOO cross-validation.
    """

    def __init__(self, sector: str, model_type: str = "ridge"):
        self.sector     = sector
        self.model_type = model_type
        self.scaler     = StandardScaler()
        self.model      = None
        self.feature_importance_: Dict = {}
        self.loo_r2_: float = None

    def fit(self, features_df: pd.DataFrame) -> "SectorPredictor":
        X = features_df[FEATURE_COLS].values
        y = features_df["target"].values

        X_scaled = self.scaler.fit_transform(X)

        if self.model_type == "ridge":
            self.model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
                                 cv=min(5, len(y))).fit(X_scaled, y)
            self.feature_importance_ = dict(zip(FEATURE_COLS, self.model.coef_))
        else:
            self.model = RandomForestRegressor(
                n_estimators=200, max_depth=3, random_state=42
            ).fit(X_scaled, y)
            self.feature_importance_ = dict(
                zip(FEATURE_COLS, self.model.feature_importances_)
            )

        if len(y) >= 4:
            from sklearn.linear_model import Ridge as R
            from scipy.stats import pearsonr
            loo = LeaveOneOut()
            # LOO Pearson IC: R² is undefined for single test observations.
            # Use leave-one-out Pearson correlation (IC) instead.
            y_pred_loo = np.empty(len(y))
            for train_idx, test_idx in loo.split(X_scaled):
                m_loo = R(alpha=1.0).fit(X_scaled[train_idx], y[train_idx])
                y_pred_loo[test_idx] = m_loo.predict(X_scaled[test_idx])
            r, p = pearsonr(y_pred_loo, y)
            self.loo_r2_ = round(float(r), 4)   # stored as "LOO IC" (Pearson r)

        return self

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(features_df[FEATURE_COLS].values)
        return self.model.predict(X_scaled)

    def predict_single(self, feature_dict: Dict) -> float:
        row = pd.DataFrame([feature_dict])[FEATURE_COLS]
        return float(self.predict(row)[0])

    def signal_strength(self, predicted_return: float) -> float:
        return float(1 / (1 + np.exp(-predicted_return * 50)))

    def summary(self) -> Dict:
        return {
            "sector":      self.sector,
            "model":       self.model_type,
            "loo_r2":      self.loo_r2_,
            "top_feature": max(self.feature_importance_,
                               key=lambda k: abs(self.feature_importance_[k])),
            "coefs":       {k: round(v, 5) for k, v in self.feature_importance_.items()},
        }


# ─────────────────────────────────────────────
# MULTI-SECTOR TRAINING
# ─────────────────────────────────────────────

def train_all_sectors(
    returns: pd.DataFrame,
    event_dates: List[pd.Timestamp],
    sectors: List[str] = None,
    target_hold: int = 3,
) -> Dict[str, SectorPredictor]:
    """
    Train one SectorPredictor per sector.

    `sectors` may contain SECTOR_MAP keys (OIL_GAS, IT …) or stock names.
    Defaults to SECTORS (all five mapped sectors).
    Returns dict {sector_name: SectorPredictor}.
    """
    if sectors is None:
        sectors = SECTORS

    predictors = {}
    for sec in sectors:
        try:
            feat_df = build_features(returns, event_dates, sec, target_hold)
            if len(feat_df) < 4:
                print(f"  ⚠  {sec}: only {len(feat_df)} samples — skipping (need ≥4)")
                continue
            pred = SectorPredictor(sec).fit(feat_df)
            predictors[sec] = pred
            print(f"  ✓  {sec}: LOO R²={pred.loo_r2_}  top={pred.summary()['top_feature']}")
        except Exception as e:
            print(f"  ✗  {sec}: {e}")

    if not predictors:
        raise RuntimeError(
            "No sector predictors could be trained. "
            "Check that event_dates fall within the returns date range and "
            "that sector stocks are present in returns.columns."
        )
    return predictors


# ─────────────────────────────────────────────
# SIGNAL ENHANCEMENT
# ─────────────────────────────────────────────

def enhanced_signal_score(
    brent_return: float,
    predictor: "SectorPredictor",
    feature_dict: Dict,
    threshold: float = 0.03,
) -> Dict:
    """Combine rule-based signal with ML-predicted return."""
    pred_ret = predictor.predict_single(feature_dict)
    strength = predictor.signal_strength(pred_ret)
    oil_sig  = abs(brent_return) > threshold
    combined = strength * (1.0 if oil_sig else 0.3)

    if combined > 0.55 and pred_ret > 0.005:
        direction = "long"
    elif combined > 0.55 and pred_ret < -0.005:
        direction = "short"
    else:
        direction = "hold"

    return {
        "predicted_return_pct": round(pred_ret * 100, 3),
        "ml_strength":          round(strength,  4),
        "combined_score":       round(combined,  4),
        "direction":            direction,
        "take_trade":           direction != "hold",
    }
