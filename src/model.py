"""
model.py
Two-model stack: Ridge (interpretable baseline) + LightGBM (non-linear upgrade).
Training uses time-based walk-forward cross-validation — no future leakage.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parent.parent
MODELS_DIR  = ROOT / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    # Macro / oil
    "oil_return_1d", "oil_return_2d", "oil_return_5d", "oil_vol_20d",
    "market_ret_1d", "market_vol_20d",
    # Sentiment
    "sentiment_1d", "sentiment_5d",
    # Interaction (the key feature)
    "oil_sent_interact", "oil_direction",
    # Regime
    "vol_regime",
    # Stock-level
    "relative_return",
    "stock_ret_1d", "stock_mom_5d", "stock_mom_20d", "stock_vol_20d",
    "oil_stock_corr_60d", "stock_beta_nifty_60d", "stock_idio_vol_20d",
]


# ─────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────

def _get_XY(panel: pd.DataFrame, feature_cols: List[str] = None):
    fc = feature_cols or FEATURE_COLS
    fc_avail = [c for c in fc if c in panel.columns]
    X = panel[fc_avail]          # keep as DataFrame so feature names flow through
    y = panel["target"].values
    return X, y, fc_avail


def walk_forward_cv(
    panel: pd.DataFrame,
    n_splits: int = 5,
    model_type: str = "ridge",
) -> Dict:
    """
    Time-series walk-forward cross-validation.
    Prevents look-ahead bias: test fold is always strictly after train fold.

    Returns dict with per-fold metrics and overall stats.
    """
    dates   = panel.index.get_level_values("date").unique().sort_values()
    n_splits = min(n_splits, len(dates) - 1)
    tscv    = TimeSeriesSplit(n_splits=n_splits)
    X, y, fc = _get_XY(panel)

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(np.arange(len(dates)))):
        train_dates = dates[train_idx]
        test_dates  = dates[test_idx]

        mask_train  = panel.index.get_level_values("date").isin(train_dates)
        mask_test   = panel.index.get_level_values("date").isin(test_dates)

        X_tr, y_tr = X.iloc[mask_train], y[mask_train]
        X_te, y_te = X.iloc[mask_test],  y[mask_test]

        if len(X_tr) < 20 or len(X_te) < 5:
            continue

        scaler = StandardScaler()
        X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
        X_te_s = pd.DataFrame(scaler.transform(X_te),     columns=X_te.columns)

        model  = _build_model(model_type)
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)

        # IC (information coefficient): rank correlation of predicted vs actual
        from scipy.stats import spearmanr
        ic, _ = spearmanr(y_pred, y_te)

        fold_results.append({
            "fold":     fold,
            "n_train":  int(mask_train.sum()),
            "n_test":   int(mask_test.sum()),
            "r2":       round(r2_score(y_te, y_pred), 4),
            "ic":       round(ic, 4),
            "hit_rate": round(float(np.mean(np.sign(y_pred) == np.sign(y_te))), 4),
        })

    df = pd.DataFrame(fold_results)
    summary = {
        "mean_ic":   round(df["ic"].mean(),   4),
        "mean_r2":   round(df["r2"].mean(),   4),
        "mean_hit":  round(df["hit_rate"].mean(), 4),
        "fold_detail": df.to_dict("records"),
    }
    return summary


def _build_model(model_type: str):
    if model_type == "ridge":
        return RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    elif model_type == "lgbm":
        try:
            import lightgbm as lgb
            return lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                num_leaves=15,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1,
            )
        except ImportError:
            log.warning("LightGBM not installed — using Ridge.")
            return RidgeCV(alphas=[0.1, 1.0, 10.0])
    raise ValueError(f"Unknown model type: {model_type}")


# ─────────────────────────────────────────────
# ALPHA MODEL CLASS
# ─────────────────────────────────────────────

class AlphaModel:
    """
    Wraps a trained model + scaler for stock return prediction.
    Supports Ridge (baseline) and LightGBM (upgrade).
    """

    def __init__(self, model_type: str = "ridge"):
        self.model_type    = model_type
        self.model         = None
        self.scaler        = StandardScaler()
        self.feature_cols_ : List[str] = []
        self.train_end_    : Optional[pd.Timestamp] = None
        self.cv_metrics_   : Dict = {}
        self.feature_importance_: Dict = {}

    # ── Training ─────────────────────────────────────────────────────────

    def fit(
        self,
        panel: pd.DataFrame,
        train_end: str = None,
    ) -> "AlphaModel":
        """
        Fit on panel data up to `train_end` (ISO date string).
        If None, uses all data.
        """
        if train_end:
            train_end_ts = pd.Timestamp(train_end)
            panel = panel.loc[panel.index.get_level_values("date") <= train_end_ts]
            self.train_end_ = train_end_ts

        X, y, fc = _get_XY(panel)
        self.feature_cols_ = fc

        # Walk-forward CV before final fit
        log.info(f"Running walk-forward CV ({self.model_type}) …")
        self.cv_metrics_ = walk_forward_cv(panel, n_splits=5, model_type=self.model_type)
        log.info(f"  CV mean IC: {self.cv_metrics_['mean_ic']}  "
                 f"hit-rate: {self.cv_metrics_['mean_hit']}")

        # Final fit on full training data
        X_scaled = self.scaler.fit_transform(X)
        self.model = _build_model(self.model_type)
        self.model.fit(X_scaled, y)

        # Feature importance
        if hasattr(self.model, "coef_"):
            self.feature_importance_ = dict(zip(fc, self.model.coef_))
        elif hasattr(self.model, "feature_importances_"):
            self.feature_importance_ = dict(zip(fc, self.model.feature_importances_))

        print("\n Top Feature Importances:")
        for k, v in sorted(self.feature_importance_.items(), key=lambda x: -abs(x[1]))[:10]:
            print(f"{k}: {round(v, 4)}")

        return self

    # ── Inference ────────────────────────────────────────────────────────

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        """
        Predict expected return for each (date, stock) row.
        Returns pd.Series with same MultiIndex as panel.
        """
        fc    = [c for c in self.feature_cols_ if c in panel.columns]
        X     = panel[fc].values
        X_s   = self.scaler.transform(X)
        preds = self.model.predict(X_s)
        return pd.Series(preds, index=panel.index, name="predicted_return")

    def predict_latest(self, live_panel: pd.DataFrame) -> pd.Series:
        """Convenience: predict on today's cross-section."""
        return self.predict(live_panel)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, name: str = "alpha_model"):
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info(f"Model saved → {path}")

    @classmethod
    def load(cls, name: str = "alpha_model") -> "AlphaModel":
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Diagnostics ──────────────────────────────────────────────────────

    def top_features(self, n: int = 10) -> pd.Series:
        fi = pd.Series(self.feature_importance_).abs().sort_values(ascending=False)
        return fi.head(n)

    def summary(self) -> Dict:
        return {
            "model_type":   self.model_type,
            "n_features":   len(self.feature_cols_),
            "train_end":    str(self.train_end_),
            "cv_mean_ic":   self.cv_metrics_.get("mean_ic"),
            "cv_mean_hit":  self.cv_metrics_.get("mean_hit"),
            "top_feature":  self.top_features(1).index[0] if self.feature_importance_ else "N/A",
        }


# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT HELPER
# ─────────────────────────────────────────────

def time_split(
    panel: pd.DataFrame,
    train_ratio: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split panel into train and test sets by time (not random).
    train_ratio = fraction of *dates* (not rows) in training set.
    """
    dates      = panel.index.get_level_values("date").unique().sort_values()
    cutoff_idx = int(len(dates) * train_ratio)
    cutoff     = dates[cutoff_idx]

    train = panel.loc[panel.index.get_level_values("date") < cutoff]
    test  = panel.loc[panel.index.get_level_values("date") >= cutoff]

    log.info(f"Train: {len(train)} rows up to {cutoff.date()}  |  "
             f"Test: {len(test)} rows from {cutoff.date()}")
    return train, test
