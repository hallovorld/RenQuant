"""Partial feature neutralization — momentum/trend only.

Strips the sector-momentum component from the trending features
(`rel_mom_*`, `trend`, `trend_long`) via rolling OLS residualization. The
mean-reversion features (`rsi`, `bbp`, `williams_r`, `cci`) are left alone
— at short horizons they carry reversion alpha that we *don't* want to
neutralize.

We use an **expanding** window for the first `expanding_warmup_days`
bars of history (so early bars still get some residualization rather
than being dropped) and then switch to a **rolling** window. β at bar
t uses strictly-prior data (bar t excluded) so a future bar cannot
leak into the past.

Public API::

    NEUTRALIZE_COLS         — default list of feature columns to neutralize
    compute_sector_momentum — build predictor frames from sector-ETF OHLCV
    neutralize_features     — replace each col with its residual vs sector
"""
from __future__ import annotations

import numpy as np
import pandas as pd


NEUTRALIZE_COLS: list[str] = ["rel_mom_20d", "rel_mom_60d", "trend", "trend_long"]

# Map each feature column to the sector-momentum predictor column.
_PREDICTOR_MAP: dict[str, str] = {
    "rel_mom_20d": "mom_20d",
    "rel_mom_60d": "mom_60d",
    "trend":       "trend",
    "trend_long":  "trend_long",
}


def compute_sector_momentum(
    sector_etf_ohlcv: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """From sector-ETF OHLCV, build {sector: DataFrame[mom_20d, mom_60d, trend, trend_long]}.

    mom_20d    = close.pct_change(20)
    mom_60d    = close.pct_change(60)
    trend      = close / EMA(50)
    trend_long = close / EMA(200)

    Indexed by the ETF's date index.
    """
    out: dict[str, pd.DataFrame] = {}
    for sector, df in sector_etf_ohlcv.items():
        close = df["close"].astype(float)
        ema50  = close.ewm(span=50,  adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        out[sector] = pd.DataFrame({
            "mom_20d":    close.pct_change(20),
            "mom_60d":    close.pct_change(60),
            "trend":      close / ema50.replace(0, np.nan),
            "trend_long": close / ema200.replace(0, np.nan),
        })
    return out


def _residualize(
    feat: pd.Series, pred: pd.Series,
    rolling_window: int, expanding_warmup_days: int,
) -> pd.Series:
    """Residual = feat_t − α_t − β_t · pred_t, where (α_t, β_t) are fit by
    OLS on strictly-prior data.

    For the first `expanding_warmup_days` observations we expand; beyond
    that we roll with `rolling_window` bars. β is undefined until we have
    at least 30 prior joint observations — those residuals are NaN.
    """
    feat = feat.astype(float)
    pred = pred.reindex(feat.index).astype(float)

    # Pair values shifted by 1 bar so "data up to t-1" becomes "rolling window ending at t".
    feat_s = feat.shift(1)
    pred_s = pred.shift(1)

    # Expanding statistics (used for t < expanding_warmup_days).
    min_obs = 30
    exp_cov  = feat_s.expanding(min_periods=min_obs).cov(pred_s)
    exp_var  = pred_s.expanding(min_periods=min_obs).var()
    exp_mf   = feat_s.expanding(min_periods=min_obs).mean()
    exp_mp   = pred_s.expanding(min_periods=min_obs).mean()

    # Rolling statistics (used once we're past warmup).
    roll_cov = feat_s.rolling(rolling_window, min_periods=min_obs).cov(pred_s)
    roll_var = pred_s.rolling(rolling_window, min_periods=min_obs).var()
    roll_mf  = feat_s.rolling(rolling_window, min_periods=min_obs).mean()
    roll_mp  = pred_s.rolling(rolling_window, min_periods=min_obs).mean()

    # bar_index (0..N-1) — switch to rolling once index ≥ expanding_warmup_days
    idx_int = np.arange(len(feat))
    use_roll = idx_int >= expanding_warmup_days

    beta  = np.where(use_roll,  roll_cov / roll_var.replace(0, np.nan),
                                exp_cov  / exp_var.replace(0, np.nan))
    # Audit fix D-2 (2026-04-25): clip β to typical equity-factor range.
    # min_obs=30 produces noisy β; without clipping, ±50 spikes inverted
    # the residualised feature for some tickers.
    beta  = np.clip(beta, -3.0, 5.0)
    mf    = np.where(use_roll,  roll_mf.values,  exp_mf.values)
    mp    = np.where(use_roll,  roll_mp.values,  exp_mp.values)
    alpha = mf - beta * mp

    residual = feat.values - alpha - beta * pred.values
    return pd.Series(residual, index=feat.index)


def neutralize_features(
    feature_frames: dict[str, pd.DataFrame],
    sector_momentum: dict[str, pd.DataFrame],
    ticker_sectors: dict[str, str],
    cols: list[str] = NEUTRALIZE_COLS,
    rolling_window: int = 252,
    expanding_warmup_days: int = 252,
) -> dict[str, pd.DataFrame]:
    """Return new feature_frames with `cols` replaced by residuals vs sector momentum.

    If a ticker's sector has no sector-momentum frame, its feature columns
    are left untouched (pass-through). If a specific `col` has no predictor
    mapping, it's also left untouched.
    """
    out: dict[str, pd.DataFrame] = {}
    for ticker, ff in feature_frames.items():
        df = ff.copy()
        sector = ticker_sectors.get(ticker)
        sec_df = sector_momentum.get(sector) if sector is not None else None
        if sec_df is None:
            out[ticker] = df
            continue
        for col in cols:
            if col not in df.columns:
                continue
            predictor_col = _PREDICTOR_MAP.get(col)
            if predictor_col is None or predictor_col not in sec_df.columns:
                continue
            df[col] = _residualize(
                df[col], sec_df[predictor_col],
                rolling_window=rolling_window,
                expanding_warmup_days=expanding_warmup_days,
            ).values
        out[ticker] = df
    return out
