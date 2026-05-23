"""Inference-time technical indicators — no training labels.

Self-contained: only numpy, pandas.  No common/ imports.
Used by models.py (feature building) and main.py (LEAN feature frame).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


# ── Individual indicators ─────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return 100 - (100 / (1 + avg_gain / avg_loss.replace(0, float("nan"))))


def compute_macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast  = close.ewm(span=fast,   adjust=False).mean()
    ema_slow  = close.ewm(span=slow,   adjust=False).mean()
    macd_line = ema_fast - ema_slow
    return macd_line - macd_line.ewm(span=signal, adjust=False).mean()


def compute_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp    = (high + low + close) / 3
    sma   = tp.rolling(period).mean()
    mad   = tp.rolling(period).apply(lambda v: np.mean(np.abs(v - v.mean())), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, float("nan")))


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14,
) -> pd.Series:
    """Wilder-smoothed Average True Range — single source of truth.

    Per Wilder 1978 *New Concepts in Technical Trading Systems* §9:

        TR_t  = max(H_t − L_t, |H_t − C_{t−1}|, |L_t − C_{t−1}|)
        ATR_t = (ATR_{t−1} × (N − 1) + TR_t) / N

    Wilder smoothing is algebraically identical to an EWMA with
    α = 1/N (verify: ATR_t = (1 − 1/N)·ATR_{t−1} + (1/N)·TR_t). We use
    pandas's ``ewm(alpha=1/period, adjust=False)`` to match RSI/MACD's
    smoothing convention in this same module and the ADX implementation
    in ``kernel/regime.py:190-193``. ``min_periods=period`` ensures the
    first ``period − 1`` values are NaN so downstream filters can detect
    insufficient history.

    Called by:
      * ``kernel/pipeline/task_sell.py::PrepareHoldingTask`` — caches
        per-position ``realized_atr_daily`` for L5 ATR-Chandelier trailing.
      * ``kernel/regime.py::compute_adx_for_df`` (internal, should be
        refactored to call this helper).

    Args:
        high / low / close: same-index pandas series.
        period: smoothing window (default 14 = Wilder canonical).

    Returns:
        ATR series, NaN-padded for the first ``period - 1`` rows.
    """
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def compute_bbp(close: pd.Series, period: int = 20) -> pd.Series:
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return (close - sma) / (2 * std.replace(0, float("nan")))


def compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    up_move  = high.diff()
    dn_move  = -low.diff()
    plus_dm  = pd.Series(
        np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(
        np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=high.index)
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    atr      = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, float("nan"))
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, float("nan"))
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def compute_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll).replace(0, float("nan"))


def compute_obv_slope(close: pd.Series, volume: pd.Series, signal_period: int = 20) -> pd.Series:
    obv     = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    obv_ema = obv.ewm(span=signal_period, adjust=False).mean()
    return obv_ema.diff(5) / obv_ema.shift(5).replace(0, float("nan"))


def compute_hurst_proxy(spy_returns: pd.Series, window: int = 20) -> pd.Series:
    """Lag-1 autocorr of SPY returns as a fast Hurst approximation."""
    def _autocorr(x: np.ndarray) -> float:
        if len(x) <= 2 or x.std() == 0:
            return 0.0
        v = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        return v if not math.isnan(v) else 0.0
    return spy_returns.rolling(window).apply(_autocorr, raw=True)


# ── All indicators at once ────────────────────────────────────────────────────

def compute_all(rows: pd.DataFrame, spec: dict | None = None) -> pd.DataFrame | None:
    """Apply all indicators to *rows* (must have open/high/low/close/volume).

    *spec* is the ``indicator_spec`` dict from strategy_config.json.
    Returns None if the result is empty.
    """
    spec  = spec or {}
    rows  = rows.copy()
    close  = rows["close"]
    high   = rows["high"]
    low    = rows["low"]
    volume = rows["volume"]

    rsi_p  = spec.get("rsi", {}).get("period", 14)
    rows["rsi"] = compute_rsi(close, rsi_p)

    macd_s = spec.get("macd", {})
    rows["macd_hist"] = compute_macd_hist(
        close,
        fast=macd_s.get("fast", 12),
        slow=macd_s.get("slow", 26),
        signal=macd_s.get("signal", 9),
    )

    rows["cci"]       = compute_cci(high, low, close, spec.get("cci", {}).get("period", 20))
    rows["bbp"]       = compute_bbp(close, spec.get("bbp", {}).get("period", 20))
    rows["adx"]       = compute_adx(high, low, close, spec.get("adx", {}).get("period", 14))
    rows["williams_r"] = compute_williams_r(high, low, close, spec.get("williams_r", {}).get("period", 14))
    rows["obv_slope"] = compute_obv_slope(close, volume, spec.get("obv", {}).get("signal_period", 20))

    ind_cols = ["rsi", "macd_hist", "cci", "bbp", "adx", "williams_r", "obv_slope"]
    rows = rows.dropna(subset=ind_cols)
    return rows if not rows.empty else None


# ── Alias for callers migrating from common/ ─────────────────────────────────

compute_indicators = compute_all


# ── Regime-context features added to inference frames ────────────────────────

def build_spy_context(spy_df: pd.DataFrame, vol_window: int = 20) -> dict:
    """Return SPY regime-context scalar features for the latest bar.

    Legacy shape — returns scalars from the LAST bar of spy_df. Used by
    the live runner path which always truncates spy_df to today before
    calling. The sim cache path should use `build_spy_context_series`
    for strict causality (scalars from the last bar broadcast across
    all rows introduces lookahead when the cache holds the full range).
    """
    spy_close   = spy_df["close"].astype(float)
    spy_returns = spy_close.pct_change().dropna()
    adx_val     = float(compute_adx(
        spy_df["high"].astype(float),
        spy_df["low"].astype(float),
        spy_close,
    ).iloc[-1]) if len(spy_df) >= 20 else 25.0
    ema50 = spy_close.ewm(span=50, adjust=False).mean()
    # Audit fix IND-1 (Round 2 deep audit, 2026-04-25): same as IND-2
    # but for the scalar form. Was lag-1 autocorr; now real Hurst.
    from kernel.regime import compute_hurst as _compute_hurst  # noqa: PLC0415
    hurst_val = (
        float(_compute_hurst(spy_returns.values[-63:]))
        if len(spy_returns) >= 63 else 0.5
    )
    return {
        "spy_realized_vol": float(spy_returns.iloc[-vol_window:].std() * (252 ** 0.5))
            if len(spy_returns) >= vol_window else 0.0,
        "spy_adx":   adx_val,
        "spy_trend": float(spy_close.iloc[-1] / ema50.iloc[-1])
            if ema50.iloc[-1] > 0 else 1.0,
        "hurst_proxy": hurst_val,
    }


def build_spy_context_series(spy_df: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """Return SPY regime-context features as PER-BAR time series.

    2026-04-24 audit: the scalar form (`build_spy_context`) computes
    from the LAST bar and broadcasts. When sim feature-cache holds the
    full OHLCV range, the "last bar" is 2026-03-26 — future info
    relative to any prior bar. Strictly-causal rolling version below.

    Each output series is causal (uses only data up to and including
    time t). Designed so `build_feature_frame(full_history)` sliced at
    any bar produces the SAME value as building on OHLCV-truncated-to-
    that-bar.
    """
    spy_close   = spy_df["close"].astype(float)
    spy_returns = spy_close.pct_change()

    # Rolling annualised realised vol (trailing vol_window bars)
    realized_vol = spy_returns.rolling(vol_window).std() * (252 ** 0.5)
    realized_vol = realized_vol.fillna(0.0)

    # ADX is already causal (uses only bars up to and including t)
    adx_series = compute_adx(
        spy_df["high"].astype(float),
        spy_df["low"].astype(float),
        spy_close,
    ).fillna(25.0)

    # Close-to-EMA50 ratio at each bar — EMA is causal
    ema50  = spy_close.ewm(span=50, adjust=False).mean()
    trend  = (spy_close / ema50.replace(0, float("nan"))).fillna(1.0)

    # Audit fix IND-2 (Round 2 deep audit, 2026-04-25): pre-fix this
    # was a 20-bar AR(1) lag-1 autocorrelation under the misleading name
    # `hurst_proxy`. After TF-3 (training/features.py) + LR-1 (live/runner.py)
    # were migrated to the real R/S Hurst exponent via
    # kernel.regime.rolling_hurst, this code path was the LAST holdout
    # — `build_spy_context_series → build_feature_frame` is what the
    # per-ticker tournament inference (Classification / QLearning /
    # XGBoost / Manual via score_artifact) feeds at scoring time.
    # Pre-fix, training fed real Hurst but inference fed lag-1 autocorr
    # → train/inference parity break per ticker model.
    from kernel.regime import rolling_hurst as _rolling_hurst  # noqa: PLC0415
    hurst = _rolling_hurst(spy_returns, window=63).fillna(0.0)

    return pd.DataFrame({
        "spy_realized_vol": realized_vol,
        "spy_adx":          adx_series,
        "spy_trend":        trend,
        "hurst_proxy":      hurst,
    })


def build_feature_frame(
    stock_rows: pd.DataFrame,
    spy_rows: pd.DataFrame,
    spec: dict,
    vol_window: int = 20,
) -> pd.DataFrame | None:
    """Build the relative-feature frame used at inference time.

    Mirrors the notebook training cell (cell 11) and LEAN's _build_feature_frame.
    Relative features: RSI/ADX ratio-ised to SPY, MACD/CCI/BBP/Williams R/OBV differenced.
    Regime context: spy_realized_vol, spy_adx, spy_trend, hurst_proxy (scalar, broadcast).
    """
    stock_ind = compute_all(stock_rows, spec)
    spy_ind   = compute_all(spy_rows,   spec)
    if stock_ind is None or spy_ind is None:
        return None

    # Regime context — PER-BAR causal time series (2026-04-24 fix).
    # Prior scalar broadcast introduced lookahead when callers passed
    # the full OHLCV range (sim feature cache). Strictly-causal series
    # makes build_feature_frame(full) equivalent to build_feature_frame
    # (truncated) when indexed at the same bar.
    common_idx = stock_ind.index.intersection(spy_ind.index)
    if len(common_idx) < 10:
        return None
    spy_slice = spy_rows.loc[:common_idx[-1]]   # only bars up to the LAST common index
    ctx_series = build_spy_context_series(spy_slice, vol_window=vol_window)
    return assemble_feature_frame_from_indicators(stock_ind, spy_ind, ctx_series)


def assemble_feature_frame_from_indicators(
    stock_ind: pd.DataFrame,
    spy_ind: pd.DataFrame,
    spy_context: pd.DataFrame,
) -> pd.DataFrame | None:
    """Assemble inference features from precomputed stock/SPY indicators.

    This is the single assembly path for both live-style one-shot feature
    building and SimAdapter's cache prebuild. The sim cache can precompute
    SPY indicators and SPY regime context once per run, then reuse them for
    every ticker without changing the mathematical contract of
    :func:`build_feature_frame`.
    """

    common_idx = stock_ind.index.intersection(spy_ind.index)
    if len(common_idx) < 10:
        return None

    stock_ind = stock_ind.loc[common_idx]
    spy_ind   = spy_ind.loc[common_idx]

    ratio_features = {"rsi", "adx"}
    diff_features  = {"macd_hist", "cci", "bbp", "williams_r", "obv_slope"}

    result = pd.DataFrame(index=common_idx)
    result["close"] = stock_ind["close"]

    for col in ratio_features | diff_features:
        if col in ratio_features:
            result[col] = stock_ind[col] / spy_ind[col].replace(0, float("nan"))
        else:
            result[col] = stock_ind[col] - spy_ind[col]

    # Trend / relative momentum
    close     = stock_ind["close"]
    spy_close = spy_ind["close"]
    ema50     = close.ewm(span=50,  adjust=False).mean()
    ema200    = close.ewm(span=200, adjust=False).mean()
    result["trend"]      = close / ema50
    result["trend_long"] = close / ema200
    rel_price = close / spy_close.replace(0, float("nan"))
    result["rel_mom_20d"] = rel_price.pct_change(20)
    result["rel_mom_60d"] = rel_price.pct_change(60)

    # Reindex to common_idx and forward-fill (rolling windows leave
    # the first N bars as NaN; downstream dropna handles them).
    for col in ("spy_realized_vol", "spy_adx", "spy_trend", "hurst_proxy"):
        if col in spy_context.columns:
            result[col] = spy_context[col].reindex(common_idx).ffill()

    result = result.dropna()
    return result if not result.empty else None
