"""Training-time feature frame construction for renquant_103.

Builds per-ticker labelled DataFrames with relative features vs SPY,
rolling regime-context columns, and forward-return labels.

Used by the tournament (Cell 7) to produce training data.
LEAN / live inference uses kernel.indicators.build_feature_frame() instead
(broadcast scalar context, no labels).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from kernel.indicators import compute_all as _compute_all

_RATIO_FEATURES = {"rsi", "adx"}
_DIFF_FEATURES  = {"macd_hist", "cci", "bbp", "williams_r", "obv_slope"}


def build_training_features(
    ticker: str,
    ohlcv: dict[str, pd.DataFrame],
    indicator_spec: dict,
    lookahead: int,
    threshold: float,
) -> pd.DataFrame | None:
    """Build a labelled feature frame for one ticker.

    close = stock/SPY*100 — relative price prevents bull-market always-buy bias.
    Regime context (spy_realized_vol, spy_adx, spy_trend, hurst_proxy) are
    rolling per-bar values — accurate for training; inference broadcasts scalars.

    Returns DataFrame with features + fwd_return + label, or None on failure.
    """
    if ticker not in ohlcv or "SPY" not in ohlcv:
        return None

    stock_ind = _compute_all(ohlcv[ticker], indicator_spec)
    spy_ind   = _compute_all(ohlcv["SPY"],  indicator_spec)
    if stock_ind is None or spy_ind is None:
        return None

    common_idx = stock_ind.index.intersection(spy_ind.index)
    s = stock_ind.loc[common_idx].copy()
    p = spy_ind.loc[common_idx].copy()

    result = pd.DataFrame(index=common_idx)

    # Relative close: stock / SPY * 100
    spy_close_aligned = ohlcv["SPY"]["close"].reindex(common_idx)
    result["close"] = s["close"] / spy_close_aligned.replace(0, np.nan) * 100

    for col in _RATIO_FEATURES:
        if col in s.columns and col in p.columns:
            result[col] = s[col] / p[col].replace(0, np.nan)
    for col in _DIFF_FEATURES:
        if col in s.columns and col in p.columns:
            result[col] = s[col] - p[col]

    close  = s["close"]
    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    result["trend"]      = close / ema50
    result["trend_long"] = close / ema200
    rel_price = close / p["close"].replace(0, np.nan)
    result["rel_mom_20d"] = rel_price.pct_change(20)
    result["rel_mom_60d"] = rel_price.pct_change(60)

    # Rolling per-bar regime context
    spy_rets = ohlcv["SPY"]["close"].pct_change().reindex(common_idx)
    spy_adx  = p["adx"] if "adx" in p.columns else pd.Series(25.0, index=common_idx)
    spy_ema50 = ohlcv["SPY"]["close"].ewm(span=50, adjust=False).mean().reindex(common_idx)
    spy_close_full = ohlcv["SPY"]["close"].reindex(common_idx)
    result["spy_realized_vol"] = spy_rets.rolling(20).std() * np.sqrt(252)
    result["spy_adx"]          = spy_adx
    result["spy_trend"]        = spy_close_full / spy_ema50.replace(0, np.nan)
    # Audit fix TF-3 (2026-04-25): pre-fix `hurst_proxy` was just lag-1
    # autocorrelation under a misleading name. Now compute the real
    # Hurst exponent (rescaled-range / R/S) over a 63-day window via
    # the existing kernel.regime helper. Same column name preserved
    # for backwards-compat with downstream model artifacts'
    # `feature_columns`. Cleaner regime context = better model.
    from kernel.regime import rolling_hurst as _rolling_hurst  # noqa: PLC0415
    result["hurst_proxy"] = _rolling_hurst(spy_rets, window=63).reindex(common_idx)

    # Supervised labels: stock outperformance vs SPY over lookahead days
    stock_fwd = ohlcv[ticker]["close"].pct_change(lookahead).shift(-lookahead)
    spy_fwd   = ohlcv["SPY"]["close"].pct_change(lookahead).shift(-lookahead)
    rel_fwd   = stock_fwd - spy_fwd.reindex(stock_fwd.index)
    result["fwd_return"] = rel_fwd.reindex(common_idx)
    result["label"] = np.where(result["fwd_return"] >  threshold,  1,
                      np.where(result["fwd_return"] < -threshold, -1, 0))

    # Audit TF-6 reconsidered (2026-04-25): the per-ticker tournament
    # consumers (Classification / Q-learning / XGBoost) require every
    # row to have non-NaN features — they have no native imputation
    # path. The panel pipeline handles its own warm-up via
    # build_panel_frame's `iloc[min_history_days:]` slice BEFORE
    # concatenating, so the panel path doesn't pay the dropna cost
    # twice. Keeping `dropna()` is correct for the tournament path
    # and harmless for the panel path. Documented here so future
    # readers don't try to "fix" this again.
    result = result.dropna()
    return result if not result.empty else None


def build_all_training_features(
    watchlist: list[str],
    ohlcv: dict[str, pd.DataFrame],
    indicator_spec: dict,
    lookahead: int,
    threshold: float,
) -> dict[str, pd.DataFrame]:
    """Build labelled feature frames for all watchlist tickers in parallel."""

    def _build(ticker: str):
        df = build_training_features(ticker, ohlcv, indicator_spec, lookahead, threshold)
        return ticker, df

    feature_frames: dict[str, pd.DataFrame] = {}
    # ThreadPoolExecutor: numpy/pandas release the GIL during computation,
    # so multiple tickers genuinely run in parallel.
    with ThreadPoolExecutor() as pool:
        futures = {pool.submit(_build, t): t for t in watchlist}
        # Collect results in original watchlist order for clean output
        results = {}
        for future in as_completed(futures):
            ticker, df = future.result()
            results[ticker] = df

    for ticker in watchlist:
        df = results.get(ticker)
        if df is None or df.empty:
            print(f"  {ticker}: no feature frame")
            continue
        lv = df["label"].value_counts().to_dict()
        print(f"  {ticker}: {len(df)} rows  "
              f"buy={lv.get(1,0)}  sell={lv.get(-1,0)}  hold={lv.get(0,0)}")
        feature_frames[ticker] = df

    print(f"\nBuilt frames for {len(feature_frames)} / {len(watchlist)} symbols.")
    return feature_frames
