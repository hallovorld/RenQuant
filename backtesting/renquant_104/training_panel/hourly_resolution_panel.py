"""Stage C — true hourly-resolution training panel for transformer.

Replaces the daily panel (1 row per (ticker, date)) with an hourly panel
(1 row per (ticker, date, hour)). This grows panel-row count by ~5-7×
(US trading session has 6.5 hours / 7 hourly bars per day; minute bars
go higher).

Why: Chen-Pelger-Zhu 2024 ship gate for transformer is panel ≥ 5000
"dates"; our daily panel has ~2500. Hourly resolution: 2500 × ~6 ≈
15,000 → cleanly above the threshold.

Pipeline contract:
  Input:  hourly OHLCV bars (already washed via kernel.intraday_wash)
  Output: pd.DataFrame indexed by (ticker, date, hour) with feature
          columns + a forward_label column.

This module is the FEATURE BUILDER for the hourly panel; the full
training pipeline plumbing (group_sizes, label horizon, transformer
backend wiring) happens at a layer above (Stage C-2 / C-3 commits).

References:
- Hasbrouck 2007 §4 — intraday return autocorr / negative serial corr
- Aït-Sahalia, Yu 2009 — high-frequency noise vs efficient price
- Chen, Pelger, Zhu 2024 — transformer panel size requirements
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from kernel.intraday_wash import (
    add_hour_of_day_features,
    add_sample_weight,
    winsorize_returns,
)

log = logging.getLogger("training_panel.hourly_resolution_panel")


HOURLY_RES_FEATURE_COLS = [
    # Core hourly indicators (computed per-bar)
    "hourly_return",         # close pct-change at hourly grid
    "hourly_log_return",     # ln(close[t]/close[t-1])
    "hourly_vol",             # rolling std of returns over 20 hourly bars
    "hourly_vwap_premium",   # close - vwap, normalized by close
    "hourly_volume_ratio",   # volume / 20-bar rolling mean volume
    # Session-relative
    "session_progress",      # 0=open, 1=close (linear in trading hours)
    "session_cum_return",    # cumulative return since session open
    "intraday_high_dist",    # (close - session_high) / session_high
    "intraday_low_dist",     # (close - session_low)  / session_low
    "overnight_gap",         # session_open vs prev_session_close (NaN for non-first bars)
    # Hour-of-day cyclic encoding (from intraday_wash)
    "hour_of_day_sin",
    "hour_of_day_cos",
]


def _compute_session_relative_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add session-relative columns (cum_return, high/low distance, gap).

    Group by trading session (calendar date), then compute features
    that depend on the session's running high/low/open/close.
    """
    out = df.copy()
    out["_session"] = out.index.normalize()
    grp = out.groupby("_session", sort=True)

    # Session running statistics
    open_per_session  = grp["open"].transform("first")
    high_per_session  = grp["high"].cummax()
    low_per_session   = grp["low"].cummin()
    out["session_cum_return"]   = out["close"] / open_per_session - 1.0
    out["intraday_high_dist"]   = (out["close"] - high_per_session) / high_per_session
    out["intraday_low_dist"]    = (out["close"] - low_per_session)  / low_per_session

    # Session progress: rank of bar within its session, normalised [0, 1]
    rank = grp.cumcount()
    n_per_session = grp["close"].transform("count")
    out["session_progress"] = rank / np.maximum(n_per_session - 1, 1)

    # Overnight gap: first bar of each session = (open - prev_session_close) / prev_session_close
    daily_open  = grp["open"].first()
    daily_close = grp["close"].last()
    prev_close  = daily_close.shift(1)
    overnight   = (daily_open - prev_close) / prev_close.replace(0, np.nan)
    overnight_first_bar = overnight.reindex(out["_session"]).values
    # Only first bar of session has the gap; later bars get NaN
    is_first_bar = (rank == 0).values
    out["overnight_gap"] = np.where(is_first_bar, overnight_first_bar, np.nan)
    return out.drop(columns=["_session"])


def compute_hourly_resolution_features(
    hourly_bars: pd.DataFrame,
    *,
    rolling_window_short: int = 20,    # ~3 sessions of hourly bars
) -> pd.DataFrame:
    """Build per-bar feature frame at hourly resolution.

    Args:
        hourly_bars: OHLCV DataFrame indexed by DatetimeIndex (timezone
            naïve; assumed market time). Must contain `open`, `high`,
            `low`, `close`, `volume`.
        rolling_window_short: number of bars for short-window rolling stats.

    Returns:
        DataFrame with all columns in HOURLY_RES_FEATURE_COLS, indexed
        by the same DatetimeIndex. NaN values appear in:
          - first few rows (warmup for rolling stats)
          - overnight_gap on non-first-bar rows (by design)
    """
    if hourly_bars is None or hourly_bars.empty:
        return pd.DataFrame(columns=HOURLY_RES_FEATURE_COLS)
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(hourly_bars.columns)
    if missing:
        raise KeyError(
            f"compute_hourly_resolution_features: input missing columns "
            f"{sorted(missing)}",
        )

    df = hourly_bars.copy().sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is not None:
        df = df.tz_convert("America/New_York").tz_localize(None)

    # 1. Core hourly indicators
    df["hourly_return"]     = df["close"].pct_change()
    df["hourly_log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["hourly_vol"]        = df["hourly_return"].rolling(
        rolling_window_short, min_periods=5,
    ).std()

    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    rolling_dollar_vol = (typical_price * df["volume"]).rolling(
        rolling_window_short, min_periods=5,
    ).sum()
    rolling_volume_sum = df["volume"].rolling(
        rolling_window_short, min_periods=5,
    ).sum()
    rolling_vwap = rolling_dollar_vol / rolling_volume_sum.replace(0, np.nan)
    df["hourly_vwap_premium"] = (df["close"] - rolling_vwap) / df["close"].replace(0, np.nan)

    rolling_volume_mean = df["volume"].rolling(
        rolling_window_short, min_periods=5,
    ).mean()
    df["hourly_volume_ratio"] = df["volume"] / rolling_volume_mean.replace(0, np.nan)

    # 2. Session-relative features
    df = _compute_session_relative_features(df)

    # 3. Hour-of-day cyclic encoding (from intraday_wash)
    df = add_hour_of_day_features(df)

    # Restrict to the canonical column set (drop OHLCV; keep features)
    keep_cols = [c for c in HOURLY_RES_FEATURE_COLS if c in df.columns]
    return df[keep_cols]


def build_hourly_resolution_panel(
    hourly_bars_per_ticker: dict[str, pd.DataFrame],
    *,
    label_horizon_bars: int = 1,
    benchmark_bars: pd.DataFrame | None = None,
    apply_wash: bool = True,
) -> pd.DataFrame:
    """Build the cross-ticker hourly panel keyed by (ticker, date, hour).

    Args:
        hourly_bars_per_ticker: dict mapping ticker → hourly OHLCV.
        label_horizon_bars: forward-return horizon (in hourly bars).
            1 = 1-hour ahead; 7 ≈ 1-day ahead for a 7h session.
        benchmark_bars: SPY hourly bars; if provided, label is excess
            return vs SPY at same horizon. Otherwise raw return.
        apply_wash: if True, apply winsorize + sample_weight at load.

    Returns:
        Long-format DataFrame with index (ticker, datetime), feature
        columns from HOURLY_RES_FEATURE_COLS, plus:
          forward_excess_return  — label
          _sample_weight          — 0 for low-vol bars (skip in training)
    """
    if not hourly_bars_per_ticker:
        return pd.DataFrame()

    benchmark_returns: pd.Series | None = None
    if benchmark_bars is not None and not benchmark_bars.empty:
        bm = benchmark_bars.copy().sort_index()
        if bm.index.tz is not None:
            bm = bm.tz_convert("America/New_York").tz_localize(None)
        benchmark_returns = bm["close"].pct_change(label_horizon_bars).shift(
            -label_horizon_bars,
        )

    parts: list[pd.DataFrame] = []
    for ticker, bars in hourly_bars_per_ticker.items():
        if bars is None or bars.empty:
            continue
        df = bars.copy().sort_index()
        if df.index.tz is not None:
            df = df.tz_convert("America/New_York").tz_localize(None)
        if apply_wash:
            df = winsorize_returns(df)
            df = add_sample_weight(df)

        feats = compute_hourly_resolution_features(df)
        # Forward return per ticker
        fwd = df["close"].pct_change(label_horizon_bars).shift(-label_horizon_bars)
        if benchmark_returns is not None:
            bm_aligned = benchmark_returns.reindex(fwd.index)
            label = fwd - bm_aligned
        else:
            label = fwd

        out = feats.copy()
        out["forward_excess_return"] = label
        out["_sample_weight"] = df.get("_sample_weight", 1.0)

        # Multi-index — (ticker, datetime)
        out["ticker"] = ticker
        out = out.set_index("ticker", append=True).reorder_levels(["ticker", out.index.names[0]])
        parts.append(out)

    if not parts:
        return pd.DataFrame()

    panel = pd.concat(parts).sort_index()
    return panel


__all__ = [
    "HOURLY_RES_FEATURE_COLS",
    "build_hourly_resolution_panel",
    "compute_hourly_resolution_features",
]
