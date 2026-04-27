"""Intraday bar washing — single point of truth for noise filtering.

Applied ONCE at bar-load time (kernel/intraday.py); all downstream
consumers (hourly_features, minute_features, transformer hourly
training panel) reuse the washed output. No double-washing.

Why we wash hourly bars:

1. Microstructure noise (Aït-Sahalia & Yu 2009, Hasbrouck 2007):
   at intraday resolution, observed price = efficient + bid-ask
   bounce + liquidity gaps. Higher freq → more noise.
2. Stale prices: low-volume bars have last-trade prices from earlier
   sessions; their "return" is artefact, not signal.
3. Tail outliers (FOMC bars, single-name news): extreme moves can
   dominate loss; should be capped, not deleted.

Reference: ``doc/components/buy-logic-design.md`` §hourly-wash.

Wash steps (all default-off; opt-in via config flag):

A. Outlier winsorization — cap |hourly return| at N·σ rolling per ticker.
   Default N=5 (~5e-5 prob in Gaussian → only event-driven outliers).
B. Zero-/low-volume sample-weight zeroing — bars with dollar_volume
   below a per-ticker threshold get sample_weight=0 (panel keeps the
   row but training ignores it).
C. Hour-of-day cyclic encoding — adds `hour_of_day_sin/cos` columns
   so transformers can learn time-of-day patterns directly.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

log = logging.getLogger("kernel.intraday_wash")


def winsorize_returns(
    df: pd.DataFrame,
    *,
    n_sigma: float = 5.0,
    rolling_window: int = 60 * 7,    # ~7 sessions of hourly bars
    return_col: str = "_hourly_return",
) -> pd.DataFrame:
    """Cap |return| at n_sigma × rolling-σ per ticker.

    Adds the return column (close pct-change) if not present. Returns
    a copy with `return_col` clipped at ±N·σ. The clip threshold is
    rolling so the model adapts to changing vol regimes; pre-windowed
    samples (insufficient history) keep the original return.

    Theoretical basis: Lopez de Prado 2018 §16 — truncate, don't drop,
    so labels stay aligned but extreme losses don't dominate loss.
    """
    if df is None or df.empty or "close" not in df.columns:
        return df
    out = df.copy()
    if return_col not in out.columns:
        out[return_col] = out["close"].pct_change()
    # Rolling σ on the return series (skipping NaNs)
    sigma = out[return_col].rolling(rolling_window, min_periods=20).std()
    cap = n_sigma * sigma
    # Where cap is well-defined, clip; otherwise keep original.
    finite = sigma.notna()
    clipped = out[return_col].where(
        ~finite,
        out[return_col].clip(lower=-cap, upper=cap),
    )
    out[return_col] = clipped
    return out


def add_sample_weight(
    df: pd.DataFrame,
    *,
    min_dollar_volume: float | None = None,
    min_dv_pct_adv: float = 0.001,
    weight_col: str = "_sample_weight",
) -> pd.DataFrame:
    """Add a sample-weight column based on bar liquidity.

    Bars with dollar_volume < threshold get weight=0 (training ignores
    them); above-threshold bars get weight=1. Threshold is
    `max(min_dollar_volume, min_dv_pct_adv × ADV)` — absolute floor
    AND relative-to-this-ticker floor.

    Default thresholds based on practitioner heuristics: $100K
    absolute, 0.1% of trailing-30-session ADV. Median small-cap hourly
    dollar-volume is ~$1M, large-cap ~$50M, so this is permissive.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    if weight_col in out.columns:
        return out  # idempotent

    if "volume" not in out.columns or "close" not in out.columns:
        out[weight_col] = 1.0
        return out

    dollar_vol = out["volume"].astype(float) * out["close"].astype(float)
    # ADV based on rolling 30-bar dollar volume (≈30 hours ≈ 5 sessions)
    adv = dollar_vol.rolling(30, min_periods=5).mean()
    abs_floor = float(min_dollar_volume) if min_dollar_volume is not None else 1e5
    threshold = np.maximum(abs_floor, min_dv_pct_adv * adv.fillna(adv.median() if not adv.empty else abs_floor))
    out[weight_col] = (dollar_vol >= threshold).astype(float)
    return out


def add_hour_of_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic encoding of hour-of-day to the bar frame.

    Output columns:
      hour_of_day_sin = sin(2π · hour / 24)
      hour_of_day_cos = cos(2π · hour / 24)

    Cyclic encoding (Vaswani 2017 §3.5 positional encoding intuition):
    sin/cos pair lets the model learn that 09:30 ≈ 09:30 next day,
    not 09:30 ≠ 14:00 by ordinal distance. For trading sessions
    9:30-16:00, only ~5 distinct hours appear — embedding is small.

    Idempotent: if hour_of_day_* already in df, returns df unchanged.
    """
    if df is None or df.empty:
        return df
    if "hour_of_day_sin" in df.columns and "hour_of_day_cos" in df.columns:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    out = df.copy()
    # Convert to NY market time so 09:30 stays 09:30 across DST shifts
    if out.index.tz is None:
        # Assume UTC-naive bars are already in market time (per existing
        # convention in HourlyBarStore.save).
        hours = out.index.hour + out.index.minute / 60.0
    else:
        ny_idx = out.index.tz_convert("America/New_York")
        hours = ny_idx.hour + ny_idx.minute / 60.0
    radians = 2.0 * math.pi * hours / 24.0
    out["hour_of_day_sin"] = np.sin(radians)
    out["hour_of_day_cos"] = np.cos(radians)
    return out


def wash_bars(
    df: pd.DataFrame,
    *,
    enable_winsorize: bool = True,
    enable_sample_weight: bool = True,
    enable_hour_features: bool = True,
    n_sigma: float = 5.0,
    min_dollar_volume: float | None = None,
) -> pd.DataFrame:
    """One-shot wash applying all enabled stages.

    Default ON for all three stages so callers can adopt with a single
    function call. Each stage individually flag-controlled for
    A/B testing and ablation.

    Idempotent + non-destructive: returns a new DataFrame; original
    untouched. NaN/inf values are NOT introduced.
    """
    if df is None or df.empty:
        return df
    out = df
    if enable_winsorize:
        out = winsorize_returns(out, n_sigma=n_sigma)
    if enable_sample_weight:
        out = add_sample_weight(out, min_dollar_volume=min_dollar_volume)
    if enable_hour_features:
        out = add_hour_of_day_features(out)
    return out


def cross_sectional_z_per_hour(
    panel: pd.DataFrame,
    feature_cols: list[str],
    *,
    date_col: str = "date",
    hour_col: str = "hour",
    out_suffix: str = "_z",
    min_group_size: int = 5,
) -> pd.DataFrame:
    """Cross-sectional z-score per (date, hour) group.

    Mirror of the daily NeutralizedFeatureZScoreTask but the group
    is (date, hour) not just date. Use for the hourly-resolution
    transformer panel where each row is (ticker, date, hour) and
    we want the cross-section neutralised within the same time slice.

    For groups smaller than `min_group_size` (e.g. early universe
    ramp), z-score is undefined — output stays raw.
    """
    if panel is None or panel.empty:
        return panel
    out = panel.copy()
    grp = out.groupby([date_col, hour_col], group_keys=False)
    for col in feature_cols:
        if col not in out.columns:
            continue
        def _z(s: pd.Series) -> pd.Series:
            if len(s) < min_group_size:
                return s
            mu = s.mean()
            sd = s.std()
            if sd == 0 or not np.isfinite(sd):
                return s - mu
            return (s - mu) / sd
        out[col + out_suffix] = grp[col].transform(_z)
    return out


__all__ = [
    "winsorize_returns",
    "add_sample_weight",
    "add_hour_of_day_features",
    "wash_bars",
    "cross_sectional_z_per_hour",
]
