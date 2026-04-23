"""Hourly-bar derived features for the panel (Plan G, 2026-04-23).

Aggregates intra-day structure into six per-(ticker, date) signals that the
daily panel can consume:

  morning_drift        — (hr1_close − open) / open
  afternoon_drift      — (close − hr1_close) / hr1_close
  vwap_premium         — (close − intraday_vwap) / intraday_vwap
  vol_ratio            — last-hour volume / first-hour volume
  intraday_realized_vol — std of the hourly log-returns across the session
  overnight_gap        — (open_today − close_prev_day) / close_prev_day

Pure pandas. No Alpaca dependency — the fetcher lives in `kernel/data.py`
(`fetch_intraday_bars`). This module operates on whatever hourly OHLCV
DataFrame the caller supplies so it's easy to unit-test.

Conventions:
  - Input: DataFrame indexed by a DatetimeIndex (tz-aware or not), with
    lowercase columns `[open, high, low, close, volume]`. Rows must be
    sorted by timestamp; multiple sessions per DataFrame OK.
  - Session grouping: rows are grouped by `index.normalize()` (calendar
    date). Sessions with fewer than 2 hourly bars are dropped from the
    output (insufficient data).
  - Output: DataFrame indexed by session date (tz-naive `pd.Timestamp`)
    with the six feature columns. Rows with NaN in critical inputs
    propagate NaN to the affected feature — the panel's
    `np.nan_to_num(0)` guard in training_panel/transformer_model.py
    handles those.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


HOURLY_FEATURE_COLS: list[str] = [
    "morning_drift",
    "afternoon_drift",
    "vwap_premium",
    "vol_ratio",
    "intraday_realized_vol",
    "overnight_gap",
]


def _safe_div(num: float, den: float) -> float:
    if den is None or den == 0 or pd.isna(den):
        return float("nan")
    try:
        return float(num) / float(den)
    except (TypeError, ValueError):
        return float("nan")


def _session_features(session: pd.DataFrame) -> dict[str, float]:
    """Compute features from a single session's hourly bars (sorted)."""
    if len(session) < 2:
        # Need at least 2 hourly bars for morning/afternoon drift.
        return {c: float("nan") for c in HOURLY_FEATURE_COLS
                if c != "overnight_gap"}

    s = session.sort_index()
    open_ = float(s["open"].iloc[0])
    hr1_close = float(s["close"].iloc[0])    # first hourly bar's close
    close = float(s["close"].iloc[-1])

    # morning_drift = change within the first hour of trading
    morning_drift = _safe_div(hr1_close - open_, open_)
    # afternoon_drift = change after the first hour to close
    afternoon_drift = _safe_div(close - hr1_close, hr1_close)

    # VWAP premium — intraday VWAP using typical price (h+l+c)/3 * volume.
    # Missing volume (pre-market crossed bars) silently drops out of the
    # weighted sum via .sum() ignoring NaN.
    typ = (s["high"] + s["low"] + s["close"]) / 3.0
    vol = s["volume"].astype(float)
    denom = vol.sum()
    if denom > 0 and not pd.isna(denom):
        vwap = float((typ * vol).sum() / denom)
    else:
        vwap = float("nan")
    vwap_premium = _safe_div(close - vwap, vwap) if not pd.isna(vwap) else float("nan")

    # Volume ratio — last hour / first hour. Guards against zero-volume
    # opens that create infinities.
    first_vol = float(s["volume"].iloc[0])
    last_vol  = float(s["volume"].iloc[-1])
    vol_ratio = _safe_div(last_vol, first_vol)

    # Intraday realized vol — std of hourly log-returns. With N bars we
    # get N-1 returns; for the common 7-bar session that's 6 values.
    closes = s["close"].astype(float)
    log_returns = np.log(closes / closes.shift(1)).dropna()
    if len(log_returns) >= 2:
        intraday_realized_vol = float(log_returns.std(ddof=1))
    else:
        intraday_realized_vol = float("nan")

    return {
        "morning_drift":         morning_drift,
        "afternoon_drift":       afternoon_drift,
        "vwap_premium":          vwap_premium,
        "vol_ratio":             vol_ratio,
        "intraday_realized_vol": intraday_realized_vol,
        # overnight_gap filled by caller (needs cross-session lookup).
    }


def compute_hourly_features(hourly_bars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly OHLCV bars into per-date feature rows.

    Returns a DataFrame indexed by session date (tz-naive at midnight) with
    columns in HOURLY_FEATURE_COLS order. Sessions with < 2 hourly bars
    are dropped.
    """
    if hourly_bars is None or hourly_bars.empty:
        return pd.DataFrame(columns=HOURLY_FEATURE_COLS)

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(hourly_bars.columns)
    if missing:
        raise KeyError(
            f"compute_hourly_features: input missing columns {sorted(missing)}"
        )

    df = hourly_bars.copy().sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)
    # Strip tz if present — caller may mix tz-aware and naive; we
    # just want per-calendar-day groupings.
    if df.index.tz is not None:
        df = df.tz_convert("America/New_York").tz_localize(None)

    df["_session"] = df.index.normalize()
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for date, session in df.groupby("_session", sort=True):
        feats = _session_features(session.drop(columns=["_session"]))
        rows[date] = feats

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "date"

    # Fill in overnight_gap using cross-session open vs previous-session close.
    # Only well-defined if we have at least 2 consecutive sessions.
    daily_open  = df.groupby("_session")["open"].first()
    daily_close = df.groupby("_session")["close"].last()
    prev_close  = daily_close.shift(1)
    overnight   = (daily_open - prev_close) / prev_close.replace(0, np.nan)
    out["overnight_gap"] = overnight

    # Drop sessions where we couldn't compute any feature (probably
    # empty/degenerate hourly data).
    out = out.dropna(how="all")
    # Guarantee column order for downstream consistency.
    out = out[HOURLY_FEATURE_COLS]
    return out


__all__ = ["HOURLY_FEATURE_COLS", "compute_hourly_features"]
