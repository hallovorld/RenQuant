"""10-minute-bar derived features for the panel (2026-04-24 extension).

Extends hourly_features.py to a finer grid. With ~39 ten-minute bars per
session (6.5 trading hours × 6 bars/hour), we get enough resolution to
measure sub-session microstructure that hourly bars can't resolve:

  morning_drift           — first-10min change (narrower than hourly)
  morning_30min_drift     — first 3 bars (matches US opening auction ramp)
  afternoon_drift         — last-10min change
  closing_30min_drift     — last 3 bars (late-day momentum / auction pressure)
  vwap_premium            — close vs intraday VWAP (more precise w/ 39 pts)
  vol_ratio               — last-30min vol / first-30min vol
  first_hour_vol_pct      — first-6-bar volume / total session volume
  intraday_realized_vol   — stdev of 10-min log-returns (39 samples)
  overnight_gap           — open vs prior close (same as hourly)
  reversal_ratio          — fraction of bar-to-bar return sign flips;
                             high → choppy/reversal; low → trending

NaN propagation: sessions with < 6 bars (not enough to compute the
30-min variants) return NaN for affected cols. Caller's downstream
z-score + sector-median fill absorbs them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


MINUTE_FEATURE_COLS: list[str] = [
    "morning_drift",
    "morning_30min_drift",
    "afternoon_drift",
    "closing_30min_drift",
    "vwap_premium",
    "vol_ratio",
    "first_hour_vol_pct",
    "intraday_realized_vol",
    "overnight_gap",
    "reversal_ratio",
]


def _safe_div(num: float, den: float) -> float:
    if den is None or den == 0 or pd.isna(den):
        return float("nan")
    try:
        return float(num) / float(den)
    except (TypeError, ValueError):
        return float("nan")


def _session_features(session: pd.DataFrame) -> dict[str, float]:
    """Compute features from one session's 10-minute bars (sorted)."""
    n = len(session)
    if n < 2:
        return {c: float("nan") for c in MINUTE_FEATURE_COLS
                if c != "overnight_gap"}

    s = session.sort_index()
    open_   = float(s["open"].iloc[0])
    close_  = float(s["close"].iloc[-1])
    bar1_c  = float(s["close"].iloc[0])  # first-10min close
    bar_last_open = float(s["open"].iloc[-1])  # last-10min open

    # Narrow morning drift (first 10 min) + 30-min variant (first 3 bars)
    morning_drift = _safe_div(bar1_c - open_, open_)
    if n >= 3:
        bar3_c = float(s["close"].iloc[2])
        morning_30min_drift = _safe_div(bar3_c - open_, open_)
    else:
        morning_30min_drift = float("nan")

    # Afternoon / closing drift — mirror structure
    afternoon_drift = _safe_div(close_ - bar_last_open, bar_last_open)
    if n >= 3:
        bar_n3_open = float(s["open"].iloc[-3])
        closing_30min_drift = _safe_div(close_ - bar_n3_open, bar_n3_open)
    else:
        closing_30min_drift = float("nan")

    # VWAP premium — intraday VWAP using typical price × volume
    typ = (s["high"] + s["low"] + s["close"]) / 3.0
    vol = s["volume"].astype(float)
    denom = vol.sum()
    if denom > 0 and not pd.isna(denom):
        vwap = float((typ * vol).sum() / denom)
    else:
        vwap = float("nan")
    vwap_premium = _safe_div(close_ - vwap, vwap) if not pd.isna(vwap) else float("nan")

    # 30-min volume ratio: last 3 bars sum / first 3 bars sum. Fall back
    # to last/first bar if we have < 6 bars (can't do two 30-min windows).
    if n >= 6:
        first_30 = float(vol.iloc[:3].sum())
        last_30  = float(vol.iloc[-3:].sum())
        vol_ratio = _safe_div(last_30, first_30)
    else:
        vol_ratio = _safe_div(float(vol.iloc[-1]), float(vol.iloc[0]))

    # First-hour volume share (first 6 bars = first hour)
    if n >= 6:
        first_hour = float(vol.iloc[:6].sum())
        total_vol  = float(vol.sum())
        first_hour_vol_pct = _safe_div(first_hour, total_vol)
    else:
        first_hour_vol_pct = float("nan")

    # Intraday realized vol — std of bar-to-bar log returns
    closes = s["close"].astype(float)
    log_returns = np.log(closes / closes.shift(1)).dropna()
    if len(log_returns) >= 2:
        intraday_realized_vol = float(log_returns.std(ddof=1))
    else:
        intraday_realized_vol = float("nan")

    # Reversal ratio — fraction of adjacent-bar return sign flips.
    # 1.0 = every bar reverses direction (pure chop); 0.0 = all same sign.
    if len(log_returns) >= 3:
        signs = np.sign(log_returns.values)
        flips = int(np.sum(signs[:-1] * signs[1:] < 0))
        reversal_ratio = flips / (len(signs) - 1)
    else:
        reversal_ratio = float("nan")

    return {
        "morning_drift":         morning_drift,
        "morning_30min_drift":   morning_30min_drift,
        "afternoon_drift":       afternoon_drift,
        "closing_30min_drift":   closing_30min_drift,
        "vwap_premium":          vwap_premium,
        "vol_ratio":             vol_ratio,
        "first_hour_vol_pct":    first_hour_vol_pct,
        "intraday_realized_vol": intraday_realized_vol,
        # overnight_gap filled by caller (needs cross-session lookup).
        "reversal_ratio":        reversal_ratio,
    }


def compute_minute_features(minute_bars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 10-minute OHLCV bars into per-session feature rows.

    Returns a DataFrame indexed by session date (tz-naive midnight) with
    columns in MINUTE_FEATURE_COLS order. Sessions with < 2 bars are
    dropped entirely.
    """
    if minute_bars is None or minute_bars.empty:
        return pd.DataFrame(columns=MINUTE_FEATURE_COLS)

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(minute_bars.columns)
    if missing:
        raise KeyError(
            f"compute_minute_features: input missing columns {sorted(missing)}"
        )

    df = minute_bars.copy().sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is not None:
        df = df.tz_convert("America/New_York").tz_localize(None)

    df["_session"] = df.index.normalize()
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for date, session in df.groupby("_session", sort=True):
        rows[date] = _session_features(session.drop(columns=["_session"]))

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "date"

    # overnight_gap from cross-session open vs prev close
    daily_open  = df.groupby("_session")["open"].first()
    daily_close = df.groupby("_session")["close"].last()
    prev_close  = daily_close.shift(1)
    overnight   = (daily_open - prev_close) / prev_close.replace(0, np.nan)
    out["overnight_gap"] = overnight

    out = out.dropna(how="all")
    out = out[MINUTE_FEATURE_COLS]
    return out


__all__ = ["MINUTE_FEATURE_COLS", "compute_minute_features"]
