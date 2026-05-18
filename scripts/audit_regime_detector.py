#!/usr/bin/env python3
"""Audit the regime detector against 5 known-objective windows.

Hypothesis (from 2026-05-17 dense panel): brief crisis windows (SVB,
DeepSeek+tariff, 2022-Q2 start) get labeled BULL_CALM because:
  - `hard_bear` needs 20-day vol > 35% (annualized) OR 20-day ret < -8%.
    Brief 1-2 week crashes don't cross either threshold.
  - `Hurst MOMENTUM + below_MA200` is the only BEAR fallback. Brief
    crashes don't break MA200.
  - CHOPPY needs `Hurst < 0.52` — SPY almost never anti-persistent.

This script replays the *exact same* per-day inputs that
`task_regime.py::BEAROverrideTask` and `RegimeFinalizeTask` use, and
prints WHICH check would have fired for each day. Then we can decide
the right calibration fix (lower vol threshold? add 5-day BEAR trigger?
resurrect CHOPPY via realized vol instead of Hurst?).

Usage:
  python scripts/audit_regime_detector.py
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.regime import compute_hurst  # noqa: E402

# Same defaults as strategy_config.golden.json::regime
BEAR_VOL_THR     = 0.35
BEAR_RET_THR     = -0.08
VOL_WINDOW       = 20
HURST_WINDOW     = 63
HURST_TRENDING   = 0.65
HURST_REVERSION  = 0.52
# 2026-05-17 fix A+C
BEAR_VOL_THR_5D  = 0.25
BEAR_RET_THR_5D  = -0.04
VOL_WINDOW_5D    = 5
CHOPPY_BASELINE_WINDOW = 60
CHOPPY_VOL_RATIO       = 1.5
CHOPPY_DRIFT_THR       = 0.02


WINDOWS = [
    ("2022_Q2_BEAR_START",  "2022-03-01", "2022-05-15"),
    ("2022_DEEP_RATEFEAR",  "2022-07-15", "2022-10-01"),  # W4 of dense — only one that fired BEAR
    ("2023_Q1_SVB",         "2023-01-15", "2023-04-01"),
    ("2024_AUG_VOL_SPIKE",  "2024-06-15", "2024-08-31"),
    ("2025_DEEPSEEK_TARIFF","2024-12-01", "2025-03-01"),
]


def load_spy(start: str, end: str) -> pd.DataFrame:
    """Load SPY OHLCV. Try local parquet first, fall back to yfinance."""
    local = REPO / "data" / "ohlcv" / "SPY" / "1d.parquet"
    if local.exists():
        df = pd.read_parquet(local)
        df.index = pd.to_datetime(df.index)
        # Need 200 bars of history BEFORE start for MA200
        load_start = (pd.to_datetime(start) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
        return df.loc[load_start:end].copy()
    # fallback
    import yfinance as yf
    load_start = (pd.to_datetime(start) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    df = yf.download("SPY", start=load_start, end=end, progress=False, auto_adjust=False)
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    return df


def predict_regime(close: float, ma50: float, ma200: float,
                   hurst: float, vol_20d: float, ret_20d: float) -> tuple[str, dict]:
    """Replay task_regime.py routing logic. Returns (regime, evidence_dict)."""
    hard_bear = False
    if math.isfinite(vol_20d) and math.isfinite(ret_20d):
        hard_bear = (vol_20d > BEAR_VOL_THR) or (ret_20d < BEAR_RET_THR)

    if hurst > HURST_TRENDING:
        hurst_regime = "MOMENTUM"
    elif hurst < HURST_REVERSION:
        hurst_regime = "REVERSION"
    else:
        hurst_regime = "AMBIGUOUS"

    below_ma50  = close < ma50  if math.isfinite(ma50)  else False
    below_ma200 = close < ma200 if math.isfinite(ma200) else False
    bearish_trend = below_ma50 and below_ma200

    # task_regime.py L230-240 — without GMM (we don't have an HMM artifact loaded)
    if hard_bear:
        regime = "BEAR"; reason = "hard_bear (vol>0.35 or ret<-0.08)"
    elif hurst_regime == "MOMENTUM":
        if bearish_trend:
            regime = "BEAR"; reason = "Hurst MOMENTUM + below both MAs"
        else:
            regime = "BULL_CALM"; reason = "Hurst MOMENTUM, not bearish trend"
    elif hurst_regime == "REVERSION":
        regime = "CHOPPY"; reason = "Hurst REVERSION"
    else:
        regime = "BULL_CALM (default)"; reason = "AMBIGUOUS Hurst, no GMM probs"

    return regime, {
        "hard_bear": hard_bear,
        "hurst_regime": hurst_regime,
        "below_ma50": below_ma50,
        "below_ma200": below_ma200,
        "reason": reason,
    }


def audit_window(label: str, start: str, end: str) -> dict:
    spy = load_spy(start, end)
    if spy.empty or len(spy) < 200:
        print(f"\n{label}: insufficient data ({len(spy)} bars)")
        return {}

    spy["ret"] = spy["close"].pct_change()
    spy["ma50"]  = spy["close"].rolling(50).mean()
    spy["ma200"] = spy["close"].rolling(200).mean()

    # rolling 20-day vol + cumulative return
    vol = spy["ret"].rolling(VOL_WINDOW).std(ddof=1) * math.sqrt(252)
    cumret = spy["ret"].rolling(VOL_WINDOW).apply(
        lambda r: np.prod(1.0 + r) - 1.0, raw=True
    )

    # rolling 63-day Hurst (expensive — do it sparsely)
    hurst = pd.Series(index=spy.index, dtype=float)
    for i in range(HURST_WINDOW, len(spy)):
        h = compute_hurst(spy["ret"].iloc[i - HURST_WINDOW:i].values, window=HURST_WINDOW)
        hurst.iloc[i] = h

    # Limit output to the OBJECTIVE window range
    mask = (spy.index >= start) & (spy.index <= end)
    out = pd.DataFrame({
        "close":   spy["close"][mask],
        "ma50":    spy["ma50"][mask],
        "ma200":   spy["ma200"][mask],
        "vol20":   vol[mask],
        "ret20":   cumret[mask],
        "hurst":   hurst[mask],
    })

    counts = {"BEAR": 0, "CHOPPY": 0, "BULL_CALM": 0, "BULL_CALM (default)": 0}
    reasons = {}
    for ts, row in out.iterrows():
        if not (np.isfinite(row["hurst"]) and np.isfinite(row["ma200"])):
            continue
        regime, ev = predict_regime(
            close=row["close"], ma50=row["ma50"], ma200=row["ma200"],
            hurst=row["hurst"], vol_20d=row["vol20"], ret_20d=row["ret20"],
        )
        counts[regime] = counts.get(regime, 0) + 1
        reasons.setdefault(regime, ev["reason"])

    n_total = sum(counts.values())
    print(f"\n=== {label}  ({start} → {end},  n={n_total} bars) ===")
    for r, n in sorted(counts.items(), key=lambda x: -x[1]):
        pct = (n / n_total * 100) if n_total else 0
        print(f"  {r:<22s} {n:>4d}  ({pct:>5.1f}%)  reason: {reasons.get(r,'')}")
    # Print headline stats for the window
    spy_start = out["close"].iloc[0]; spy_end = out["close"].iloc[-1]
    print(f"  SPY: {spy_start:.2f} → {spy_end:.2f}  ({(spy_end/spy_start-1)*100:+.1f}%)")
    print(f"  vol20 min/max:  {out['vol20'].min():.2%} / {out['vol20'].max():.2%}    (BEAR thr={BEAR_VOL_THR:.0%})")
    print(f"  ret20 min/max:  {out['ret20'].min():+.2%} / {out['ret20'].max():+.2%}    (BEAR thr={BEAR_RET_THR:+.0%})")
    print(f"  hurst min/max:  {out['hurst'].min():.3f} / {out['hurst'].max():.3f}  (MOM={HURST_TRENDING}, REV={HURST_REVERSION})")
    return {"counts": counts, "window": (start, end)}


def main():
    print("=== Regime detector audit — replay routing on 5 known windows ===")
    print(f"Defaults: BEAR_VOL_THR={BEAR_VOL_THR}, BEAR_RET_THR={BEAR_RET_THR}, "
          f"VOL_WINDOW={VOL_WINDOW}, HURST=({HURST_REVERSION},{HURST_TRENDING})")
    results = {}
    for label, start, end in WINDOWS:
        results[label] = audit_window(label, start, end)
    print()
    print("=== summary ===")
    print(f"{'window':<24} {'BEAR':>6} {'CHOPPY':>6} {'BULL':>6} {'BUL_def':>8}")
    for label in results:
        c = results[label].get("counts", {})
        print(f"{label:<24} {c.get('BEAR',0):>6} {c.get('CHOPPY',0):>6} "
              f"{c.get('BULL_CALM',0):>6} {c.get('BULL_CALM (default)',0):>8}")


if __name__ == "__main__":
    main()
