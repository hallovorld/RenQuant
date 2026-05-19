"""HMM-style per-date regime labels — stateless approximation for IC eval.

Per user 2026-05-18 23:10 mandate: bull_regime_IC uses internal HMM
categorical labels {BULL_CALM, BULL_VOLATILE, BEAR, CHOPPY}, not the
SPY-derived 9-grid. The full stateful detector in
`backtesting/renquant_104/kernel/regime.py::detect_regime` is too heavy
to replay per-date (needs GMM artifact + state + config). This module
provides a stateless approximation reusing the same thresholds.

NOTE: BULL_STRONG appears in some golden config entries but is NOT
emitted by the production detector (only BULL_CALM, BULL_VOLATILE,
CHOPPY, BEAR per kernel/config.py::REGIMES). So `bull_regime_ic`
aggregates {BULL_CALM, BULL_VOLATILE} — the 2 bull labels the detector
actually emits.

Categorical labels (matching kernel/config.py::REGIMES):
  - BEAR: vol_20d > 0.35 OR ret_20d < -0.08 OR vol_5d > 0.25 OR ret_5d < -0.04
  - CHOPPY: vol_5d > vol_60d × 1.5 AND |drift_20d| < 0.02 AND not BEAR
  - BULL_CALM: hurst > 0.65 (trending up, low chop) AND not BEAR/CHOPPY
  - BULL_VOLATILE: everything else

Thresholds source: kernel/regime.py::detect_regime defaults (2026-05-17
post-detector-fix versions).
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import pandas as pd


# Thresholds — keep in sync with kernel/regime.py defaults
BEAR_VOL_20D_THR = 0.35
BEAR_RET_20D_THR = -0.08
BEAR_VOL_5D_THR  = 0.25
BEAR_RET_5D_THR  = -0.04
CHOPPY_VOL_RATIO = 1.5
CHOPPY_DRIFT_TH  = 0.02
HURST_TREND_THR  = 0.65


def _compute_hurst(returns: np.ndarray, window: int = 63) -> float:
    """Simplified rescaled-range Hurst exponent. Returns 0.5 if insufficient."""
    if len(returns) < window:
        return 0.5
    r = returns[-window:]
    mean = r.mean()
    cum = np.cumsum(r - mean)
    R = cum.max() - cum.min()
    S = r.std(ddof=1)
    if S < 1e-9:
        return 0.5
    # Single-window RS heuristic; full Hurst uses multi-window log-log.
    # For BULL_CALM detection, this monotonic transform is sufficient.
    rs = R / (S * math.sqrt(window))
    # Map RS to Hurst-ish in [0, 1]
    return float(np.clip(0.5 + (rs - 1.0) * 0.3, 0.0, 1.0))


def compute_hmm_regime_labels(spy_path: Path,
                                lookback_days: int = 252) -> pd.DataFrame:
    """Per-date HMM-style regime label from SPY OHLCV.

    Returns: DataFrame with columns [date, regime] where regime ∈
    {BULL_CALM, BULL_VOLATILE, BEAR, CHOPPY}.
    """
    spy = pd.read_parquet(spy_path)
    spy.index = pd.to_datetime(spy.index)
    spy = spy.sort_index()
    spy["ret"] = np.log(spy["close"] / spy["close"].shift(1))
    spy = spy.dropna(subset=["ret"])

    out = []
    rets = spy["ret"].values
    dates = spy.index
    for i in range(len(rets)):
        if i < 30:
            out.append({"date": dates[i], "regime": "BULL_CALM"})
            continue
        # 20-day stats
        w20 = rets[max(0, i - 20):i]
        vol_20d = float(np.std(w20, ddof=1) * math.sqrt(252)) if len(w20) >= 5 else 0.0
        ret_20d = float(np.prod(1.0 + w20) - 1.0)
        # 5-day stats
        w5 = rets[max(0, i - 5):i]
        vol_5d = float(np.std(w5, ddof=1) * math.sqrt(252)) if len(w5) >= 3 else 0.0
        ret_5d = float(np.prod(1.0 + w5) - 1.0)
        # 60-day vol baseline
        w60 = rets[max(0, i - 60):i]
        vol_60d = float(np.std(w60, ddof=1) * math.sqrt(252)) if len(w60) >= 30 else vol_20d
        # 20-day drift
        drift_20d = ret_20d  # cumulative 20d return as drift signal
        # Hurst
        hurst = _compute_hurst(rets[:i + 1], window=63)

        # BEAR override (highest priority)
        if (vol_20d > BEAR_VOL_20D_THR or ret_20d < BEAR_RET_20D_THR or
            vol_5d > BEAR_VOL_5D_THR or ret_5d < BEAR_RET_5D_THR):
            regime = "BEAR"
        # CHOPPY: vol cluster + low drift
        elif (vol_60d > 1e-6 and vol_5d > vol_60d * CHOPPY_VOL_RATIO
              and abs(drift_20d) < CHOPPY_DRIFT_TH):
            regime = "CHOPPY"
        # BULL_CALM: trending up
        elif hurst > HURST_TREND_THR:
            regime = "BULL_CALM"
        else:
            regime = "BULL_VOLATILE"
        out.append({"date": dates[i], "regime": regime})

    return pd.DataFrame(out)


def per_hmm_regime_ic(preds_df: pd.DataFrame, hmm_labels: pd.DataFrame,
                      min_samples_per_day: int = 5,
                      min_days_per_regime: int = 5) -> dict[str, float]:
    """Compute per-HMM-regime mean cross-sectional Spearman IC.

    Args:
      preds_df: columns date, pred, label (one row per (date, ticker))
      hmm_labels: from compute_hmm_regime_labels()
      min_samples_per_day: skip days with fewer tickers
      min_days_per_regime: regimes with fewer days excluded

    Returns:
      dict regime → mean IC across that regime's days
    """
    from scipy.stats import spearmanr

    preds_df = preds_df.copy()
    preds_df["date"] = pd.to_datetime(preds_df["date"])
    hmm_labels = hmm_labels.copy()
    hmm_labels["date"] = pd.to_datetime(hmm_labels["date"])
    merged = preds_df.merge(hmm_labels, on="date", how="left")

    by_regime: dict[str, list[float]] = {}
    for _, g in merged.groupby("date"):
        if len(g) < min_samples_per_day:
            continue
        x = g["pred"].values
        y = g["label"].values
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            continue
        r, _ = spearmanr(x, y)
        if np.isnan(r):
            continue
        regime = g["regime"].iloc[0]
        if pd.isna(regime):
            continue
        by_regime.setdefault(str(regime), []).append(float(r))

    return {regime: float(np.mean(ics))
            for regime, ics in by_regime.items()
            if len(ics) >= min_days_per_regime}


def bull_regime_ic(per_regime: dict[str, float]) -> float:
    """Aggregate {BULL_CALM, BULL_VOLATILE} ICs into single criterion.

    User mandate 2026-05-18 23:10: PatchTST↔XGB swap decision uses
    bull_regime_IC. Returns nan if neither bull regime present.
    """
    bull_ics = [per_regime[r] for r in ("BULL_CALM", "BULL_VOLATILE")
                if r in per_regime]
    if not bull_ics:
        return float("nan")
    return float(np.mean(bull_ics))


__all__ = ["compute_hmm_regime_labels", "per_hmm_regime_ic",
           "bull_regime_ic"]
