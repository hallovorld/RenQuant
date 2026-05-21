"""SPY-derived per-date regime labels — shared across IC eval, paired-returns
eval, and Optuna objective.

Extracted from scripts/ic_eval_news_sentiment_regime.py + matches the
scheme used by scripts/eval_regime_stratified.py. Single source of
truth so callers can't drift.

Scheme: 9-regime grid = TREND × VOL
  TREND (rolling 60d SPY Sharpe): LOW (<0.5) / MED (0.5-1.5) / HIGH (>1.5)
  VOL   (20d realized vol vs 252d history percentile):
        CALM (<33%) / NORMAL (33-66%) / SPIKED (>66%)
  → 9 buckets: LOW_CALM, LOW_NORMAL, ..., HIGH_SPIKED

PRIME DIRECTIVE (CLAUDE.md §🔴): RenQuant is regime-conditional.
Pooled metrics across regimes are MISLEADING. Use this labeller to
stratify every objective function, IC eval, or A/B verdict.

Reference: Asness-Moskowitz-Pedersen 2013 "Value and Momentum
Everywhere" *J. Finance* 68(3):929 — factor returns are
regime-dependent; conditional analysis reveals structure.
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import pandas as pd


def compute_spy_regime_labels(spy_path: Path,
                              trend_window: int = 60,
                              vol_window: int = 20,
                              vol_hist_window: int = 252) -> pd.DataFrame:
    """Returns DataFrame with columns: date, regime (str like 'HIGH_CALM')."""
    spy = pd.read_parquet(spy_path)
    spy.index = pd.to_datetime(spy.index)
    spy = spy.sort_index()
    spy["ret"] = np.log(spy["close"] / spy["close"].shift(1))
    spy["roll_mean"] = spy["ret"].rolling(
        trend_window, min_periods=trend_window // 2).mean()
    spy["roll_vol"] = spy["ret"].rolling(
        trend_window, min_periods=trend_window // 2).std(ddof=1)
    spy["sharpe60"] = (spy["roll_mean"] / spy["roll_vol"]) * math.sqrt(252.0)
    spy["vol20"] = spy["ret"].rolling(
        vol_window, min_periods=vol_window // 2).std(ddof=1) * math.sqrt(252.0)

    def _pct_rank(s: pd.Series) -> float:
        if len(s) < 30 or pd.isna(s.iloc[-1]):
            return float("nan")
        last = s.iloc[-1]
        return float((s < last).sum()) / float(len(s) - 1)

    spy["vol_pct"] = spy["vol20"].rolling(
        vol_hist_window, min_periods=30).apply(_pct_rank, raw=False)
    spy["trend_label"] = pd.cut(
        spy["sharpe60"],
        bins=[-np.inf, 0.5, 1.5, np.inf],
        labels=["LOW", "MED", "HIGH"])
    spy["vol_label"] = pd.cut(
        spy["vol_pct"],
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["CALM", "NORMAL", "SPIKED"])
    spy["regime"] = spy["trend_label"].astype(str) + "_" + spy["vol_label"].astype(str)
    out = spy[["regime"]].copy()
    out.index.name = "date"
    return out.reset_index()


def per_regime_cs_ic(preds_df: pd.DataFrame,
                     regime_df: pd.DataFrame,
                     min_samples_per_day: int = 5,
                     min_days_per_regime: int = 10) -> dict[str, float]:
    """Compute per-regime mean cross-sectional Spearman IC.

    Args:
      preds_df: columns date, pred, label (one row per (date, ticker))
      regime_df: columns date, regime
      min_samples_per_day: skip days with fewer cross-sectional samples
      min_days_per_regime: regimes with fewer eligible days → excluded
        from result (under-sampled = unreliable)

    Returns:
      dict regime → mean IC across that regime's days. Excludes
      under-sampled regimes (n_days < min_days_per_regime).
    """
    from scipy.stats import spearmanr

    preds_df = preds_df.copy()
    preds_df["date"] = pd.to_datetime(preds_df["date"])
    regime_df = regime_df.copy()
    regime_df["date"] = pd.to_datetime(regime_df["date"])
    merged = preds_df.merge(regime_df, on="date", how="left")

    out: dict[str, list[float]] = {}
    for d, g in merged.groupby("date"):
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
        if pd.isna(regime) or regime == "nan_nan":
            continue
        out.setdefault(str(regime), []).append(float(r))

    return {regime: float(np.mean(ics))
            for regime, ics in out.items()
            if len(ics) >= min_days_per_regime}


def min_across_regimes(per_regime: dict[str, float]) -> float:
    """Robustness objective: worst-case regime IC. Use as Optuna objective
    so model selection picks regime-robust models, not pooled-mean winners."""
    if not per_regime:
        return float("nan")
    return float(min(per_regime.values()))


__all__ = ["compute_spy_regime_labels", "per_regime_cs_ic",
           "min_across_regimes"]
