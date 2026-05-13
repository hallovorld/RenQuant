"""SpyRegimeLabelTask — objective SPY-derived regime label.

Computes a 2D regime label per bar using SPY price data:
  TREND_LABEL = SPY rolling 60d Sharpe (annualized)
    LOW (<0.5), MED (0.5-1.5), HIGH (>1.5)
  VOL_LABEL = SPY 20d realized vol percentile in 252d history
    CALM (<33pct), NORMAL (33-66pct), SPIKED (>66pct)
  ctx.spy_regime = "<TREND>_<VOL>"  e.g. "HIGH_CALM"

Purpose (per doc/research/2026-05-12-findings-and-next.md):
  The strategy's existing GMM-based regime detector labels 95% of OOS
  days BULL_CALM regardless of true conditions. This task adds a
  PARALLEL, objective signal derived purely from SPY price action that
  can label HIGH_SPIKED periods (carry trade unwinds, Fed shocks) the
  GMM detector misses. Used downstream for regime-conditional ranking
  feature deployment (ApplyGrinoldKahnTransformTask with
  ranking.alpha_to_mu.regime_overrides).

Default: DISABLED. Opt-in via:
  config.regime.spy_regime.enabled = true

When disabled, ctx.spy_regime is set to None. Downstream tasks must
fall back to the global config (no regime override applied).

Reference:
  Asness-Moskowitz-Pedersen 2013 "Value and Momentum Everywhere"
  *J. Finance* 68(3):929 — factor returns are regime-dependent;
  conditional analysis reveals the structure.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from kernel.pipeline.pipeline import Task

log = logging.getLogger("kernel.pipeline.spy_regime")


# Default thresholds (matches scripts/eval_regime_stratified.py)
_DEFAULTS = {
    "trend_window":    60,    # rolling Sharpe window (trading days)
    "vol_window":      20,    # realized vol window
    "vol_hist_window": 252,   # vol percentile lookback (1 year)
    "trend_low":       0.5,   # < 0.5 = LOW trend
    "trend_high":      1.5,   # > 1.5 = HIGH trend
    "vol_calm":        0.33,  # < 33pct = CALM
    "vol_spiked":      0.66,  # > 66pct = SPIKED
}


def compute_spy_regime_label(
    spy_closes: "list | tuple | np.ndarray",
    trend_window:    int   = 60,
    vol_window:      int   = 20,
    vol_hist_window: int   = 252,
    trend_low:       float = 0.5,
    trend_high:      float = 1.5,
    vol_calm:        float = 0.33,
    vol_spiked:      float = 0.66,
) -> str | None:
    """Compute single SPY regime label from a close-price series.

    Args:
        spy_closes: 1-D sequence of SPY closes, latest LAST.
        (other args: see _DEFAULTS)

    Returns:
        "<TREND>_<VOL>" string, or None if insufficient data
        (need at least ``vol_window + vol_hist_window`` closes).

    Fail-open: any malformed input → returns None (caller treats as
    "no regime info available, fall back to global config").
    """
    closes = np.asarray(list(spy_closes), dtype=float)
    closes = closes[np.isfinite(closes)]
    n = len(closes)
    min_needed = max(trend_window, vol_window + vol_hist_window) + 1
    if n < min_needed:
        return None
    # Daily log-returns
    rets = np.diff(np.log(closes))
    # Trend = rolling Sharpe of latest `trend_window` returns
    recent = rets[-trend_window:]
    mu = float(np.mean(recent))
    sd = float(np.std(recent, ddof=1))
    if sd <= 0 or not math.isfinite(sd):
        return None
    trend_sharpe = (mu / sd) * math.sqrt(252.0)
    # Vol = latest `vol_window` annualized vol
    vol_recent = rets[-vol_window:]
    if len(vol_recent) < 2:
        return None
    vol_now = float(np.std(vol_recent, ddof=1)) * math.sqrt(252.0)
    # Vol percentile vs `vol_hist_window` history of rolling 20d vols
    vol_history = []
    start_pos = max(0, n - 1 - vol_hist_window)
    for end in range(start_pos + vol_window, n - 1):  # exclude today (vol_now)
        window = rets[end - vol_window:end]
        if len(window) >= 2:
            v = float(np.std(window, ddof=1)) * math.sqrt(252.0)
            if math.isfinite(v) and v > 0:
                vol_history.append(v)
    if len(vol_history) < 30:
        return None
    vol_pct = float(np.mean(np.asarray(vol_history) < vol_now))
    # Discretize
    if trend_sharpe < trend_low:
        trend_label = "LOW"
    elif trend_sharpe > trend_high:
        trend_label = "HIGH"
    else:
        trend_label = "MED"
    if vol_pct < vol_calm:
        vol_label = "CALM"
    elif vol_pct > vol_spiked:
        vol_label = "SPIKED"
    else:
        vol_label = "NORMAL"
    return f"{trend_label}_{vol_label}"


class SpyRegimeLabelTask(Task):
    """Pipeline task that writes ``ctx.spy_regime`` per bar.

    Reads:  ctx.ohlcv['SPY'] (close column), config.regime.spy_regime
    Writes: ctx.spy_regime (str like "HIGH_CALM", or None if disabled
            / insufficient data)

    OFF by default. Opt-in via:
        config.regime.spy_regime.enabled = true

    Wired into RegimeJob as a sibling of GMMTask/BEAROverrideTask —
    runs alongside, doesn't replace. Downstream regime-conditional
    ranking tasks read ctx.spy_regime to choose IC, vol-target, etc.
    """
    name = "SpyRegimeLabelTask"

    def run(self, ctx) -> bool | None:
        cfg = (ctx.config or {}).get("regime", {}).get("spy_regime", {})
        if not cfg.get("enabled", False):
            ctx.spy_regime = None  # explicit None so downstream can detect
            return None
        spy_df = (ctx.ohlcv or {}).get("SPY")
        if spy_df is None or "close" not in spy_df.columns or len(spy_df) == 0:
            log.warning("SpyRegimeLabelTask: SPY OHLCV missing — skipping")
            ctx.spy_regime = None
            return None
        params = {k: cfg.get(k, v) for k, v in _DEFAULTS.items()}
        try:
            label = compute_spy_regime_label(spy_df["close"].values, **params)
        except Exception as exc:
            log.warning("SpyRegimeLabelTask: compute failed (%s) — fall back", exc)
            ctx.spy_regime = None
            return None
        ctx.spy_regime = label
        log.info("SpyRegimeLabelTask: spy_regime=%s", label or "INSUFFICIENT_DATA")


__all__ = ["SpyRegimeLabelTask", "compute_spy_regime_label"]
