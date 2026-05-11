"""R-03 (2026-05-11) — Hurst-Ooi-Pedersen 2017 SPY-12M trend overlay.

Hurst, Ooi & Pedersen 2017 ("A Century of Evidence on Trend-Following
Investing", JPM 44(1):15-29) document that a simple long/short rule
based on **12-month past return** has captured trend-following profits
across asset classes and decades. AQR's "TS-Mom" production strategy
uses the same 12-month look-back as a high-confidence regime filter.

Applied as a SPY overlay: when 12-month SPY total return is ≤ 0 (or the
configured threshold), force the portfolio into BEAR regime regardless
of the GMM / Hurst-rolling layers. This is a SOFT, opt-in safety net —
the existing BEAROverrideTask handles severe acute drawdowns
(realized-vol spike + 20d cumret crash). The 12-month rule covers the
slow-motion bear (e.g. 2022 grinding-down regime) that the 20-day
window misses entirely.

Wiring: this Task runs **between BEAROverrideTask and
RegimeFinalizeTask**. It only escalates — once `state.hard_bear=True`,
RegimeFinalizeTask routes to BEAR via the canonical branch. It never
overrides a True back to False.

Config block::

    "regime": {
        "trend_overlay": {
            "enabled":          true,
            "lookback_days":    252,
            "threshold":        0.0
        }
    }

Defaults disable the overlay — golden behaviour preserved.

Fail-open contract: missing SPY frame, fewer than `lookback_days` of
OHLCV, or non-finite return → leave `state.hard_bear` untouched.
"""
from __future__ import annotations

import logging
import math

from .context import InferenceContext
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.trend_overlay")


def compute_spy_trend_return(close_series, lookback_days: int) -> float | None:
    """Total SPY return over `lookback_days` calendar bars.

    `close_series` is a pandas-like Series (the SPY OHLCV `close` column).
    Returns None if too few rows or non-finite arithmetic.
    """
    if close_series is None:
        return None
    try:
        n = len(close_series)
    except TypeError:
        return None
    if n < lookback_days + 1:
        return None
    p_now  = float(close_series.iloc[-1])
    p_then = float(close_series.iloc[-(lookback_days + 1)])
    if not (math.isfinite(p_now) and math.isfinite(p_then) and p_then > 0):
        return None
    ret = (p_now / p_then) - 1.0
    if not math.isfinite(ret):
        return None
    return ret


class TrendOverlayTask(Task):
    """Force ``state.hard_bear=True`` when SPY 12M return ≤ threshold.

    Runs between BEAROverrideTask and RegimeFinalizeTask so the
    finalize step picks up the override through the existing
    ``state.hard_bear or BEAR-GMM > 0.5`` branch (no extra coupling).
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        cfg = ctx.config.get("regime", {}).get("trend_overlay") or {}
        if not cfg.get("enabled", False):
            return None

        lookback  = int  (cfg.get("lookback_days", 252))
        threshold = float(cfg.get("threshold",     0.0))

        spy_df = ctx.ohlcv.get("SPY")
        if spy_df is None or len(spy_df) == 0:
            log.warning(
                "TrendOverlayTask: SPY OHLCV missing — skipping overlay "
                "(fail-open; state.hard_bear unchanged).",
            )
            return None

        if "close" not in spy_df.columns:
            return None

        ret_12m = compute_spy_trend_return(spy_df["close"], lookback)
        if ret_12m is None:
            return None

        triggered = ret_12m <= threshold
        if triggered:
            ctx.regime_state.hard_bear = True
            log.info(
                "TrendOverlayTask: SPY %d-day return %.2f%% ≤ %.2f%% — "
                "forcing hard_bear=True (Hurst-Ooi-Pedersen overlay).",
                lookback, ret_12m * 100, threshold * 100,
            )
        else:
            log.debug(
                "TrendOverlayTask: SPY %d-day return %.2f%% > %.2f%% — "
                "no override (overlay disabled this bar).",
                lookback, ret_12m * 100, threshold * 100,
            )
        return None
