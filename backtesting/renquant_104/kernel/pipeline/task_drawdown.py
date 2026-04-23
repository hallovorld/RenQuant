"""Drawdown circuit breaker tasks."""
from __future__ import annotations

import logging

from .context import InferenceContext
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.drawdown")


class HWMUpdateTask(Task):
    """Advance high-water mark: hwm = max(hwm, portfolio_value)."""

    def run(self, ctx: InferenceContext) -> bool | None:
        ctx.hwm = max(ctx.hwm, ctx.portfolio_value)
        log.debug("HWMUpdateTask: hwm=%.2f  portfolio=%.2f", ctx.hwm, ctx.portfolio_value)


class DrawdownCircuitTask(Task):
    """Re-evaluate drawdown circuit breaker: set ctx.skip_buys each bar.

    Bug history: before this Task reset skip_buys on recovery, the flag was
    one-way — once drawdown ≥ halt_pct fired a single bar, skip_buys stayed
    True forever (the adapter persists it across bars via ctx.skip_buys).
    In a 2024-2026 sim that produced a 133-day+ no-trade streak in BULL_CALM.

    Now: skip_buys is RECOMPUTED each bar from the current drawdown, so buys
    resume automatically once portfolio value recovers above the threshold.
    The HWM itself is ratcheted by HWMUpdateTask.

    Drawdown resume_pct hysteresis is optional — set regime_params.<regime>
    `drawdown_resume_pct` to a value < halt_pct to require extra recovery
    before re-enabling buys (prevents oscillation on borderline drawdowns).
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        regime_p = ctx.config.get("regime_params", {}).get(ctx.regime, {})
        halt_pct = float(regime_p.get("drawdown_halt_pct", 0.0))

        if ctx.hwm <= 0 or halt_pct <= 0:
            return

        drawdown = (ctx.hwm - ctx.portfolio_value) / ctx.hwm

        if ctx.skip_buys:
            # Already halted — keep halted until drawdown recovers below
            # `drawdown_resume_pct` (defaults to halt_pct for no hysteresis).
            resume_pct = float(regime_p.get("drawdown_resume_pct", halt_pct))
            if drawdown < resume_pct:
                ctx.skip_buys = False
                log.info("DrawdownCircuitTask: resumed "
                         "(drawdown=%.1f%% < resume=%.1f%%)",
                         drawdown * 100, resume_pct * 100)
            return

        if drawdown >= halt_pct:
            ctx.skip_buys = True
            log.info("DrawdownCircuitTask: halt triggered "
                     "(drawdown=%.1f%% ≥ halt=%.1f%%)",
                     drawdown * 100, halt_pct * 100)
