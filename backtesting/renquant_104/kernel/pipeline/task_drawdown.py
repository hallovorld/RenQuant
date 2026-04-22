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
    """Fire drawdown circuit breaker if drawdown ≥ halt_pct; set ctx.skip_buys."""

    def run(self, ctx: InferenceContext) -> bool | None:
        regime_p = ctx.config.get("regime_params", {}).get(ctx.regime, {})
        halt_pct = float(regime_p.get("drawdown_halt_pct", 0.0))

        if ctx.hwm > 0 and halt_pct > 0:
            drawdown = (ctx.hwm - ctx.portfolio_value) / ctx.hwm
            if drawdown >= halt_pct:
                ctx.skip_buys = True
                log.info("DrawdownCircuitTask: halt triggered "
                         "(drawdown=%.1f%% ≥ halt=%.1f%%)",
                         drawdown * 100, halt_pct * 100)
