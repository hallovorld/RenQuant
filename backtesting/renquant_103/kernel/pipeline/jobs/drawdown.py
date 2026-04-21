"""DrawdownJob — portfolio drawdown circuit breaker.

Reads:  ctx.portfolio_value, ctx.hwm, ctx.regime, ctx.config
Writes: ctx.hwm, ctx.skip_buys
"""
from __future__ import annotations

from ..context import InferenceContext
from ..pipeline import Job


class DrawdownJob(Job):
    """Update HWM and set skip_buys if the drawdown circuit breaker fires."""

    def run(self, ctx: InferenceContext) -> None:
        from kernel.portfolio import update_drawdown_circuit_breaker  # noqa: PLC0415

        regime_p = ctx.config.get("regime_params", {}).get(ctx.regime, {})
        halt_pct = float(regime_p.get("drawdown_halt_pct", 0.0))

        ctx.hwm, ctx.skip_buys = update_drawdown_circuit_breaker(
            ctx.portfolio_value, ctx.hwm, halt_pct
        )
