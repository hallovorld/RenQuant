"""Drawdown circuit breaker tasks."""
from __future__ import annotations

import logging

from .context import InferenceContext
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.drawdown")


class HWMUpdateTask(Task):
    """Advance high-water mark: hwm = max(hwm, portfolio_value)."""

    def run(self, ctx: InferenceContext) -> bool | None:
        # Audit fix DC-1 (Round 2 deep audit, 2026-04-25): pre-fix, NaN/inf
        # ctx.portfolio_value silently corrupted hwm via `max(hwm, NaN) =
        # NaN`. Once hwm = NaN, drawdown calc returned NaN, and
        # `drawdown >= halt_pct` evaluated False → drawdown circuit
        # breaker permanently disabled, with NO log signal.
        # Same pattern as E-5 (kernel/exits.py NaN price → HWM corruption).
        # Now: skip the update on non-finite portfolio_value; keep prior
        # hwm intact so the drawdown gate stays armed.
        import math
        if not math.isfinite(ctx.portfolio_value):
            log.warning(
                "HWMUpdateTask: portfolio_value=%s is non-finite — "
                "skipping HWM update (kept hwm=%.2f)",
                ctx.portfolio_value, ctx.hwm,
            )
            return
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
        import math
        regime_p = ctx.config.get("regime_params", {}).get(ctx.regime, {})
        halt_pct = float(regime_p.get("drawdown_halt_pct", 0.0))

        if ctx.hwm <= 0 or halt_pct <= 0:
            return

        # 2026-05-04 audit Issue 07 fix: NaN/inf guard on portfolio_value.
        # Pre-fix: NaN portfolio_value → drawdown = NaN → `NaN >= halt_pct`
        # is False → halt silently doesn't fire. Same NaN-propagation
        # pattern as DC-1 (already fixed in HWMUpdateTask), but the same
        # guard was missing here. Fail-SAFE: block buys on non-finite
        # input, the same way HWMUpdateTask preserves prior HWM.
        if not math.isfinite(ctx.hwm) or not math.isfinite(ctx.portfolio_value):
            ctx.skip_buys = True
            log.warning("DrawdownCircuitTask: non-finite hwm=%s or "
                        "portfolio_value=%s — fail-SAFE forcing skip_buys=True",
                        ctx.hwm, ctx.portfolio_value)
            return

        # 2026-05-11: delegate to single-source-of-truth DD helper.
        from .task_drawdown_rebalance import (  # noqa: PLC0415
            compute_portfolio_drawdown,
        )
        drawdown = compute_portfolio_drawdown(ctx.hwm, ctx.portfolio_value)

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
