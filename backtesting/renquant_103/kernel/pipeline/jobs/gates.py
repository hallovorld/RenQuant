"""BuyGatesJob — pre-buy gate checks (drawdown, transition, BEAR, velocity, EMA50).

Reads:  ctx.skip_buys, ctx.regime_state, ctx.regime, ctx.spy_returns,
        ctx.ohlcv["SPY"], ctx.config, ctx.holdings
Writes: ctx.buy_blocked, ctx.bear_only
        ctx.counters["transition_blocks"], ctx.counters["velocity_blocks"]
"""
from __future__ import annotations

import logging

from ..context import InferenceContext
from ..pipeline import Job
from kernel.config import BEAR

log = logging.getLogger("kernel.pipeline.gates")


class BuyGatesJob(Job):
    """Apply all pre-buy gates; set ctx.buy_blocked or ctx.bear_only."""

    def run(self, ctx: InferenceContext) -> None:
        from kernel.market_gates import check_spy_velocity_crash, check_spy_ema_trend  # noqa: PLC0415

        # Gate 0 — drawdown circuit breaker already set by DrawdownJob
        if ctx.skip_buys:
            ctx.buy_blocked = True
            log.info("BuyGates: drawdown circuit breaker — buys blocked")
            return

        # Gate 1 — transition uncertainty window
        if ctx.regime_state is not None and ctx.regime_state.in_transition:
            ctx.counters["transition_blocks"] = ctx.counters.get("transition_blocks", 0) + 1
            ctx.buy_blocked = True
            log.info("BuyGates: transition window — buys blocked")
            return

        # Gate 2 — BEAR branch (not fully blocked, but restricted to defensives)
        if ctx.regime == BEAR:
            ctx.bear_only = True
            log.info("BuyGates: BEAR regime — defensives only")
            return

        # Gate 3 — SPY velocity crash
        config   = ctx.config
        regime_p = config.get("regime_params", {}).get(ctx.regime, {})
        v_halt   = float(regime_p.get("spy_velocity_halt_pct", 0.0))
        v_look   = int(regime_p.get("spy_velocity_lookback_days", 3))

        if check_spy_velocity_crash(ctx.spy_returns, v_look, v_halt):
            ctx.counters["velocity_blocks"] = ctx.counters.get("velocity_blocks", 0) + 1
            ctx.buy_blocked = True
            log.info("BuyGates: SPY velocity crash — buys blocked")
            return

        # Gate 4 — SPY EMA50 trend gate
        spy_df = ctx.ohlcv.get("SPY")
        if spy_df is not None and check_spy_ema_trend(spy_df["close"]):
            ctx.buy_blocked = True
            log.info("BuyGates: SPY below EMA50 — buys blocked")
            return
