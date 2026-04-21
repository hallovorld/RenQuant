"""Pre-buy gate tasks — each returns False to short-circuit and block buys.

Reads:  ctx.skip_buys, ctx.regime_state, ctx.regime, ctx.spy_returns,
        ctx.ohlcv["SPY"], ctx.config
Writes: ctx.buy_blocked, ctx.bear_only
        ctx.counters["transition_blocks"], ctx.counters["velocity_blocks"]
"""
from __future__ import annotations

import logging

from ..context import InferenceContext
from ..pipeline import Task
from kernel.config import BEAR

log = logging.getLogger("kernel.pipeline.gates")


class DrawdownGateTask(Task):
    """Gate 0: if drawdown circuit breaker already fired, block buys."""

    def run(self, ctx: InferenceContext) -> bool | None:
        if ctx.skip_buys:
            ctx.buy_blocked = True
            log.info("DrawdownGateTask: drawdown circuit breaker — buys blocked")
            return False


class TransitionWindowTask(Task):
    """Gate 1: CUSUM uncertainty window — no new buys during regime transition."""

    def run(self, ctx: InferenceContext) -> bool | None:
        if ctx.regime_state is not None and ctx.regime_state.in_transition:
            ctx.counters["transition_blocks"] = ctx.counters.get("transition_blocks", 0) + 1
            ctx.buy_blocked = True
            log.info("TransitionWindowTask: CUSUM transition window — buys blocked")
            return False


class BEARBranchTask(Task):
    """Gate 2: BEAR regime — allow defensive tickers only; stop normal scan."""

    def run(self, ctx: InferenceContext) -> bool | None:
        if ctx.regime == BEAR:
            ctx.bear_only = True
            log.info("BEARBranchTask: BEAR regime — defensives only")
            return False  # stop chain; bear_only candidates handled separately


class VelocityCrashTask(Task):
    """Gate 3: SPY velocity crash — down > threshold% over lookback days."""

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.market_gates import check_spy_velocity_crash  # noqa: PLC0415

        regime_p = ctx.config.get("regime_params", {}).get(ctx.regime, {})
        v_halt   = float(regime_p.get("spy_velocity_halt_pct", 0.0))
        v_look   = int(regime_p.get("spy_velocity_lookback_days", 3))

        if check_spy_velocity_crash(ctx.spy_returns, v_look, v_halt):
            ctx.counters["velocity_blocks"] = ctx.counters.get("velocity_blocks", 0) + 1
            ctx.buy_blocked = True
            log.info("VelocityCrashTask: SPY velocity crash — buys blocked")
            return False


class EMA50GateTask(Task):
    """Gate 4: SPY below 50-day EMA — macro downtrend, block new entries."""

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.market_gates import check_spy_ema_trend  # noqa: PLC0415

        spy_df = ctx.ohlcv.get("SPY")
        if spy_df is not None and check_spy_ema_trend(spy_df["close"]):
            ctx.buy_blocked = True
            log.info("EMA50GateTask: SPY below EMA50 — buys blocked")
            return False
