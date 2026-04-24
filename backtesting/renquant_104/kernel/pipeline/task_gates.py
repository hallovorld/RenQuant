"""Pre-buy gate tasks — each returns False to short-circuit and block buys."""
from __future__ import annotations

import logging

from .context import InferenceContext
from .pipeline import Task
from kernel.config import BEAR, BULL_VOLATILE

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


class ConfidenceVetoTask(Task):
    """Gate 1b: GMM confidence veto — if regime confidence is too low, treat as BEAR.

    When confidence < regime.confidence_veto_threshold (default 0.55), offensive
    buys are blocked and only defensive slots can be filled — same effect as
    BEARBranchTask but driven by uncertainty rather than a detected BEAR label.
    Skipped if the regime is already BEAR (BEARBranchTask handles it).
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        if ctx.regime == BEAR:
            return None
        regime_cfg = ctx.config.get("regime", {})
        threshold  = float(regime_cfg.get("confidence_veto_threshold", 0.0))
        if threshold > 0.0 and ctx.confidence < threshold:
            ctx.counters["confidence_veto_blocks"] = ctx.counters.get("confidence_veto_blocks", 0) + 1
            ctx.bear_only = True
            log.info("ConfidenceVetoTask: confidence %.2f < %.2f — defensives only",
                     ctx.confidence, threshold)
            return False


class BullVolOffensiveBlockTask(Task):
    """Gate 1c — AA-surfaced: BULL_VOLATILE ranker Spearman IC = -0.172 on
    real decision-trace data (445 rows). The panel anti-predicts during
    vol spikes — we'd be buying the worst names. Block offensive buys in
    BULL_VOLATILE when `regime.bull_vol_block_offensive` is true.

    When on, behaves like BEARBranchTask for BULL_VOL: flips `bear_only=True`
    so the selection loop only admits defensive tickers. Set
    `regime.bull_vol_defensives_too = true` to block defensives as well
    (pure cash position during BULL_VOL).

    Default OFF to preserve current behaviour until A/B validates.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        if ctx.regime != BULL_VOLATILE:
            return None
        regime_cfg = ctx.config.get("regime", {})
        if not bool(regime_cfg.get("bull_vol_block_offensive", False)):
            return None
        ctx.counters["bull_vol_blocks"] = ctx.counters.get("bull_vol_blocks", 0) + 1
        if bool(regime_cfg.get("bull_vol_defensives_too", False)):
            ctx.buy_blocked = True
            log.info("BullVolOffensiveBlockTask: BULL_VOLATILE — all buys blocked")
            return False
        ctx.bear_only = True
        log.info("BullVolOffensiveBlockTask: BULL_VOLATILE — defensives only")
        return False


class BEARBranchTask(Task):
    """Gate 2: BEAR regime — allow defensive tickers only; stop normal scan."""

    def run(self, ctx: InferenceContext) -> bool | None:
        if ctx.regime == BEAR:
            ctx.bear_only = True
            log.info("BEARBranchTask: BEAR regime — defensives only")
            return False


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
