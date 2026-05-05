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
    """Gate 1: CUSUM uncertainty window — no new buys during regime transition.

    CUSUM-v2 Design C (user-locked): when `regime.cusum_cooldown_mode`
    is `"wall_time"`, this gate is a no-op — the cooldown is enforced
    instead by SizeAndEmitTask via `max_pct × cooldown_progress`.
    Under Design C, Kelly sizing does the scaling rather than a hard block.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        # Design C — soft cooldown (no hard block here)
        mode = str(ctx.config.get("regime", {}).get("cusum_cooldown_mode", "bar_count"))
        if mode == "wall_time":
            return None
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

    2026-04-24: this Task no longer short-circuits the chain — it sets
    `bear_only=True` and returns None so VelocityCrash + EMA50 still
    fire (those set `buy_blocked` which combined with `bear_only` means
    "defensives only AND macro halt").
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        if ctx.regime == BEAR:
            return None
        regime_cfg = ctx.config.get("regime", {})
        threshold  = float(regime_cfg.get("confidence_veto_threshold", 0.0))
        # Audit fix G-1 (Round 2 deep audit, 2026-04-25): pre-fix, NaN
        # ctx.confidence (regime classifier failed / GMM returned uniform
        # prior) slipped past `confidence < threshold` because NaN < X
        # is False → veto NOT triggered → offensive buys went through
        # in a regime we couldn't classify. That's the OPPOSITE of the
        # intended safety semantics: NaN confidence means "we don't know
        # the regime", which is precisely when defensives-only is safer
        # than allowing offensive buys into uncertainty.
        # Now: NaN/inf confidence routes to the same defensives-only
        # branch as low confidence (fail-SAFE).
        import math
        conf = ctx.confidence
        non_finite = (conf is None or not math.isfinite(conf))
        if non_finite or (threshold > 0.0 and conf < threshold):
            ctx.counters["confidence_veto_blocks"] = ctx.counters.get("confidence_veto_blocks", 0) + 1
            ctx.bear_only = True
            log.info("ConfidenceVetoTask: confidence=%s%s — defensives only",
                     "non-finite" if non_finite else f"{conf:.2f}",
                     "" if non_finite else f" < {threshold:.2f}")
            # Continue chain so velocity/EMA50 macros can still fire.


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
        # Continue chain so velocity/EMA50 still set buy_blocked when applicable.


class BEARBranchTask(Task):
    """Gate 2: BEAR regime — allow defensive tickers only.

    2026-04-24: no longer short-circuits the chain so VelocityCrash +
    EMA50 still fire (set `buy_blocked` if applicable). Combined with
    `bear_only=True`, the downstream `_buy_universe` returns defensives
    when `buy_blocked AND bear_only` — defensives can still be entered
    in BEAR even during a velocity crash, which is the intended behaviour
    (defensives like GLD/TLT exist precisely for those conditions).
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        if ctx.regime == BEAR:
            ctx.bear_only = True
            log.info("BEARBranchTask: BEAR regime — defensives only")
            # Continue chain (velocity/EMA macros may still apply).


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
        # 2026-05-04 audit Issue 06 fix: fail-SAFE on missing SPY data.
        # Pre-fix: returned None (no block) so a SPY data outage let
        # offensive buys flow even though all other macro gates default
        # to "block on missing data" (DrawdownGate, VelocityCrash). With
        # Issue 05 (VelocityCrash silent on NaN), a SPY outage could
        # disable both macro gates in BULL while offensive buys flowed.
        # Now: missing SPY = block buys this bar.
        if spy_df is None or "close" not in spy_df.columns or spy_df.empty:
            ctx.buy_blocked = True
            log.warning("EMA50GateTask: SPY OHLCV missing — fail-SAFE blocking "
                        "buys this bar (data outage)")
            return False
        if check_spy_ema_trend(spy_df["close"]):
            ctx.buy_blocked = True
            log.info("EMA50GateTask: SPY below EMA50 — buys blocked")
            return False
