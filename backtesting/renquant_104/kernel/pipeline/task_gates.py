"""Pre-buy gate tasks — each returns False to short-circuit and block buys."""
from __future__ import annotations

import logging

from .context import InferenceContext
from .pipeline import Task
from kernel.config import BEAR, BULL_VOLATILE

log = logging.getLogger("kernel.pipeline.gates")


class FlattenCooldownGateTask(Task):
    """Gate -1 (2026-05-11): post-flatten cooldown.

    When :class:`DrawdownFlattenTask` fires a HARD FLATTEN, it stamps
    ``ctx.monitor_state["flatten_last_date_iso"]`` and
    ``flatten_cooldown_bars``. This gate blocks buys for that many
    business days regardless of DrawdownCircuit's resume threshold.

    Solves the S-3 death-spiral pathology: when flatten fires and the
    drawdown immediately recovers below ``drawdown_resume_pct``,
    DrawdownGate re-enables buys; new positions are bought into a still-
    fragile market, then the next drop triggers another flatten,
    realising fresh losses on every cycle (observed: 38× flatten cycles
    in S-3 sim → 96% MaxDD vs 44% golden).

    Disabled when ``risk.drawdown_flatten.cooldown_bars`` is unset or 0.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        ms = ctx.monitor_state if isinstance(ctx.monitor_state, dict) else None
        if not ms:
            return None
        last_iso = ms.get("flatten_last_date_iso")
        cd_bars  = ms.get("flatten_cooldown_bars")
        if not last_iso or not cd_bars:
            return None
        try:
            cd_n = int(cd_bars)
        except (TypeError, ValueError):
            return None
        if cd_n <= 0:
            return None
        import datetime as _dt  # noqa: PLC0415
        try:
            last_dt = _dt.date.fromisoformat(str(last_iso))
        except ValueError:
            return None
        today = ctx.today
        # ctx.today is a date in sim, datetime in live — normalize.
        if isinstance(today, _dt.datetime):
            today = today.date()
        if not isinstance(today, _dt.date):
            return None
        # Cooldown window: [last_flatten_date + 1, last_flatten_date + cd_n]
        # inclusive of cd_n business days. Use calendar-day arithmetic
        # since SimAdapter ticks on business days only; on weekends this
        # function isn't invoked anyway.
        days_since = (today - last_dt).days
        if days_since <= 0:
            # Same bar — flatten just fired; ensure buys still blocked.
            ctx.skip_buys = True
            ctx.buy_blocked = True
            return False
        if days_since <= cd_n:
            ctx.skip_buys = True
            ctx.buy_blocked = True
            log.info(
                "FlattenCooldownGateTask: post-flatten cooldown active "
                "(day %d of %d since flatten %s) — buys blocked.",
                days_since, cd_n, last_iso,
            )
            return False
        # Expired — clear the cooldown stamp so subsequent bars don't
        # re-check forever.
        ms.pop("flatten_last_date_iso", None)
        ms.pop("flatten_cooldown_bars", None)
        return None


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


class RegimeAlphaGateTask(Task):
    """Gate 1d (2026-05-20): block new buys in regimes where PROD has no
    truly-OOS alpha. Sourced from artifacts/prod/truly_oos_eval/eval_truly_oos.json
    (train cutoff 2024-07-01, eval 404 dates strictly post-cutoff):

      BEAR:          IC +0.345  top10_α +0.696  → KEEP buys
      CHOPPY:        IC +0.103  top10_α +0.259  → KEEP buys
      BULL_VOLATILE: IC +0.105  top10_α +0.129  → KEEP buys
      BULL_STRONG:   IC +0.060  top10_α +0.245  → KEEP buys
      BULL_CALM:     IC +0.005  top10_α -0.045  → optional block knob

    Knob: `regime_params[<regime>].disable_new_buys = True`. PRIME DIRECTIVE
    requires per-regime knobs, not a global. Default False so legacy
    configs are unchanged. Production BULL_CALM was re-enabled by operator
    override on 2026-05-21; this task remains available for future regime
    risk policy changes.

    SELL logic unaffected: only blocks NEW buys (sets ctx.buy_blocked=True).
    Existing positions can still exit via stop-loss / trailing / model_sell /
    QP trim per usual sell-side tasks.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        regime_p = (ctx.config.get("regime_params", {}) or {}).get(ctx.regime, {})
        if not bool(regime_p.get("disable_new_buys", False)):
            return None
        ctx.counters["regime_alpha_blocks"] = (
            ctx.counters.get("regime_alpha_blocks", 0) + 1
        )
        ctx.buy_blocked = True
        log.info(
            "RegimeAlphaGateTask: regime=%s with disable_new_buys=True — "
            "buys blocked (truly-OOS top10 alpha non-positive). Existing "
            "holdings may still exit.",
            ctx.regime,
        )
        return False


class BEARBranchTask(Task):
    """Gate 2: BEAR regime — allow defensive tickers only.

    2026-04-24: no longer short-circuits the chain so VelocityCrash +
    EMA50 still fire (set `buy_blocked` if applicable). Combined with
    `bear_only=True`, the downstream `_buy_universe` returns defensives
    when `buy_blocked AND bear_only` — defensives can still be entered
    in BEAR even during a velocity crash, which is the intended behaviour
    (defensives like GLD/TLT exist precisely for those conditions).

    2026-05-14 Soft-BEAR fix (Kaminski-Lo 2014 + Garleanu-Pedersen 2013):
    Pre-fix this task fired on a SINGLE bar of regime==BEAR, regardless
    of detector confidence or persistence. With the 2026-05-14 direction-
    aware Hurst detector, bull windows (Q10/Q11/Q15) get 5-11% transient
    BEAR mis-labels. Each one triggered a full defensive switch → strategy
    sold positions → drawdown_halt liquidated → missed the V-recovery.
    Net Panel A regression: −4.10pt mean Δ_APY vs original GMM baseline.

    Soft fix (config-gated, default ON per PRIME DIRECTIVE):
      * Skip bear_only when in_transition=True (cooldown window after
        regime switch — confidence is flat 0.5).
      * Skip bear_only when confidence < `regime.bear_branch_min_confidence`
        (default 0.60). Real BEARs reach ≥0.8 confidence in 2-3 bars
        (hard_bear or GMM-BEAR>0.5 path → confidence=1.0); transient
        Hurst-MOMENTUM-down mis-fires sit at 0.5.

    Set `regime.bear_branch_legacy_mode=true` to restore pre-fix behavior.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        if ctx.regime != BEAR:
            return None
        # Restore pre-2026-05-14 hard-switch behavior if explicitly requested
        regime_cfg = ctx.config.get("regime", {}) or {}
        if bool(regime_cfg.get("bear_branch_legacy_mode", False)):
            ctx.bear_only = True
            log.info("BEARBranchTask: BEAR regime (legacy mode) — defensives only")
            return None
        # Soft path: require confidence > threshold AND not in transition
        rs = getattr(ctx, "regime_state", None)
        in_transition = bool(getattr(rs, "in_transition", False)) if rs is not None \
                        else False
        min_conf = float(regime_cfg.get("bear_branch_min_confidence", 0.60))
        import math
        # Defensive: pre-soft-gate test fixtures (e.g. test_audit_2026_04_24_fixes)
        # construct ctx via SimpleNamespace() without setting confidence — treat
        # missing attr as non-finite (same fail-SAFE branch as None/NaN/inf).
        conf = getattr(ctx, "confidence", None)
        non_finite = (conf is None or not isinstance(conf, (int, float))
                      or not math.isfinite(float(conf)))
        if in_transition:
            ctx.counters["bear_branch_skipped_transition"] = (
                ctx.counters.get("bear_branch_skipped_transition", 0) + 1
            )
            log.info("BEARBranchTask: regime=BEAR but in_transition — bear_only NOT set")
            return None
        if non_finite or float(conf) < min_conf:
            ctx.counters["bear_branch_skipped_lowconf"] = (
                ctx.counters.get("bear_branch_skipped_lowconf", 0) + 1
            )
            log.info("BEARBranchTask: regime=BEAR but conf=%s < %.2f — bear_only NOT set",
                     "non-finite" if non_finite else f"{float(conf):.2f}", min_conf)
            return None
        ctx.bear_only = True
        log.info("BEARBranchTask: BEAR regime (conf=%.2f ≥ %.2f) — defensives only",
                 float(conf), min_conf)


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
    """Gate 4: SPY below 50-day EMA — macro downtrend, block new entries.

    2026-05-13: gated by ``gates.ema50_gate.enabled`` (default True so
    baseline behaviour is unchanged). Setting to False allows research
    sims to test offense-only configurations without code edits.
    Disabling in production is NOT recommended — diagnosis showed the
    gate adds ~12pt mean alpha in bear regimes.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.market_gates import check_spy_ema_trend  # noqa: PLC0415

        gate_cfg = (getattr(ctx, "config", None) or {}).get(
            "gates", {}).get("ema50_gate", {})
        if not gate_cfg.get("enabled", True):
            return None

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
