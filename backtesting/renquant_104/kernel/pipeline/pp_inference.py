"""InferencePipeline and SellOnlyPipeline — renquant_103 inference orchestrators.

Phase layout:
  Phase 1  global sequential  RegimeJob → DrawdownJob → BuyGatesJob
  Phase 2a parallel sell      TickerSellJob (one per held ticker)
  Phase 2b parallel buy scan  TickerCandidateJob (one per universe ticker)
  Phase 3  global sequential  RankingJob → SelectionJob
"""
from __future__ import annotations

import logging
import time

from .context  import InferenceContext, TickerInferenceContext
from .pipeline import run_parallel
from .job_regime         import RegimeJob
from .job_drawdown       import DrawdownJob
from .job_gates          import BuyGatesJob
from .job_sell           import TickerSellJob
from .job_candidates     import TickerCandidateJob
from .job_ranking        import RankingJob
from .job_rotation       import RotationJob
from .job_selection      import SelectionJob
from .job_joint_actions  import JointActionJob
from .job_panel_veto     import PanelRankVetoJob
from .job_score_distribution import ScoreDistributionJob

# PanelScoringJob is imported lazily inside run() to avoid a circular import:
# kernel.panel_pipeline.__init__ pulls in this module via
# kernel.pipeline.context, which would trigger us before
# kernel.panel_pipeline finishes initializing.

log = logging.getLogger("kernel.pipeline")


# ── Context builders ───────────────────────────────────────────────────────────

def _build_exit_params(regime_p: dict, config: dict) -> dict:
    tax_cfg = config.get("tax", {})
    return {
        "trailing_stop_trigger_pct": regime_p.get("trailing_stop_trigger_pct", 0),
        "trailing_stop_trail_pct":   regime_p.get("trailing_stop_trail_pct",   0),
        "stop_loss_pct":             regime_p.get("stop_loss_pct",             0),
        # 2026-05-10 σ-aware stop_loss (Fix #0a revive) — set to 0 to disable;
        # typical industry value 2.0 (Almgren-Chriss / Edwards-Magee).
        "stop_n_sigma":              regime_p.get("stop_n_sigma",              0),
        # 2026-05-11 L5: Chandelier exit multiplier (Wilder 1978 + Le Beau 1993).
        # When > 0 and ATR(14) is available, effective trailing trail-pct
        # becomes max(trailing_stop_trail_pct, k × ATR / HWM). Typical k=3.
        "atr_n_multiplier":          regime_p.get("atr_n_multiplier",          0),
        "max_single_day_loss_pct":   regime_p.get("max_single_day_loss_pct",   0),
        # Existing σ-aware SDL plumbing — previously wired in check_single_day_loss
        # but the config key was never threaded through _build_exit_params,
        # so it was dead. Now config-driven (industry value 2.0-2.5).
        "sdl_n_sigma":               regime_p.get("sdl_n_sigma",               0),
        "max_hold_days":             regime_p.get("max_hold_days",             0),
        "consecutive_sell_signals":  int(config.get("consecutive_sell_signals", 3)),
        "min_hold_days":             int(config.get("min_hold_days", 0)),
        "lt_hold_gate_days":         int(config.get("lt_hold_gate_days", 0)),
        "lt_hold_min_gain":          float(config.get("lt_hold_min_gain", 0.10)),
        # #18 fix: config-driven LT threshold (not hardcoded 365 in compute_exits).
        "lt_hold_threshold_days":    int(tax_cfg.get("long_term_threshold_days", 365)),
    }


def _make_sell_tctx(ctx: InferenceContext, ticker: str) -> TickerInferenceContext:
    regime_p    = ctx.config.get("regime_params", {}).get(ctx.regime, {})
    exit_params = _build_exit_params(regime_p, ctx.config)
    return TickerInferenceContext(
        ticker=ticker,
        ohlcv=ctx.ohlcv,
        model=ctx.models.get(ticker),
        config=ctx.config,
        today=ctx.today,
        regime=ctx.regime,
        regime_params=regime_p,
        exit_params=exit_params,
        holding=ctx.holdings[ticker],
        price=ctx.prices.get(ticker, 0.0),
        # earnings_calendar plumbed to sell tctx (2026-05-01) so
        # EarningsBlackoutSellTask can veto model-driven exits inside the
        # event-blackout window. Buy-side has had this since the original
        # candidate pipeline; sell-side was missing it and let CAT exit on
        # 2026-05-01 the day after a +9.88% earnings rip.
        earnings_calendar=ctx.earnings_calendar,
        feature_cache_frame=ctx.feature_cache.get(ticker) if ctx.feature_cache else None,
    )


def _make_cand_tctx(ctx: InferenceContext, ticker: str) -> TickerInferenceContext:
    regime_p = ctx.config.get("regime_params", {}).get(ctx.regime, {})
    return TickerInferenceContext(
        ticker=ticker,
        ohlcv=ctx.ohlcv,
        model=ctx.models.get(ticker),
        config=ctx.config,
        today=ctx.today,
        regime=ctx.regime,
        regime_params=regime_p,
        exit_params={},
        holding=None,
        price=ctx.prices.get(ticker, 0.0),
        earnings_calendar=ctx.earnings_calendar,
        last_sell_dates=ctx.last_sell_dates,
        last_sell_pls=ctx.last_sell_pls,
        feature_cache_frame=ctx.feature_cache.get(ticker) if ctx.feature_cache else None,
    )


def _buy_universe(ctx: InferenceContext) -> list[str]:
    held = set(ctx.holdings.keys())
    # Audit fix BROKER-PRECHECK (2026-04-26): exclude tickers with pending
    # orders at the broker. Pre-fix, these were filtered AT SUBMIT TIME,
    # AFTER the pipeline already built features + scored + sized them →
    # wasted compute AND distorted cash budget. Now: never enter the buy
    # universe if the broker already has a pending order for them.
    pending_at_broker = set(getattr(ctx, "pending_broker_tickers", None) or [])
    if ctx.bear_only:
        # Defensives also need OHLCV — without it BuildFeaturesTask
        # short-circuits anyway, but earlier tasks (EarningsFilter / WashSale)
        # would still run on an empty frame and the parallel worker would
        # spin up uselessly. Match the non-bear branch's gate.
        defensives = set(ctx.config.get("defensive_tickers", []))
        return [t for t in defensives
                if t in ctx.models and t not in held
                and t not in pending_at_broker and t in ctx.ohlcv]
    return [t for t in ctx.models if t not in held
            and t not in pending_at_broker and t in ctx.ohlcv]


# ── InferencePipeline ──────────────────────────────────────────────────────────

class InferencePipeline:
    """Full buy+sell inference pipeline."""

    def run(self, ctx: InferenceContext) -> None:
        # Lazy import — see module docstring note above.
        from kernel.panel_pipeline.job_panel_scoring import PanelScoringJob  # noqa: PLC0415

        t0 = time.monotonic()
        log.info("InferencePipeline START  date=%s", ctx.today)

        # Audit fix #8 (2026-04-26): if challenger.enabled=true in
        # strategy_config but the live wiring (Phase 4b) hasn't landed
        # yet, an operator could believe shadow-mode is active when
        # nothing is being recorded. Surface this once per run start.
        try:
            ch_cfg = (ctx.config.get("acceptance") or {}).get("challenger") or {}
            if ch_cfg.get("enabled") and not getattr(ctx, "_challenger_warned_once", False):
                log.warning(
                    "acceptance.challenger.enabled=true BUT live wiring not yet "
                    "in pp_inference.py (Phase 4b deferred). Shadow scoring will "
                    "not record decisions to challenger_decisions table this run. "
                    "See doc/components/model-selection.md §Tier 4."
                )
                ctx._challenger_warned_once = True   # noqa: SLF001
        except Exception:
            pass    # never let observability break the pipeline

        # 2026-05-03 P0 incident: panel pipeline ingested through Thursday
        # only, model + inference ran on stale data, 6 live orders went out
        # Sunday based on Thursday closes 3 trading days behind Friday.
        # This gate is the LAST line of defense against silent staleness
        # leaking into broker submissions. Disable in backtest configs via
        # data_freshness.enabled=false. See task_data_freshness.py.
        from .task_data_freshness import DataFreshnessGateTask  # noqa: PLC0415
        DataFreshnessGateTask().run(ctx)

        RegimeJob().run(ctx)
        DrawdownJob().run(ctx)
        BuyGatesJob().run(ctx)

        sell_tctxs = [_make_sell_tctx(ctx, t) for t in list(ctx.holdings.keys())]
        run_parallel(sell_tctxs, TickerSellJob())
        for tc in sell_tctxs:
            ctx.holdings[tc.ticker] = tc.holding
            if tc.exit_signal is not None and tc.exit_signal.should_exit:
                ctx.exits.append((tc.ticker, tc.exit_signal))
            elif tc.exit_signal is not None and getattr(tc.exit_signal, "blocked_streak", False):
                ctx.counters["blocked_streak"] = ctx.counters.get("blocked_streak", 0) + 1
        log.info("Phase 2a (sell): %d exits from %d held", len(ctx.exits), len(sell_tctxs))

        # S-2 (2026-05-11) — HARD FLATTEN kill switch at drawdown
        # threshold. Runs AFTER the parallel TickerSellJob so path-rule
        # exits are already in ctx.exits; this task augments the list
        # with flatten signals for every still-held ticker when
        # portfolio drawdown ≥ risk.drawdown_flatten.flatten_pct.
        # Disabled by default — golden behaviour preserved.
        from .task_dd_flatten import DrawdownFlattenTask  # noqa: PLC0415
        DrawdownFlattenTask().run(ctx)

        # P4.4 (2026-05-11) — meta-label veto on path-rule exits.
        # López de Prado AFML ch.20: trained binary classifier predicts
        # P(profitable_exit) and drops false-positive stop_loss /
        # trailing_stop / single_day_loss / max_hold triggers. Runs
        # AFTER DrawdownFlattenTask so a hard-flatten event can't be
        # vetoed (the kill switch overrides everything). Disabled when
        # config.ranking.meta_label.enabled=false OR adapter hasn't
        # loaded a predictor (§5.13.10 fallback).
        from kernel.meta_label.task_meta_label_veto import MetaLabelVetoTask  # noqa: PLC0415
        MetaLabelVetoTask().run(ctx)

        # 2026-04-26 round-7 audit fix MAX-SELLS-PER-BAR:
        # portfolio-level cap on simultaneous model_sell exits. Risk
        # rules (stop_loss / trailing / SDL / max_hold) exempt — only
        # model_sell goes through the cap. Default off (knob = 0).
        from .task_limit_sells import LimitSellsPerBarTask  # noqa: PLC0415
        LimitSellsPerBarTask().run(ctx)

        if not (ctx.buy_blocked and not ctx.bear_only):
            universe   = _buy_universe(ctx)
            cand_tctxs = [_make_cand_tctx(ctx, t) for t in universe]
            run_parallel(cand_tctxs, TickerCandidateJob())
            for tc in cand_tctxs:
                if tc.candidate is not None:
                    ctx.candidates.append(tc.candidate)
            log.info("Phase 2b (buy scan): %d candidates from %d tickers",
                     len(ctx.candidates), len(universe))

        # G8 (2026-05-04 post-stop blackout): drop candidates whose ticker
        # had a path-rule exit (trailing_stop / stop_loss / single_day_loss /
        # max_hold / gap_down) within `risk.post_stop_cooldown.bars`. Off
        # by default — opt-in via that config block. Adapter populates
        # ctx.last_stop_exit_dates when the exit_type is in the
        # DEFAULT_STOP_EXIT_TYPES set.
        from .task_post_stop_cooldown import (  # noqa: PLC0415
            PostStopCooldownFilterTask,
        )
        PostStopCooldownFilterTask().run(ctx)

        # 2026-05-03 P0 risk gates (added same evening as DataFreshnessGate):
        # universe-admission filters keep small/illiquid names out of the
        # watchlist, but once admitted nothing at runtime checks
        #   • realized vol — could be 100% annualized and still pass
        #   • current concentration — could already hold 30% of portfolio
        # These two gates drop violators from ctx.candidates before
        # Phase 3 (PanelScoring/Ranking) so the QP never sees them. See
        # task_risk_gates.py for invariants + defaults.
        from .task_risk_gates import (  # noqa: PLC0415
            RealizedVolGateTask, PositionConcentrationGateTask,
        )
        RealizedVolGateTask().run(ctx)
        PositionConcentrationGateTask().run(ctx)

        # 2026-04-24: honour Job.should_skip on the Phase-3 jobs. Each
        # Job declares should_skip() guards (no candidates, bear_only,
        # rotation disabled, …) but the framework's Job.run() never
        # consulted them. Without this, RankingJob runs even when
        # ctx.candidates is empty and SelectionJob runs even when
        # ctx.ranked is empty — wasted work, not a correctness break.
        # Wiring it here keeps the docstring promise. getattr fallback
        # so tests that monkey-patch Jobs with bare callables (no
        # should_skip method) still drive the pipeline.
        # Phase 2 (2026-04-25): JointActionJob replaces RotationJob +
        # SelectionJob when `rotation.joint_actions.enabled = true`. When
        # the flag is false (default), JointActionJob.should_skip returns
        # True and the legacy chain runs unchanged. When the flag is
        # true, the legacy Rotation + Selection Jobs are bypassed in this
        # bar (their state is already handled by JointActionJob).
        joint_enabled = bool((ctx.config.get("rotation", {})
                                       .get("joint_actions", {})
                                       .get("enabled", False)))
        # PanelRankVetoJob runs after PanelScoringJob (so rank_score is
        # on holdings) but BEFORE the action jobs so vetoed exits are
        # already removed from ctx.exits by the time joint/rotate/select
        # see them. Default off — opt-in via model_sell.panel_veto.enabled.
        # ScoreDistributionJob (2026-04-26 round-5) runs LAST in Phase 3
        # so it captures FINAL rank_score values (post-calibration, post-
        # NGBoost). Default off — opt-in via score_db.enabled. Doesn't
        # affect decisions; only persists the distribution to runs.db
        # for percentile-based admission in a future Phase 2.
        if joint_enabled and not ctx.bear_only:
            phase3_jobs = (PanelScoringJob(), PanelRankVetoJob(),
                           RankingJob(), JointActionJob(),
                           ScoreDistributionJob())
        else:
            phase3_jobs = (PanelScoringJob(), PanelRankVetoJob(),
                           RankingJob(),
                           RotationJob(), SelectionJob(),
                           ScoreDistributionJob())
        for job in phase3_jobs:
            skip_fn = getattr(job, "should_skip", None)
            if callable(skip_fn) and skip_fn(ctx):
                log.debug("%s skipped by should_skip", type(job).__name__)
                continue
            job.run(ctx)

        # Plan C: Kelly-driven top-up for existing holdings whose panel
        # score has improved beyond kelly_target_pct. No-op unless
        # ranking.kelly_sizing.enabled. Runs after SelectionJob so we
        # don't double-buy a fresh pick — only adds to pre-existing
        # positions.
        from .task_topup import TopUpHeldTask  # noqa: PLC0415
        TopUpHeldTask().run(ctx)

        # Plan AB-trim: Kelly-driven partial sell for over-weight holdings
        # whose current_pct > kelly_target + trim_threshold. Runs AFTER
        # TopUpHeldTask so trim and top-up are never emitted for the same
        # ticker in one bar (TopUp skips over-target, Trim skips under).
        from .task_trim import TrimHeldTask  # noqa: PLC0415
        TrimHeldTask().run(ctx)

        # Monitor: persistent no-trade periods are treated as a hard signal,
        # not a silent state. See task_monitor.MonitorIdleStreakTask.
        from .task_monitor import MonitorIdleStreakTask  # noqa: PLC0415
        MonitorIdleStreakTask().run(ctx)

        # P4.1 (2026-05-11) — meta-label training data capture.
        # MetaLabelLoggingJob.should_skip returns True unless
        # config.meta_label_training.enabled = true AND
        # ctx.snapshot_logger is set by the adapter. No-op in prod.
        from kernel.meta_label.job_meta_label_log import MetaLabelLoggingJob  # noqa: PLC0415
        _ml_job = MetaLabelLoggingJob()
        if not _ml_job.should_skip(ctx):
            _ml_job.run(ctx)

        # Audit fix ROT-COUNTER (Bug L, 2026-04-25): pre-fix this logged
        # `len(ctx.rotations)` which is "pairs CONSIDERED by find_rotation_pairs",
        # not "pairs EMITTED to broker". Iter3 produced rotations=1 in the log
        # while EmitRotationsTask actually skipped the pair (Kelly=0). Now
        # log both — counters["rotations"] is incremented per EMITTED pair.
        n_considered = len(ctx.rotations)
        n_emitted    = int(ctx.counters.get("rotations", 0))
        n_blocked    = len(getattr(ctx, "rotations_blocked", []) or [])
        log.info(
            "InferencePipeline DONE  total=%.2fs  rotations_emitted=%d "
            "(considered=%d  blocked=%d)",
            time.monotonic() - t0, n_emitted, n_considered, n_blocked,
        )


# ── SellOnlyPipeline ───────────────────────────────────────────────────────────

class SellOnlyPipeline:
    """Intraday exit-only variant."""

    def run(self, ctx: InferenceContext) -> None:
        log.info("SellOnlyPipeline START  date=%s", ctx.today)
        t0 = time.monotonic()

        # 2026-05-03 P0: even sell-only paths must refuse stale data.
        from .task_data_freshness import DataFreshnessGateTask  # noqa: PLC0415
        DataFreshnessGateTask().run(ctx)

        RegimeJob().run(ctx)
        DrawdownJob().run(ctx)

        sell_tctxs = [_make_sell_tctx(ctx, t) for t in list(ctx.holdings.keys())]
        run_parallel(sell_tctxs, TickerSellJob())
        for tc in sell_tctxs:
            ctx.holdings[tc.ticker] = tc.holding
            if tc.exit_signal is not None and tc.exit_signal.should_exit:
                ctx.exits.append((tc.ticker, tc.exit_signal))

        # 2026-04-26 round-7 audit fix MAX-SELLS-PER-BAR:
        # also cap intraday sell-only bursts. Same task, same config.
        from .task_limit_sells import LimitSellsPerBarTask  # noqa: PLC0415
        LimitSellsPerBarTask().run(ctx)

        # Audit #6: also advance the no-trade monitor on sell-only bars.
        # An intraday-trip-only window is still an "active" decision —
        # the streak counter should reflect that we DID trade (or
        # explicitly chose not to) on this bar.
        from .task_monitor import MonitorIdleStreakTask  # noqa: PLC0415
        MonitorIdleStreakTask().run(ctx)

        log.info("SellOnlyPipeline DONE  total=%.2fs", time.monotonic() - t0)
