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
from .job_regime     import RegimeJob
from .job_drawdown   import DrawdownJob
from .job_gates      import BuyGatesJob
from .job_sell       import TickerSellJob
from .job_candidates import TickerCandidateJob
from .job_ranking    import RankingJob
from .job_rotation   import RotationJob
from .job_selection  import SelectionJob

# PanelScoringJob is imported lazily inside run() to avoid a circular import:
# kernel.panel_pipeline.__init__ pulls in this module via
# kernel.pipeline.context, which would trigger us before
# kernel.panel_pipeline finishes initializing.

log = logging.getLogger("kernel.pipeline")


# ── Context builders ───────────────────────────────────────────────────────────

def _build_exit_params(regime_p: dict, config: dict) -> dict:
    return {
        "trailing_stop_trigger_pct": regime_p.get("trailing_stop_trigger_pct", 0),
        "trailing_stop_trail_pct":   regime_p.get("trailing_stop_trail_pct",   0),
        "stop_loss_pct":             regime_p.get("stop_loss_pct",             0),
        "max_single_day_loss_pct":   regime_p.get("max_single_day_loss_pct",   0),
        "max_hold_days":             regime_p.get("max_hold_days",             0),
        "consecutive_sell_signals":  int(config.get("consecutive_sell_signals", 3)),
        "min_hold_days":             int(config.get("min_hold_days", 0)),
        "lt_hold_gate_days":         int(config.get("lt_hold_gate_days", 0)),
        "lt_hold_min_gain":          float(config.get("lt_hold_min_gain", 0.10)),
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
    )


def _buy_universe(ctx: InferenceContext) -> list[str]:
    held = set(ctx.holdings.keys())
    if ctx.bear_only:
        defensives = set(ctx.config.get("defensive_tickers", []))
        return [t for t in defensives if t in ctx.models and t not in held]
    return [t for t in ctx.models if t not in held and t in ctx.ohlcv]


# ── InferencePipeline ──────────────────────────────────────────────────────────

class InferencePipeline:
    """Full buy+sell inference pipeline."""

    def run(self, ctx: InferenceContext) -> None:
        # Lazy import — see module docstring note above.
        from kernel.panel_pipeline.job_panel_scoring import PanelScoringJob  # noqa: PLC0415

        t0 = time.monotonic()
        log.info("InferencePipeline START  date=%s", ctx.today)

        RegimeJob().run(ctx)
        DrawdownJob().run(ctx)
        BuyGatesJob().run(ctx)

        sell_tctxs = [_make_sell_tctx(ctx, t) for t in list(ctx.holdings.keys())]
        run_parallel(sell_tctxs, TickerSellJob())
        for tc in sell_tctxs:
            ctx.holdings[tc.ticker] = tc.holding
            if tc.exit_signal is not None and tc.exit_signal.should_exit:
                ctx.exits.append((tc.ticker, tc.exit_signal))
            elif tc.exit_signal is not None and getattr(tc.exit_signal, "_blocked_streak", False):
                ctx.counters["blocked_streak"] = ctx.counters.get("blocked_streak", 0) + 1
        log.info("Phase 2a (sell): %d exits from %d held", len(ctx.exits), len(sell_tctxs))

        if not (ctx.buy_blocked and not ctx.bear_only):
            universe   = _buy_universe(ctx)
            cand_tctxs = [_make_cand_tctx(ctx, t) for t in universe]
            run_parallel(cand_tctxs, TickerCandidateJob())
            for tc in cand_tctxs:
                if tc.candidate is not None:
                    ctx.candidates.append(tc.candidate)
            log.info("Phase 2b (buy scan): %d candidates from %d tickers",
                     len(ctx.candidates), len(universe))

        PanelScoringJob().run(ctx)
        RankingJob().run(ctx)
        RotationJob().run(ctx)
        SelectionJob().run(ctx)

        # Plan C: Kelly-driven top-up for existing holdings whose panel
        # score has improved beyond kelly_target_pct. No-op unless
        # ranking.kelly_sizing.enabled. Runs after SelectionJob so we
        # don't double-buy a fresh pick — only adds to pre-existing
        # positions.
        from .task_topup import TopUpHeldTask  # noqa: PLC0415
        TopUpHeldTask().run(ctx)

        # Monitor: persistent no-trade periods are treated as a hard signal,
        # not a silent state. See task_monitor.MonitorIdleStreakTask.
        from .task_monitor import MonitorIdleStreakTask  # noqa: PLC0415
        MonitorIdleStreakTask().run(ctx)

        log.info("InferencePipeline DONE  total=%.2fs  rotations=%d",
                 time.monotonic() - t0, len(ctx.rotations))


# ── SellOnlyPipeline ───────────────────────────────────────────────────────────

class SellOnlyPipeline:
    """Intraday exit-only variant."""

    def run(self, ctx: InferenceContext) -> None:
        log.info("SellOnlyPipeline START  date=%s", ctx.today)
        t0 = time.monotonic()

        RegimeJob().run(ctx)
        DrawdownJob().run(ctx)

        sell_tctxs = [_make_sell_tctx(ctx, t) for t in list(ctx.holdings.keys())]
        run_parallel(sell_tctxs, TickerSellJob())
        for tc in sell_tctxs:
            ctx.holdings[tc.ticker] = tc.holding
            if tc.exit_signal is not None and tc.exit_signal.should_exit:
                ctx.exits.append((tc.ticker, tc.exit_signal))

        log.info("SellOnlyPipeline DONE  total=%.2fs", time.monotonic() - t0)
