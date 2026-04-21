"""Job ABC, InferencePipeline, SellOnlyPipeline, and run_parallel helper.

Self-contained: only stdlib.  No common/ imports.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import current_thread

from .context import InferenceContext, TickerInferenceContext

log = logging.getLogger("kernel.pipeline")


# ── Job ABCs ───────────────────────────────────────────────────────────────────

class Job(ABC):
    """Global pipeline stage — reads/writes InferenceContext."""

    @abstractmethod
    def run(self, ctx: InferenceContext) -> None: ...

    def should_skip(self, ctx: InferenceContext) -> bool:
        return False


class TickerJob(ABC):
    """Per-ticker pipeline stage — reads/writes TickerInferenceContext."""

    @abstractmethod
    def run(self, tc: TickerInferenceContext) -> None: ...


# ── Parallel executor ──────────────────────────────────────────────────────────

def run_parallel(
    ticker_ctxs: list[TickerInferenceContext],
    job: TickerJob,
    max_workers: int = 8,
) -> None:
    """Run job.run(tc) for each tc in parallel; faults are logged, not raised.

    Each thread is named 'infer-SYMBOL' so logs identify the ticker and phase.
    """
    if not ticker_ctxs:
        return
    job_name = type(job).__name__
    n = min(max_workers, len(ticker_ctxs))
    log.info("run_parallel: %s  %d tickers  %d workers", job_name, len(ticker_ctxs), n)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n, thread_name_prefix="infer") as ex:
        futures = {ex.submit(_wrapped_run, job, tc): tc.ticker for tc in ticker_ctxs}
        for fut in as_completed(futures):
            ticker = futures[fut]
            exc = fut.exception()
            if exc:
                log.error("run_parallel [%s] %s ERROR — %s: %s",
                          ticker, job_name, type(exc).__name__, exc)
    log.info("run_parallel: %s DONE  %.2fs", job_name, time.monotonic() - t0)


def _wrapped_run(job: TickerJob, tc: TickerInferenceContext) -> None:
    """Thin wrapper so thread name appears in logs."""
    log.debug("[%s|%s] %s START", tc.ticker, current_thread().name, type(job).__name__)
    t0 = time.monotonic()
    job.run(tc)
    log.debug("[%s|%s] %s DONE  %.2fs", tc.ticker, current_thread().name,
              type(job).__name__, time.monotonic() - t0)


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
    regime_p  = ctx.config.get("regime_params", {}).get(ctx.regime, {})
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
    """Tickers eligible for buy scoring this bar."""
    held = set(ctx.holdings.keys())
    if ctx.bear_only:
        defensives = set(ctx.config.get("defensive_tickers", []))
        return [t for t in defensives if t in ctx.models and t not in held]
    return [t for t in ctx.models if t not in held and t in ctx.ohlcv]


# ── InferencePipeline ──────────────────────────────────────────────────────────

class InferencePipeline:
    """7-phase pipeline.

    Phases 1 and 3 are global (sequential).
    Phase 2 runs per-ticker jobs in parallel via ThreadPoolExecutor.
    """

    def run(self, ctx: InferenceContext) -> None:
        from .jobs.regime     import RegimeJob
        from .jobs.drawdown   import DrawdownJob
        from .jobs.gates      import BuyGatesJob
        from .jobs.sell       import TickerSellJob
        from .jobs.candidates import TickerCandidateJob
        from .jobs.ranking    import RankingJob
        from .jobs.selection  import SelectionJob

        t0 = time.monotonic()
        log.info("InferencePipeline START  date=%s", ctx.today)

        # ── Phase 1: global sequential ─────────────────────────────────────────
        RegimeJob().run(ctx)
        DrawdownJob().run(ctx)
        BuyGatesJob().run(ctx)

        # ── Phase 2a: parallel sell evaluation ─────────────────────────────────
        sell_tctxs = [_make_sell_tctx(ctx, t) for t in list(ctx.holdings.keys())]
        run_parallel(sell_tctxs, TickerSellJob())
        for tc in sell_tctxs:
            ctx.holdings[tc.ticker] = tc.holding   # updated streak + HWM
            if tc.exit_signal is not None and tc.exit_signal.should_exit:
                ctx.exits.append((tc.ticker, tc.exit_signal))
            elif tc.exit_signal is not None and getattr(tc.exit_signal, "_blocked_streak", False):
                ctx.counters["blocked_streak"] = ctx.counters.get("blocked_streak", 0) + 1
        log.info("Phase 2a (sell): %d exits from %d held", len(ctx.exits), len(sell_tctxs))

        # ── Phase 2b: parallel candidate scoring ───────────────────────────────
        if not (ctx.buy_blocked and not ctx.bear_only):
            universe   = _buy_universe(ctx)
            cand_tctxs = [_make_cand_tctx(ctx, t) for t in universe]
            run_parallel(cand_tctxs, TickerCandidateJob())
            for tc in cand_tctxs:
                if tc.candidate is not None:
                    ctx.candidates.append(tc.candidate)
            log.info("Phase 2b (buy scan): %d candidates from %d tickers", len(ctx.candidates), len(universe))

        # ── Phase 3: global aggregation ────────────────────────────────────────
        RankingJob().run(ctx)
        SelectionJob().run(ctx)

        log.info("InferencePipeline DONE  total=%.2fs", time.monotonic() - t0)


# ── SellOnlyPipeline ───────────────────────────────────────────────────────────

class SellOnlyPipeline:
    """Intraday exit-only variant: global gates + parallel sell evaluation."""

    def run(self, ctx: InferenceContext) -> None:
        from .jobs.regime   import RegimeJob
        from .jobs.drawdown import DrawdownJob
        from .jobs.sell     import TickerSellJob

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
