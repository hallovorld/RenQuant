"""JointPortfolioQPJob — Job that orchestrates atoms + domain Tasks.

User mandate (2026-05-04 §1c): Job is "where complexity lives" —
sequence/concurrent/conditional. Atoms handle generic boilerplate
(skip gates, vector building, counters, logging). Domain Tasks handle
QP-specific math (tax cost, Σ, solve, emit).

Composition:
    SkipIfConfigDisabledTask    × 2  [atom]   solver==qp + enabled
    SkipIfFieldEqualsTask              [atom]   bear_only != True
    StableTickerOrderTask              [atom]   build _qp_tickers
    BuildVectorFromMappingTask  × 2  [atom]   _qp_mu, _qp_sigma
    BuildWeightVectorTask              [domain] _qp_w_current (NAV math)
    ComputeFullSigmaTask               [domain] _qp_Sigma_full from corr
    ComputeBrownSmithTaxCostTask       [domain] _qp_tax_cost
    ComputeWashSaleMaskTask            [domain] _qp_wash_mask
    ComputeQPConstraintsTask           [domain] _qp_w_upper, dw_max, etc
    SolveMarkowitzQPTask               [domain] _qp_solution
    EmitOrdersFromQPSolutionTask       [domain] ctx.orders / ctx.exits
    IncrementCounterTask        × 2  [atom]   qp_buys, qp_sells
    LogSummaryTask                     [atom]   one-line summary
"""
from __future__ import annotations

import logging

from kernel.pipeline.atoms import (
    BuildVectorFromMappingTask,
    IncrementCounterTask,
    LogSummaryTask,
    SkipIfConfigDisabledTask,
    SkipIfFieldEqualsTask,
    StableTickerOrderTask,
)
from kernel.pipeline.context import InferenceContext
from kernel.pipeline.pipeline import Job, Task

from .tasks import (
    BuildADVVectorTask,
    BuildWeightVectorTask,
    ComputeBrownSmithTaxCostTask,
    ComputeFullSigmaTask,
    ComputeQPConstraintsTask,
    ComputeWashSaleMaskTask,
    EmitOrdersFromQPSolutionTask,
    ShrinkSigmaLedoitWolfTask,
    SolveMarkowitzQPTask,
)

log = logging.getLogger("kernel.portfolio_qp.job")


class _BuildMuVectorTask(BuildVectorFromMappingTask):
    """Specialized: μ from candidates first, holdings second, panel_score
    fallback. Inherits all NaN-handling from the atom."""

    def __init__(self):
        super().__init__(
            tickers_field="_qp_tickers",
            source_field="_qp_mu_source_map",
            attr="mu", target="_qp_mu",
            default=0.0, fallback_attr="panel_score",
        )

    @property
    def name(self) -> str:
        return "BuildMuVectorTask"


class _BuildSigmaVectorTask(BuildVectorFromMappingTask):
    """Specialized: σ from candidates+holdings union; default 5%."""

    def __init__(self):
        super().__init__(
            tickers_field="_qp_tickers",
            source_field="_qp_mu_source_map",   # same source dict
            attr="sigma", target="_qp_sigma",
            default=0.05,
        )

    @property
    def name(self) -> str:
        return "BuildSigmaVectorTask"


class _BuildSourceMapTask(Task):
    """Build dict {ticker: candidate_or_holding} for vector tasks to consume.
    Candidates win when both have the ticker (latest scoring data)."""
    name = "BuildSourceMapTask"

    def run(self, ctx) -> bool | None:
        src: dict = {}
        for t, hs in (ctx.holdings or {}).items():
            src[t] = hs
        for c in (ctx.candidates or []):
            t = getattr(c, "ticker", None)
            if t:
                src[t] = c   # candidate wins (newer scores)
        ctx._qp_mu_source_map = src   # noqa: SLF001


class JointPortfolioQPJob(Job):
    """5-phase QP optimization composed of atoms + 7 domain Tasks.

    Order is load-bearing — every domain Task depends on outputs of
    upstream ones via the documented `ctx._qp_*` private fields.
    """

    name = "JointPortfolioQPJob"

    def should_skip(self, ctx: InferenceContext) -> bool:
        # The atom-based skip gates inside `tasks` cover this too, but
        # short-circuiting at Job level avoids per-task method calls.
        joint = (ctx.config.get("rotation", {})
                            .get("joint_actions", {}))
        if not joint.get("enabled", False):
            return True
        if str(joint.get("solver", "greedy")).lower() != "qp":
            return True
        if getattr(ctx, "bear_only", False):
            return True
        return False

    @property
    def tasks(self) -> list[Task]:
        return [
            # ── Phase 1: ticker order + source map (atoms) ─────────────
            StableTickerOrderTask("holdings", "candidates", "_qp_tickers"),
            _BuildSourceMapTask(),

            # ── Phase 2: build vectors (atom + domain) ─────────────────
            _BuildMuVectorTask(),
            _BuildSigmaVectorTask(),
            BuildWeightVectorTask(),
            ComputeFullSigmaTask(),
            ShrinkSigmaLedoitWolfTask(),           # G5: LW shrinkage (off by default)

            # ── Phase 3: tax + constraints (domain) ────────────────────
            ComputeBrownSmithTaxCostTask(),
            ComputeWashSaleMaskTask(),
            BuildADVVectorTask(),                  # G3: per-asset ADV from ohlcv
            ComputeQPConstraintsTask(),

            # ── Phase 4: solve (domain) ────────────────────────────────
            SolveMarkowitzQPTask(),

            # ── Phase 5: emit (domain) ─────────────────────────────────
            EmitOrdersFromQPSolutionTask(),

            # ── Phase 6: telemetry (atoms) ─────────────────────────────
            IncrementCounterTask("qp_buys",  amount="_qp_n_buys"),
            IncrementCounterTask("qp_sells", amount="_qp_n_sells"),
            # Log line — n=count of tickers (not the list itself); use %s
            # so the % formatting tolerates list/None gracefully and never
            # falls into LogSummaryTask's silent except path. Audit P2-1
            # 2026-05-04: previously `%d` on a list silently spammed
            # "LogSummaryTask: format failed" once per QP bar.
            LogSummaryTask(
                "JointPortfolioQPJob: buys=%s sells=%s obj=%s iter=%s",
                fields=(
                    "_qp_n_buys", "_qp_n_sells",
                    "_qp_solution.objective",
                    "_qp_solution.n_iter",
                ),
                level="info",
                logger="kernel.portfolio_qp.job",
            ),
        ]


__all__ = ["JointPortfolioQPJob"]
