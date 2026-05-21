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
    ApplyConvictionCapTask,
    ApplyExposureScalingTask,
    ApplyGrinoldKahnTransformTask,
    ForceMuSourceTask,
    BuildADVVectorTask,
    BuildCorrelationGroupConstraintTask,
    BuildSectorConstraintMatrixTask,
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
    Candidates win when both have the ticker (latest scoring data).

    2026-05-13 Long-Short Phase 2A: also include ctx.short_candidates
    (bottom-of-rank tickers) when long_short.enabled. The QP will
    optimize over the joint long+short candidate set + holdings.
    """
    name = "BuildSourceMapTask"

    def run(self, ctx) -> bool | None:
        src: dict = {}
        for t, hs in (ctx.holdings or {}).items():
            src[t] = hs
        for c in (ctx.candidates or []):
            t = getattr(c, "ticker", None)
            if t:
                src[t] = c   # candidate wins (newer scores)
        # Phase 2B fix (2026-05-14): short candidates OVERRIDE long candidates
        # for the same ticker. ctx.candidates is the BROAD admission pool
        # (60-70 names that passed earnings/wash-sale gates) while
        # ctx.short_candidates is the bottom-decile of the FULL universe by
        # panel score. They overlap at the bottom of the admission pool.
        # Pre-fix the `if t not in src` check left the long-side positive
        # mu in place → QP never allocated negative weights → "longshort"
        # sims ran as 130% long-only (leverage from gross_max=1.30), giving
        # false Tier 3 readings on 2026-05-14. Override ensures the short
        # candidate's signed panel_score reaches the QP.
        for c in (getattr(ctx, "short_candidates", None) or []):
            t = getattr(c, "ticker", None)
            if t:
                src[t] = c
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
        rotation = ctx.config.get("rotation", {})
        allowed_regimes = set(rotation.get("enabled_regimes", []) or [])
        if allowed_regimes and getattr(ctx, "regime", None) not in allowed_regimes:
            return True
        joint = rotation.get("joint_actions", {})
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
            # 2026-05-12: Option A NGBoost validator (off by default).
            # When ngboost.enabled=true AND ranking.qp_mu_source='panel_score',
            # forces μ_QP back to the LTR z-score scale so we can isolate
            # whether NGBoost's σ (in Kelly path) adds value independent
            # of the destructive μ-scale mismatch.
            ForceMuSourceTask(),
            # 2026-05-12: Grinold-Kahn α→μ transform (off by default).
            # Normalizes ANY scoring source (LTR panel_score / NGBoost μ /
            # custom) to σ-scale, decoupling QP risk-penalty calibration
            # from input scale. See doc/AUDIT_2026-05-12_dead_paths.md
            # §NGBoost SUSPECT — μ-scale mismatch.
            ApplyGrinoldKahnTransformTask(),
            BuildWeightVectorTask(),
            ComputeFullSigmaTask(),
            ShrinkSigmaLedoitWolfTask(),           # G5: LW shrinkage (off by default)

            # ── Phase 3: tax + constraints (domain) ────────────────────
            ComputeBrownSmithTaxCostTask(),
            ComputeWashSaleMaskTask(),
            BuildADVVectorTask(),                  # G3: per-asset ADV from ohlcv
            ComputeQPConstraintsTask(),            # ← per-name caps, w_upper, …
            # 2026-05-12 dead-path fix: hoist vol-target + DD-Kelly scaling
            # out of the dormant Kelly path into the QP bounds. Composes
            # multiplicatively with conviction & sector caps below. See
            # doc/AUDIT_2026-05-12_dead_paths.md.
            ApplyExposureScalingTask(),
            # 2026-05-11 A2: per-ticker conviction shrink of w_upper.
            # OFF by default; opt-in via
            #   rotation.joint_actions.qp_conviction_cap_enabled=true
            # MUST run BEFORE sector/correlation tasks (they anchor on
            # _qp_w_upper.max()).
            ApplyConvictionCapTask(),
            # 2026-05-10 C2: hard sector + correlation pair caps. MUST run
            # AFTER ComputeQPConstraintsTask so the sector / corr Tasks can
            # read ctx._qp_w_upper for cap anchoring.
            BuildSectorConstraintMatrixTask(),
            BuildCorrelationGroupConstraintTask(),

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
