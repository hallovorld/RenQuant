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
import math

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
from kernel.pipeline.task_benchmark_sleeve import (
    benchmark_sleeve_ticker,
    exclude_benchmark_sleeve_from_alpha,
)

from .tasks import (
    ApplyConvictionCapTask,
    ApplyExitOnlyTopupGuardTask,
    ApplyExposureScalingTask,
    ApplyGrinoldKahnTransformTask,
    ApplyProportionalTradeTask,
    ApplySectorMetadataGuardTask,
    AlignQPHorizonUnitsTask,
    ForceMuSourceTask,
    BuildADVVectorTask,
    BuildConstraintSnapshotTask,
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
    ValidateQPMuContractTask,
    _qp_buy_admission_block_reason,
    _qp_cfg,
    _qp_max_positions,
)

log = logging.getLogger("kernel.portfolio_qp.job")


class _BuildMuVectorTask(BuildVectorFromMappingTask):
    """Specialized: μ from candidates first, holdings second.

    Do not fall back to raw panel/rank scores here. QP μ is an
    expected-return-like quantity; raw score sources must be requested
    explicitly through ForceMuSourceTask and normalized by alpha_to_mu.
    """

    def __init__(self):
        super().__init__(
            tickers_field="_qp_tickers",
            source_field="_qp_mu_source_map",
            attr="mu", target="_qp_mu",
            default=0.0,
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
        sleeve_ticker = (
            benchmark_sleeve_ticker(ctx)
            if exclude_benchmark_sleeve_from_alpha(ctx) else None
        )
        holdings = ctx.holdings or {}
        exit_only_tickers: set[str] = set(
            getattr(ctx, "_qp_exit_only_tickers", set()) or set()
        )
        for t, hs in holdings.items():
            if t == sleeve_ticker:
                continue
            src[t] = hs
        admitted_new_tickers: set[str] = set()
        blocked_map = getattr(ctx, "_blocked_by_ticker", None)
        if blocked_map is None:
            blocked_map = {}
            ctx._blocked_by_ticker = blocked_map  # noqa: SLF001
        short_tickers = {
            getattr(c, "ticker", None)
            for c in (getattr(ctx, "short_candidates", None) or [])
        }
        eligible_new_candidates: list = []
        for c in self._ordered_long_candidates(ctx):
            t = getattr(c, "ticker", None)
            if not t:
                continue
            if t == sleeve_ticker:
                blocked_map.setdefault(t, "benchmark_sleeve_excluded_from_alpha_qp")
                continue
            if t in holdings:
                src[t] = c   # candidate wins (newer scores)
                continue
            if t in short_tickers:
                # The short-candidate phase below owns this ticker. Do not let
                # a broad long-side candidate consume a scarce long slot first.
                continue
            reason = self._new_candidate_block_reason(
                ctx,
                src,
                t,
                c,
                admitted_new_tickers=admitted_new_tickers,
                ignore_slots=True,
            )
            if reason:
                blocked_map.setdefault(t, reason)
                log.info(
                    "QP_SOLVER_UNIVERSE_EXCLUDED %-6s %s "
                    "(QP only optimizes admitted buy alpha)",
                    t, reason,
                )
                continue
            eligible_new_candidates.append(c)

        selected_new, slot_rejected = self._select_new_candidates_for_slots(
            ctx, src, eligible_new_candidates,
        )
        for c in selected_new:
            t = getattr(c, "ticker", None)
            if not t:
                continue
            admitted_new_tickers.add(t)
            src[t] = c
        for c in slot_rejected:
            t = getattr(c, "ticker", None)
            if not t:
                continue
            blocked_map.setdefault(t, "qp_admission_no_slot")
            log.info(
                "QP_SOLVER_UNIVERSE_EXCLUDED %-6s %s "
                "(QP only optimizes admitted buy alpha)",
                t, "qp_admission_no_slot",
            )
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
            if t and t != sleeve_ticker:
                src[t] = c
                exit_only_tickers.discard(t)
        ctx._qp_mu_source_map = src   # noqa: SLF001
        ctx._qp_exit_only_tickers = exit_only_tickers  # noqa: SLF001
        self._sync_ticker_order(ctx, src)

    @staticmethod
    def _new_candidate_block_reason(
        ctx,
        src: dict,
        ticker: str,
        cand,
        *,
        admitted_new_tickers: set[str] | None = None,
        ignore_slots: bool = False,
    ) -> str | None:
        if bool(getattr(ctx, "buy_blocked", False)):
            return "buy_blocked"
        if bool(getattr(ctx, "skip_buys", False)):
            return "skip_buys"
        joint = _qp_cfg(ctx)
        env = {
            "cfg": joint,
            "holdings_set": set((ctx.holdings or {}).keys()),
            "holdings": ctx.holdings or {},
            "preexisting_exit_tickers": {
                t for t, _ in (getattr(ctx, "exits", None) or [])
            },
            "max_positions": _qp_max_positions(ctx),
            "score_sources": {**src, ticker: cand},
            "cands": {ticker: cand},
            "admitted_new_tickers": set(admitted_new_tickers or set()),
            "ignore_slots": bool(ignore_slots),
        }
        return _qp_buy_admission_block_reason(ctx, env, ticker)

    @staticmethod
    def _select_new_candidates_for_slots(ctx, src: dict, candidates: list) -> tuple[list, list]:
        if not candidates:
            return [], []
        joint = _qp_cfg(ctx)
        gate = joint.get("qp_admission_gate") or {}
        if not bool(gate.get("enabled", False)):
            return candidates, []
        if not bool(gate.get("respect_open_slots", True)):
            return candidates, []

        held_after_exits = set((ctx.holdings or {}).keys()) - {
            t for t, _ in (getattr(ctx, "exits", None) or [])
        }
        open_slots = max(0, _qp_max_positions(ctx) - len(held_after_exits))
        if open_slots >= len(candidates):
            return candidates, []
        if open_slots <= 0:
            return [], list(candidates)

        mode = _BuildSourceMapTask._resolve_slot_priority_mode(ctx, gate)
        ordered = sorted(
            candidates,
            key=lambda c: _BuildSourceMapTask._candidate_slot_priority(c, gate, mode=mode),
            reverse=True,
        )
        selected = ordered[:open_slots]
        selected_tickers = {getattr(c, "ticker", None) for c in selected}
        rejected = [
            c for c in candidates
            if getattr(c, "ticker", None) not in selected_tickers
        ]
        return selected, rejected

    @staticmethod
    def _ordered_long_candidates(ctx) -> list:
        """Return buy candidates in the RankingJob order.

        QP is a sizing/rebalance layer. It must not reconstruct alpha
        selection from the broad candidate pool after RankingJob has already
        produced ``ctx.ranked``. Falling back to ``ctx.candidates`` preserves
        direct unit-call behaviour when RankingJob has not run.
        """
        candidates = list(getattr(ctx, "candidates", None) or [])
        if not candidates:
            return []
        ranked = list(getattr(ctx, "ranked", None) or [])
        if not ranked:
            for idx, cand in enumerate(candidates):
                if not math.isfinite(_finite_attr(cand, "_ranking_order_index")):
                    setattr(cand, "_ranking_order_index", idx)
            return candidates

        seen: set[str] = set()
        ordered: list = []
        candidate_by_ticker = {
            getattr(c, "ticker", None): c
            for c in candidates
            if getattr(c, "ticker", None)
        }
        for cand in ranked:
            ticker = getattr(cand, "ticker", None)
            if not ticker or ticker in seen:
                continue
            canonical = candidate_by_ticker.get(ticker, cand)
            if canonical is not cand:
                for attr in (
                    "_ranking_composite",
                    "_ranking_norm_rank",
                    "_ranking_norm_rs",
                    "_ranking_order_index",
                ):
                    if hasattr(cand, attr):
                        setattr(canonical, attr, getattr(cand, attr))
            ordered.append(canonical)
            seen.add(ticker)
        for idx, cand in enumerate(ordered):
            if not math.isfinite(_finite_attr(cand, "_ranking_order_index")):
                setattr(cand, "_ranking_order_index", idx)
        return ordered

    @staticmethod
    def _resolve_slot_priority_mode(ctx, gate: dict) -> str:
        configured = str(gate.get("slot_priority", "")).strip().lower()
        if configured:
            return configured
        try:
            _, w_rs = getattr(ctx, "_blend_w", (1.0, 0.0))
            if float(w_rs) > 0.0:
                return "ranking_composite"
        except (TypeError, ValueError):
            pass
        return "rank_score"

    @staticmethod
    def _candidate_slot_priority(cand, gate: dict, *, mode: str | None = None) -> tuple[float, float, float, float, float, float]:
        mode = str(mode or gate.get("slot_priority", "rank_score")).strip().lower()
        if mode in {"kelly", "kelly_target", "kelly_target_pct"}:
            primary = _finite_attr(cand, "kelly_target_pct")
        elif mode in {"mu_over_sigma", "edge_over_sigma"}:
            mu = _finite_attr(cand, "mu")
            sigma = _finite_attr(cand, "sigma")
            primary = (
                mu / max(sigma, 1e-12)
                if math.isfinite(mu) and math.isfinite(sigma) and sigma > 0
                else float("-inf")
            )
        elif mode in {"panel", "panel_score"}:
            primary = _finite_attr(cand, "panel_score")
        elif mode in {"ranking", "ranking_composite", "composite", "blend", "blended"}:
            primary = _finite_attr(cand, "_ranking_composite")
        elif mode in {"ranking_order", "ranked_order"}:
            idx = _finite_attr(cand, "_ranking_order_index")
            primary = -idx if math.isfinite(idx) else float("-inf")
        else:
            primary = _finite_attr(cand, "rank_score")

        rank = _finite_attr(cand, "rank_score")
        panel = _finite_attr(cand, "panel_score")
        mu = _finite_attr(cand, "mu")
        sigma = _finite_attr(cand, "sigma")
        idx = _finite_attr(cand, "_ranking_order_index")
        if not math.isfinite(primary):
            primary = rank
        return (
            primary if math.isfinite(primary) else float("-inf"),
            -idx if math.isfinite(idx) else float("-inf"),
            rank if math.isfinite(rank) else float("-inf"),
            panel if math.isfinite(panel) else float("-inf"),
            mu if math.isfinite(mu) else float("-inf"),
            -sigma if math.isfinite(sigma) else float("-inf"),
        )

    @staticmethod
    def _sync_ticker_order(ctx, src: dict) -> None:
        ctx._qp_tickers = list(src)  # noqa: SLF001


def _finite_attr(obj, name: str) -> float:
    value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


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
        if getattr(ctx, "_calibrator_contract_failed", False):
            return True
        if getattr(ctx, "_panel_scoring_contract_failed", False):
            log.info(
                "JointPortfolioQPJob: skipped because panel scoring contract "
                "already failed; QP is a sizing layer, not an alpha fallback"
            )
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
            # QP expects μ and σ/Σ to share one rebalance horizon. Calibrator
            # μ is 60d in prod; realized-vol fallback is annualized, so align
            # σ before either covariance construction or alpha-to-mu scaling.
            AlignQPHorizonUnitsTask(),
            # 2026-05-12: Grinold-Kahn α→μ transform (off by default).
            # Normalizes ANY scoring source (LTR panel_score / NGBoost μ /
            # custom) to σ-scale, decoupling QP risk-penalty calibration
            # from input scale. See doc/AUDIT_2026-05-12_dead_paths.md
            # §NGBoost SUSPECT — μ-scale mismatch.
            ApplyGrinoldKahnTransformTask(),
            # QP expects μ to be expected-return-like. Strict by default:
            # raw rank/panel scores cannot silently reach the optimizer.
            ValidateQPMuContractTask(),
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
            # Held names that are not current buy candidates stay available
            # for trims/closes, but QP must not add fresh risk to them.
            ApplyExitOnlyTopupGuardTask(),
            # Missing sector metadata cannot be an implicit exemption from
            # sector constraints. Cap unmapped names at current weight before
            # building sector/correlation matrices.
            ApplySectorMetadataGuardTask(),
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

            # ── Phase 3b: freeze the assembled constraint state ────────
            # Step 1c of §8 plan (PR #125). The snapshot is the contract
            # downstream allocators (current QP, simplified-QP, Hybrid,
            # MPO, …) consume. Constructor failure short-circuits the
            # Job before SolveMarkowitzQPTask runs — fail loud on a
            # contradictory constraint state instead of feeding it to
            # cvxpy.
            BuildConstraintSnapshotTask(),

            # ── Phase 4: solve (domain) ────────────────────────────────
            SolveMarkowitzQPTask(),

            # ── Phase 4b: partial-rebalance (research B, default off) ──
            # Gârleanu-Pedersen 2013: if regime_params.<R>.qp_partial_trade
            # _horizon_days > 1, shrink the QP target by 1/N for smooth
            # multi-day rebalancing. No-op when N ≤ 1 (legacy behaviour).
            ApplyProportionalTradeTask(),

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
