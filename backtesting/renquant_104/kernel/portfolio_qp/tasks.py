"""Portfolio-QP pipeline Tasks — atom-composed.

User mandate (2026-05-04 §1c): Tasks are reusable atoms; domain Tasks
glue them with QP-specific math. This file holds the QP-specific
domain Tasks; reusable building blocks live in
`kernel/pipeline/atoms/`.

Job composition (in `job_qp.py`):

    JointPortfolioQPJob
    ├── SkipIfConfigDisabledTask("rotation.joint_actions.enabled")     [atom]
    ├── SkipIfFieldEqualsTask("bear_only", True)                        [atom]
    ├── StableTickerOrderTask("holdings", "candidates", "_qp_tickers")  [atom]
    ├── BuildWeightVectorTask                                           [domain]
    ├── BuildVectorFromMappingTask × N (mu, sigma)                      [atom]
    ├── ComputeFullSigmaTask                                            [domain]
    ├── ComputeBrownSmithTaxCostTask                                    [domain]
    ├── ComputeWashSaleMaskTask                                         [domain — uses BuildMaskFromConditionTask atom]
    ├── ComputeQPConstraintsTask                                        [domain]
    ├── SolveMarkowitzQPTask                                            [domain]
    ├── EmitOrdersFromQPSolutionTask                                    [domain]
    ├── IncrementCounterTask × 2                                        [atom]
    └── LogSummaryTask                                                  [atom]

Each domain Task here is ≤30 lines body, single-responsibility.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from kernel.pipeline.atoms.ctx_ops import _get_path, _set_path
from kernel.pipeline.context import InferenceContext
from kernel.pipeline.order_attribution import stamp_order_attribution
from kernel.pipeline.pipeline import Task

log = logging.getLogger("kernel.portfolio_qp.tasks")


def _ensure_blocked_map(ctx) -> dict:
    blocked_map = getattr(ctx, "_blocked_by_ticker", None)
    if blocked_map is None:
        blocked_map = {}
        ctx._blocked_by_ticker = blocked_map  # noqa: SLF001
    return blocked_map


def _stamp_qp_ticker_block(ctx, ticker: str, reason: str) -> None:
    if not ticker:
        return
    _ensure_blocked_map(ctx).setdefault(str(ticker), reason)


def _stamp_all_qp_blocks(ctx, reason: str) -> None:
    for ticker in (_get_path(ctx, "_qp_tickers") or []):
        _stamp_qp_ticker_block(ctx, str(ticker), reason)


# Module-level single source of truth for "the QP solution can drive orders".
# Both ``SolveMarkowitzQPTask`` (failure-stamping branch) and
# ``EmitOrdersFromQPSolutionTask`` (emit branch) consult this set; keeping
# them in sync prevents the codex #75/#10 blocker class where a synthetic
# success status is treated as a failure upstream while the emit task
# happily emits orders from it.
#
# Add a new status here when a new synthetic-but-successful solver path
# lands (e.g., ``cap_compliance_fallback`` for the audit #2 / issue #70
# force-sell escape).
QP_EMITTABLE_STATUSES: frozenset[str] = frozenset({
    "optimal",
    "cap_compliance_fallback",
})


def _stamp_qp_failure_counter(ctx, status: str) -> None:
    """Single source for the no-trade observability counters (codex PR #48
    review #1). Every QP failure path that sets ``ctx._qp_status`` /
    ``ctx._qp_failure_reason`` MUST also call this helper so that
    ``live.runner._why_no_trade()`` surfaces the binding QP failure instead of
    falling through to an earlier upstream drop.

    Idempotent per ctx (codex PR #48 v2 review): SolveMarkowitzQPTask stamps
    on non-optimal status but does NOT return False; the job then runs
    EmitOrdersFromQPSolutionTask which sees the same status and would stamp
    again — producing ``qp_infeasible=2`` for a single bar. Once stamped,
    subsequent calls within the same context are no-ops via the
    ``ctx._qp_failure_counter_stamped`` sentinel.

    Paths covered:
      - ``ComputeFullSigmaTask._fail_full_sigma`` (infeasible:<cov-reason>,
        returns False so emit never runs)
      - ``SolveMarkowitzQPTask`` unsupported-cvxportfolio branch
        (sets infeasible solution, returns False)
      - ``SolveMarkowitzQPTask`` non-optimal solver outcome
        (does NOT return False — emit still runs but is now idempotent)
      - ``EmitOrdersFromQPSolutionTask`` non-optimal handling
        (catches paths that reach emit directly; idempotent against Solve)

    Counter keys mirror live.runner._why_no_trade() precedence:
      qp_infeasible / qp_missing_solution / qp_optimal_no_signal /
      qp_other_nonoptimal.
    """
    counters = getattr(ctx, "counters", None)
    if not isinstance(counters, dict):
        return
    if getattr(ctx, "_qp_failure_counter_stamped", False):
        return                              # idempotent: one stamp per QP failure event
    s = (status or "").strip()
    if not s:
        return
    if "infeasible" in s:
        key = "qp_infeasible"
    elif s == "missing_solution":
        key = "qp_missing_solution"
    elif s == "optimal_no_signal":
        key = "qp_optimal_no_signal"
    elif s.startswith("optimal"):
        return                              # successful path; do not stamp a failure counter
    else:
        key = "qp_other_nonoptimal"
    counters[key] = counters.get(key, 0) + 1
    try:
        ctx._qp_failure_counter_stamped = True  # noqa: SLF001
    except Exception:                        # context may be read-only mock
        pass


# ── 1. Build w_current from shares × prices / NAV ────────────────────────────

class BuildWeightVectorTask(Task):
    """Compute current portfolio weight vector from holdings.

    Reads:  ctx._qp_tickers (list[str]), ctx.holdings (dict),
             ctx.prices (dict), ctx.portfolio_value (float)
    Writes: ctx._qp_w_current (np.ndarray)
    """
    name = "BuildWeightVectorTask"

    def run(self, ctx) -> bool | None:
        tickers = _get_path(ctx, "_qp_tickers") or []
        if not tickers:
            return False
        nav = float(_get_path(ctx, "portfolio_value", 0.0) or 0.0)
        if nav <= 0:
            return False
        prices = _get_path(ctx, "prices") or {}
        holdings = _get_path(ctx, "holdings") or {}
        w = np.zeros(len(tickers))
        for i, t in enumerate(tickers):
            hs = holdings.get(t)
            if hs is None:
                continue
            shares = float(getattr(hs, "shares", 0.0) or 0.0)
            px = float(prices.get(t, 0.0) or 0.0)
            if px > 0:
                w[i] = shares * px / nav
        ctx._qp_w_current = w  # noqa: SLF001


# ── 2. Build full Σ from a cached correlation matrix ────────────────────────

class ComputeFullSigmaTask(Task):
    """Build n×n Σ_full = ρ × σ_i × σ_j from loaded/configured correlations.

    Reads:  ctx._qp_tickers, ctx._qp_sigma, ctx.corr_matrix,
             ctx.config['_strategy_dir'], ctx.config['regime']['correlation_artifact'],
             ctx.config['rotation']['joint_actions']['qp_use_full_sigma']
    Writes: ctx._qp_Sigma_full (np.ndarray | None — None only when full
             covariance is disabled, or when an explicit diagnostic
             diagonal fallback is enabled)
    """
    name = "ComputeFullSigmaTask"

    def run(self, ctx) -> bool | None:
        cfg = _qp_cfg(ctx)
        if not bool(cfg.get("qp_use_full_sigma", True)):
            ctx._qp_Sigma_full = None  # noqa: SLF001
            return
        tickers = _get_path(ctx, "_qp_tickers") or []
        n = len(tickers)
        sig = self._coerce_sigma(ctx, n)
        if sig is None:
            return self._fail_full_sigma(ctx, "qp_full_sigma_invalid_sigma")

        Sigma = self._diagonal_sigma(sig)
        if n <= 1:
            ctx._qp_Sigma_full = Sigma + 1e-8 * np.eye(n)  # noqa: SLF001
            return

        corr = self._resolve_corr(ctx)
        if not corr:
            return self._handle_missing_corr(ctx, cfg)
        allow_diag = _allow_diagonal_sigma_fallback(cfg)
        missing_pairs, invalid_pairs = self._fill_corr_covariances(
            Sigma, tickers, sig, corr, allow_diag=allow_diag,
        )
        if missing_pairs or invalid_pairs:
            return self._fail_full_sigma(
                ctx,
                "qp_full_sigma_incomplete_corr",
                missing_pairs=missing_pairs,
                invalid_pairs=invalid_pairs,
            )
        self._stamp_all_zero_corr_fallback(ctx, Sigma, allow_diag=allow_diag)
        ctx._qp_Sigma_full = Sigma + 1e-8 * np.eye(n)  # noqa: SLF001

    @staticmethod
    def _coerce_sigma(ctx, n: int):
        try:
            sig = np.asarray(_get_path(ctx, "_qp_sigma"), dtype=float)
        except (TypeError, ValueError):
            return None
        if sig.shape != (n,) or not np.isfinite(sig).all() or (sig <= 0).any():
            return None
        return sig

    @staticmethod
    def _diagonal_sigma(sig):
        Sigma = np.zeros((len(sig), len(sig)))
        for i in range(len(sig)):
            Sigma[i, i] = sig[i] ** 2
        return Sigma

    def _resolve_corr(self, ctx):
        corr = getattr(ctx, "corr_matrix", None)
        return corr or self._load_corr_from_artifact(ctx)

    def _handle_missing_corr(self, ctx, cfg: dict) -> bool | None:
        if _allow_diagonal_sigma_fallback(cfg):
            ctx._qp_Sigma_full = None  # noqa: SLF001
            ctx._qp_covariance_fallback_reason = "qp_full_sigma_missing_corr"  # noqa: SLF001
            log.warning(
                "ComputeFullSigmaTask: qp_use_full_sigma=true but no "
                "correlation matrix was loaded; explicit diagonal "
                "fallback enabled."
            )
            return None
        return self._fail_full_sigma(ctx, "qp_full_sigma_missing_corr")

    @staticmethod
    def _fill_corr_covariances(Sigma, tickers, sig, corr, *, allow_diag: bool):
        missing_pairs: list[str] = []
        invalid_pairs: list[str] = []
        for i, ti in enumerate(tickers):
            for j in range(i + 1, len(tickers)):
                tj = tickers[j]
                rho_f = _coerce_corr_for_qp(
                    corr, ti, tj, allow_diag=allow_diag,
                    missing_pairs=missing_pairs,
                    invalid_pairs=invalid_pairs,
                )
                if rho_f is None:
                    continue
                cov = rho_f * sig[i] * sig[j]
                Sigma[i, j] = cov
                Sigma[j, i] = cov
        return missing_pairs, invalid_pairs

    @staticmethod
    def _stamp_all_zero_corr_fallback(ctx, Sigma, *, allow_diag: bool) -> None:
        if not allow_diag or len(Sigma) <= 1:
            return
        n_off_diag = int(
            (np.abs(Sigma) > 1e-12).sum()
            - np.count_nonzero(np.diag(Sigma))
        )
        if n_off_diag == 0:
            ctx._qp_covariance_fallback_reason = "qp_full_sigma_all_zero_corr"  # noqa: SLF001

    @staticmethod
    def _fail_full_sigma(
        ctx,
        reason: str,
        *,
        missing_pairs: list[str] | None = None,
        invalid_pairs: list[str] | None = None,
    ) -> bool:
        ctx._qp_Sigma_full = None  # noqa: SLF001
        ctx._qp_status = f"infeasible:{reason}"  # noqa: SLF001
        ctx._qp_failure_reason = reason  # noqa: SLF001
        ctx._qp_n_buys = 0  # noqa: SLF001
        ctx._qp_n_sells = 0  # noqa: SLF001
        ctx._qp_covariance_issue = {  # noqa: SLF001
            "reason": reason,
            "missing_pairs": list(missing_pairs or [])[:25],
            "invalid_pairs": list(invalid_pairs or [])[:25],
        }
        _stamp_all_qp_blocks(ctx, reason)
        _stamp_qp_failure_counter(ctx, ctx._qp_status)  # noqa: SLF001 (codex PR #48 #1)
        log.error(
            "ComputeFullSigmaTask: full covariance required but unavailable "
            "(reason=%s missing_pairs=%d invalid_pairs=%d); blocking QP orders",
            reason,
            len(missing_pairs or []),
            len(invalid_pairs or []),
        )
        return False

    @staticmethod
    def _load_corr_from_artifact(ctx) -> dict | None:
        sd = (ctx.config or {}).get("_strategy_dir", "")
        if not sd:
            return None
        rel = (
            (ctx.config or {})
            .get("regime", {})
            .get("correlation_artifact", "prod/watchlist-correlation.json")
        )
        rel_path = Path(str(rel))
        path = rel_path if rel_path.is_absolute() else Path(sd) / "artifacts" / rel_path
        if not path.exists():
            ctx._qp_corr_load_error = f"missing:{path}"  # noqa: SLF001
            return None
        try:
            raw = json.loads(path.read_text())
            from kernel.walk_forward import (  # noqa: PLC0415
                assert_correlation_no_leakage,
                parse_correlation_artifact,
            )
            corr, as_of = parse_correlation_artifact(raw)
            config = ctx.config or {}
            assert_correlation_no_leakage(
                as_of,
                config.get("backtest_start"),
                is_live_mode=bool(config.get("_is_live_mode", False)),
                allow_legacy_without_as_of=bool(
                    (config.get("regime", {}) or {})
                    .get("allow_legacy_correlation_without_as_of", False)
                ),
                context="ComputeFullSigmaTask corr",
            )
            return corr
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            ctx._qp_corr_load_error = f"{path}:{type(exc).__name__}:{exc}"  # noqa: SLF001
            log.warning("ComputeFullSigmaTask: corr load failed from %s (%s)", path, exc)
            return None


def _clamp_w_upper_at_w_current(ctx) -> None:
    """Hard-cap-aware hold-flat clamp (2026-06-02 v3 fix — codex #123 review).

    Used at the SOFT-scaling sites (ApplyExposureScalingTask,
    ApplyConvictionCapTask). Preserves the hold-flat invariant
    Δw=0 feasible **for holdings already within the hard cap**, while
    keeping over-cap holdings exposed to the solver as a hard-cap
    violation so ``_retry_for_per_asset_cap_compliance()`` can fire.

    The hard cap is ``ctx._qp_w_upper_hard``, stamped once by
    ``ComputeQPConstraintsTask`` before any soft scaling runs. Two cases:

    * ``w_current[i] <= w_upper_hard[i]`` — the holding is within hard
      cap, so raising ``_qp_w_upper[i]`` up to ``w_current[i]`` is
      safe: it relaxes the SOFT target back to "hold". The hard cap
      is unchanged; cap-compliance retry still has the original ceiling.
    * ``w_current[i] >  w_upper_hard[i]`` — the holding is **over**
      hard cap. Set ``_qp_w_upper[i] = w_upper_hard[i]`` so the
      solver returns ``infeasible`` for that asset and
      ``_retry_for_per_asset_cap_compliance()`` fires, force-selling
      back to the hard cap. The soft-scaled value is intentionally
      DISCARDED on this branch: conviction × vol-target × drawdown
      multipliers are ≤ 1, so the soft-scaled cap is below the hard
      cap; cap-compliance docstring says we sell back to the
      *risk* cap (hard), not the soft target. Keeping the soft value
      would force-sell ORCL from 22% straight to 7.5% under a
      low-conviction multiplier — codex #123 v4 review. (v3 kept the
      soft value here, which had this bug.)

    Reads/writes ``ctx._qp_w_upper`` in place.
    """
    w_upper = _get_path(ctx, "_qp_w_upper")
    w_curr  = _get_path(ctx, "_qp_w_current")
    w_hard  = _get_path(ctx, "_qp_w_upper_hard")
    if w_upper is None or w_curr is None or len(w_upper) != len(w_curr):
        return
    w_upper_arr = np.asarray(w_upper, dtype=float)
    w_curr_arr  = np.asarray(w_curr, dtype=float)
    if w_hard is None or len(w_hard) != len(w_upper_arr):
        # No hard-cap snapshot: skip the clamp entirely. v3 invariant —
        # ComputeQPConstraintsTask must always stamp ``_qp_w_upper_hard``
        # before any soft scaling runs. Silently widening to ``w_current``
        # without the hard cap is the bug codex caught on #123 v2 (raising
        # a 15% hard cap up to a 22% over-cap holding). Skip = strict
        # behaviour; the missing stamp is a contract bug, not a soft
        # degradation surface.
        return
    w_hard_arr = np.asarray(w_hard, dtype=float)
    # Per-asset behaviour:
    #   within hard cap (w_curr ≤ hard) → raise soft cap to max(soft, current)
    #                                     so hold-flat is feasible.
    #   over    hard cap (w_curr >  hard) → restore w_hard (DISCARD soft) so
    #                                     cap-compliance retry sells back
    #                                     to the hard cap, not the soft cap.
    safe_to_raise = w_curr_arr <= w_hard_arr
    raised = np.maximum(w_upper_arr, w_curr_arr)
    ctx._qp_w_upper = np.where(safe_to_raise, raised, w_hard_arr)  # noqa: SLF001


def _lookup_corr_explicit_none(corr: dict, left: str, right: str, *, default: float = 0.0):
    """Symmetric corr lookup that treats 0.0 as real data, not missing."""
    row = corr.get(left)
    if isinstance(row, dict):
        value = row.get(right)
        if value is not None:
            return value
    row = corr.get(right)
    if isinstance(row, dict):
        value = row.get(left)
        if value is not None:
            return value
    return default


def _lookup_corr_required(corr: dict, left: str, right: str):
    """Symmetric corr lookup that returns None only when both directions miss."""
    row = corr.get(left)
    if isinstance(row, dict) and right in row and row.get(right) is not None:
        return row.get(right)
    row = corr.get(right)
    if isinstance(row, dict) and left in row and row.get(left) is not None:
        return row.get(left)
    return None


def _coerce_corr_for_qp(
    corr: dict,
    left: str,
    right: str,
    *,
    allow_diag: bool,
    missing_pairs: list[str],
    invalid_pairs: list[str],
) -> float | None:
    rho = _lookup_corr_required(corr, left, right)
    if rho is None:
        if allow_diag:
            return 0.0
        missing_pairs.append(f"{left}|{right}")
        return None
    try:
        return max(-0.99, min(0.99, float(rho)))
    except (TypeError, ValueError):
        if allow_diag:
            return 0.0
        invalid_pairs.append(f"{left}|{right}")
        return None


def _allow_diagonal_sigma_fallback(cfg: dict) -> bool:
    """Explicit diagnostic escape hatch for running QP without full Sigma."""
    if bool(cfg.get("qp_allow_diagonal_sigma_fallback", False)):
        return True
    policy = str(cfg.get("qp_full_sigma_fallback_policy", "strict")).lower()
    return policy in {"diagonal", "diag", "allow_diagonal", "diagnostic_diagonal"}


class AlignQPHorizonUnitsTask(Task):
    """Align QP σ to the same single-period horizon as μ.

    Markowitz 1952 and Boyd-Vandenberghe 2004 portfolio objectives assume
    μ and Σ describe the same rebalance period. In 104, calibrator μ is a
    forward-return estimate over `panel_ltr.lookahead_days`, while the
    realized-vol fallback is explicitly annualized. This task converts σ
    before Σ is built so risk and expected-return units match.
    """
    name = "AlignQPHorizonUnitsTask"
    TRADING_DAYS_PER_YEAR = 252.0

    def run(self, ctx) -> bool | None:
        cfg = _qp_cfg(ctx)
        mode = str(cfg.get("qp_sigma_horizon_mode", "none")).lower()
        sigma = _get_path(ctx, "_qp_sigma")
        if sigma is None or mode in {"none", "off", "disabled"}:
            return
        horizon = _resolve_qp_mu_horizon_days(ctx, cfg)
        unit = str(cfg.get("qp_sigma_unit", "horizon")).lower()
        if horizon is None:
            return _record_qp_horizon_issue(ctx, cfg, "missing_mu_horizon")
        if getattr(ctx, "_qp_sigma_horizon_scaled", False):
            return
        scale = _qp_sigma_horizon_scale(unit, horizon)
        if scale is None:
            return _record_qp_horizon_issue(ctx, cfg, f"unknown_sigma_unit:{unit}")
        sig = np.asarray(sigma, dtype=float)
        if not np.isfinite(sig).all() or (sig <= 0).any():
            return _record_qp_horizon_issue(ctx, cfg, "non_positive_sigma")
        ctx._qp_sigma_raw = sig.copy()  # noqa: SLF001
        ctx._qp_sigma = sig * scale     # noqa: SLF001
        ctx._qp_sigma_horizon_scaled = True  # noqa: SLF001
        ctx._qp_horizon_contract = {  # noqa: SLF001
            "ok": True, "sigma_unit": unit, "mu_horizon_days": int(horizon),
            "scale": float(scale),
        }


# ── 2b. Ledoit-Wolf 2004 Σ shrinkage (post-step on full Σ) ──────────────────

class ShrinkSigmaLedoitWolfTask(Task):
    """Apply Ledoit-Wolf 2004 shrinkage to Σ_full toward scalar identity.

        Σ_shrunk = (1 - λ) · Σ_full + λ · F     with F = (trace(Σ)/n) · I

    Effect: pulls off-diagonal correlation toward zero AND equalises
    diagonal variances toward the average — reducing noise on small-n
    correlation estimates. λ=0 → no change; λ=1 → identity·avg_var
    (no correlation, equal variance).

    **2026-05-10 default bumped 0.0 → 0.2** (Track C3). λ=0.2 is the
    industry-standard mid-of-range from Ledoit & Wolf 2004 ("Honey, I
    Shrunk the Sample Covariance Matrix", J. Portfolio Management 30(4):
    110-119): they show on a 169-stock universe (matching ours) the OAS
    (oracle approximating shrinkage) optimum sits in [0.13, 0.27].
    Choosing λ=0.2 (mid of that range) is conservative, robust, and
    config-overridable. Set 0.0 to disable; 1.0 → diagonal.

    Eigenvalue floor: post-shrinkage we clip Σ's eigenvalues to ≥1e-8
    (per CLAUDE.md §5.13.12) — guarantees CLARABEL/OSQP/SCS see a strict
    PSD matrix and do not stall on numerical near-singularity (a real
    failure mode pre-fix when correlation_artifact NaN cells leak into
    Σ_full and the LW blend doesn't fully wash them out).

    Reads:  ctx._qp_Sigma_full,
             ctx.config['rotation']['joint_actions']['qp_ledoit_wolf_lambda']
    Writes: ctx._qp_Sigma_full (in place; None if upstream produced None
             — diagonal-Σ fallback in solver is unaffected)
    """
    name = "ShrinkSigmaLedoitWolfTask"

    # Default λ=0.2: ledoit-wolf 2004, see class docstring. Override via
    # config['rotation']['joint_actions']['qp_ledoit_wolf_lambda'].
    DEFAULT_LAMBDA = 0.2
    EIGEN_FLOOR    = 1e-8

    def run(self, ctx) -> bool | None:
        cfg = _qp_cfg(ctx)
        lam = float(cfg.get("qp_ledoit_wolf_lambda", self.DEFAULT_LAMBDA))
        if not math.isfinite(lam) or lam <= 0.0:
            return                                      # off
        lam = min(lam, 1.0)
        S = _get_path(ctx, "_qp_Sigma_full")
        if S is None:
            return                                      # diagonal-Σ path
        n = S.shape[0]
        if n == 0:
            return
        avg_var = float(np.trace(S)) / max(n, 1)
        F = avg_var * np.eye(n)
        S_blend = (1.0 - lam) * S + lam * F
        # §5.13.12 — clamp eigenvalues so the solver always sees a sane
        # PSD matrix. Symmetrize first to absorb any asymmetric float
        # noise before eigh (which assumes Hermitian input).
        S_sym = 0.5 * (S_blend + S_blend.T)
        eigvals, eigvecs = np.linalg.eigh(S_sym)
        if (eigvals < self.EIGEN_FLOOR).any():
            eigvals = np.maximum(eigvals, self.EIGEN_FLOOR)
            S_blend = eigvecs @ np.diag(eigvals) @ eigvecs.T
            # Re-symmetrize: V·diag·V^T floats can drift ~1e-16 off-symmetric.
            S_blend = 0.5 * (S_blend + S_blend.T)
        ctx._qp_Sigma_full = S_blend  # noqa: SLF001


# ── 3. Brown-Smith dynamic tax + Berkin-Jeffrey loss-harvest ────────────────

class ComputeBrownSmithTaxCostTask(Task):
    """Per-asset tax cost vector. Brown-Smith (2011) LT-bridge for
    winners; Berkin-Jeffrey (1990) loss-harvest credit (negative cost)
    for losers when ctx.ytd_realized_gain_dollar > 0.

    Reads:  ctx._qp_tickers, ctx._qp_w_current, ctx.holdings, ctx.prices,
             ctx.portfolio_value, ctx.today, ctx.ytd_realized_gain_dollar,
             ctx.config['rotation']['joint_actions']['qp_tax_*']
    Writes: ctx._qp_tax_cost (np.ndarray)
    """
    name = "ComputeBrownSmithTaxCostTask"

    def run(self, ctx) -> bool | None:
        tickers = _get_path(ctx, "_qp_tickers") or []
        n = len(tickers)
        cost = np.zeros(n)
        cfg = _qp_cfg(ctx)
        if not cfg.get("qp_tax_aware", False):
            ctx._qp_tax_cost = cost  # noqa: SLF001
            return
        st_rate = float(cfg.get("qp_tax_rate_st", 0.30))
        lt_rate = float(cfg.get("qp_tax_rate_lt", 0.15))
        lt_days = int(cfg.get("qp_lt_threshold_days", 365))
        bridge_w = int(cfg.get("qp_lt_bridge_window_days", 30))
        # G7: tax-lot disposal method. "fifo"/"hifo" → per-lot accounting;
        # "avg" → legacy single-cost-basis path (kill-switch).
        lot_method = str(cfg.get("qp_tax_lot_method", "fifo")).lower()
        offset = max(0.0, float(getattr(ctx, "ytd_realized_gain_dollar", 0.0) or 0.0))
        nav = float(_get_path(ctx, "portfolio_value", 0.0) or 0.0)
        w_current = _get_path(ctx, "_qp_w_current")
        prices = _get_path(ctx, "prices") or {}
        holdings = _get_path(ctx, "holdings") or {}
        today = ctx.today
        for i, t in enumerate(tickers):
            hs = holdings.get(t)
            if hs is None or w_current[i] <= 0:
                continue
            if lot_method == "avg":
                cost[i], offset = _per_asset_tax(
                    hs, prices.get(t, 0.0), w_current[i], nav, today,
                    st_rate, lt_rate, lt_days, bridge_w, offset,
                )
            else:
                cost[i], offset = _per_asset_tax_lots(
                    hs, prices.get(t, 0.0), w_current[i], nav, today,
                    st_rate, lt_rate, lt_days, bridge_w, offset, lot_method,
                )
        ctx._qp_tax_cost = cost  # noqa: SLF001


# ── 4. Wash-sale mask (uses atom + predicate) ───────────────────────────────

class ComputeWashSaleMaskTask(Task):
    """Wash-sale mask: tickers sold within wash_sale_days where §1091 BLOCKS
    the re-entry get Δw_i ≤ 0 in the QP.

    Cost-aware per IRC §1091 (mirrors `WashSaleFilterTask` in candidate path):
      - Sale outside the wash_sale_days window → not blocked
      - Sale was a GAIN (or unknown — fail-conservative) → §1091 N/A → not blocked
      - Sale was a LOSS → §1091 applies → blocked (forces Δw ≤ 0)

    2026-05-09 audit Phase 2.2 fix: pre-fix this task ignored
    `ctx.last_sell_pls` and applied a binary 30-day block. The candidate
    filter (`WashSaleFilterTask`) was correctly cost-aware, but tickers
    that passed the filter (e.g. just sold for a gain) hit the binary QP
    mask and were silently locked from increases. Result: post-gain re-
    entries were architecturally impossible despite §1091 not applying.

    2026-05-18 ANTI-CHURN: `min_reentry_days` additionally blocks recent
    sold tickers regardless of gain/loss, preventing immediate same-name
    QP rebuys unless enough time has passed for new information.

    Reads:  ctx._qp_tickers, ctx.last_sell_dates, ctx.last_sell_pls,
             ctx.config['wash_sale_days']
    Writes: ctx._qp_wash_mask (np.ndarray of bool)

    References:
      - IRC §1091 wash-sale; §1091(d) basis adjustment; §1223(3) holding period
      - kernel/selection.py::is_wash_sale_blocked_with_cost (single-source-of-truth)
    """
    name = "ComputeWashSaleMaskTask"

    def run(self, ctx) -> bool | None:
        wash_days = int((ctx.config or {}).get("wash_sale_days", 0))
        min_reentry = int((ctx.config or {}).get("min_reentry_days", 0))
        tickers = _get_path(ctx, "_qp_tickers") or []
        if (wash_days <= 0 and min_reentry <= 0) or not tickers:
            ctx._qp_wash_mask = np.zeros(len(tickers), dtype=bool)  # noqa: SLF001
            return
        held_tickers = set(ctx.holdings.keys()) if getattr(ctx, "holdings", None) else set()
        mask, n_wash, n_churn, n_sat = _compute_qp_wash_mask(
            tickers=tickers,
            today=ctx.today,
            last_sell_dates=_get_path(ctx, "last_sell_dates") or {},
            last_sell_pls=_get_path(ctx, "last_sell_pls") or {},
            wash_days=wash_days,
            min_reentry=min_reentry,
            held_tickers=held_tickers,
            calibrator_saturated=bool(getattr(ctx, "_calibrator_saturated", False)),
        )
        ctx._qp_wash_mask = mask  # noqa: SLF001
        if n_wash or n_churn or n_sat:
            import logging
            logging.getLogger("kernel.portfolio_qp.tasks").info(
                "ComputeWashSaleMaskTask: blocked %d wash + %d churn + "
                "%d calibrator-saturation-abstain (min_reentry=%dd) of %d tickers",
                n_wash, n_churn, n_sat, min_reentry, len(tickers))


# ── 5. Position caps + scalar constraints ──────────────────────────────────

class ComputeQPConstraintsTask(Task):
    """Per-asset weight caps (regime × confidence-scaled) + scalar limits.

    Reads:  ctx._qp_tickers, ctx.regime, ctx.confidence, ctx.regime_state,
             ctx.config (regime_params, regime, rotation.joint_actions)
    Writes: ctx._qp_w_upper (np.ndarray), ctx._qp_w_upper_hard (np.ndarray),
             ctx._qp_w_lower (float),
             ctx._qp_dw_max (np.ndarray), ctx._qp_cash_reserve (float),
             ctx._qp_drawdown (float), ctx._qp_drawdown_limit (float),
             ctx._qp_turnover_max (float | None)

    ``_qp_w_upper_hard`` is the IMMUTABLE per-asset hard cap snapshot
    (regime × confidence-scaled max_position_pct). Soft-scaling Tasks
    (ApplyExposureScalingTask, ApplyConvictionCapTask) may lower or
    re-raise ``_qp_w_upper`` for hold-flat purposes but MUST NOT raise
    it above ``_qp_w_upper_hard``. The cap-compliance fallback path
    (``_retry_for_per_asset_cap_compliance``) trusts the hard cap to
    drive deterministic over-cap sell-downs. See codex #123 review.
    """
    name = "ComputeQPConstraintsTask"

    def run(self, ctx) -> bool | None:
        from kernel.regime import confidence_to_size_multiplier
        cfg = _qp_cfg(ctx)
        tickers = _get_path(ctx, "_qp_tickers") or []
        n = len(tickers)
        rp = (ctx.config.get("regime_params", {})
                          .get(getattr(ctx, "regime", None), {}))
        max_pct = float(rp.get("max_position_pct",
                                ctx.config.get("max_position_pct", 0.20)))
        scale = confidence_to_size_multiplier(getattr(ctx, "confidence", None))
        hard_cap = np.full(n, max_pct * scale)
        # _qp_w_upper_hard is the immutable hard cap. Soft scalers can never
        # raise _qp_w_upper above this; cap-compliance fallback keys off it.
        ctx._qp_w_upper_hard = hard_cap.copy()  # noqa: SLF001
        ctx._qp_w_upper = hard_cap  # noqa: SLF001
        self._resolve_short_constraints(ctx, scale)
        ctx._qp_dw_max = np.full(n, float(cfg.get("qp_dw_max", 0.50)))  # noqa: SLF001
        ctx._qp_cash_reserve = float(rp.get(  # noqa: SLF001
            "cash_reserve_pct",
            ctx.config.get("cash_reserve_pct", 0.0),
        ))
        rs = getattr(ctx, "regime_state", None)
        ctx._qp_drawdown = (  # noqa: SLF001
            0.0 if rs is None
            else float(rs.get("drawdown", 0.0) or 0.0) if isinstance(rs, dict)
            else float(getattr(rs, "drawdown", 0.0) or 0.0)
        )
        ctx._qp_drawdown_limit = float(cfg.get(  # noqa: SLF001
            "qp_drawdown_limit",
            ctx.config.get("regime", {}).get("drawdown_halt_pct", 0.20),
        ))
        tm = cfg.get("qp_turnover_max", 0.30)
        try:
            ctx._qp_turnover_max = float(tm) if tm else None  # noqa: SLF001
        except (TypeError, ValueError):
            ctx._qp_turnover_max = None  # noqa: SLF001

    def _resolve_short_constraints(self, ctx, scale: float) -> None:
        """Set ``_qp_w_lower`` and ``_qp_gross_max`` per the long/short policy.

        PRIME DIRECTIVE: every knob resolves through regime overlay first,
        global second. See CLAUDE.md + doc/roadmap.md P1.

        Resolution: ``regime_params.<regime>.long_short_enabled``
                    > ``long_short.enabled``
                    > False

        BEAR hybrid (option γ, 2026-05-14 LOCKED):
          * shorts disabled globally → long-only (w_lower=0, gross unlimited)
          * regime=BEAR + hard_bear=False → DEFENSIVE: still no shorts
            (bear_defensive_slots picks up GLD/TLT)
          * regime=BEAR + hard_bear=True  → OFFENSIVE: shorts allowed
            (longs already blocked by max_position_pct=0)
          * otherwise (BULL_*, CHOPPY)    → shorts at -max_short_pct

        SAFETY: ``max_gross_exposure`` hard-capped at 1.0 (no leverage
        authorized — see tests/test_no_leverage_invariant.py).
        """
        from kernel.regime_resolver import resolve_regime_knob
        _LEVERAGE_HARDCAP = 1.0

        shorts_enabled = bool(resolve_regime_knob(
            ctx, "long_short", "enabled", default=False,
        ))
        regime = getattr(ctx, "regime", None)
        hard_bear = bool(getattr(getattr(ctx, "regime_state", None),
                                 "hard_bear", False))
        if not shorts_enabled or (regime == "BEAR" and not hard_bear):
            ctx._qp_w_lower = 0.0  # noqa: SLF001
            ctx._qp_gross_max = None  # noqa: SLF001
            return

        max_short_pct = float(resolve_regime_knob(
            ctx, "long_short", "max_short_pct", default=0.05,
        ))
        ctx._qp_w_lower = -float(max_short_pct) * scale  # noqa: SLF001
        _cfg_gross = float(resolve_regime_knob(
            ctx, "long_short", "max_gross_exposure",
            default=_LEVERAGE_HARDCAP,
        ))
        ctx._qp_gross_max = min(_cfg_gross, _LEVERAGE_HARDCAP)  # noqa: SLF001


class ApplySectorMetadataGuardTask(Task):
    """Prevent QP from adding risk to tickers without sector metadata.

    The sector cap matrix can only constrain tickers that have a sector row.
    A missing sector therefore used to be an implicit exemption from sector
    diversification. Upstream candidate gates now block most missing-sector
    entries, but the QP must defend its own contract because holdings can
    enter through broker state and future paths may pass candidates directly.

    Invariant: when sector caps are enabled, an unmapped ticker may be held
    or reduced, but QP cannot increase its post-trade weight.
    """
    name = "ApplySectorMetadataGuardTask"

    def run(self, ctx) -> bool | None:
        cfg = _qp_cfg(ctx)
        if not bool(cfg.get("qp_sector_cap_enabled", True)):
            return None
        tickers = _get_path(ctx, "_qp_tickers") or []
        sector_map = (ctx.config or {}).get("sector_map", {}) or {}
        w_upper = _get_path(ctx, "_qp_w_upper")
        w_current = _get_path(ctx, "_qp_w_current")
        if not tickers or w_upper is None or w_current is None:
            return None
        missing = [
            i for i, t in enumerate(tickers)
            if not isinstance(sector_map.get(t), str) or not sector_map.get(t)
        ]
        if not missing:
            ctx._qp_missing_sector_tickers = []  # noqa: SLF001
            return None
        w_upper_arr = np.asarray(w_upper, dtype=float).copy()
        w_current_arr = np.asarray(w_current, dtype=float)
        blocked: list[str] = []
        blocked_map = getattr(ctx, "_blocked_by_ticker", None)
        if blocked_map is None:
            blocked_map = {}
            ctx._blocked_by_ticker = blocked_map  # noqa: SLF001
        candidate_tickers = {
            getattr(c, "ticker", None)
            for c in (getattr(ctx, "candidates", None) or [])
        }
        for i in missing:
            if i >= len(w_upper_arr) or i >= len(w_current_arr):
                continue
            ticker = tickers[i]
            w_upper_arr[i] = min(float(w_upper_arr[i]), max(float(w_current_arr[i]), 0.0))
            blocked.append(ticker)
            if ticker in candidate_tickers:
                blocked_map.setdefault(ticker, "missing_sector_map")
        ctx._qp_w_upper = w_upper_arr  # noqa: SLF001
        ctx._qp_missing_sector_tickers = blocked  # noqa: SLF001
        _inc_counter(ctx, "qp_missing_sector_guard", len(blocked))
        log.warning(
            "ApplySectorMetadataGuardTask: capped %d missing-sector ticker(s) "
            "at current weight: %s",
            len(blocked), blocked[:10],
        )


class ApplyExitOnlyTopupGuardTask(Task):
    """Prevent held-but-unadmitted names from becoming QP top-ups.

    **What the counter ``qp_exit_only_topup_guard`` measures**

    Emits a counter equal to the number of held tickers whose QP
    ``w_upper`` was capped at their current weight in this bar. A
    ticker is capped when:

      * it appears in ``ctx._qp_exit_only_tickers`` — meaning it is
        currently held but did NOT pass through the buy/candidate
        admission gate this bar (e.g. ``regime_admission`` blocked the
        regime, the model is ``promotion_status=gated_buys``, etc.)
      * it has a valid (ticker_index, w_upper, w_current) entry — the
        cap only fires when the QP universe row resolves cleanly

    Reading the counter:

      * ``qp_exit_only_topup_guard = N`` means N holdings are in
        "exit-only" mode this bar. The QP solver can still REDUCE or
        CLOSE these names; it simply cannot ADD to them. The 2026-06-02
        audit found this counter on every BULL_CALM run today (always
        equal to the held-ticker count) because the daily artifact is
        ``gated_buys`` and the regime gate blocks every candidate,
        which leaves every holding in exit-only mode.

      * ``qp_exit_only_topup_guard = 0`` either means no holdings exist
        or every holding got admitted via the buy-candidate path this
        bar (rare under the current gated_buys policy).

    Each capped ticker is also stamped via ``_stamp_qp_ticker_block``
    with the reason ``qp_universe_exit_only`` (or a more specific reason
    from ``ctx._qp_exit_only_reasons``) so the decision trace shows the
    block reason, not just the count.

    Holdings remain in the QP universe so the optimizer can reduce or
    close them. That exit permission must not imply fresh alpha
    admission. If the current buy/candidate path did not admit a held
    ticker, cap its upper weight at current weight before solve; the
    emission path has a second fail-closed check.
    """
    name = "ApplyExitOnlyTopupGuardTask"

    def run(self, ctx) -> bool | None:
        exit_only = set(getattr(ctx, "_qp_exit_only_tickers", set()) or set())
        if not exit_only:
            return None
        tickers = _get_path(ctx, "_qp_tickers") or []
        w_upper = _get_path(ctx, "_qp_w_upper")
        w_current = _get_path(ctx, "_qp_w_current")
        if not tickers or w_upper is None or w_current is None:
            return None
        w_upper_arr = np.asarray(w_upper, dtype=float).copy()
        w_current_arr = np.asarray(w_current, dtype=float)
        reason_map = dict(getattr(ctx, "_qp_exit_only_reasons", {}) or {})
        capped: list[str] = []
        for i, ticker in enumerate(tickers):
            if (
                ticker not in exit_only
                or i >= len(w_upper_arr)
                or i >= len(w_current_arr)
            ):
                continue
            w_upper_arr[i] = min(float(w_upper_arr[i]), max(float(w_current_arr[i]), 0.0))
            capped.append(ticker)
            _stamp_qp_ticker_block(
                ctx, ticker, reason_map.get(ticker, "qp_universe_exit_only"),
            )
        ctx._qp_w_upper = w_upper_arr  # noqa: SLF001
        _inc_counter(ctx, "qp_exit_only_topup_guard", len(capped))


_BuildADVVectorTask = None  # lazy class, defined below


def _resolve_regime_override(base_cfg: dict, ctx) -> dict:
    """P1 (2026-05-12): if `base_cfg` has `regime_overrides` AND ctx.spy_regime
    is set, return base_cfg merged with the regime-specific override.

    Resolution order (highest precedence first):
      1. regime_overrides[ctx.spy_regime]  (if regime label exists in overrides)
      2. base_cfg                          (fallback)

    Keys in the override block fully override base_cfg keys (shallow merge);
    this means an override CAN flip `enabled: true → false` to disable a
    feature in toxic regimes.

    Returns base_cfg unmodified if:
      - no `regime_overrides` block
      - ctx.spy_regime is None (SpyRegimeLabelTask disabled)
      - current regime not in overrides

    Fail-open: any error → return base_cfg (no override).
    """
    if not isinstance(base_cfg, dict):
        return {}
    overrides = base_cfg.get("regime_overrides")
    if not isinstance(overrides, dict) or not overrides:
        return base_cfg
    regime = getattr(ctx, "spy_regime", None)
    if regime is None or regime not in overrides:
        return base_cfg
    override = overrides.get(regime)
    if not isinstance(override, dict):
        return base_cfg
    # Shallow merge — override wins on key collision
    merged = dict(base_cfg)
    merged.update(override)
    return merged


def _has_finite_attr(obj: Any, attr: str) -> bool:
    if obj is None:
        return False
    try:
        value = obj.get(attr) if isinstance(obj, dict) else getattr(obj, attr)
        return math.isfinite(float(value))
    except (AttributeError, TypeError, ValueError):
        return False


def _inc_counter(ctx, key: str, amount: int = 1) -> None:
    counters = getattr(ctx, "counters", None)
    if not isinstance(counters, dict):
        counters = {}
        ctx.counters = counters
    counters[key] = int(counters.get(key, 0)) + int(amount)


# ── 4a. Force μ_QP source (Option A: validate NGBoost theory) ───────────────

class ForceMuSourceTask(Task):
    """Override `_qp_mu` from a specific candidate attribute, independent
    of the NGBoost mu/panel_score fallback chain in BuildMuVectorTask.

    Enables Option A validation (CLAUDE.md §2b NGBoost audit, 2026-05-12):
    when `ngboost.enabled=true` (so σ flows through to Kelly + risk), we
    can still force μ_QP = panel_score so the QP's risk/return tradeoff
    stays in its calibrated z-score scale.

    This isolates the contribution of NGBoost σ from the destructive
    μ-scale mismatch that broke E55 (455 trades vs 303, APY +2.99% vs
    +6.77%).

    Config (off by default):
        ranking.qp_mu_source = "panel_score"  (or "rank_score" / "mu")
                            default: "mu"  → no-op, preserves baseline

    Stage A → if NGB-on + force panel_score beats NGB-off baseline,
    NGBoost σ is contributing real value. Then we know option C
    (Grinold-Kahn normalization of NGB μ) is the right architecture.
    """
    name = "ForceMuSourceTask"

    def run(self, ctx) -> bool | None:
        source = str((ctx.config or {}).get("ranking", {}).get("qp_mu_source", "mu")).lower()
        if source == "mu":
            return None  # no-op: keep mu from BuildMuVectorTask
        source_attr = {
            "panel_score": "panel_score",
            "panel": "panel_score",
            "rank_score": "rank_score",
            "rank": "rank_score",
            "rs_score": "rs_score",
            "rs": "rs_score",
            "ranking_composite": "_ranking_composite",
            "composite": "_ranking_composite",
            "blend": "_ranking_composite",
            "blended": "_ranking_composite",
        }.get(source)
        if source_attr is None:
            log.warning("ForceMuSource: unknown source '%s' — no-op", source)
            return None
        tickers   = _get_path(ctx, "_qp_tickers") or []
        src_map   = _get_path(ctx, "_qp_mu_source_map") or {}
        new_mu    = np.full(len(tickers), np.nan)
        n_set     = 0
        missing: list[str] = []
        for i, t in enumerate(tickers):
            obj = src_map.get(t)
            if obj is None:
                missing.append(str(t))
                continue
            val = getattr(obj, source_attr, None)
            try:
                v = float(val) if val is not None else math.nan
            except (TypeError, ValueError):
                v = math.nan
            if math.isfinite(v):
                new_mu[i] = v
                n_set += 1
            else:
                missing.append(str(t))
        ctx._qp_mu = new_mu  # noqa: SLF001
        ctx._qp_forced_mu_source = source  # noqa: SLF001
        ctx._qp_forced_mu_missing_tickers = missing  # noqa: SLF001
        log.info(
            "ForceMuSource: μ_QP ← %s for %d/%d tickers (missing=%d, μ̄=%.4f, σ_μ=%.4f)",
            source, n_set, len(tickers),
            len(missing),
            float(np.nanmean(new_mu)) if n_set else 0.0,
            float(np.nanstd(new_mu, ddof=1)) if n_set > 1 else 0.0,
        )


# ── 4b. Grinold-Kahn α→μ transform (scale-normalizes any score) ─────────────

class ApplyGrinoldKahnTransformTask(Task):
    """Convert raw `_qp_mu` into σ-scale natural units via Grinold-Kahn.

    Reference: Grinold 1989 "The Fundamental Law of Active Management"
    (*J. Portfolio Management*); Grinold-Kahn 1999 *Active Portfolio
    Management* ch.5. Formula:

        μ_i  =  IC  ×  σ_i  ×  z(score_i)

    Where z(score) is the cross-sectional z-score of the raw score, σ_i
    the asset volatility (from `_qp_sigma`), IC the information
    coefficient (use calibrator's pool_ic).

    Fixes §5.13.10 NGBoost μ-scale-mismatch bug class: swapping between
    LTR `panel_score` (~±2 z-units) and NGBoost μ (~1e-3 raw return) used
    to silently change the QP's risk/return tradeoff because λ_risk and
    transaction-cost weights are anchored to one input scale.

    With this transform, μ is always in σ-units (the natural scale of the
    quadratic risk term), so swapping signal sources is safe.

    Config (off by default — opt-in to preserve baseline):
        ranking.alpha_to_mu.enabled = true
        ranking.alpha_to_mu.ic      = 0.094   (default: calibrator pool_ic)
        ranking.alpha_to_mu.regime_overrides = {                # P1 2026-05-12
            "HIGH_CALM":   {"enabled": true, "ic": 0.094},
            "HIGH_SPIKED": {"enabled": false},                  # disable in toxic regime
            "LOW_SPIKED":  {"enabled": true, "ic": 0.15},       # different IC per regime
        }

    When `regime_overrides` is set AND `ctx.spy_regime` is non-None
    (set by SpyRegimeLabelTask, off by default), this task selects the
    override for the current regime. Falls back to global if regime
    not in overrides. Allows regime-conditional deployment per
    doc/research/2026-05-12-findings-and-next.md.
    """
    name = "ApplyGrinoldKahnTransformTask"

    def run(self, ctx) -> bool | None:
        base_cfg = (ctx.config or {}).get("ranking", {}).get("alpha_to_mu", {})
        cfg = _resolve_regime_override(base_cfg, ctx)
        if not cfg.get("enabled", False):
            return None
        ic = float(cfg.get("ic", 0.094))
        if not math.isfinite(ic):
            return None
        mu_arr = _get_path(ctx, "_qp_mu")
        sigma_arr = _get_path(ctx, "_qp_sigma")
        if mu_arr is None or sigma_arr is None:
            return None
        mu_arr = np.asarray(mu_arr, dtype=float)
        sigma_arr = np.asarray(sigma_arr, dtype=float)
        if len(mu_arr) < 2 or len(mu_arr) != len(sigma_arr):
            return None
        finite = np.isfinite(mu_arr)
        if int(finite.sum()) < 2:
            return None
        m  = float(mu_arr[finite].mean())
        sd = float(mu_arr[finite].std(ddof=1))
        if not math.isfinite(sd) or sd <= 0:
            return None
        z = np.zeros_like(mu_arr)
        z[finite] = (mu_arr[finite] - m) / sd
        ctx._qp_mu = ic * sigma_arr * z  # noqa: SLF001
        ctx._qp_mu_transformed = True  # noqa: SLF001
        log.info(
            "ApplyGrinoldKahnTransform: IC=%.3f raw_μ̄=%.4f raw_σ_μ=%.4f → μ̄_QP=%.4f",
            ic, m, sd, float(np.abs(ctx._qp_mu).mean()),
        )


class ValidateQPMuContractTask(Task):
    """Guard QP μ semantics.

    The QP objective expects μ to be an expected-return-like quantity. A
    raw ranking score is acceptable only after the Grinold-Kahn
    ``alpha_to_mu`` transform normalizes it to volatility units. Default
    mode is ``strict``: sim/WF results are invalid if QP falls back to raw
    score semantics.
    """
    name = "ValidateQPMuContractTask"

    def run(self, ctx) -> bool | None:
        cfg = _qp_cfg(ctx)
        mode = str(cfg.get("qp_mu_contract", "strict")).lower()
        if mode in {"off", "disabled", "none"}:
            return None

        alpha_cfg = (ctx.config or {}).get("ranking", {}).get("alpha_to_mu", {})
        alpha_cfg = _resolve_regime_override(alpha_cfg, ctx)
        alpha_applied = bool(getattr(ctx, "_qp_mu_transformed", False))
        forced = str((ctx.config or {}).get("ranking", {}).get("qp_mu_source", "mu")).lower()
        tickers = _get_path(ctx, "_qp_tickers") or []
        src = _get_path(ctx, "_qp_mu_source_map") or {}
        forced_missing = list(getattr(ctx, "_qp_forced_mu_missing_tickers", []) or [])
        missing_mu = [
            t for t in tickers
            if not _has_finite_attr(src.get(t), "mu")
        ]
        missing_sigma = [
            t for t in tickers
            if not _has_finite_attr(src.get(t), "sigma")
        ]
        expected_horizon = _resolve_qp_mu_horizon_days(ctx, cfg)
        horizon_mismatch = []
        if expected_horizon is not None and not alpha_applied:
            horizon_mismatch = [
                t for t in tickers
                if _has_finite_attr(src.get(t), "mu")
                and _source_positive_int(src.get(t), "mu_horizon_days")
                != expected_horizon
            ]
        forced_raw = forced not in {"", "none", "mu"}
        ok = (not forced_missing) and (
            alpha_applied or (not missing_mu and not forced_raw)
        ) and not missing_sigma and not horizon_mismatch
        ctx._qp_mu_contract = {  # noqa: SLF001
            "ok": ok,
            "mode": mode,
            "alpha_to_mu_enabled": bool(alpha_cfg.get("enabled", False)),
            "alpha_to_mu_applied": alpha_applied,
            "expected_mu_horizon_days": expected_horizon,
            "forced_source": forced,
            "forced_source_missing_count": len(forced_missing),
            "forced_source_missing_sample": forced_missing[:10],
            "missing_mu_count": len(missing_mu),
            "missing_mu_sample": missing_mu[:10],
            "missing_sigma_count": len(missing_sigma),
            "missing_sigma_sample": missing_sigma[:10],
            "mu_horizon_mismatch_count": len(horizon_mismatch),
            "mu_horizon_mismatch_sample": horizon_mismatch[:10],
        }
        if ok:
            return None

        affected = sorted({
            str(t)
            for t in (
                missing_mu + missing_sigma + forced_missing + horizon_mismatch
            )
        })
        affected_count = len(affected) or int(forced_raw)
        _inc_counter(ctx, "qp_mu_contract_fallback", affected_count)
        msg = (
            "ValidateQPMuContract: QP μ contract failed "
            f"(missing_mu={len(missing_mu)}, forced_source={forced}, "
            f"forced_missing={len(forced_missing)}, "
            f"missing_sigma={len(missing_sigma)}, "
            f"mu_horizon_mismatch={len(horizon_mismatch)}, "
            f"alpha_to_mu_applied={alpha_applied})"
        )
        if mode in {"strict", "hard", "error", "enforce"}:
            _inc_counter(ctx, "qp_mu_contract_block", 1)
            if affected:
                for ticker in affected:
                    reason = (
                        "qp_mu_contract_block"
                        if ticker in {str(t) for t in missing_mu + forced_missing}
                        else "qp_mu_horizon_contract_block"
                        if ticker in {str(t) for t in horizon_mismatch}
                        else "qp_sigma_contract_block"
                    )
                    _stamp_qp_ticker_block(ctx, str(ticker), reason)
            else:
                for ticker in tickers:
                    _stamp_qp_ticker_block(ctx, str(ticker), "qp_mu_contract_block")
            log.error("%s — stopping QP job", msg)
            return False
        log.warning("%s — continuing in warn mode", msg)
        return None


# ── 5a. Exposure scaling (vol-target + DD-Kelly) ────────────────────────────

class ApplyExposureScalingTask(Task):
    """Scale per-asset `_qp_w_upper` by basket-level exposure modifiers.

    Composes Moskowitz-Ooi-Pedersen 2012 volatility-targeting and
    Grossman-Zhou 1993 drawdown-conditioned Kelly scaling at the QP
    upper-bound level, INDEPENDENT of the Kelly sizing path (which is
    dead when NGB is off — see doc/AUDIT_2026-05-12_dead_paths.md).

    Invariant pinned:
        _qp_w_upper ≡ max_pos × confidence × vol_target_scale × dd_scale

    Config (read from BOTH legacy and new locations for backward compat):
        ranking.kelly_sizing.vol_target.{enabled,target_vol,window_days,...}
        ranking.kelly_sizing.drawdown_scaling.{enabled,dd_max,exponent}
        exposure_scaling.vol_target.*        (new top-level path)
        exposure_scaling.drawdown_scaling.*  (new top-level path)

    Both helpers fail-open (return 1.0 on malformed input).
    """
    name = "ApplyExposureScalingTask"

    def run(self, ctx) -> bool | None:
        w_upper = _get_path(ctx, "_qp_w_upper")
        if w_upper is None or len(w_upper) == 0:
            ctx._vol_target_scale = 1.0  # noqa: SLF001
            ctx._dd_kelly_scale = 1.0    # noqa: SLF001
            return None
        cfg = ctx.config or {}
        legacy = cfg.get("ranking", {}).get("kelly_sizing", {})
        topl   = cfg.get("exposure_scaling", {})
        vt_cfg = topl.get("vol_target")        or legacy.get("vol_target")        or {}
        dd_cfg = topl.get("drawdown_scaling")  or legacy.get("drawdown_scaling")  or {}
        # P1 (2026-05-12): regime-conditional override per ctx.spy_regime
        vt_cfg = _resolve_regime_override(vt_cfg, ctx)
        dd_cfg = _resolve_regime_override(dd_cfg, ctx)
        vt_scale = _compute_vt_scale(ctx, vt_cfg) if vt_cfg.get("enabled", False) else 1.0
        dd_scale = _compute_dd_scale(ctx, dd_cfg) if dd_cfg.get("enabled", False) else 1.0
        ctx._vol_target_scale = float(vt_scale)  # noqa: SLF001
        ctx._dd_kelly_scale   = float(dd_scale)  # noqa: SLF001
        combined = vt_scale * dd_scale
        if combined != 1.0:
            ctx._qp_w_upper = np.asarray(w_upper) * float(combined)  # noqa: SLF001
            _clamp_w_upper_at_w_current(ctx)
            log.info(
                "ApplyExposureScalingTask: w_upper scaled by vt=%.3f × dd=%.3f = %.3f (hold-flat-clamped)",
                vt_scale, dd_scale, combined,
            )


def _compute_vt_scale(ctx, vt_cfg: dict) -> float:
    from kernel.vol_target import compute_vol_target_scale  # noqa: PLC0415
    return compute_vol_target_scale(
        getattr(ctx, "spy_returns", None) or [],
        target_vol  = float(vt_cfg.get("target_vol",  0.15)),
        window_days = int  (vt_cfg.get("window_days", 60)),
        floor       = float(vt_cfg.get("floor",       0.30)),
        ceiling     = float(vt_cfg.get("ceiling",     1.50)),
    )


def _compute_dd_scale(ctx, dd_cfg: dict) -> float:
    from kernel.kelly import compute_kelly_dd_scale  # noqa: PLC0415
    from kernel.pipeline.task_drawdown_rebalance import compute_portfolio_drawdown  # noqa: PLC0415
    hwm = float(getattr(ctx, "hwm", 0.0) or 0.0)
    pv  = float(getattr(ctx, "portfolio_value", 0.0) or 0.0)
    dd  = compute_portfolio_drawdown(hwm, pv)
    return compute_kelly_dd_scale(
        dd,
        dd_max   = float(dd_cfg.get("dd_max",   0.30)),
        exponent = float(dd_cfg.get("exponent", 1.0)),
    )


# ── 5b. Conviction-scaled per-name cap ──────────────────────────────────────

class ApplyConvictionCapTask(Task):
    """Shrink per-ticker `_qp_w_upper` by conviction multiplier.

    Parity with greedy paths (`task_selection`, `task_rotation`,
    `task_joint_actions`) which multiply position size by
    `conviction_multiplier(panel_score, sizing_cfg)`. Without this, the
    QP path treats every name with the same regime+confidence cap
    regardless of model conviction — high- and low-rank candidates can
    both saturate at `max_position_pct`.

    Wiring: runs AFTER `ComputeQPConstraintsTask` (which writes
    `_qp_w_upper` as a uniform vector) and BEFORE the sector/correlation
    constraint Tasks (which anchor their caps on `_qp_w_upper.max()`).

    Reads:  ctx._qp_tickers, ctx._qp_mu_source_map, ctx._qp_w_upper,
             ctx.config["rotation"]["joint_actions"]["qp_conviction_cap_enabled"],
             ctx.config["ranking"]["panel_scoring"]["sizing"]
    Writes: ctx._qp_w_upper (in-place per-ticker scaling)
             ctx._qp_conviction_caps (list[float] for diagnostics)

    Default: disabled. Opt-in via `qp_conviction_cap_enabled=true`. No
    promotion until sim shows positive APY delta (CLAUDE.md §2a).
    """
    name = "ApplyConvictionCapTask"

    def run(self, ctx) -> bool | None:
        cfg = _qp_cfg(ctx)
        if not bool(cfg.get("qp_conviction_cap_enabled", False)):
            return None
        sizing_cfg = ((ctx.config or {}).get("ranking", {})
                       .get("panel_scoring", {})
                       .get("sizing", {}))
        if not sizing_cfg or not sizing_cfg.get("enabled", False):
            return None

        # Local import to keep qp module decoupled from kernel.sizing.
        from kernel.sizing import conviction_multiplier

        tickers = _get_path(ctx, "_qp_tickers") or []
        w_upper = _get_path(ctx, "_qp_w_upper")
        src     = _get_path(ctx, "_qp_mu_source_map") or {}
        if w_upper is None or len(tickers) == 0 or len(w_upper) != len(tickers):
            return None

        caps: list[float] = []
        for i, t in enumerate(tickers):
            obj = src.get(t)
            ps = getattr(obj, "panel_score", None) if obj is not None else None
            mult = conviction_multiplier(ps, sizing_cfg)
            # Defensive: conviction_multiplier returns 1.0 on bad input
            # (None / NaN / inf / malformed cfg). Clip to [0, 1] in case
            # of future-config changes — w_upper must remain ≤ original.
            try:
                m = float(mult)
            except (TypeError, ValueError):
                m = 1.0
            if not math.isfinite(m):
                m = 1.0
            m = max(0.0, min(1.0, m))
            w_upper[i] = float(w_upper[i]) * m
            caps.append(m)

        # 2026-06-02 fix: never shrink below w_current — see helper docstring.
        _clamp_w_upper_at_w_current(ctx)

        ctx._qp_conviction_caps = caps  # noqa: SLF001
        return None


# ── 5a. Sector cap → per-sector indicator matrix + cap vector ───────────────

class BuildSectorConstraintMatrixTask(Task):
    """Construct hard linear sector-cap constraint inputs for the QP.

    Per CLAUDE.md §5.13.5 (single source of truth), sector_map and
    `max_positions_per_sector` come from THE SAME config keys the buy-side
    `passes_sector_guard` uses (`config['sector_map']`,
    `config['max_positions_per_sector']`). The QP enforcing the same caps
    closes the audit gap: once a holding is in the book, the buy-side
    filter never sees it again, but a stress reallocation could still pile
    weight on top of it. The solver constraint catches that.

    Per-sector weight cap = max_per_sector × max_position_pct × confidence.
    Defensive tickers (`config['defensive_tickers']`) are included in the
    indicator (they get the same cap) — divergence from buy-side which
    *bypasses* the count-of-positions cap is intentional: the QP
    constraint is on *weight*, not count, and an unbounded defensive
    sleeve would defeat the diversification goal.

    Reads:  ctx._qp_tickers, ctx.config['sector_map'],
             ctx.config['max_positions_per_sector'],
             ctx._qp_w_upper (anchors per-name cap × sector_count),
             ctx.config['rotation']['joint_actions']['qp_sector_cap_enabled']
    Writes: ctx._qp_sector_indicator (m × n np.ndarray, 0/1 ints) — None
             when constraint disabled / no sectors mapped,
             ctx._qp_sector_cap_vec (m-length np.ndarray of weight caps),
             ctx._qp_sector_names (list[str]) — for diagnostics.
    """
    name = "BuildSectorConstraintMatrixTask"

    def run(self, ctx) -> bool | None:
        cfg = _qp_cfg(ctx)
        if not bool(cfg.get("qp_sector_cap_enabled", True)):
            ctx._qp_sector_indicator = None  # noqa: SLF001
            ctx._qp_sector_cap_vec   = None  # noqa: SLF001
            ctx._qp_sector_names     = []    # noqa: SLF001
            return
        tickers = _get_path(ctx, "_qp_tickers") or []
        n = len(tickers)
        sector_map = (ctx.config or {}).get("sector_map", {}) or {}
        max_per_sector = int((ctx.config or {}).get("max_positions_per_sector", 0))
        if n == 0 or not sector_map or max_per_sector <= 0:
            ctx._qp_sector_indicator = None  # noqa: SLF001
            ctx._qp_sector_cap_vec   = None  # noqa: SLF001
            ctx._qp_sector_names     = []    # noqa: SLF001
            return
        sector_to_idx = self._build_sector_index(tickers, sector_map)
        if not sector_to_idx:
            ctx._qp_sector_indicator = None  # noqa: SLF001
            ctx._qp_sector_cap_vec   = None  # noqa: SLF001
            ctx._qp_sector_names     = []    # noqa: SLF001
            return
        # Per-name cap (post-confidence/scaling/conviction) is in
        # ctx._qp_w_upper. Anchor only on names that actually belong to a
        # mapped sector row; an unmapped broker holding capped at current
        # weight must not inflate every mapped sector's group limit.
        w_upper = _get_path(ctx, "_qp_w_upper")
        mapped_idx = [j for idxs in sector_to_idx.values() for j in idxs]
        per_name_cap = _max_upper_for_indices(
            w_upper, mapped_idx,
            fallback=float((ctx.config or {}).get("max_position_pct", 0.20)),
        )
        sector_names = sorted(sector_to_idx.keys())
        m = len(sector_names)
        S = np.zeros((m, n), dtype=float)
        for row, name in enumerate(sector_names):
            for j in sector_to_idx[name]:
                S[row, j] = 1.0
        legacy_cap = max_per_sector * per_name_cap
        cap, source = _resolve_sector_weight_cap(ctx, legacy_cap)
        cap_vec = np.full(m, cap, dtype=float)
        ctx._qp_sector_indicator = S            # noqa: SLF001
        ctx._qp_sector_cap_vec   = cap_vec      # noqa: SLF001
        ctx._qp_sector_names     = sector_names # noqa: SLF001
        ctx._qp_sector_cap_source = source       # noqa: SLF001

    @staticmethod
    def _build_sector_index(tickers, sector_map) -> dict[str, list[int]]:
        """Return {sector_name: [ticker_indices]} for sectors with ≥1 member."""
        out: dict[str, list[int]] = {}
        for j, t in enumerate(tickers):
            sec = sector_map.get(t)
            if not sec or not isinstance(sec, str):
                continue
            out.setdefault(sec, []).append(j)
        return out


def _resolve_sector_weight_cap(ctx, legacy_cap: float) -> tuple[float, str]:
    """Return QP sector cap with regime override support.

    Resolution:
      regime_params.<regime>.max_sector_weight_pct
        > config.max_sector_weight_pct
        > max_positions_per_sector * per_name_cap

    The final cap is min(configured, legacy_count_cap), so count-based
    diversification remains a hard ceiling while regime-level exposure
    tightening can reduce concentration in dominant regimes.
    """
    from kernel.regime_resolver import resolve_regime_knob  # noqa: PLC0415
    cap = legacy_cap
    source = "count_x_per_name"
    configured = resolve_regime_knob(
        ctx, None, "max_sector_weight_pct", default=None,
    )
    try:
        cfg_cap = float(configured) if configured is not None else float("nan")
    except (TypeError, ValueError):
        cfg_cap = float("nan")
    if math.isfinite(cfg_cap) and cfg_cap > 0:
        cap = min(float(legacy_cap), cfg_cap)
        source = "regime_or_global_max_sector_weight_pct"
    return float(max(0.0, cap)), source


# ── 5a-bis. High-correlation pair group cap ────────────────────────────────

class BuildCorrelationGroupConstraintTask(Task):
    """Build (i, j, group_cap) triples for high-correlation pairs.

    For every pair (i, j) where |corr[i, j]| ≥ correlation_guard_threshold,
    add a linear constraint `wp[i] + wp[j] ≤ 2 × per_name_cap` (group
    bound). This is the convex linear approximation of the non-convex
    `wp[i] · wp[j] ≤ pair_cap`. Tradeoff documented in qp_solver.py.

    §5.13.5 single-source-of-truth: the `correlation_guard_threshold` is
    read from `config['regime']['correlation_guard_threshold']` — same key
    `passes_correlation_guard` uses in selection.py. Behaviour-equivalent
    when the candidate filter and QP both fire (no double-blocking; the
    QP just ensures any *internal* re-shuffling can't recreate the pair
    concentration).

    Reads:  ctx._qp_tickers, ctx.corr_matrix (pre-loaded by SimAdapter),
             ctx.config['regime']['correlation_guard_threshold'],
             ctx._qp_w_upper, ctx.config['rotation']['joint_actions']
                 ['qp_correlation_cap_enabled']
    Writes: ctx._qp_corr_group_pairs (list[tuple[int, int, float]] | None)
    """
    name = "BuildCorrelationGroupConstraintTask"

    def run(self, ctx) -> bool | None:
        cfg = _qp_cfg(ctx)
        if not bool(cfg.get("qp_correlation_cap_enabled", True)):
            ctx._qp_corr_group_pairs = None  # noqa: SLF001
            return
        tickers = _get_path(ctx, "_qp_tickers") or []
        n = len(tickers)
        if n < 2:
            ctx._qp_corr_group_pairs = None  # noqa: SLF001
            return
        corr_matrix = getattr(ctx, "corr_matrix", None)
        if not corr_matrix:
            self._cap_missing_corr_tickers(
                ctx, tickers, set(tickers), reason="missing_correlation_matrix",
            )
            ctx._qp_corr_group_pairs = None  # noqa: SLF001
            return
        thr = float(((ctx.config or {}).get("regime", {}) or {}).get(
            "correlation_guard_threshold", 0.70,
        ))
        if not math.isfinite(thr) or thr <= 0.0 or thr >= 1.0:
            ctx._qp_corr_group_pairs = None  # noqa: SLF001
            return
        w_upper = _get_path(ctx, "_qp_w_upper")
        fallback_cap = float((ctx.config or {}).get("max_position_pct", 0.20))
        pairs, missing_tickers = self._collect_pairs(
            tickers, corr_matrix, thr, w_upper, fallback_cap,
        )
        if missing_tickers:
            self._cap_missing_corr_tickers(
                ctx, tickers, missing_tickers, reason="missing_correlation_pair",
            )
        ctx._qp_corr_group_pairs = pairs if pairs else None  # noqa: SLF001

    @staticmethod
    def _collect_pairs(tickers, corr_matrix, thr, w_upper, fallback_cap):
        """Walk the upper-triangle of the corr matrix; return (i, j, cap)."""
        pairs: list[tuple[int, int, float]] = []
        missing_tickers: set[str] = set()
        for i in range(len(tickers)):
            ti = tickers[i]
            row = corr_matrix.get(ti)
            for j in range(i + 1, len(tickers)):
                tj = tickers[j]
                rho = None
                if isinstance(row, dict):
                    rho = row.get(tj)
                if rho is None:
                    other = corr_matrix.get(tj)
                    if isinstance(other, dict):
                        rho = other.get(ti)
                if rho is None:
                    missing_tickers.update({ti, tj})
                    continue
                try:
                    rho_f = float(rho)
                except (TypeError, ValueError):
                    missing_tickers.update({ti, tj})
                    continue
                if not math.isfinite(rho_f):
                    # Fail-conservative: NaN correlation → treat as high.
                    rho_f = 1.0
                if abs(rho_f) >= thr:
                    group_cap = _pair_upper_cap(w_upper, i, j, fallback_cap)
                    pairs.append((i, j, group_cap))
        return pairs, missing_tickers

    @staticmethod
    def _cap_missing_corr_tickers(ctx, tickers, missing_tickers: set[str], reason: str):
        w_upper = _get_path(ctx, "_qp_w_upper")
        w_current = _get_path(ctx, "_qp_w_current")
        if w_upper is None or w_current is None:
            return
        w_upper_arr = np.asarray(w_upper, dtype=float).copy()
        w_current_arr = np.asarray(w_current, dtype=float)
        blocked: list[str] = []
        blocked_map = getattr(ctx, "_blocked_by_ticker", None)
        if blocked_map is None:
            blocked_map = {}
            ctx._blocked_by_ticker = blocked_map  # noqa: SLF001
        candidate_tickers = {
            getattr(c, "ticker", None)
            for c in (getattr(ctx, "candidates", None) or [])
        }
        for i, ticker in enumerate(tickers):
            if ticker not in missing_tickers:
                continue
            if i >= len(w_upper_arr) or i >= len(w_current_arr):
                continue
            w_upper_arr[i] = min(
                float(w_upper_arr[i]), max(float(w_current_arr[i]), 0.0),
            )
            blocked.append(ticker)
            if ticker in candidate_tickers:
                blocked_map.setdefault(ticker, reason)
        if not blocked:
            return
        ctx._qp_w_upper = w_upper_arr  # noqa: SLF001
        ctx._qp_missing_correlation_tickers = blocked  # noqa: SLF001
        _inc_counter(ctx, "qp_missing_correlation_guard", len(blocked))
        log.warning(
            "BuildCorrelationGroupConstraintTask: capped %d ticker(s) "
            "at current weight due to incomplete correlation metadata: %s",
            len(blocked), blocked[:10],
        )


def _max_upper_for_indices(w_upper, indices: list[int], *, fallback: float) -> float:
    """Max finite positive upper bound over a constrained group."""
    if w_upper is None:
        return float(fallback)
    arr = np.asarray(w_upper, dtype=float)
    vals = [
        float(arr[i]) for i in indices
        if 0 <= i < len(arr) and math.isfinite(float(arr[i])) and float(arr[i]) >= 0
    ]
    return max(vals) if vals else float(fallback)


def _pair_upper_cap(w_upper, i: int, j: int, fallback: float) -> float:
    """Linear high-correlation cap from the two assets' own upper bounds."""
    if w_upper is None:
        return 2.0 * float(fallback)
    arr = np.asarray(w_upper, dtype=float)
    vals: list[float] = []
    for idx in (i, j):
        if 0 <= idx < len(arr):
            val = float(arr[idx])
            vals.append(val if math.isfinite(val) and val >= 0 else float(fallback))
        else:
            vals.append(float(fallback))
    return float(vals[0] + vals[1])


# ── 5b. Per-asset 20-day ADV (Almgren-Chriss participation) ─────────────────

class BuildADVVectorTask(Task):
    """Per-asset average daily dollar volume (ADV) over `qp_adv_window` days.

    ADV_i = mean(close_t × volume_t) over the last `window` rows of the
    asset's OHLCV frame. Used by Stage G3 sqrt-impact: missing or
    too-short data → NaN entry → solver disables impact for that asset.

    Reads:  ctx._qp_tickers, ctx.ohlcv,
             ctx.config['rotation']['joint_actions']['qp_adv_window']
    Writes: ctx._qp_v_daily_dollar (np.ndarray, $; NaN for unavailable)
    """
    name = "BuildADVVectorTask"

    def run(self, ctx) -> bool | None:
        cfg = _qp_cfg(ctx)
        window = max(1, int(cfg.get("qp_adv_window", 20)))
        tickers = _get_path(ctx, "_qp_tickers") or []
        ohlcv = _get_path(ctx, "ohlcv") or {}
        v = np.full(len(tickers), np.nan)
        for i, t in enumerate(tickers):
            df = ohlcv.get(t)
            if df is None or len(df) == 0:
                continue
            try:
                tail = df.tail(window)
                cv = (tail["close"] * tail["volume"]).mean()
                v[i] = float(cv) if math.isfinite(float(cv)) else math.nan
            except (KeyError, AttributeError, ValueError, TypeError):
                continue
        ctx._qp_v_daily_dollar = v  # noqa: SLF001


# ── 5b. Snapshot the assembled constraint state ─────────────────────────────

class BuildConstraintSnapshotTask(Task):
    """Stamp ``ctx._qp_constraint_snapshot`` from the upstream Tasks' output.

    Step 1c of the §8 plan (PR #125). Strictly additive — runs AFTER
    the existing 4-Task constraint-composition pipeline
    (``ComputeQPConstraintsTask → ApplyExposureScalingTask →
    ApplyConvictionCapTask → sector/correlation``) and freezes the
    assembled constraint state into an immutable
    :class:`ConstraintSnapshot` that downstream allocators consume via
    the contract instead of via free-form ``ctx._qp_*`` reads.

    On invariant violation (snapshot constructor raises
    ``ValueError`` — e.g. soft > hard cap, shape mismatch, non-finite
    entries) this Task logs the failure and returns ``False`` so the
    Job short-circuits before the solver runs. Better to fail loud
    here than to feed contradictory constraints to ``cvxpy``.

    Reads:  every ctx._qp_* field built by Tasks 1-5.
    Writes: ctx._qp_constraint_snapshot (ConstraintSnapshot | None).
    """

    name = "BuildConstraintSnapshotTask"
    FAILURE_STATUS = "infeasible:qp_constraint_snapshot_invalid"
    FAILURE_REASON = "qp_constraint_snapshot_invalid"

    def run(self, ctx) -> bool | None:  # noqa: D401
        from kernel.portfolio_qp.constraint_snapshot import build_snapshot_from_ctx

        # The Job has already short-circuited if there are no tickers
        # to optimize over, but defend just in case.
        tickers = _get_path(ctx, "_qp_tickers") or ()
        if not tickers:
            ctx._qp_constraint_snapshot = None  # noqa: SLF001
            return None
        try:
            snap = build_snapshot_from_ctx(ctx)
        except ValueError as exc:
            # The snapshot's __post_init__ failed — one of the upstream
            # Tasks produced a contradictory or malformed constraint
            # state. Stamp this as a first-class QP failure path so
            # ``live.runner._why_no_trade`` and downstream telemetry can
            # see and attribute the failure (codex #129 review).
            log.error(
                "BuildConstraintSnapshotTask: constraint state invalid "
                "— %s",
                exc,
            )
            ctx._qp_constraint_snapshot = None  # noqa: SLF001
            ctx._qp_constraint_snapshot_error = str(exc)  # noqa: SLF001
            ctx._qp_status = self.FAILURE_STATUS  # noqa: SLF001
            ctx._qp_failure_reason = self.FAILURE_REASON  # noqa: SLF001
            ctx._qp_n_buys = 0  # noqa: SLF001
            ctx._qp_n_sells = 0  # noqa: SLF001
            _stamp_all_qp_blocks(ctx, self.FAILURE_REASON)
            _stamp_qp_failure_counter(ctx, ctx._qp_status)  # noqa: SLF001
            return False
        ctx._qp_constraint_snapshot = snap  # noqa: SLF001


# ── 6. Solve the QP ─────────────────────────────────────────────────────────

class SolveMarkowitzQPTask(Task):
    """Call solve_portfolio_qp with the prepared inputs.

    Reads:  every ctx._qp_* field built by upstream Tasks
    Writes: ctx._qp_solution (QPSolution dataclass)
    """
    name = "SolveMarkowitzQPTask"

    def run(self, ctx) -> bool | None:
        cfg = _qp_cfg(ctx)
        backend, _solve = self._pick_backend(cfg)
        kwargs = self._build_solver_kwargs(ctx, cfg)
        if backend == "cvxportfolio":
            unsupported = self._unsupported_cvxportfolio_constraints(kwargs)
            if unsupported:
                sol = self._unsupported_cvxportfolio_solution(kwargs, unsupported)
                ctx._qp_solution = sol  # noqa: SLF001
                ctx._qp_status = sol.status  # noqa: SLF001
                ctx._qp_diagnostics = dict(sol.diagnostics)  # noqa: SLF001
                ctx._qp_failure_reason = f"qp_global:{sol.status}"  # noqa: SLF001
                _stamp_all_qp_blocks(ctx, ctx._qp_failure_reason)
                _stamp_qp_failure_counter(ctx, ctx._qp_status)  # noqa: SLF001 (codex PR #48 #1)
                ctx._qp_n_buys = 0  # noqa: SLF001
                ctx._qp_n_sells = 0  # noqa: SLF001
                log.error(
                    "cvxportfolio backend cannot enforce hard QP constraints "
                    "%s — strict policy blocks QP orders for this bar",
                    unsupported,
                )
                return False
            self._strip_kwargs_for_cvxportfolio(kwargs, ctx)
        sol = _solve(**kwargs)
        sol = _retry_with_relaxed_c2_caps(
            sol,
            kwargs,
            _solve,
            policy=str(cfg.get("qp_c2_infeasible_policy", "strict")),
        )
        # Audit #2 / issue #70: when QP is infeasible AND at least one
        # holding is over its per-asset cap, fall back to a deterministic
        # force-sell-to-cap for the over-cap holdings. Opt-in via the
        # ``allow_cap_compliance_sells_on_infeasible`` config knob (default
        # False, preserves strict behavior). The fallback emits SELLS only
        # — buys remain blocked by upstream gates (regime_admission etc.).
        if bool(cfg.get("allow_cap_compliance_sells_on_infeasible", False)):
            sol = _retry_for_per_asset_cap_compliance(sol, kwargs, _solve)
        ctx._qp_solution = sol  # noqa: SLF001
        ctx._qp_status = str(getattr(sol, "status", "missing_solution"))  # noqa: SLF001
        ctx._qp_diagnostics = dict(getattr(sol, "diagnostics", {}) or {})  # noqa: SLF001
        # codex #75/#10: treat any QP_EMITTABLE_STATUSES outcome as a
        # successful solve, NOT a failure. Pre-fix, ``cap_compliance_fallback``
        # was both stamped here as ``qp_global:cap_compliance_fallback`` +
        # all-symbol-blocked + failure-counter-incremented AND simultaneously
        # emitted by ``EmitOrdersFromQPSolutionTask`` — contradictory state
        # that broke observability/no-trade-attribution for the exact path
        # the fallback is supposed to rescue.
        if sol.status not in QP_EMITTABLE_STATUSES:
            reason = "qp_no_signal" if sol.status == "optimal_no_signal" else f"qp_global:{sol.status}"
            ctx._qp_failure_reason = reason  # noqa: SLF001
            _stamp_qp_failure_counter(ctx, ctx._qp_status)  # noqa: SLF001 (codex PR #48 #1)
            _stamp_all_qp_blocks(ctx, reason)
        ctx._qp_n_buys = 0  # noqa: SLF001
        ctx._qp_n_sells = 0  # noqa: SLF001

    @staticmethod
    def _pick_backend(cfg: dict):
        """Choose cvxpy (default) vs cvxportfolio (opt-in, Boyd ref).

        2026-05-06: both backends accept the same kwargs; cvxportfolio
        uses Boyd's reference policy classes verbatim.
        """
        backend = str(cfg.get("qp_solver_backend", "cvxpy")).lower()
        if backend == "cvxportfolio":
            from kernel.portfolio_qp.cvxportfolio_backend import (  # noqa: PLC0415
                solve_portfolio_qp_cvxportfolio as _solve,
            )
        else:
            from kernel.portfolio_qp.qp_solver import (  # noqa: PLC0415
                solve_portfolio_qp as _solve,
            )
        return backend, _solve

    @staticmethod
    def _build_solver_kwargs(ctx, cfg: dict) -> dict:
        """Marshal ctx + cfg into solve_portfolio_qp's kwargs.

        Single source of truth for every QP knob; future additions go here.
        See `qp_solver.solve_portfolio_qp` docstring for parameter semantics.
        """
        return dict(
            w_current=_get_path(ctx, "_qp_w_current"),
            mu=_get_path(ctx, "_qp_mu"),
            sigma=_get_path(ctx, "_qp_sigma"),
            Sigma=_get_path(ctx, "_qp_Sigma_full"),
            risk_aversion=float(cfg.get("qp_risk_aversion", 3.0)),
            cost_kappa=_effective_qp_cost_kappa(cfg),
            cash_reserve=_get_path(ctx, "_qp_cash_reserve"),
            w_upper=_get_path(ctx, "_qp_w_upper"),
            w_lower=_get_path(ctx, "_qp_w_lower"),
            dw_max=_get_path(ctx, "_qp_dw_max"),
            wash_sale_mask=_get_path(ctx, "_qp_wash_mask"),
            signal_decay=float(cfg.get("qp_signal_decay", 0.0)),
            drawdown=_get_path(ctx, "_qp_drawdown"),
            drawdown_limit=_get_path(ctx, "_qp_drawdown_limit"),
            robust_mu_kappa=float(cfg.get("qp_robust_mu_kappa", 0.0)),
            tax_cost_per_sell=_get_path(ctx, "_qp_tax_cost"),
            turnover_max=_get_path(ctx, "_qp_turnover_max"),
            cvar_lambda=float(cfg.get("qp_cvar_lambda", 0.0)),
            cvar_alpha=float(cfg.get("qp_cvar_alpha", 0.05)),
            impact_coef=float(cfg.get("qp_impact_coef", 0.0)),
            v_daily_dollar=_get_path(ctx, "_qp_v_daily_dollar"),
            nav_dollar=float(_get_path(ctx, "portfolio_value", 0.0) or 0.0),
            fixed_cost_per_trade=float(cfg.get("qp_fixed_cost_per_trade", 0.0)),
            fixed_cost_beta=float(cfg.get("qp_fixed_cost_beta", 200.0)),
            budget_mode=str(cfg.get("qp_budget_mode", "inequality")),
            min_invested_pct=_effective_min_invested_pct(ctx, cfg),
            cash_drag_lambda=float(cfg.get("qp_cash_drag_lambda", 0.05)),
            sector_indicator=_get_path(ctx, "_qp_sector_indicator"),
            sector_cap_vec=_get_path(ctx, "_qp_sector_cap_vec"),
            corr_group_pairs=_get_path(ctx, "_qp_corr_group_pairs"),
            gross_max=_get_path(ctx, "_qp_gross_max"),
            allow_optimal_inaccurate=bool(cfg.get("qp_allow_optimal_inaccurate", False)),
        )

    @staticmethod
    def _unsupported_cvxportfolio_constraints(kwargs: dict) -> list[str]:
        unsupported: list[str] = []
        if kwargs.get("sector_indicator") is not None:
            unsupported.append("sector_cap")
        if kwargs.get("corr_group_pairs"):
            unsupported.append("correlation_cap")
        if kwargs.get("gross_max") is not None:
            unsupported.append("gross_max")
        tax_cost = kwargs.get("tax_cost_per_sell")
        if tax_cost is not None:
            try:
                tax_arr = np.asarray(tax_cost, dtype=float)
                if np.isfinite(tax_arr).any() and np.nanmax(np.abs(tax_arr)) > 1e-12:
                    unsupported.append("tax_cost_per_sell")
            except (TypeError, ValueError):
                unsupported.append("tax_cost_per_sell")
        min_invested = float(kwargs.get("min_invested_pct") or 0.0)
        cash_drag = float(kwargs.get("cash_drag_lambda") or 0.0)
        if min_invested > 0.0 and cash_drag > 0.0:
            unsupported.append("cash_drag_min_invested")
        fixed_cost = float(kwargs.get("fixed_cost_per_trade") or 0.0)
        if fixed_cost > 0.0:
            unsupported.append("fixed_cost_per_trade")
        return unsupported

    @staticmethod
    def _unsupported_cvxportfolio_solution(kwargs: dict, unsupported: list[str]):
        from kernel.portfolio_qp.qp_solver import QPSolution  # noqa: PLC0415
        w_current = np.asarray(kwargs.get("w_current"), dtype=float)
        if w_current.ndim != 1:
            w_current = np.zeros(0)
        return QPSolution(
            delta_w=np.zeros_like(w_current),
            target_w=w_current.copy(),
            objective=0.0,
            n_iter=-1,
            status="infeasible:cvxportfolio_unsupported_constraints",
            diagnostics={
                "backend": "cvxportfolio",
                "unsupported_hard_constraints": list(unsupported),
            },
        )

    @staticmethod
    def _strip_kwargs_for_cvxportfolio(kwargs: dict, ctx) -> None:
        """Strip kwargs unsupported by cvxportfolio after hard-constraint precheck.

        Sector/correlation/gross constraints must already be absent here.
        The strict precheck above blocks if they are present, so this helper
        only removes explicit ``None`` placeholders and optional metadata.
        """
        kwargs["tickers"] = _get_path(ctx, "_qp_tickers")
        kwargs.pop("sector_indicator", None)
        kwargs.pop("sector_cap_vec", None)
        kwargs.pop("corr_group_pairs", None)
        kwargs.pop("gross_max", None)
        kwargs.pop("allow_optimal_inaccurate", None)


# ── Optional diagnostic fallback for C2 hard constraints ──────────────────

def _retry_with_relaxed_c2_caps(sol, kwargs, solve_fn, *, policy: str = "strict"):
    """Handle infeasible QP solves when C2 caps are active.

    Production default is strict fail-closed: sector and correlation caps are
    hard risk constraints, so an infeasible solve means "no QP trade this bar",
    not "retry with weaker diversification." This follows the convex
    optimization contract in Boyd & Vandenberghe: constraints define the
    feasible set; preferences belong in the objective.

    Diagnostic-only policies:
      - "relax": multiply C2 caps by 1.5 and re-solve once.
      - "drop": after the relax retry also drop C2 caps entirely.

    Returns the final QPSolution. Status carries `infeasible:*` only when
    the selected policy does not find a feasible solution.
    """
    if not sol.status.startswith("infeasible"):
        return sol
    has_c2 = (kwargs.get("sector_indicator") is not None
              or kwargs.get("corr_group_pairs"))
    if not has_c2:
        return sol
    mode = str(policy or "strict").strip().lower()
    if mode in {"", "strict", "fail_closed", "fail-closed", "none", "off"}:
        sol.diagnostics = {
            **(getattr(sol, "diagnostics", {}) or {}),
            "c2_infeasible_policy": "strict",
        }
        log.error(
            "QP infeasible with C2 caps — strict policy keeps sector/corr "
            "constraints and blocks QP orders for this bar",
        )
        return sol
    log.warning("QP infeasible with C2 caps — retrying with caps relaxed ×1.5")
    relaxed = dict(kwargs)
    cap_v = relaxed.get("sector_cap_vec")
    if cap_v is not None:
        relaxed["sector_cap_vec"] = np.asarray(cap_v) * 1.5
    pairs = relaxed.get("corr_group_pairs")
    if pairs:
        relaxed["corr_group_pairs"] = [
            (i, j, float(c) * 1.5) for (i, j, c) in pairs
        ]
    sol = solve_fn(**relaxed)
    if not sol.status.startswith("infeasible"):
        sol.diagnostics = {
            **(getattr(sol, "diagnostics", {}) or {}),
            "c2_infeasible_policy": "relax",
        }
        return sol
    if mode not in {"drop", "relax_then_drop", "drop_after_relax"}:
        sol.diagnostics = {
            **(getattr(sol, "diagnostics", {}) or {}),
            "c2_infeasible_policy": "relax",
        }
        return sol
    log.warning(
        "QP still infeasible after relax — dropping C2 caps for this bar "
        "(sector + corr-pair constraints removed)",
    )
    last_resort = dict(kwargs)
    last_resort["sector_indicator"] = None
    last_resort["sector_cap_vec"]   = None
    last_resort["corr_group_pairs"] = None
    sol = solve_fn(**last_resort)
    sol.diagnostics = {
        **(getattr(sol, "diagnostics", {}) or {}),
        "c2_infeasible_policy": "drop",
    }
    return sol


def _retry_for_per_asset_cap_compliance(sol, kwargs, solve_fn):
    """When QP is infeasible and at least one holding is over its per-asset
    cap, attempt a sells-only re-solve that targets cap-compliance.

    This is the §7.6 force-sell-to-cap escape for the audit #2 / issue #70
    failure mode:

      * artifact is ``promotion_status=gated_buys`` (no buy candidates
        admitted via regime_admission)
      * one or more holdings drift above ``regime_params.<R>.max_position_pct``
        due to price appreciation
      * strict QP can't redistribute (no buy slack to absorb the freed
        weight) → status=infeasible → no orders emitted → over-cap
        position stays above cap indefinitely

    Resolution: bring the over-cap holdings back to exactly the cap via a
    deterministic Δw = (cap - current_weight) for each violating asset.
    Other assets get Δw = 0 (hold). No QP optimization is needed for a
    cap-compliance reduction — the action is fully determined by the cap
    + current weight.

    This preserves the spirit of ``gated_buys`` (no fresh BUYS) while
    enforcing the per-asset risk discipline that the strict QP was
    refusing to act on.

    Returns a synthetic QPSolution with ``status="cap_compliance_fallback"``
    when sells were generated, or the original infeasible ``sol`` if no
    holding is actually over cap.
    """
    if not sol.status.startswith("infeasible"):
        return sol
    w_current = kwargs.get("w_current")
    w_upper   = kwargs.get("w_upper")
    if w_current is None or w_upper is None:
        return sol
    w_current = np.asarray(w_current, dtype=float)
    n = w_current.size
    if n == 0:
        return sol
    if np.isscalar(w_upper):
        w_upper_arr = np.full(n, float(w_upper))
    else:
        w_upper_arr = np.asarray(w_upper, dtype=float)
        if w_upper_arr.size != n:
            return sol
    over_mask = w_current > (w_upper_arr + 1e-9)
    if not over_mask.any():
        return sol
    delta_w = np.zeros(n)
    delta_w[over_mask] = w_upper_arr[over_mask] - w_current[over_mask]  # negative (sell)
    target_w = w_current + delta_w
    n_sold = int(over_mask.sum())
    total_sold = float(-np.sum(delta_w[over_mask]))
    log.warning(
        "QP cap-compliance fallback: forcing %d over-cap holding(s) "
        "to per-asset cap (total Δw = -%.3f). Preserves gated_buys policy "
        "(no buys) while enforcing risk discipline.",
        n_sold, total_sold,
    )
    diagnostics = dict(getattr(sol, "diagnostics", {}) or {})
    diagnostics["c2_infeasible_policy"] = "cap_compliance_fallback"
    diagnostics["cap_compliance_n_sold"] = n_sold
    diagnostics["cap_compliance_total_sold"] = total_sold
    return sol.__class__(
        delta_w=delta_w,
        target_w=target_w,
        objective=0.0,
        n_iter=-1,
        status="cap_compliance_fallback",
        diagnostics=diagnostics,
    )


# ── 7. Translate Δw → orders / exits ───────────────────────────────────────

# ── Helper functions for EmitOrdersFromQPSolutionTask (split per §1c) ──────

def _passes_no_trade_band(
    dw: float, sig_i: float, min_dw: float, no_trade_factor: float,
    band_cap: float = 0.05,
    *,
    # Davis-Norman closed-form path (2026-05-30 C, default off):
    band_method: str = "legacy",
    dn_eps_oneway: float = 0.0,
    dn_gamma: float = 0.0,
    dn_pi_star: float = 0.0,
    dn_floor: float = 0.0,
    dn_ceiling: float = 1.0,
) -> tuple[bool, bool]:
    """No-trade band gate. Two band-computation modes:

    Legacy (default, ``band_method='legacy'``):
        Davis-Norman / Constantinides ad-hoc form. Skip inside
        ``max(min_dw, min(band_cap, no_trade_factor × σ_i))``.

        2026-05-09 BUG #7 fix: band_cap protects high-σ names from
        unreachable 10-30% bands (BA at σ=0.24).

    Closed-form Davis-Norman (``band_method='davis_norman'``):
        Threshold from the literature 1/3-power formula
        ``δ* = (1.5/γ · ε · π·(1-π)² · σ²)^(1/3)`` (Davis-Norman 1990,
        Janeček-Shreve 2004), clamped to ``[dn_floor, dn_ceiling]``.
        Eliminates the three hand-tuned knobs in favor of literature
        scaling. Enable via ``rotation.joint_actions.qp_band_method`` and
        provide γ (risk_aversion), π* (current/target weight), ε (one-way
        cost). The 2026-05-30 research report verified DN gives ≈ 1.1%
        at our typical params vs the hand-tuned 2% floor — ~half the
        threshold, more trades pass through.

    Returns (pass, was_in_band).
    """
    if band_method == "davis_norman":
        from .davis_norman import davis_norman_band_clamped  # noqa: PLC0415
        threshold = davis_norman_band_clamped(
            eps_oneway=dn_eps_oneway, sigma=sig_i, gamma=dn_gamma,
            pi_star=dn_pi_star, floor=dn_floor, ceiling=dn_ceiling,
        )
        # Preserve the legacy min_dw floor (operator can still force min absolute Δw).
        threshold = max(min_dw, threshold)
        if abs(dw) < threshold:
            return False, abs(dw) >= min_dw
        return True, False
    # Legacy path (unchanged).
    sigma_band = min(band_cap, no_trade_factor * sig_i)
    threshold = max(min_dw, sigma_band)
    if abs(dw) < threshold:
        return False, abs(dw) >= min_dw
    return True, False


def _gate_buy_or_block(
    t: str, dw: float, today, earnings_cal, earn_buf: int,
    buys_gated: bool,
) -> str | None:
    """If dw>0 (buy/top-up): return blocked_reason if any gate fires.
    Returns None if buy is allowed."""
    if dw <= 0:
        return None
    if buys_gated:
        return "buys_gated"
    from kernel.selection import is_earnings_blocked  # noqa: PLC0415
    if today is not None and is_earnings_blocked(t, today, earnings_cal, earn_buf):
        return "earnings"
    return None


def _shares_from_dw(dw: float, nav: float, px: float) -> int:
    """Convert Δw fraction into integer share count, with finite checks."""
    import math as _m  # noqa: PLC0415
    if not (_m.isfinite(dw) and _m.isfinite(px) and _m.isfinite(nav)):
        return 0
    if px <= 0 or nav <= 0:
        return 0
    return int(abs(dw) * nav / px)


def _long_sell_credit(env: dict, ticker: str, shares: int, px: float) -> float:
    """Estimated same-bar cash released by selling an existing long."""
    if shares <= 0 or px <= 0:
        return 0.0
    hs = (env.get("holdings") or {}).get(ticker)
    if hs is None:
        return 0.0
    try:
        held = float(getattr(hs, "shares", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(held) or held <= 0.0:
        return 0.0
    credit = min(float(shares), held) * float(px)
    return credit if math.isfinite(credit) and credit > 0.0 else 0.0


def _array_float_at(value, idx: int) -> float | None:
    if value is None:
        return None
    try:
        out = float(value[idx])
    except (TypeError, ValueError, IndexError):
        return None
    return out if math.isfinite(out) else None


def _qp_max_positions(ctx) -> int:
    regime_params = (
        (ctx.config.get("regime_params", {}) or {})
        .get(getattr(ctx, "regime", None), {})
        or {}
    )
    return int(regime_params.get(
        "max_concurrent_positions",
        ctx.config.get("max_concurrent_positions", 8),
    ))


def _qp_buy_admission_block_reason(ctx, env: dict, ticker: str) -> str | None:
    """Fail closed when QP tries to add risk without alpha admission.

    QP solves portfolio weights; it must not become the model-selection layer.
    The gate is intentionally applied at order emission so the solver can still
    trim or close holdings, while new risk additions require finite calibrated
    score evidence and optional raw-panel support.
    """
    gate = (env.get("cfg", {}) or {}).get("qp_admission_gate", {}) or {}
    if not bool(gate.get("enabled", False)):
        return None

    is_held = ticker in env.get("holdings_set", set())
    if is_held and ticker in set(env.get("exit_only_tickers", set()) or set()):
        return (
            (env.get("exit_only_reasons") or {}).get(ticker)
            or "qp_universe_exit_only"
        )
    if (
        not is_held
        and bool(gate.get("respect_open_slots", True))
        and not bool(env.get("ignore_slots", False))
    ):
        held_after_exits = set(env.get("holdings_set", set())) - set(
            env.get("preexisting_exit_tickers", set())
        )
        admitted_new = set(env.get("admitted_new_tickers", set()) or set())
        emitted_new = set(env.get("emitted_new_tickers", set()) or set())
        used_slots = len(held_after_exits | admitted_new | emitted_new)
        if used_slots >= int(env.get("max_positions", 0) or 0):
            return "qp_admission_no_slot"

    source = (
        (env.get("score_sources") or {}).get(ticker)
        or (env.get("cands") or {}).get(ticker)
        or (env.get("holdings") or {}).get(ticker)
    )
    if source is None:
        return "qp_admission_missing_score"

    rank_floor = gate.get(
        "topup_min_rank_score" if is_held else "min_rank_score",
        gate.get("min_rank_score"),
    )
    rank = _source_float(source, "rank_score")
    if rank_floor is not None:
        floor = float(rank_floor)
        if not math.isfinite(rank) or rank < floor:
            return "qp_admission_rank"

    panel_floor = gate.get(
        "topup_min_panel_score" if is_held else "min_panel_score",
        gate.get("min_panel_score"),
    )
    panel = _source_float(source, "panel_score")
    if panel_floor is not None:
        floor = float(panel_floor)
        if not math.isfinite(panel) or panel < floor:
            return "qp_admission_panel"

    sigma_cap = _qp_admission_gate_value(
        gate,
        "topup_max_sigma" if is_held else "max_sigma",
        getattr(ctx, "regime", None),
    )
    if sigma_cap is not None:
        cap = float(sigma_cap)
        sigma = _source_float(source, "sigma")
        if not math.isfinite(sigma) or sigma > cap:
            return "qp_admission_sigma"

    er_floor = _qp_admission_expected_return_floor(gate, is_held, getattr(ctx, "regime", None))
    if er_floor is not None:
        floor = float(er_floor)
        expected_return = _source_float(source, "expected_return")
        horizon_block = _qp_admission_horizon_block_reason(
            ctx,
            env,
            source,
            metric="expected_return",
        )
        if horizon_block:
            return horizon_block
        if not math.isfinite(expected_return) or expected_return < floor:
            return "qp_admission_expected_return"

    er_over_sigma_floor, er_over_sigma_metric = _qp_admission_expected_return_over_sigma_floor(
        gate,
        is_held,
        getattr(ctx, "regime", None),
    )
    if er_over_sigma_floor is not None:
        floor = float(er_over_sigma_floor)
        signal = _source_float(source, er_over_sigma_metric)
        horizon_block = _qp_admission_horizon_block_reason(
            ctx,
            env,
            source,
            metric=er_over_sigma_metric,
        )
        if horizon_block:
            return horizon_block
        sigma = _qp_admission_sigma(ctx, env, ticker, source)
        ratio = (
            signal / sigma
            if math.isfinite(signal) and math.isfinite(sigma) and sigma > 0
            else float("nan")
        )
        if not math.isfinite(ratio) or ratio < floor:
            return "qp_admission_expected_return_over_sigma"

    return None


def _qp_admission_gate_value(gate: dict, key: str, regime: str | None):
    by_regime = gate.get(f"{key}_by_regime")
    if isinstance(by_regime, dict) and regime in by_regime:
        return by_regime[regime]
    return gate.get(key)


def _qp_admission_expected_return_floor(
    gate: dict,
    is_held: bool,
    regime: str | None,
):
    keys = (
        (
            "topup_min_expected_return",
            "topup_min_expected_excess_return",
            "min_expected_return",
            "min_expected_excess_return",
        )
        if is_held else
        (
            "min_expected_return",
            "min_expected_excess_return",
        )
    )
    for key in keys:
        value = _qp_admission_gate_value(gate, key, regime)
        if value is not None:
            return value
    return None


def _qp_admission_expected_return_over_sigma_floor(
    gate: dict,
    is_held: bool,
    regime: str | None,
):
    keys = (
        (
            "topup_min_expected_return_over_sigma",
            "topup_min_mu_over_sigma",
            "topup_min_edge_over_sigma",
            "min_expected_return_over_sigma",
            "min_mu_over_sigma",
            "min_edge_over_sigma",
        )
        if is_held else
        (
            "min_expected_return_over_sigma",
            "min_mu_over_sigma",
            "min_edge_over_sigma",
        )
    )
    for key in keys:
        value = _qp_admission_gate_value(gate, key, regime)
        if value is not None:
            metric = "mu" if "_mu_" in key or key.startswith("min_mu") else "expected_return"
            return value, metric
    return None, "expected_return"


def _qp_admission_expected_horizon(ctx, env: dict) -> int | None:
    cfg = env.get("cfg", {}) or {}
    candidates = [
        cfg.get("qp_mu_horizon_days"),
        cfg.get("target_horizon_days"),
    ]
    full_cfg = getattr(ctx, "config", None) or {}
    candidates.extend([
        ((full_cfg.get("rotation", {}) or {}).get("joint_actions", {}) or {}).get(
            "qp_mu_horizon_days",
        ),
        (full_cfg.get("rotation", {}) or {}).get("target_horizon_days"),
        (full_cfg.get("panel_ltr", {}) or {}).get("lookahead_days"),
    ])
    for value in candidates:
        try:
            out = int(value)
        except (TypeError, ValueError):
            continue
        if out > 0:
            return out
    return None


def _source_positive_int(source: object, name: str) -> int | None:
    value = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _qp_admission_horizon_block_reason(
    ctx,
    env: dict,
    source: object,
    *,
    metric: str,
) -> str | None:
    expected_horizon = _qp_admission_expected_horizon(ctx, env)
    if expected_horizon is None:
        return None
    field = "mu_horizon_days" if metric == "mu" else "expected_return_horizon_days"
    actual_horizon = _source_positive_int(source, field)
    if actual_horizon != expected_horizon:
        suffix = "mu_horizon" if metric == "mu" else "expected_return_horizon"
        return f"qp_admission_{suffix}"
    return None


def _source_float(source: object, name: str) -> float:
    value = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _qp_admission_sigma(ctx, env: dict, ticker: str, source: object) -> float:
    """Return σ in the same horizon units QP uses for μ admission.

    The optional edge-over-risk gate compares a horizon return estimate to a
    volatility estimate. If QP is configured to scale annual/daily σ to the
    μ horizon, admission must use the same scale; otherwise this gate blocks
    good candidates for a pure unit mismatch.
    """
    tickers = list(env.get("tickers") or [])
    sigma_vec = env.get("sigma_vec")
    if sigma_vec is not None and ticker in tickers:
        idx = tickers.index(ticker)
        try:
            out = float(sigma_vec[idx])
        except (TypeError, ValueError, IndexError):
            out = float("nan")
        if math.isfinite(out) and out > 0:
            return out

    raw = _source_float(source, "sigma")
    if not math.isfinite(raw) or raw <= 0:
        return float("nan")
    cfg = env.get("cfg", {}) or {}
    mode = str(cfg.get("qp_sigma_horizon_mode", "none")).lower()
    if mode in {"none", "off", "disabled"}:
        return raw
    horizon = _qp_admission_expected_horizon(ctx, env)
    if horizon is None:
        return raw
    unit = str(cfg.get("qp_sigma_unit", "horizon")).lower()
    scale = _qp_sigma_horizon_scale(unit, horizon)
    return raw * scale if scale is not None else raw


def _buy_cost_multiplier(config: dict) -> float:
    """Return conservative cash multiplier for a buy order."""
    exec_cfg = (config or {}).get("execution", {}) or {}
    if bool(exec_cfg.get("legacy_no_fees", False)):
        return 1.0
    if not bool(exec_cfg.get("enabled", True)):
        return 1.0
    bps = (
        float(exec_cfg.get("half_spread_bps", 2.0) or 0.0)
        + float(exec_cfg.get("commission_bps", 0.0) or 0.0)
        + float(exec_cfg.get("qp_buy_cash_buffer_bps", 1.0) or 0.0)
    )
    return 1.0 + max(0.0, bps) / 10000.0


def _cap_buy_shares_to_cash(
    shares: int,
    px: float,
    cash_left: float,
    cost_multiplier: float,
) -> tuple[int, float]:
    """Cap buy shares so emitted QP orders fit free cash."""
    if shares <= 0 or px <= 0 or cash_left <= 0:
        return 0, 0.0
    unit_cost = px * max(1.0, float(cost_multiplier))
    capped = min(int(shares), int(cash_left // unit_cost))
    return capped, capped * unit_cost


def _actual_qp_buy_target_pct(ctx, ticker: str, shares: int, px: float) -> float:
    """Return post-fill target weight implied by emitted shares.

    QP's solver target_w is the desired total weight before integer share and
    cash caps. The adapters execute the emitted whole-share order; target_pct
    is retained as audit metadata and must match the capped shares.
    """
    nav = float(getattr(ctx, "portfolio_value", 0.0) or 0.0)
    if nav <= 0 or px <= 0 or shares <= 0:
        return 0.0
    hs = (getattr(ctx, "holdings", None) or {}).get(ticker)
    held_shares = float(getattr(hs, "shares", 0.0) or 0.0) if hs is not None else 0.0
    return max(0.0, (held_shares + float(shares)) * float(px) / nav)


def _qp_soft_sell_block_reason(ctx, ticker: str, sol, i: int) -> str | None:
    """Apply model-soft-exit guards to QP long trims/closes.

    QP sells are optimizer-driven, not hard risk exits, so they respect the
    same thesis-age horizon gate as panel-conviction exits. Tax-aware soft
    sell gates are different: the production contract says `qp_tax_aware=false`
    means no QP tax-driven sell/hold logic, including the order-emission stage.
    """
    cfg = _qp_cfg(ctx)
    guard_cfg = cfg.get("qp_soft_sell_guard", {})
    if isinstance(guard_cfg, dict) and guard_cfg.get("enabled") is False:
        return None
    target_w = float(sol.target_w[i])
    if target_w < -1e-9:
        return None
    hs = (getattr(ctx, "holdings", None) or {}).get(ticker)
    if hs is None:
        return None
    panel_cfg = _qp_soft_sell_effective_panel_cfg(
        ((getattr(ctx, "config", {}) or {}).get("risk", {}) or {}).get("panel_exit", {}) or {},
        guard_cfg,
    )
    from kernel.pipeline.soft_exit_guards import (  # noqa: PLC0415
        configured_soft_exit_min_days,
        lt_gate_suppression,
        resolve_current_price,
        soft_exit_horizon_suppression,
        soft_exit_thesis_regime,
        tax_adjusted_soft_exit_suppression,
        trading_holding_days,
    )
    thesis_regime = soft_exit_thesis_regime(hs, getattr(ctx, "regime", None))
    suppress, why = soft_exit_horizon_suppression(
        panel_cfg=panel_cfg,
        regime=thesis_regime,
        today=getattr(ctx, "today", None),
        holding=hs,
    )
    if suppress:
        return "qp_soft_sell_horizon:" + why
    min_days = configured_soft_exit_min_days(panel_cfg, thesis_regime)
    if min_days > 0:
        pending_shares = (_get_path(ctx, "_qp_pending_sell_shares") or {}).get(ticker)
        lot_days = _disposed_lot_min_holding_days(
            holding=hs,
            shares=pending_shares,
            today=getattr(ctx, "today", None),
            lot_method=str(cfg.get("qp_tax_lot_method", "fifo")).lower(),
        )
        if lot_days is not None and lot_days < min_days:
            return (
                "qp_soft_sell_lot_horizon:"
                f"lot_days={lot_days} < {min_days} "
                f"regime={thesis_regime} "
                f"method={str(cfg.get('qp_tax_lot_method', 'fifo')).lower()}"
            )
    current_price = resolve_current_price(ctx, hs, ticker)
    if not _qp_soft_sell_tax_gates_enabled(cfg, guard_cfg):
        return None
    suppress, why = lt_gate_suppression(
        config=getattr(ctx, "config", {}) or {},
        today=getattr(ctx, "today", None),
        holding=hs,
        current_price=current_price,
    )
    if suppress:
        return "qp_soft_sell_lt_gate:" + why
    mu_vec = _get_path(ctx, "_qp_mu")
    mu_i = None
    if mu_vec is not None and i < len(mu_vec):
        mu_i = float(mu_vec[i])
    suppress, why = tax_adjusted_soft_exit_suppression(
        panel_cfg=panel_cfg,
        tax_cfg=(getattr(ctx, "config", {}) or {}).get("tax") or {},
        today=getattr(ctx, "today", None),
        holding=hs,
        current_price=current_price,
        mu=mu_i,
    )
    if suppress:
        return "qp_soft_sell_tax:" + why
    return None


def _qp_soft_sell_effective_panel_cfg(
    panel_cfg: dict[str, Any],
    guard_cfg: Any,
) -> dict[str, Any]:
    """QP-specific soft-sell guard config.

    QP trims are optimizer-driven soft exits but they do not need to share
    every threshold with the cross-sectional panel-conviction exit. Let the
    QP guard override the thesis-age horizon while inheriting the shared
    panel-exit defaults for LT/tax helpers.
    """
    merged = dict(panel_cfg or {})
    if not isinstance(guard_cfg, dict):
        return merged
    for key in ("min_holding_days", "min_holding_days_by_regime"):
        if key in guard_cfg:
            merged[key] = guard_cfg[key]
    return merged


def _disposed_lot_min_holding_days(
    *,
    holding: Any,
    shares: Any,
    today: Any,
    lot_method: str,
) -> int | None:
    """Minimum age among lots a QP soft sell would actually dispose."""
    from kernel.pipeline.soft_exit_guards import trading_holding_days  # noqa: PLC0415

    if not isinstance(today, _dt.date):
        return None
    try:
        target = float(shares)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(target) or target <= 0:
        return None
    lots = list(getattr(holding, "lots", None) or [])
    if not lots:
        return trading_holding_days(today, holding)

    method = str(lot_method or "fifo").lower()
    if method == "hifo":
        ordered = sorted(lots, key=lambda lot: -float(getattr(lot, "price", 0.0) or 0.0))
    elif method == "avg":
        return trading_holding_days(today, holding)
    else:
        ordered = lots

    consumed = 0.0
    min_days: int | None = None
    for lot in ordered:
        if consumed >= target - 1e-12:
            break
        try:
            lot_shares = float(getattr(lot, "shares", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(lot_shares) or lot_shares <= 0:
            continue
        lot_date = getattr(lot, "date", None)
        if not isinstance(lot_date, _dt.date):
            continue
        take = min(lot_shares, target - consumed)
        if take <= 0:
            continue
        from types import SimpleNamespace  # noqa: PLC0415

        age = trading_holding_days(today, SimpleNamespace(entry_date=lot_date))
        if age is None:
            return None
        min_days = age if min_days is None else min(min_days, age)
        consumed += take
    return min_days


class ApplyProportionalTradeTask(Task):
    """Gârleanu-Pedersen 2013 partial-rebalance (research B).

    Mutates ``ctx._qp_solution`` in place: replaces ``target_w`` (the
    QP's frictionless one-shot target) with ``current + (target - current) / N``
    and recomputes ``delta_w`` from the new target. Skipped (no-op) when N ≤ 1
    or the per-regime knob is absent — preserves all-or-nothing legacy behavior.

    N comes from ``regime_params.<REGIME>.qp_partial_trade_horizon_days`` per
    PRIME DIRECTIVE, with a global default ``rotation.joint_actions.
    qp_partial_trade_horizon_days``.

    Reference: cvxportfolio.ProportionalTradeToTargets.values_in_time
    (15-line verbatim form of GP-2013's optimal trade-rate matrix's
    scalar projection).
    """

    def run(self, ctx) -> bool | None:
        import numpy as np  # noqa: PLC0415
        from .proportional_trade import (  # noqa: PLC0415
            proportional_trade_target,
            resolve_trade_horizon_days,
        )

        sol = _get_path(ctx, "_qp_solution")
        if sol is None or not hasattr(sol, "target_w"):
            return None
        current = _get_path(ctx, "_qp_w_current")
        if current is None:
            return None

        regime = str(getattr(ctx, "regime", "") or "")
        regime_params = (ctx.config or {}).get("regime_params") or {}
        cfg = ((ctx.config or {}).get("rotation") or {}).get("joint_actions") or {}
        default_n = cfg.get("qp_partial_trade_horizon_days")

        n_days = resolve_trade_horizon_days(
            regime=regime, regime_params=regime_params, default_days=default_n,
        )
        if n_days <= 1.0:
            # Legacy all-or-nothing — preserve QP target as-is.
            ctx._qp_partial_trade_applied = False  # noqa: SLF001
            return None

        current_arr = np.asarray(current, dtype=float)
        target_arr = np.asarray(sol.target_w, dtype=float)
        partial = proportional_trade_target(
            current_w=current_arr, target_w=target_arr, n_days=n_days,
        )
        sol.target_w = partial
        sol.delta_w = partial - current_arr
        ctx._qp_partial_trade_applied = True  # noqa: SLF001
        ctx._qp_partial_trade_n_days = float(n_days)  # noqa: SLF001
        log.info(
            "ApplyProportionalTradeTask: regime=%s N=%.1f — shrank QP target by 1/N "
            "(Gârleanu-Pedersen 2013 partial-rebalance, research B)",
            regime, n_days,
        )
        return True


class EmitOrdersFromQPSolutionTask(Task):
    """Translate Δw → ctx.orders (buys/top-ups) + ctx.exits (closes/trims).

    Reads:  ctx._qp_solution, ctx._qp_tickers, ctx.prices, ctx.holdings,
             ctx.portfolio_value, ctx.candidates,
             ctx.config['rotation']['joint_actions']['qp_min_dw_pct']
    Writes: ctx.orders (append), ctx.exits (append),
             ctx._qp_n_buys, ctx._qp_n_sells (counters for atom-side LogSummary)

    Logic split per CLAUDE.md §1c (2026-05-06):
      1. Setup gate flags + helper closures
      2. Per-ticker loop calls _passes_no_trade_band, _gate_buy_or_block,
         _shares_from_dw, then _emit_qp_buy / _emit_qp_sell
      3. Log summary of blocked/skipped counters

    Bug fixes pinned by tests (do NOT regress):
      - Bug 3 (wl183 2026-05-05): buy_blocked / skip_buys suppress top-ups
      - Bug 4 (wl183 2026-05-05): earnings blackout suppresses top-ups
      - Bug 9 (2026-05-05): non-finite Δw skipped instead of crashing
      - Davis-Norman no-trade band (2026-05-05 cash-drag fix)
    """
    name = "EmitOrdersFromQPSolutionTask"

    # Module-level ``QP_EMITTABLE_STATUSES`` is the single source of truth
    # for "the QP solution can drive orders". Re-exported here as a class
    # attribute for backward compatibility with tests that import via the
    # class.
    _EMITTABLE_STATUSES = QP_EMITTABLE_STATUSES

    def run(self, ctx) -> bool | None:
        sol = _get_path(ctx, "_qp_solution")
        if sol is None or sol.status not in QP_EMITTABLE_STATUSES:
            status = str(sol.status if sol else "missing_solution")
            reason = (
                "qp_missing_solution" if sol is None
                else ("qp_no_signal" if sol.status == "optimal_no_signal"
                      else f"qp_global:{sol.status}")
            )
            ctx._qp_status = status  # noqa: SLF001
            ctx._qp_failure_reason = reason  # noqa: SLF001
            if sol is not None:
                ctx._qp_diagnostics = dict(getattr(sol, "diagnostics", {}) or {})  # noqa: SLF001
            _stamp_all_qp_blocks(ctx, reason)
            # Codex PR #48 #1: route through shared helper so emit + earlier
            # short-circuit paths (ComputeFullSigma._fail_full_sigma,
            # SolveMarkowitzQP unsupported-cvxportfolio branch, non-optimal
            # solver result) all stamp the same counter set.
            _stamp_qp_failure_counter(ctx, status)
            log.warning("EmitOrdersFromQPSolutionTask: status=%s — skip", status)
            return False
        env = self._build_env(ctx, sol)
        self._log_holding_solves(env)
        nb, ns, counters = self._emit_orders_loop(ctx, env)
        for key, value in counters.items():
            if value:
                ckey = f"qp_{key}"
                ctx.counters[ckey] = ctx.counters.get(ckey, 0) + int(value)
        self._log_summary(
            n_blocked_buys=counters["blocked_buys"], buy_blocked=env["buy_blocked"],
            n_blocked_earnings=counters["blocked_earnings"], earn_buf=env["earn_buf"],
            n_defensive_non_bear=counters["defensive_non_bear"],
            n_skipped_nonfinite=counters["skipped_nonfinite"],
            n_skipped_band=counters["skipped_band"], min_dw=env["min_dw"],
            no_trade_factor=env["no_trade_factor"],
            n_delta_below_min_dw=counters["delta_below_min_dw"],
            n_zero_shares=counters["zero_shares"],
            n_no_buy_delta=counters["no_buy_delta"],
            n_not_selected=counters["not_selected"],
            n_cash_capped=counters["cash_capped"],
            n_cash_exhausted=counters["cash_exhausted"],
            n_soft_sell_blocked=counters["soft_sell_blocked"],
            n_preexisting_exit=counters["preexisting_exit"],
            n_admission_blocked=counters["admission_blocked"],
        )
        ctx._qp_n_buys = nb  # noqa: SLF001
        ctx._qp_n_sells = ns  # noqa: SLF001

    @staticmethod
    def _build_env(ctx, sol) -> dict:
        """Snapshot the per-run gates + thresholds in one dict so each
        downstream helper sees a coherent view."""
        cfg = _qp_cfg(ctx)
        buy_blocked = bool(getattr(ctx, "buy_blocked", False))
        skip_buys = bool(getattr(ctx, "skip_buys", False))
        from kernel.pipeline.task_benchmark_sleeve import (  # noqa: PLC0415
            benchmark_sleeve_alpha_funding_capacity,
            benchmark_sleeve_cash_reserve_credit,
        )
        cash = float(_get_path(
            ctx, "cash", _get_path(ctx, "portfolio_value", 0.0),
        ) or 0.0)
        alpha_funding_cash = float(benchmark_sleeve_alpha_funding_capacity(ctx))
        cash_reserve = float(_get_path(ctx, "_qp_cash_reserve", 0.0) or 0.0)
        reserve_credit = float(benchmark_sleeve_cash_reserve_credit(ctx))
        effective_cash_reserve = max(0.0, cash_reserve - reserve_credit)
        ctx._qp_alpha_funding_cash = alpha_funding_cash  # noqa: SLF001
        ctx._qp_cash_reserve_effective = effective_cash_reserve  # noqa: SLF001
        return dict(
            cfg=cfg,
            sol=sol,
            tickers=_get_path(ctx, "_qp_tickers") or [],
            prices=_get_path(ctx, "prices") or {},
            nav=float(_get_path(ctx, "portfolio_value", 0.0) or 0.0),
            cash=cash + alpha_funding_cash,
            cash_actual=cash,
            alpha_funding_cash=alpha_funding_cash,
            cash_reserve=effective_cash_reserve,
            cash_reserve_configured=cash_reserve,
            cash_reserve_credit=reserve_credit,
            buy_cost_multiplier=_buy_cost_multiplier(ctx.config or {}),
            min_dw=float(cfg.get("qp_min_dw_pct", 0.005)),
            no_trade_factor=float(cfg.get("qp_no_trade_band_factor", 0.0)),
            band_cap=float(cfg.get("qp_no_trade_band_cap", 0.05)),
            # Davis-Norman closed-form path (research C, 2026-05-30).
            # Default 'legacy' preserves current behaviour.
            band_method=str(cfg.get("qp_band_method", "legacy")),
            dn_eps_oneway=float(cfg.get("qp_band_dn_eps_oneway",
                                         (float(cfg.get("qp_cost_kappa", 0.002)) or 0.002) / 2.0)),
            dn_gamma=float(cfg.get("qp_band_dn_gamma",
                                     cfg.get("qp_risk_aversion", 3.0))),
            dn_floor=float(cfg.get("qp_band_dn_floor", 0.005)),
            dn_ceiling=float(cfg.get("qp_band_dn_ceiling",
                                       cfg.get("qp_no_trade_band_cap", 0.05))),
            sigma_vec=_get_path(ctx, "_qp_sigma"),
            cands={c.ticker: c for c in (ctx.candidates or [])},
            score_sources=_get_path(ctx, "_qp_mu_source_map") or {},
            buy_blocked=buy_blocked,
            buys_gated=buy_blocked or skip_buys,
            earnings_cal=getattr(ctx, "earnings_calendar", None) or {},
            earn_buf=int((ctx.config.get("regime", {}) or {})
                          .get("earnings_buffer_days", 3)),
            today=getattr(ctx, "today", None),
            holdings_set=set((ctx.holdings or {}).keys()),
            holdings=(ctx.holdings or {}),
            max_positions=_qp_max_positions(ctx),
            # 2026-05-24 safety hardening: disabled by default. Whole-share
            # rounding must not turn a sub-1-share optimizer target into an
            # over-target trade. Use fractional shares or a true MIP lot-size
            # optimizer before re-enabling this exploratory override.
            min_share_floor_pct=float(cfg.get("qp_min_share_floor_pct", 0.0)),
            min_share_ceiling_pct=float(cfg.get("qp_min_share_ceiling_pct", 0.15)),
            defensive_set=set((ctx.config or {}).get("defensive_tickers", []) or []),
            bear_only=bool(getattr(ctx, "bear_only", False)),
            preexisting_exit_tickers={
                t for t, _ in (getattr(ctx, "exits", None) or [])
            },
            exit_only_tickers=set(getattr(ctx, "_qp_exit_only_tickers", set()) or set()),
            exit_only_reasons=dict(getattr(ctx, "_qp_exit_only_reasons", {}) or {}),
            emitted_new_tickers=set(),
        )

    @staticmethod
    def _log_holding_solves(env: dict) -> None:
        """2026-05-09 BA QP audit: log every holding's per-asset solution
        so we can see why a name (e.g. high-negative-μ̂ BA) wasn't sold
        even after BUG #7 band-cap fix. Holdings only — buys are visible
        via QP_BUY. Diagnostic-only; no behavior change."""
        import math as _m  # noqa: PLC0415
        sol = env["sol"]; sigma_vec = env["sigma_vec"]
        for i, t in enumerate(env["tickers"]):
            if t not in env["holdings_set"]:
                continue
            tw = float(sol.target_w[i]) if hasattr(sol, "target_w") else float("nan")
            dw_h = float(sol.delta_w[i]) if hasattr(sol, "delta_w") else float("nan")
            sig_h = float(sigma_vec[i]) if (sigma_vec is not None and i < len(sigma_vec)) else float("nan")
            eff_band = max(env["min_dw"], min(env["band_cap"],
                                                env["no_trade_factor"] * (sig_h if _m.isfinite(sig_h) else 0)))
            will_skip = (abs(dw_h) < eff_band) if _m.isfinite(dw_h) else None
            log.info(
                "QP_HOLDING_SOLVE %s: target_w=%+.4f Δw=%+.4f σ=%.3f "
                "eff_band=%.4f will_skip=%s",
                t, tw, dw_h, sig_h, eff_band, will_skip,
            )

    @staticmethod
    def _emit_orders_loop(ctx, env: dict) -> tuple[int, int, dict]:
        """Iterate tickers, apply no-trade-band + earnings/halt gates,
        emit buys/sells. Returns (n_buys, n_sells, counters)."""
        import math as _m  # noqa: PLC0415
        sol = env["sol"]; sigma_vec = env["sigma_vec"]
        nb = ns = 0
        candidate_tickers = set(env["cands"].keys())
        emitted_candidates: set[str] = set()
        blocked_map = getattr(ctx, "_blocked_by_ticker", None)
        if blocked_map is None:
            blocked_map = {}
            ctx._blocked_by_ticker = blocked_map  # noqa: SLF001

        def stamp(ticker: str, reason: str) -> None:
            if ticker in candidate_tickers or ticker in env["holdings_set"]:
                blocked_map.setdefault(ticker, reason)

        c = dict(blocked_buys=0, blocked_earnings=0, defensive_non_bear=0,
                 skipped_nonfinite=0, skipped_band=0,
                 delta_below_min_dw=0, zero_shares=0,
                 no_buy_delta=0, not_selected=0,
                 cash_capped=0, cash_exhausted=0, soft_sell_blocked=0,
                 preexisting_exit=0, admission_blocked=0,
                 sell_credit_events=0)
        buy_cash_left = max(0.0, env["cash"] - env["nav"] * env["cash_reserve"])
        pending_sell_shares: dict[str, float] = {}
        _set_path(ctx, "_qp_pending_sell_shares", pending_sell_shares)
        for i, t in enumerate(env["tickers"]):
            dw = float(sol.delta_w[i])
            if not _m.isfinite(dw):
                c["skipped_nonfinite"] += 1
                stamp(t, "qp_nonfinite_delta")
                continue
            if t in env["preexisting_exit_tickers"]:
                c["preexisting_exit"] += 1
                stamp(t, "qp_preexisting_exit")
                log.info(
                    "QP_TRADE_SUPPRESSED %-6s preexisting_exit "
                    "(QP must not double-act on an already exiting ticker)",
                    t,
                )
                continue
            sig_i = 0.0
            if sigma_vec is not None and i < len(sigma_vec):
                s = float(sigma_vec[i])
                if _m.isfinite(s) and s > 0:
                    sig_i = s
            # π* for DN band: target weight from QP solve (the asset's
            # frictionless optimum in DN-1990 terms). Fall back to a small
            # constant when target is non-finite or zero (e.g. blocked names).
            pi_star_i = 0.0
            try:
                tw_i = float(sol.target_w[i]) if hasattr(sol, "target_w") else 0.0
                if _m.isfinite(tw_i) and tw_i > 0:
                    pi_star_i = tw_i
            except Exception:  # noqa: BLE001
                pi_star_i = 0.0
            if pi_star_i <= 0:
                # Use a typical-name fallback so DN gives a meaningful threshold
                # for sell-only or near-zero-target names. 0.05 = 5% portfolio weight.
                pi_star_i = 0.05
            ok, in_band = _passes_no_trade_band(
                dw, sig_i, env["min_dw"],
                env["no_trade_factor"], band_cap=env["band_cap"],
                band_method=env.get("band_method", "legacy"),
                dn_eps_oneway=env.get("dn_eps_oneway", 0.0),
                dn_gamma=env.get("dn_gamma", 0.0),
                dn_pi_star=pi_star_i,
                dn_floor=env.get("dn_floor", 0.0),
                dn_ceiling=env.get("dn_ceiling", 1.0),
            )
            if not ok:
                if in_band:
                    c["skipped_band"] += 1
                    stamp(t, "qp_no_trade_band")
                else:
                    c["delta_below_min_dw"] += 1
                    stamp(t, "qp_delta_below_min_dw")
                continue
            px = env["prices"].get(t, 0.0)
            shares = _shares_from_dw(dw, env["nav"], px)
            if shares <= 0 and dw > 0 and px > 0 and env["nav"] > 0:
                # 2026-05-17 min_share_floor for high-price stocks (EQIX/META class).
                # Without this, any candidate whose share price exceeds the QP's
                # dollar budget (target_w × NAV) gets silently dropped — for a
                # $10k account this blocks EQIX ($1059), BKNG ($5k), NVR ($8k),
                # etc. entirely, biasing the strategy toward low-price names.
                # 2026-05-24 audit: this is now an explicit experiment only
                # (default floor=0). Markowitz/Boyd-style constrained QP
                # weights are the contract; integer execution may round down
                # to stay feasible, but rounding up to one share can exceed
                # the optimizer target/cap and manufacture an unintended
                # trade. Mature fixes are fractional-share execution or a
                # mixed-integer lot-size optimizer, not silent over-allocation.
                floor   = env["min_share_floor_pct"]
                ceiling = env["min_share_ceiling_pct"]
                if floor > 0:
                    one_share_pct = px / env["nav"]
                    if floor <= one_share_pct <= ceiling:
                        shares = 1
                        log.info(
                            "QP_MIN_SHARE_FLOOR %s: dw=%+.4f → 0 shares "
                            "(px=$%.2f > target $%.0f) — buy 1 share "
                            "(1 share = %.1f%% NAV, floor=%.1f%%, ceil=%.1f%%)",
                            t, dw, px, abs(dw) * env["nav"],
                            one_share_pct * 100, floor * 100, ceiling * 100,
                        )
            if shares <= 0:
                if t in candidate_tickers:
                    if dw > 0:
                        c["zero_shares"] += 1
                        stamp(t, "qp_zero_shares")
                    else:
                        c["no_buy_delta"] += 1
                        stamp(t, "qp_no_buy_delta")
                continue
            if dw > 0:
                admission_block = _qp_buy_admission_block_reason(ctx, env, t)
                if admission_block:
                    c["admission_blocked"] += 1
                    stamp(t, admission_block)
                    log.info(
                        "QP_BUY_SUPPRESSED %-6s %s "
                        "(QP only sizes pre-qualified alpha)",
                        t, admission_block,
                    )
                    continue
                if t in env["defensive_set"] and not env["bear_only"]:
                    c["defensive_non_bear"] += 1
                    stamp(t, "defensive_non_bear")
                    log.info(
                        "QP_BUY_SUPPRESSED %-6s defensive_non_bear "
                        "(regime=%s)",
                        t, getattr(ctx, "regime", None),
                    )
                    continue
                blocked = _gate_buy_or_block(
                    t, dw, env["today"], env["earnings_cal"], env["earn_buf"],
                    env["buys_gated"],
                )
                if blocked == "buys_gated":
                    c["blocked_buys"] += 1
                    stamp(t, "buy_blocked" if env["buy_blocked"] else "skip_buys")
                    continue
                if blocked == "earnings":
                    c["blocked_earnings"] += 1
                    stamp(t, "earnings")
                    continue
                capped_shares, used_cash = _cap_buy_shares_to_cash(
                    shares, px, buy_cash_left, env["buy_cost_multiplier"],
                )
                if capped_shares <= 0:
                    c["cash_exhausted"] += 1
                    stamp(t, "qp_cash_exhausted")
                    continue
                if capped_shares < shares:
                    c["cash_capped"] += 1
                    stamp(t, "qp_cash_capped")
                    shares = capped_shares
                buy_cash_left = max(0.0, buy_cash_left - used_cash)
                _emit_qp_buy(
                    ctx, t, shares, env["prices"].get(t, 0.0),
                    sol, i, env["score_sources"],
                )
                emitted_candidates.add(t)
                if t not in env["holdings_set"]:
                    env["emitted_new_tickers"].add(t)
                nb += 1
            else:
                pending_sell_shares[t] = float(shares)
                soft_block = _qp_soft_sell_block_reason(ctx, t, sol, i)
                pending_sell_shares.pop(t, None)
                if soft_block:
                    c["soft_sell_blocked"] += 1
                    stamp(t, soft_block)
                    log.info("QP_SELL_SUPPRESSED %-6s  Δw=%+.4f  %s",
                             t, dw, soft_block)
                    continue
                if _emit_qp_sell(ctx, t, shares, dw, sol, i):
                    ns += 1
                    credit = _long_sell_credit(env, t, shares, px)
                    if credit > 0.0:
                        buy_cash_left += credit
                        c["sell_credit_events"] += 1
                        log.info(
                            "QP_SELL_CREDIT %-6s  credited=$%.0f  "
                            "buy_cash_left=$%.0f",
                            t, credit, buy_cash_left,
                        )
                else:
                    stamp(t, "qp_no_sell_position")
        for ticker in candidate_tickers - emitted_candidates:
            if ticker not in blocked_map:
                c["not_selected"] += 1
                blocked_map[ticker] = "qp_not_selected"
        return nb, ns, c

    @staticmethod
    def _log_summary(
        *, n_blocked_buys, buy_blocked, n_blocked_earnings, earn_buf,
        n_defensive_non_bear,
        n_skipped_nonfinite, n_skipped_band, min_dw, no_trade_factor,
        n_delta_below_min_dw, n_zero_shares, n_no_buy_delta, n_not_selected,
        n_cash_capped, n_cash_exhausted, n_soft_sell_blocked,
        n_preexisting_exit, n_admission_blocked,
    ) -> None:
        if n_blocked_buys:
            reason = ("buy_blocked=True" if buy_blocked
                      else "skip_buys=True (drawdown halt)")
            log.info(
                "EmitOrdersFromQPSolutionTask: %s — suppressed %d QP top-ups",
                reason, n_blocked_buys,
            )
        if n_blocked_earnings:
            log.info(
                "EmitOrdersFromQPSolutionTask: suppressed %d top-ups within "
                "±%d earnings days", n_blocked_earnings, earn_buf,
            )
        if n_defensive_non_bear:
            log.info(
                "EmitOrdersFromQPSolutionTask: suppressed %d defensive "
                "QP buy/top-up(s) outside BEAR regime",
                n_defensive_non_bear,
            )
        if n_skipped_nonfinite:
            log.warning(
                "EmitOrdersFromQPSolutionTask: skipped %d non-finite Δw "
                "(investigate Σ conditioning)", n_skipped_nonfinite,
            )
        if n_skipped_band:
            log.info(
                "EmitOrdersFromQPSolutionTask: skipped %d trades by "
                "no-trade band (min_dw=%.2f%%, factor=%.1fσ — Davis-Norman)",
                n_skipped_band, min_dw * 100, no_trade_factor,
            )
        if n_delta_below_min_dw:
            log.info(
                "EmitOrdersFromQPSolutionTask: skipped %d trades below "
                "minimum Δw %.2f%%",
                n_delta_below_min_dw, min_dw * 100,
            )
        if n_zero_shares:
            log.info(
                "EmitOrdersFromQPSolutionTask: skipped %d candidate buy(s) "
                "because Δw rounded to 0 shares",
                n_zero_shares,
            )
        if n_no_buy_delta:
            log.info(
                "EmitOrdersFromQPSolutionTask: skipped %d candidate buy(s) "
                "because QP assigned no positive buy delta",
                n_no_buy_delta,
            )
        if n_not_selected:
            log.info(
                "EmitOrdersFromQPSolutionTask: %d candidate(s) received no "
                "QP allocation reason after solve",
                n_not_selected,
            )
        if n_cash_capped:
            log.info(
                "EmitOrdersFromQPSolutionTask: capped %d QP buy(s) to available cash",
                n_cash_capped,
            )
        if n_cash_exhausted:
            log.info(
                "EmitOrdersFromQPSolutionTask: skipped %d QP buy(s) because cash was exhausted",
                n_cash_exhausted,
            )
        if n_soft_sell_blocked:
            log.info(
                "EmitOrdersFromQPSolutionTask: suppressed %d QP soft sell(s) "
                "by horizon/LT/tax guards",
                n_soft_sell_blocked,
            )
        if n_preexisting_exit:
            log.info(
                "EmitOrdersFromQPSolutionTask: suppressed %d QP trade(s) "
                "for ticker(s) already carrying an exit intent",
                n_preexisting_exit,
            )
        if n_admission_blocked:
            log.info(
                "EmitOrdersFromQPSolutionTask: suppressed %d QP buy/top-up(s) "
                "by alpha-admission gate",
                n_admission_blocked,
            )


# ── helpers ────────────────────────────────────────────────────────────────

# Keys that support per-regime override (2026-05-16 B-track):
# Reading order:
#   regime_params.<ctx.regime>.<KEY>  →  rotation.joint_actions.<KEY>
# Pattern matches CLAUDE.md PRIME DIRECTIVE (regime-conditional strategy).
# Test pin: tests/test_qp_cfg_per_regime_override.py
_QP_PER_REGIME_KEYS = (
    "qp_cvar_lambda",
    "qp_cvar_alpha",
    "qp_turnover_max",
    "qp_risk_aversion",
    "qp_cost_kappa",
    "qp_cost_kappa_floor_round_trip",
    "qp_dw_max",
    "qp_min_dw_pct",
    "qp_no_trade_band_factor",
    "qp_no_trade_band_cap",
    "qp_min_invested_pct",
    "qp_cash_drag_lambda",
    "qp_min_invested_requires_positive_edge",
    "qp_min_invested_edge_floor",
    "qp_mu_horizon_days",
    "qp_sigma_unit",
    "qp_sigma_horizon_mode",
    "qp_horizon_contract",
    "qp_admission_gate",
    "qp_c2_infeasible_policy",
)


def _qp_cfg(ctx) -> dict:
    base = dict((ctx.config.get("rotation", {}).get("joint_actions", {})) or {})
    regime = getattr(ctx, "regime", None)
    if regime:
        regime_p = (ctx.config.get("regime_params", {}) or {}).get(regime, {}) or {}
        for key in _QP_PER_REGIME_KEYS:
            if key in regime_p:
                base[key] = regime_p[key]
    return base


def _resolve_qp_mu_horizon_days(ctx, cfg: dict) -> int | None:
    raw = cfg.get("qp_mu_horizon_days")
    if raw is None:
        raw = (ctx.config.get("panel_ltr", {}) or {}).get("lookahead_days")
    if raw is None:
        raw = ctx.config.get("lookahead_days")
    try:
        horizon = int(raw)
    except (TypeError, ValueError):
        return None
    return horizon if horizon > 0 else None


def _qp_sigma_horizon_scale(unit: str, horizon_days: int) -> float | None:
    if unit in {"horizon", "period", "matched"}:
        return 1.0
    if unit in {"annual", "annualized", "ann"}:
        return math.sqrt(float(horizon_days) / 252.0)
    if unit == "daily":
        return math.sqrt(float(horizon_days))
    return None


def _record_qp_horizon_issue(ctx, cfg: dict, reason: str) -> bool | None:
    contract = str(cfg.get("qp_horizon_contract", cfg.get("qp_mu_contract", "warn"))).lower()
    report = {"ok": False, "reason": reason}
    ctx._qp_horizon_contract = report  # noqa: SLF001
    counters = getattr(ctx, "counters", None)
    if counters is not None:
        key = "qp_horizon_contract_block" if contract == "strict" else "qp_horizon_contract_warn"
        counters[key] = counters.get(key, 0) + 1
    log.warning("AlignQPHorizonUnitsTask: %s", reason)
    return False if contract == "strict" else None


def _effective_min_invested_pct(ctx, cfg: dict) -> float:
    base = float(cfg.get("qp_min_invested_pct", 0.0) or 0.0)
    if base <= 0.0 or not bool(cfg.get("qp_min_invested_requires_positive_edge", False)):
        return base
    mu = np.asarray(_get_path(ctx, "_qp_mu"), dtype=float)
    finite = mu[np.isfinite(mu)]
    best_mu = float(np.max(finite)) if finite.size else float("-inf")
    floor = float(cfg.get("qp_min_invested_edge_floor", _round_trip_cost(cfg)))
    blocked = best_mu <= floor
    ctx._qp_min_invested_contract = {  # noqa: SLF001
        "base": base, "effective": 0.0 if blocked else base,
        "best_mu": best_mu, "edge_floor": floor, "blocked": blocked,
    }
    return 0.0 if blocked else base


def _round_trip_cost(cfg: dict) -> float:
    fee = float(cfg.get("fee_pct", cfg.get("qp_cost_kappa", 0.0)) or 0.0)
    slip = float(cfg.get("slippage_pct", 0.0) or 0.0)
    return 2.0 * (fee + slip)


def _effective_qp_cost_kappa(cfg: dict) -> float:
    """L1 turnover penalty used by the QP objective.

    Gârleanu-Pedersen 2013 shows proportional transaction costs create a
    no-trade region around the current portfolio. In this single-period QP,
    the convex proxy is the L1 turnover penalty. When the floor flag is on,
    stale configs cannot underprice trading below explicit fee+slippage.
    """
    raw = float(cfg.get("qp_cost_kappa", cfg.get("fee_pct", 0.0005)) or 0.0)
    if bool(cfg.get("qp_cost_kappa_floor_round_trip", False)):
        return max(raw, _round_trip_cost(cfg))
    return raw


def _qp_soft_sell_tax_gates_enabled(cfg: dict, guard_cfg: object) -> bool:
    """Return whether QP order emission may suppress sells for tax reasons."""
    if isinstance(guard_cfg, dict) and "apply_tax_gates" in guard_cfg:
        return bool(guard_cfg.get("apply_tax_gates"))
    return bool(cfg.get("qp_tax_aware", False))


def _compute_qp_wash_mask(
    *,
    tickers: list[str],
    today,
    last_sell_dates: dict,
    last_sell_pls: dict,
    wash_days: int,
    min_reentry: int,
    held_tickers: set[str],
    calibrator_saturated: bool,
) -> tuple[np.ndarray, int, int, int]:
    """Build QP block mask for wash-sale, anti-churn, and saturation abstain."""
    from kernel.selection import is_wash_sale_blocked_with_cost  # noqa: PLC0415
    mask = np.zeros(len(tickers), dtype=bool)
    n_wash = n_churn = n_sat = 0
    for i, t in enumerate(tickers):
        if wash_days > 0:
            blocked, _, _ = is_wash_sale_blocked_with_cost(
                ticker=t,
                today=today,
                last_sell_dates=last_sell_dates,
                last_sell_pls=last_sell_pls,
                wash_sale_days=wash_days,
            )
            if blocked:
                mask[i] = True
                n_wash += 1
                continue
        if min_reentry > 0:
            last = last_sell_dates.get(t)
            if last is not None:
                if isinstance(last, str):
                    try:
                        last = _dt.date.fromisoformat(last[:10])
                    except (ValueError, TypeError):
                        continue
                days_since = (today - last).days
                if 0 <= days_since < min_reentry:
                    mask[i] = True
                    n_churn += 1
                    continue
        if calibrator_saturated and t not in held_tickers:
            mask[i] = True
            n_sat += 1
    return mask, n_wash, n_churn, n_sat


def _per_asset_tax(hs, price, w_i, nav, today, st_rate, lt_rate,
                    lt_days, bridge_w, offset_left) -> tuple[float, float]:
    """Brown-Smith dynamic tax + Berkin-Jeffrey loss-harvest credit (legacy).

    Uses a single average entry_price/entry_date — kept for back-compat
    when `qp_tax_lot_method == "avg"`. Lot-aware path is `_per_asset_tax_lots`.
    """
    entry_p = float(getattr(hs, "entry_price", 0.0) or 0.0)
    entry_d = getattr(hs, "entry_date", None)
    if entry_p <= 0 or entry_d is None or price <= 0:
        return 0.0, offset_left
    gain = (price - entry_p) / entry_p
    try:
        days_held = (today - entry_d).days
    except Exception:
        days_held = 0
    if gain > 0:
        if days_held >= lt_days:
            return gain * lt_rate, offset_left
        days_to_lt = max(0, lt_days - days_held)
        if days_to_lt <= bridge_w:
            rate = lt_rate + (st_rate - lt_rate) * (
                days_to_lt / max(1, bridge_w)
            )
            return gain * rate, offset_left
        return gain * st_rate, offset_left
    if gain < 0 and offset_left > 0:
        est_loss = w_i * abs(gain) * nav
        used = min(est_loss, offset_left)
        if used > 0:
            savings = used * st_rate
            cost = -(savings / max(nav, 1.0) / max(w_i, 1e-6))
            return cost, offset_left - used
    return 0.0, offset_left


def _bridge_rate(st_rate, lt_rate, lt_days, days_held, bridge_w):
    """ST/LT bridge: between (lt_days - bridge_w) and lt_days, rate
    decays linearly from ST toward LT. Outside the bridge: pure ST or LT.
    """
    if days_held >= lt_days:
        return lt_rate
    days_to_lt = max(0, lt_days - days_held)
    if days_to_lt <= bridge_w:
        return lt_rate + (st_rate - lt_rate) * (
            days_to_lt / max(1, bridge_w)
        )
    return st_rate


def _per_asset_tax_lots(hs, price, w_i, nav, today, st_rate, lt_rate,
                         lt_days, bridge_w, offset_left, lot_method
                         ) -> tuple[float, float]:
    """Lot-aware Brown-Smith tax cost.

    Iterates `hs.lots` in disposal order (HIFO → highest-cost lot first
    minimises realized gain; FIFO → oldest first, broker default), and
    accumulates dollar tax across the lots that would be touched to fund
    a 1-NAV-fraction sell of asset i. Returns (cost_per_unit_w, offset_left).

    Loss harvest: same Berkin-Jeffrey credit as legacy — when a lot has
    gain_per_share < 0 AND offset_left > 0, the harvested loss reduces
    `offset_left` and credits a NEGATIVE cost component (savings).
    """
    from kernel.exits import ensure_lots
    if hs is None or price <= 0 or w_i <= 0:
        return 0.0, offset_left
    ensure_lots(hs)
    lots = hs.lots or []
    if not lots:
        return 0.0, offset_left
    method = (lot_method or "fifo").lower()
    if method == "hifo":
        order = sorted(lots, key=lambda L: -L.price)
    else:   # FIFO — preserve insertion order (older first)
        order = list(lots)
    target_shares = (w_i * nav) / max(price, 1e-9)
    cost_dollar = 0.0
    consumed = 0.0
    for L in order:
        if consumed >= target_shares - 1e-12:
            break
        take = min(float(L.shares), target_shares - consumed)
        if take <= 0:
            continue
        gain_per_share = price - float(L.price)
        try:
            held_days = (today - L.date).days
        except Exception:
            held_days = 0
        if gain_per_share > 0:
            rate = _bridge_rate(st_rate, lt_rate, lt_days, held_days, bridge_w)
            cost_dollar += take * gain_per_share * rate
        elif gain_per_share < 0 and offset_left > 0:
            harvest = take * abs(gain_per_share)
            used = min(harvest, offset_left)
            if used > 0:
                cost_dollar += -used * st_rate          # savings (negative)
                offset_left -= used
        consumed += take
    if not math.isfinite(cost_dollar):
        return 0.0, offset_left
    cost_per_unit_w = cost_dollar / max(w_i * nav, 1.0)
    return cost_per_unit_w, offset_left


def _emit_qp_buy(ctx, ticker, shares, px, sol, i, score_sources):
    cand = score_sources.get(ticker)
    actual_target_pct = _actual_qp_buy_target_pct(ctx, ticker, shares, px)
    qp_mu_used = _array_float_at(_get_path(ctx, "_qp_mu"), i)
    qp_sigma_used = _array_float_at(_get_path(ctx, "_qp_sigma"), i)
    qp_mu_source = str(
        getattr(
            ctx,
            "_qp_forced_mu_source",
            (ctx.config or {}).get("ranking", {}).get("qp_mu_source", "mu"),
        )
    )
    ctx.orders.append(stamp_order_attribution({
        "ticker": ticker, "shares": shares, "price": px,
        "invest": shares * px,
        "target_pct": actual_target_pct,
        "regime": getattr(ctx, "regime", None),
        "confidence": getattr(ctx, "confidence", None),
        "rank_score": getattr(cand, "rank_score", None),
        "rs_score": getattr(cand, "rs_score", None),
        "panel_score": getattr(cand, "panel_score", None),
        "mu": getattr(cand, "mu", None),
        "sigma": getattr(cand, "sigma", None),
        "kelly_target_pct": getattr(cand, "kelly_target_pct", None),
        "detail": getattr(cand, "detail", ""),
        "order_type": "QP_BUY",
        "source": "qp",
    }, ctx=ctx, source_job="JointPortfolioQPJob",
        source_task="EmitOrdersFromQPSolutionTask",
        acceptance_reason="qp_target_weight_increase",
        source_obj=cand,
        decision_inputs={
            "delta_w": float(sol.delta_w[i]),
            "target_w": float(sol.target_w[i]),
            "actual_target_w": float(actual_target_pct),
            "solver_status": getattr(sol, "status", None),
            "expected_return_horizon_days": getattr(
                cand, "expected_return_horizon_days", None,
            ),
            "mu_horizon_days": getattr(cand, "mu_horizon_days", None),
            "qp_mu_used": qp_mu_used,
            "qp_sigma_used": qp_sigma_used,
            "qp_mu_source": qp_mu_source,
            "alpha_to_mu_applied": bool(getattr(ctx, "_qp_mu_transformed", False)),
        }))
    log.info("QP_BUY  %-6s  Δw=%+.4f  shares=%d  px=%.2f  invest=$%.0f",
             ticker, float(sol.delta_w[i]), shares, px, shares * px)


def _qp_solver_decision_inputs(ctx, sol, i) -> dict:
    qp_mu_source = str(
        getattr(
            ctx,
            "_qp_forced_mu_source",
            (ctx.config or {}).get("ranking", {}).get("qp_mu_source", "mu"),
        )
    )
    return {
        "solver_status": getattr(sol, "status", None),
        "qp_mu_used": _array_float_at(_get_path(ctx, "_qp_mu"), i),
        "qp_sigma_used": _array_float_at(_get_path(ctx, "_qp_sigma"), i),
        "qp_mu_source": qp_mu_source,
        "alpha_to_mu_applied": bool(getattr(ctx, "_qp_mu_transformed", False)),
    }


def _holding_score_snapshot(hs) -> dict:
    return {
        "rank_score": getattr(hs, "rank_score", None),
        "rs_score": getattr(hs, "rs_score", None),
        "panel_score": getattr(hs, "panel_score", None),
        "mu": getattr(hs, "mu", None),
        "mu_horizon_days": getattr(hs, "mu_horizon_days", None),
        "sigma": getattr(hs, "sigma", None),
        "expected_return": getattr(hs, "expected_return", None),
        "expected_return_horizon_days": getattr(
            hs, "expected_return_horizon_days", None,
        ),
        "kelly_target_pct": getattr(hs, "kelly_target_pct", None),
        "model_type": getattr(hs, "model_type", None),
        "sector": getattr(hs, "sector", None),
    }


def _qp_sell_decision_inputs(ctx, ticker, qty, held, dw, target_w, sol, i, hs) -> dict:
    inputs = {
        "delta_w": float(dw),
        "target_w": float(target_w),
        "shares": float(qty),
        "held_shares": float(held),
        "expected_return_horizon_days": getattr(
            hs, "expected_return_horizon_days", None,
        ),
        "mu_horizon_days": getattr(hs, "mu_horizon_days", None),
    }
    inputs.update(_qp_solver_decision_inputs(ctx, sol, i))
    inputs.update(_holding_score_snapshot(hs))
    return inputs


def _emit_qp_sell(ctx, ticker, shares, dw, sol, i) -> bool:
    """Emit SELL signal, including SHORT-OPEN when target_w < 0.

    Three cases:
    1. Closing a long (current shares > 0, target_w ≥ 0): emit qp_sell
       up to held shares, capped at held.
    2. Closing-and-flipping a long to short (current > 0, target_w < 0):
       emit qp_close for the full long portion (held shares), THEN
       emit qp_short_open for the remaining magnitude needed to reach
       target_w.
    3. Opening fresh short (current = 0 or None, target_w < 0): emit
       qp_short_open with magnitude |shares|.

    Phase 2A wiring fix (2026-05-14): pre-fix this function bailed when
    holdings.get(ticker) was None or when qty went negative, so even
    when the QP requested negative target weights, no short orders were
    ever generated. Sim and live ran long-only regardless.
    """
    from kernel.exits import ExitSignal
    target_w = float(sol.target_w[i])
    hs = (ctx.holdings or {}).get(ticker)
    source_obj = (getattr(ctx, "_qp_mu_source_map", None) or {}).get(ticker, hs)
    held = int(getattr(hs, "shares", 0) or 0) if hs is not None else 0
    requested = int(shares)  # always positive; sign comes from target_w

    # Case A: target ≥ 0 → just close-down/no-op of existing long
    if target_w >= -1e-9:
        if held <= 0:
            return False
        qty = min(requested, held)
        if qty <= 0:
            return False
        exit_type = "qp_sell" if target_w > 1e-4 else "qp_close"
        ctx.exits.append((ticker, ExitSignal(
            should_exit=True, exit_type=exit_type,
            quantity=float(qty), reason=f"qp_dw={dw:+.4f}",
        )))
        sig = ctx.exits[-1][1]
        sig.source_job = "JointPortfolioQPJob"
        sig.source_task = "EmitOrdersFromQPSolutionTask"
        sig.decision_inputs = _qp_sell_decision_inputs(
            ctx, ticker, qty, held, dw, target_w, sol, i, source_obj,
        )
        log.info("QP_SELL %-6s  Δw=%+.4f  shares=%d  reason=%s",
                 ticker, dw, qty, exit_type)
        return True

    # Case B/C: target_w < 0 → final position is short.
    # Total |Δshares| comes from QP's |delta_w[i]| × NAV / price, which
    # caller already converted to `shares`. We split into close-long
    # and short-open portions.
    long_close = min(held, requested) if held > 0 else 0
    short_open = max(0, requested - long_close)

    emitted = False
    if long_close > 0:
        ctx.exits.append((ticker, ExitSignal(
            should_exit=True, exit_type="qp_close",
            quantity=float(long_close), reason=f"qp_dw={dw:+.4f}",
        )))
        sig = ctx.exits[-1][1]
        sig.source_job = "JointPortfolioQPJob"
        sig.source_task = "EmitOrdersFromQPSolutionTask"
        sig.decision_inputs = _qp_sell_decision_inputs(
            ctx, ticker, long_close, held, dw, target_w, sol, i, source_obj,
        )
        log.info("QP_SELL %-6s  Δw=%+.4f  shares=%d  reason=qp_close",
                 ticker, dw, long_close)
        emitted = True
    if short_open > 0:
        # Append a SHORT-OPEN order. SimAdapter.commit reads ctx.orders
        # for buys; for shorts we use ctx.exits with a special exit_type
        # so the downstream consumer can route to a short-open code path
        # in _apply_sell when shares > held.
        ctx.exits.append((ticker, ExitSignal(
            should_exit=True, exit_type="qp_short_open",
            quantity=float(short_open), reason=f"qp_dw={dw:+.4f} target_w={target_w:+.4f}",
        )))
        sig = ctx.exits[-1][1]
        sig.source_job = "JointPortfolioQPJob"
        sig.source_task = "EmitOrdersFromQPSolutionTask"
        sig.decision_inputs = _qp_sell_decision_inputs(
            ctx, ticker, short_open, held, dw, target_w, sol, i, source_obj,
        )
        log.info("QP_SHORT_OPEN %-6s  Δw=%+.4f  shares=%d  target_w=%+.4f",
                 ticker, dw, short_open, target_w)
        emitted = True
    return emitted


__all__ = [
    "BuildWeightVectorTask",
    "ComputeFullSigmaTask",
    "ShrinkSigmaLedoitWolfTask",
    "ComputeBrownSmithTaxCostTask",
    "ComputeWashSaleMaskTask",
    "BuildADVVectorTask",
    "ComputeQPConstraintsTask",
    "ApplySectorMetadataGuardTask",
    "ApplyConvictionCapTask",
    "BuildSectorConstraintMatrixTask",
    "BuildCorrelationGroupConstraintTask",
    "SolveMarkowitzQPTask",
    "EmitOrdersFromQPSolutionTask",
]
