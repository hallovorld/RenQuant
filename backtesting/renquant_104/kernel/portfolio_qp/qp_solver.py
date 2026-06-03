"""Portfolio QP solver — cvxpy + CLARABEL (Boyd/Stanford cvxportfolio idiom).

Solves the single-period mean-variance optimization with:

    maximize     μᵀwp
                 - γ_risk · wpᵀΣwp                    (risk)
                 - cvar_λ · z_α/√α · ‖Σ^½ wp‖₂        (CVaR tail penalty, RU2002)
                 - κ · ‖Δw‖₁                          (linear transaction cost)
                 - Σᵢ tax_i · max(0, -Δwᵢ)            (Brown-Smith tax-aware sells)
                 - b · Σᵢ σᵢ · sqrt(NAV/Vᵢ) · |Δwᵢ|^1.5  (Almgren-Chriss impact)
                 - λ_cash · max(0, target_invested - Σwp)  (SOFT cash-drag penalty)

    subject to   Σwp ≤ 1 - cash_reserve                (budget, hard)
                 w_lower ≤ wp ≤ w_upper                (per-asset cap, hard)
                 -dw_max ≤ Δw ≤ dw_max                 (per-bar slippage, hard)
                 Δwᵢ ≤ 0 ∀ i ∈ wash_sale_mask          (wash-sale, hard)
                 ‖Δw‖₁ ≤ τ_max                         (turnover cap, hard)

The ONLY hard constraints are physics (budget, box bounds, slippage cap,
wash-sale, turnover). All preferences (cash-drag, tax, impact, CVaR, robust μ,
drawdown scaler) are SOFT terms in the objective. This is the Boyd /
cvxportfolio textbook formulation: an over-determined hard-constraint set is
infeasible; a soft-penalty objective is always solvable and the trade-off
is exposed via the penalty coefficients.

References (read prior to design — CLAUDE.md §5.12, §5.12a):
  - Boyd & Vandenberghe 2004 §10.4 — interior-point convex QP
  - Markowitz 1952 — mean-variance portfolio selection
  - Garleanu-Pedersen 2013 — dynamic trading with predictable returns + costs
  - Almgren-Chriss 2000 §2 — sqrt-impact transaction cost
  - Rockafellar-Uryasev 2002 — CVaR (tail risk) closed form
  - Garlappi-Uppal-Wang 2007 — robust μ subtraction
  - Berkin-Jefferey 1990 — after-tax portfolio optimization
  - cvxportfolio 1.5 (Boyd/Stanford) — `SinglePeriodOpt` reference impl

Solver chain: CLARABEL (primary, interior point) → OSQP (alternative IP) →
SCS (large-scale fallback). All three are convex-QP-optimal; the chain
exists to maximize success probability across solver-specific edge cases.
`optimal_inaccurate` is rejected by default; pass the explicit diagnostic
flag only for local analysis.

Status semantics: `optimal` (clean solve), `optimal_no_signal` (μ ≈ 0 →
solver returned Δw ≈ 0; valid but caller should fall through to a Kelly
default), `infeasible` (constraints contradict — diagnostic in log).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import cvxpy as cp

log = logging.getLogger("kernel.portfolio_qp.qp_solver")


@dataclass
class QPSolution:
    """Output of solve_portfolio_qp."""

    delta_w:        np.ndarray            # n-vector of weight changes
    target_w:       np.ndarray            # n-vector of post-trade weights
    objective:      float                  # final objective value (max form)
    n_iter:         int                    # solver iterations (-1 if unknown)
    status:         str                    # "optimal" / "optimal_no_signal" / "infeasible"
    diagnostics:    dict                   # solver internals + binding hints


def _solve_cvx(
    prob,
    primary,
    fallbacks,
    *,
    verbose: bool = False,
    allow_optimal_inaccurate: bool = False,
) -> str:
    """Solve a cvxpy problem with a primary solver + ordered fallbacks.

    Returns the final `prob.status`. Catches solver exceptions per attempt;
    last attempt's status (or "exception") is returned. The cvxportfolio
    pattern: never let one solver's quirks define correctness.
    """
    chain: list = [primary] + list(fallbacks)
    last_status = "exception"
    for solver in chain:
        try:
            prob.solve(solver=solver, verbose=verbose)
            last_status = prob.status
            if last_status == "optimal":
                return last_status
            if last_status == "optimal_inaccurate":
                if allow_optimal_inaccurate:
                    return last_status
                log.warning(
                    "QP solver %s returned optimal_inaccurate; strict status "
                    "policy rejects it and tries the next solver",
                    solver,
                )
        except (cp.error.SolverError, Exception) as exc:  # noqa: BLE001
            last_status = f"exception:{type(exc).__name__}"
            log.debug("QP solver %s raised %s; trying next", solver, exc)
    return last_status


def solve_portfolio_qp_from_snapshot(
    snap,
    *,
    mu: Sequence[float],
    sigma: Sequence[float] | None = None,
    Sigma: np.ndarray | None = None,
    risk_aversion: float = 3.0,
    cost_kappa: float = 0.0001,
    signal_decay: float = 0.0,
    robust_mu_kappa: float = 0.0,
    cvar_lambda: float = 0.0,
    cvar_alpha: float = 0.05,
    tax_cost_per_sell: Sequence[float] | None = None,
    impact_coef: float = 0.0,
    v_daily_dollar: Sequence[float] | None = None,
    nav_dollar: float = 0.0,
    fixed_cost_per_trade: float = 0.0,
    fixed_cost_beta: float = 100.0,
    budget_mode: str = "inequality",
    min_invested_pct: float = 0.0,
    cash_drag_lambda: float = 0.05,
    allow_optimal_inaccurate: bool = False,
) -> "QPSolution":
    """Solve the QP from a :class:`ConstraintSnapshot` + forecast / cost kwargs.

    This is the contract entry point introduced as Step 2 of the §8 plan
    (PR #125). Strictly delegates to :func:`solve_portfolio_qp` by
    unpacking the snapshot's hard-constraint fields into the matching
    kwargs — no behaviour change. The pre-existing entry point stays
    untouched; callers migrate one at a time.

    Why this exists: the snapshot is the immutable contract every
    candidate allocator (current QP, simplified-QP, Hybrid, MPO, …)
    consumes. Routing the solver through it removes the kwargs-shaped
    surface where #123 v1/v2/v3 hid their bug (the soft cap > hard cap
    state never reaches this wrapper because the snapshot constructor
    rejects it at build time).
    """
    return solve_portfolio_qp(
        # Constraint fields from the snapshot — these are the kwargs
        # the snapshot owns.
        w_current=snap.w_current,
        w_upper=snap.w_upper,
        w_lower=snap.w_lower,
        dw_max=snap.dw_max,
        cash_reserve=snap.cash_reserve,
        wash_sale_mask=snap.wash_sale_mask,
        drawdown=snap.drawdown,
        drawdown_limit=snap.drawdown_limit,
        turnover_max=snap.turnover_max,
        gross_max=snap.gross_max,
        sector_indicator=snap.sector_indicator,
        sector_cap_vec=snap.sector_cap_vec,
        corr_group_pairs=tuple(snap.corr_group_pairs) or None,
        # Forecast + cost kwargs the snapshot does NOT own.
        mu=mu,
        sigma=sigma,
        Sigma=Sigma,
        risk_aversion=risk_aversion,
        cost_kappa=cost_kappa,
        signal_decay=signal_decay,
        robust_mu_kappa=robust_mu_kappa,
        cvar_lambda=cvar_lambda,
        cvar_alpha=cvar_alpha,
        tax_cost_per_sell=tax_cost_per_sell,
        impact_coef=impact_coef,
        v_daily_dollar=v_daily_dollar,
        nav_dollar=nav_dollar,
        fixed_cost_per_trade=fixed_cost_per_trade,
        fixed_cost_beta=fixed_cost_beta,
        budget_mode=budget_mode,
        min_invested_pct=min_invested_pct,
        cash_drag_lambda=cash_drag_lambda,
        allow_optimal_inaccurate=allow_optimal_inaccurate,
    )


def solve_portfolio_qp(
    *,
    w_current:      Sequence[float],
    mu:             Sequence[float],
    sigma:          Sequence[float] | None = None,
    Sigma:          np.ndarray | None = None,
    risk_aversion:  float = 3.0,
    cost_kappa:     float = 0.0001,
    cash_reserve:   float = 0.0,
    w_upper:        Sequence[float] | float = 0.20,
    w_lower:        Sequence[float] | float = 0.0,
    dw_max:         Sequence[float] | float = 0.50,
    wash_sale_mask: Sequence[bool] | None = None,
    signal_decay:   float = 0.0,
    drawdown:       float = 0.0,
    drawdown_limit: float = 0.20,
    robust_mu_kappa: float = 0.0,
    cvar_lambda:    float = 0.0,
    cvar_alpha:     float = 0.05,
    tax_cost_per_sell: Sequence[float] | None = None,
    turnover_max:   float | None = None,
    impact_coef:    float = 0.0,
    v_daily_dollar: Sequence[float] | None = None,
    nav_dollar:     float = 0.0,
    fixed_cost_per_trade: float = 0.0,    # legacy kwarg, ignored (not DCP-compliant)
    fixed_cost_beta:      float = 100.0,  # legacy kwarg, ignored
    budget_mode: str = "inequality",      # legacy kwarg, treated as "≤" always
    min_invested_pct:     float = 0.0,    # SOFT target now; was hard floor pre-2026-05-06
    cash_drag_lambda:     float = 0.05,   # NEW: penalty coefficient on cash-drag
    gross_max:            float | None = None,  # Long-Short Phase 2A: cap Σ|wp| (set when shorts enabled)
    allow_optimal_inaccurate: bool = False,
    # ── 2026-05-10 industrial-grade constraints (Track C2) ───────────────
    # Sector cap as hard linear constraint: S @ wp ≤ sector_cap_vec.
    # `sector_indicator` is an m × n indicator matrix (m sectors), `sector_cap_vec`
    # is the per-sector weight cap. None / empty → constraint omitted.
    # On infeasibility, caller (BuildSectorConstraintMatrixTask) detects the
    # `infeasible:sector` status and re-solves with relaxed caps.
    sector_indicator: np.ndarray | None = None,
    sector_cap_vec:   Sequence[float] | None = None,
    # Correlation group cap: list of (i, j, group_cap) tuples for pairs whose
    # |corr| ≥ correlation_guard_threshold. Adds wp[i] + wp[j] ≤ group_cap
    # (linear approximation of pair non-convex `wp[i] · wp[j] ≤ pair_cap`).
    # Reference: Boyd & Vandenberghe 2004 §4.4 (linear group bounds).
    corr_group_pairs: Sequence[tuple[int, int, float]] | None = None,
) -> QPSolution:
    """Convex Markowitz QP via cvxpy + CLARABEL (cvxportfolio idiom).

    Replaces the 2026-04 SLSQP implementation. Drops SLSQP entirely after
    the V4/V5 alpha158_linear sim demonstrated SLSQP failed every bar with
    numerical errors (LSQ subproblem overflow, degenerate warm-starts) and
    the cvxpy fallback was firing too rarely.

    `min_invested_pct` is now a SOFT target driving a cash-drag penalty
    `cash_drag_lambda · max(0, target - Σwp)`. The previous HARD floor was
    structurally infeasible whenever sum(per-asset hi-bounds) +
    turnover_max < min_invested_pct (a typical from-cash situation with
    confidence-multiplier-tight caps). cvxportfolio's reference policy
    (`SinglePeriodOpt`) follows the same soft-penalty pattern.

    Tuning `cash_drag_lambda`:
      - 0.0 → no cash-drag preference (legacy `min_invested_pct=0` parity)
      - 0.05 (default) → moderate push to deploy; signal of ~50bp net wins
      - 0.50 → aggressive deployment; only strong negative signal stays cash
    """
    w_current = np.asarray(w_current, dtype=float)
    mu_raw    = np.asarray(mu,        dtype=float)
    n         = len(w_current)
    if len(mu_raw) != n:
        raise ValueError(f"len(mu)={len(mu_raw)} != len(w_current)={n}")

    # ── Σ resolution (full or diagonal) ───────────────────────────────────
    if Sigma is None:
        if sigma is None:
            raise ValueError("must provide either Sigma (n×n) or sigma (n-vector)")
        sigma_arr = np.asarray(sigma, dtype=float)
        if len(sigma_arr) != n:
            raise ValueError(f"len(sigma)={len(sigma_arr)} != n={n}")
        sigma_arr = np.where(np.isfinite(sigma_arr), sigma_arr, 0.05)
        sigma_arr = np.clip(sigma_arr, 1e-6, None)
        Sigma_mat = np.diag(sigma_arr ** 2)
    else:
        Sigma_mat = np.asarray(Sigma, dtype=float)
        if Sigma_mat.shape != (n, n):
            raise ValueError(f"Sigma shape {Sigma_mat.shape} != (n={n}, n={n})")
        if not np.isfinite(Sigma_mat).all():
            n_bad = int(np.sum(~np.isfinite(Sigma_mat)))
            log.warning("QP: Σ has %d non-finite cells — sanitising", n_bad)
            Sigma_mat = np.where(np.isfinite(Sigma_mat), Sigma_mat, 0.0)
            Sigma_mat += 1e-8 * np.eye(n)

    # ── Per-asset bound vectors ───────────────────────────────────────────
    # NOTE: solver treats `w_upper` as a HARD risk cap (see solver
    # contract docstring above). If `w_current > w_upper` (over-cap
    # holding), the solver MUST keep this infeasible so
    # `SolveMarkowitzQPTask._retry_for_per_asset_cap_compliance()` can
    # remediate via the cap-compliance retry path. The hold-flat-
    # feasibility clamp that addresses today's daily-104 bug lives in
    # the SOFT-scaling tasks (ApplyExposureScalingTask +
    # ApplyConvictionCapTask) — see CLAUDE.md daily-full memo. Hard caps
    # (max_position_pct, sector cap, corr cap) stay as hard constraints
    # here.
    w_upper_arr = (np.full(n, float(w_upper)) if np.isscalar(w_upper)
                    else np.asarray(w_upper, dtype=float))
    w_lower_arr = (np.full(n, float(w_lower)) if np.isscalar(w_lower)
                    else np.asarray(w_lower, dtype=float))
    dw_max_arr  = (np.full(n, float(dw_max))  if np.isscalar(dw_max)
                    else np.asarray(dw_max, dtype=float))

    # ── μ cleanup + Garleanu-Pedersen 2013 signal-decay scaling ──────────
    finite_mu = np.isfinite(mu_raw)
    mu_clean  = np.where(finite_mu, mu_raw, 0.0)
    sd = float(signal_decay)
    if sd > 0.0:
        sd = min(sd, 0.99)
        mu_clean = mu_clean * (1.0 / (1.0 - sd))

    # ── Garlappi-Uppal-Wang 2007 robust μ adjustment ─────────────────────
    if robust_mu_kappa != 0.0:
        sigma_diag = np.sqrt(np.maximum(np.diag(Sigma_mat), 0.0))
        mu_clean = mu_clean - float(robust_mu_kappa) * sigma_diag

    # ── Grossman-Zhou 1993 drawdown scaler on γ ──────────────────────────
    dd        = float(max(0.0, drawdown))
    dd_lim    = float(max(1e-6, drawdown_limit))
    dd_factor = max(1e-3, 1.0 - dd / dd_lim)
    gamma_eff = float(risk_aversion) / dd_factor

    # ── Tax-cost vector (Brown-Smith) ─────────────────────────────────────
    if tax_cost_per_sell is not None:
        tax_arr = np.asarray(tax_cost_per_sell, dtype=float)
        if len(tax_arr) != n:
            raise ValueError(f"tax_cost_per_sell length {len(tax_arr)} != n={n}")
        tax_arr = np.where(np.isfinite(tax_arr), tax_arr, 0.0)
    else:
        tax_arr = np.zeros(n)

    # ── Almgren-Chriss 2000 sqrt-impact coefficients ─────────────────────
    b_impact = float(max(0.0, impact_coef))
    if (b_impact > 0.0 and v_daily_dollar is not None
            and float(nav_dollar) > 0.0):
        v_arr = np.asarray(v_daily_dollar, dtype=float)
        if len(v_arr) != n:
            raise ValueError(f"v_daily_dollar length {len(v_arr)} != n={n}")
        v_safe = np.where((np.isfinite(v_arr)) & (v_arr > 0.0), v_arr, np.inf)
        sigma_diag_g3   = np.sqrt(np.maximum(np.diag(Sigma_mat), 0.0))
        impact_coef_arr = b_impact * sigma_diag_g3 * np.sqrt(
            float(nav_dollar) / v_safe,
        )
        impact_coef_arr = np.where(
            np.isfinite(impact_coef_arr), impact_coef_arr, 0.0,
        )
    else:
        impact_coef_arr = np.zeros(n)

    # ── cvxpy decision variable + post-trade weight expression ───────────
    dw = cp.Variable(n)
    wp = w_current + dw

    # ── HARD constraints (physics only) ──────────────────────────────────
    constraints = [
        cp.sum(wp) <= 1.0 - float(cash_reserve),     # budget upper
        wp >= w_lower_arr,                            # per-asset floor
        wp <= w_upper_arr,                            # per-asset cap
        dw >= -dw_max_arr,                            # slippage band lo
        dw <= dw_max_arr,                             # slippage band hi
    ]
    if wash_sale_mask is not None:
        wsm = np.asarray(wash_sale_mask, dtype=bool)
        # Δwᵢ ≤ 0 ∀ i ∈ wash_sale_mask  → cannot re-buy
        if wsm.any():
            constraints.append(dw[wsm] <= 0.0)
    if turnover_max is not None and float(turnover_max) > 0.0:
        constraints.append(cp.norm(dw, 1) <= float(turnover_max))
    # ── Long-Short Phase 2A: gross-exposure cap (Σ|wp| ≤ gross_max) ──────
    # When shorts are enabled (w_lower < 0), without this constraint the
    # optimizer can naively select max long AND max short on every name,
    # blowing gross beyond Reg-T 150% limit. cvxpy norm(wp, 1) is convex
    # so feasible under CLARABEL. Skip when gross_max is None (long-only
    # path; sum(wp) <= 1 bound is sufficient).
    if gross_max is not None and float(gross_max) > 0.0:
        constraints.append(cp.norm(wp, 1) <= float(gross_max))

    # ── Sector cap (hard linear): S @ wp ≤ sector_cap_vec ────────────────
    # Reference: Garleanu-Pedersen 2013 §3.2 budget-with-group-bounds; same
    # form as cvxportfolio's `MaxWeightsAtSectors`. NaN-safe: rows / caps
    # with non-finite entries are dropped.
    n_sector_rows = 0
    if sector_indicator is not None and sector_cap_vec is not None:
        S = np.asarray(sector_indicator, dtype=float)
        cap_v = np.asarray(sector_cap_vec, dtype=float)
        if S.size and cap_v.size:
            if S.ndim != 2 or S.shape[1] != n or S.shape[0] != cap_v.shape[0]:
                raise ValueError(
                    f"sector_indicator shape {S.shape} incompatible "
                    f"with n={n} and cap_vec len={cap_v.shape[0]}",
                )
            finite_caps = np.isfinite(cap_v) & (cap_v >= 0.0)
            finite_rows = np.isfinite(S).all(axis=1)
            keep = finite_caps & finite_rows
            if keep.any():
                S_keep = S[keep]
                cap_keep = cap_v[keep]
                n_sector_rows = int(S_keep.shape[0])
                constraints.append(S_keep @ wp <= cap_keep)

    # ── Correlation group cap (hard linear): wp[i] + wp[j] ≤ group_cap ───
    # Linearization of non-convex pair-product cap. Catches the case where
    # two highly-correlated holdings together exceed the diversification
    # budget for the group. Reference: Boyd & Vandenberghe 2004 §4.4.
    n_corr_pairs = 0
    if corr_group_pairs:
        for triple in corr_group_pairs:
            try:
                i_idx, j_idx, gcap = triple
                ii = int(i_idx); jj = int(j_idx); gc = float(gcap)
            except (TypeError, ValueError, IndexError):
                continue
            if not math.isfinite(gc) or gc < 0:
                continue
            if not (0 <= ii < n and 0 <= jj < n) or ii == jj:
                continue
            constraints.append(wp[ii] + wp[jj] <= gc)
            n_corr_pairs += 1

    # ── Objective (maximize utility) ──────────────────────────────────────
    # Σ_psd_wrap protects against tiny negative eigenvalues from finite
    # precision in shrinkage covariance.
    obj_terms = [mu_clean @ wp, -gamma_eff * cp.quad_form(wp, cp.psd_wrap(Sigma_mat))]
    # Linear transaction cost
    if cost_kappa > 0:
        obj_terms.append(-float(cost_kappa) * cp.norm(dw, 1))
    # Tax-aware cost on sells: tax_i · max(0, -Δw_i)
    if np.any(np.abs(tax_arr) > 1e-12):
        obj_terms.append(-tax_arr @ cp.pos(-dw))
    # Almgren-Chriss sqrt-impact: Σᵢ coef_i · |Δwᵢ|^1.5
    if np.any(impact_coef_arr > 0.0):
        # cp.power on |dw| is DCP-compliant for p=1.5 (convex, increasing on R+)
        obj_terms.append(-impact_coef_arr @ cp.power(cp.abs(dw), 1.5))
    # Rockafellar-Uryasev CVaR Gaussian closed form: λ · (φ(z_α)/α) · ‖Σ^½ wp‖₂
    if cvar_lambda > 0.0:
        from scipy.stats import norm  # noqa: PLC0415
        z_alpha   = float(norm.ppf(1.0 - float(cvar_alpha)))
        phi_z     = float(norm.pdf(z_alpha))
        cvar_mult = phi_z / max(float(cvar_alpha), 1e-6)
        # Sigma^(1/2) wp via psd-sqrt; cvxpy norm(Sigma_sqrt @ wp, 2) is convex.
        # Use eigendecomp once: Σ = V D V', Σ^½ = V D^½ V'.
        eigvals, eigvecs = np.linalg.eigh(Sigma_mat)
        eigvals = np.maximum(eigvals, 0.0)
        Sigma_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        obj_terms.append(-float(cvar_lambda) * cvar_mult
                          * cp.norm(Sigma_sqrt @ wp, 2))
    # SOFT cash-drag penalty: λ_cash · max(0, target - Σwp)
    # When λ_cash > 0 and target > 0, the solver pays this to leave cash
    # idle. Replaces the hard `Σwp ≥ target` floor that caused V4/V5 0-trade
    # infeasibility.
    if min_invested_pct > 0.0 and cash_drag_lambda > 0.0:
        obj_terms.append(-float(cash_drag_lambda)
                          * cp.pos(float(min_invested_pct) - cp.sum(wp)))

    # ── Solve with chained solver fallback ────────────────────────────────
    obj  = cp.Maximize(cp.sum(obj_terms))
    prob = cp.Problem(obj, constraints)
    allow_inaccurate = bool(allow_optimal_inaccurate)
    status = _solve_cvx(
        prob,
        cp.CLARABEL,
        [cp.OSQP, cp.SCS],
        allow_optimal_inaccurate=allow_inaccurate,
    )

    ok_statuses = {"optimal"}
    if allow_inaccurate:
        ok_statuses.add("optimal_inaccurate")
    if status not in ok_statuses:
        log.warning(
            "QP infeasible: status=%s  n=%d  sum(w_current)=%.3f  "
            "cash_slack=%.3f  per_asset_cap_max=%.3f  turnover_max=%s  "
            "min_invested_pct=%.3f  cash_drag_lambda=%.4f",
            status, n, float(np.sum(w_current)),
            (1.0 - float(cash_reserve)) - float(np.sum(w_current)),
            float(np.max(w_upper_arr - w_current)) if n else 0.0,
            "None" if turnover_max is None else f"{turnover_max:.3f}",
            float(min_invested_pct), float(cash_drag_lambda),
        )
        # Return zero-trade fallback rather than raising — pipeline knows to
        # log no-trade alert and the run continues. Same semantic the SLSQP
        # path used.
        delta_w_val = np.zeros(n)
        return QPSolution(
            delta_w=delta_w_val, target_w=w_current.copy(),
            objective=0.0, n_iter=-1, status=f"infeasible:{status}",
            diagnostics={"n_assets": n, "primary": "CLARABEL",
                          "fallback_chain": ["OSQP", "SCS"],
                          "solver_status": status,
                          "allow_optimal_inaccurate": bool(allow_inaccurate),
                          "n_sector_constraints": n_sector_rows,
                          "n_corr_pair_constraints": n_corr_pairs},
        )

    delta_w  = np.asarray(dw.value, dtype=float)
    target_w = w_current + delta_w
    # Numerical clean-up: interior-point solvers leave |x| < 1e-9 noise
    # at constraint boundaries (e.g. wp ≈ -3e-10 when w_lower=0). Clip
    # to the box bounds so callers don't see "shorts" that are floating-
    # point artifacts.
    target_w = np.clip(target_w, w_lower_arr, w_upper_arr)
    delta_w  = target_w - w_current

    nonzero_mu = int(np.sum(np.abs(mu_clean) > 1e-12))
    if nonzero_mu == 0:
        # All-zero μ → solver returns Δw ≈ 0 by definition. Tag the status
        # so the caller can branch (e.g. fall through to Kelly default).
        result_status = "optimal_no_signal"
    else:
        result_status = "optimal"

    return QPSolution(
        delta_w=delta_w,
        target_w=target_w,
        objective=float(prob.value) if prob.value is not None else 0.0,
        n_iter=int(prob.solver_stats.num_iters) if prob.solver_stats else -1,
        status=result_status,
        diagnostics={
            "n_assets":         n,
            "risk_aversion":    risk_aversion,
            "gamma_effective":  gamma_eff,
            "dd_factor":        dd_factor,
            "signal_decay":     sd,
            "robust_kappa":     float(robust_mu_kappa),
            "cvar_lambda":      float(cvar_lambda),
            "cvar_alpha":       float(cvar_alpha),
            "cost_kappa":       cost_kappa,
            "cash_reserve":     cash_reserve,
            "n_finite_mu":      int(finite_mu.sum()),
            "n_wash_blocked":   (int(np.asarray(wash_sale_mask).sum())
                                  if wash_sale_mask is not None else 0),
            "tax_cost_max":     float(tax_arr.max()) if tax_arr.size else 0.0,
            "tax_cost_mean":    float(tax_arr.mean()) if tax_arr.size else 0.0,
            "turnover_max":     float(turnover_max) if turnover_max is not None else None,
            "actual_turnover":  float(np.sum(np.abs(delta_w))),
            "impact_coef":      b_impact,
            "impact_cost_max":  float(impact_coef_arr.max()) if impact_coef_arr.size else 0.0,
            "min_invested_pct": float(min_invested_pct),
            "cash_drag_lambda": float(cash_drag_lambda),
            "allow_optimal_inaccurate": bool(allow_inaccurate),
            "solver_status":    status,
            "primary":          "CLARABEL",
            # Diagnostic: # of non-zero off-diagonal Σ entries — non-zero
            # means the QP is using cross-asset correlation (full Σ path
            # rather than diagonal). Pinned by tests to verify shrinkage
            # / Ledoit-Wolf wiring is live.
            "sigma_off_diag_nonzero": int(
                (np.abs(Sigma_mat) > 1e-12).sum()
                - np.count_nonzero(np.diag(Sigma_mat))
            ),
            # 2026-05-10 industrial-grade C2 deliverables — pinned by
            # tests/test_qp_sector_constraint.py + test_qp_correlation_constraint.py
            "n_sector_constraints": n_sector_rows,
            "n_corr_pair_constraints": n_corr_pairs,
        },
    )


# ── Back-compat aliases ──────────────────────────────────────────────────
# Pre-2026-05-06 code imported `_solve_via_cvxpy_fallback` and
# `_clamp_min_invested_floor` directly. Keep stubs that delegate to the
# new core solver so existing tests/callers don't break — but mark them
# DEPRECATED. New callers should use `solve_portfolio_qp` only.

def _solve_via_cvxpy_fallback(
    *, w_current, mu, Sigma, risk_aversion, cost_kappa, cash_reserve,
    w_lower_arr, w_upper_arr, dw_max_arr,
    min_invested_pct=0.0, turnover_max=None,
) -> np.ndarray | None:
    """DEPRECATED: pre-2026-05-06 cvxpy fallback. Now a thin shim — the
    new core solver IS cvxpy. Kept only so the old test suite + any
    direct importer keeps working. Will be removed once tests migrate."""
    sol = solve_portfolio_qp(
        w_current=w_current, mu=mu, Sigma=Sigma,
        risk_aversion=risk_aversion, cost_kappa=cost_kappa,
        cash_reserve=cash_reserve,
        w_upper=w_upper_arr, w_lower=w_lower_arr, dw_max=dw_max_arr,
        min_invested_pct=min_invested_pct,
        turnover_max=turnover_max,
        # Hard-floor semantics in old fallback → use a stiff penalty so
        # behaviour roughly matches when the constraint set IS feasible.
        # When infeasible, soft penalty deploys what it can rather than
        # returning None (the new behaviour is strictly better).
        cash_drag_lambda=10.0,
    )
    if sol.status.startswith("infeasible"):
        return None
    return sol.delta_w


def _clamp_min_invested_floor(
    *,
    min_invested_pct: float,
    w_current: np.ndarray,
    cash_reserve: float,
    hi_bounds: np.ndarray,
    turnover_max: float | None = None,
    safety_eps: float = 0.01,
) -> tuple[float, str]:
    """DEPRECATED: pre-2026-05-06 capacity/turnover floor clamp.

    The new convex QP makes this unnecessary — `min_invested_pct` is a
    SOFT target driven by `cash_drag_lambda`, not a hard floor. There is
    no infeasibility to clamp. Kept ONLY for back-compat with the
    regression tests that captured the V4/V5 bug. Returns the same
    `(floor, reason)` tuple."""
    floor = float(min_invested_pct)
    if floor <= 0:
        return 0.0, "none"
    sum_w = float(np.sum(w_current))
    cash_slack = (1.0 - float(cash_reserve)) - sum_w
    reason = "none"
    if floor > cash_slack:
        floor = max(0.0, cash_slack)
        reason = "cash_slack"
    max_capacity = float(np.sum(hi_bounds)) + sum_w
    if floor > max_capacity - safety_eps:
        floor = max(0.0, max_capacity - safety_eps)
        reason = "capacity"
    if turnover_max is not None and float(turnover_max) > 0:
        max_reachable = sum_w + float(turnover_max)
        if floor > max_reachable - safety_eps:
            floor = max(0.0, max_reachable - safety_eps)
            reason = "turnover"
    return floor, reason
