"""Portfolio QP solver — joint Markowitz w/ linear-cost transaction model.

Solves the single-period mean-variance optimization with proportional
transaction costs:

    max_Δw   r̂' (w + Δw) - γ (w + Δw)' Σ (w + Δw) - κ |Δw|_1
    s.t.     1' (w + Δw)  ≤ 1 - cash_reserve
              w_lower    ≤ w + Δw   ≤ w_upper       (per-position cap)
              -dw_max    ≤ Δw       ≤  dw_max       (slippage cap)
              Δw[wash]   ≤ 0                         (wash-sale: cannot re-buy)

Output: Δw vector — sign IS the action (positive=buy, negative=sell).
The current `JointActionTask` greedy heap maps to a special case of this
when Σ is diagonal and κ is small.

References:
- Markowitz 1952; Pogue 1970 (cost extension); Constantinides 1986
  (no-trade band); Garleanu-Pedersen 2013 (cost-aware partial moves).

Implementation: scipy.optimize.minimize with method='SLSQP'. For our
scale (≤101 variables) solves in ~5-10 ms. cvxpy/ECOS would be faster
on larger problems but requires extra dependency; defer to Stage 1.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize

log = logging.getLogger("kernel.portfolio_qp.qp_solver")


def _solve_via_cvxpy_fallback(
    *, w_current, mu, Sigma, risk_aversion, cost_kappa, cash_reserve,
    w_lower_arr, w_upper_arr, dw_max_arr,
    min_invested_pct=0.0, turnover_max=None,
) -> np.ndarray | None:
    """cvxpy/CLARABEL fallback for QP infeasibility cases.

    Phase A' (2026-05-06): used only when SLSQP fails with degenerate-
    starting-point errors ("Positive directional derivative for linesearch").
    Pattern from cvxportfolio/cvxportfolio (Boyd group, Stanford).

    Returns delta_w numpy array, or None if cvxpy also fails. Imports
    cvxpy lazily to avoid import-time cost on the fast SLSQP path.
    """
    try:
        import cvxpy as cp  # noqa: PLC0415
    except ImportError:
        log.warning("cvxpy not installed — cannot fall back. pip install cvxpy")
        return None

    n = len(w_current)
    dw = cp.Variable(n)
    wp = w_current + dw

    # Objective (matches SLSQP path's objective up to convexified parts):
    # max  μ' wp  -  γ wp' Σ wp  -  κ |Δw|_1
    # Σ_psd_wrap protects against tiny negative eigenvalues in Σ
    obj = mu @ wp - risk_aversion * cp.quad_form(wp, cp.psd_wrap(Sigma))
    if cost_kappa > 0:
        obj = obj - cost_kappa * cp.norm(dw, 1)

    constraints = [
        cp.sum(wp) <= 1.0 - cash_reserve,
        wp >= w_lower_arr,
        wp <= w_upper_arr,
        dw >= -dw_max_arr,
        dw <= dw_max_arr,
    ]
    if min_invested_pct > 0:
        constraints.append(cp.sum(wp) >= float(min_invested_pct))
    if turnover_max is not None and float(turnover_max) > 0:
        constraints.append(cp.norm(dw, 1) <= float(turnover_max))

    prob = cp.Problem(cp.Maximize(obj), constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        # Fall back to OSQP if CLARABEL not available or fails
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except Exception as exc:
            log.warning("cvxpy fallback solver chain exhausted: %s", exc)
            return None

    if prob.status not in ("optimal", "optimal_inaccurate"):
        log.warning("cvxpy fallback status=%s — giving up", prob.status)
        return None
    return np.asarray(dw.value, dtype=float)


@dataclass
class QPSolution:
    """Output of solve_portfolio_qp."""

    delta_w:        np.ndarray            # n-vector of weight changes
    target_w:       np.ndarray            # n-vector of post-trade weights
    objective:      float                  # final objective value
    n_iter:         int                    # iterations used
    status:         str                    # "optimal", "infeasible", etc.
    diagnostics:    dict                   # solver internals (κ, γ, …)


def solve_portfolio_qp(
    *,
    w_current:      Sequence[float],        # n-vector — current weights
    mu:             Sequence[float],        # n-vector — predicted returns
    sigma:          Sequence[float] | None = None,  # n-vector — used to build diagonal Σ if `Sigma` not given
    Sigma:          np.ndarray | None = None,       # n×n — full covariance (preferred when available)
    risk_aversion:  float = 3.0,            # γ
    cost_kappa:     float = 0.0001,         # linear cost coeff
    cash_reserve:   float = 0.0,            # min fraction held as cash (0 = invest fully)
    w_upper:        Sequence[float] | float = 0.20,
    w_lower:        Sequence[float] | float = 0.0,
    dw_max:         Sequence[float] | float = 0.50,
    wash_sale_mask: Sequence[bool] | None = None,
    # Stage 2 — Garleanu-Pedersen 2013 partial-move (signal-decay-aware)
    signal_decay:   float = 0.0,            # φ ∈ [0, 1); 0 = no decay (myopic Markowitz)
    # Stage 4 — Grossman-Zhou 1993 drawdown scaler
    drawdown:       float = 0.0,            # current DD as positive fraction (e.g. 0.05 = 5%)
    drawdown_limit: float = 0.20,           # α — DD limit; γ_eff = γ_base / max(eps, 1 - DD/α)
    # Stage 5 — Garlappi-Uppal-Wang 2007 robust μ adjustment
    robust_mu_kappa: float = 0.0,           # μ_robust_i = μ_i - κ · σ_i (worst-case ball)
    # Stage 7 — Rockafellar-Uryasev 2002 CVaR risk term
    # When > 0, adds a tail-risk penalty using the Gaussian-CVaR closed
    # form: CVaR_α(w'r) ≈ -μ'w + φ(z_α)/α · √(w'Σw). For our use case
    # (intraday position adjustment) we approximate with a per-asset
    # CVaR contribution proportional to σ_i and a tail multiplier.
    cvar_lambda:    float = 0.0,            # weight on CVaR term (0 = pure variance)
    cvar_alpha:     float = 0.05,           # tail probability (e.g. 5%)
    # Stage 8 (2026-04-29) — Tax-basis aware cost. Per-asset additional
    # cost on SELLS (Δw_i < 0) reflecting realized-gain tax drag. Fixes
    # the "QP doesn't see tax basis" structural bug — pre-fix high-gain
    # positions looked as cheap to liquidate as zero-gain positions.
    # Caller passes one cost-per-unit-sell value per asset, in same units
    # as the rest of the objective (NAV-fraction). Cost only applied to
    # the sell side: cost_i × max(0, -Δw_i).
    tax_cost_per_sell: Sequence[float] | None = None,
    # Stage 9 (2026-04-29) — Turnover hard constraint. Σ |Δw_i| ≤ τ_max.
    # Pre-fix the only churn brake was the soft cost penalty κ·‖Δw‖₁ which
    # at κ=0.0001 was negligible vs typical μ. With turnover_max set, the
    # solver can never trade more than τ_max NAV-fraction per bar.
    turnover_max:   float | None = None,
    # Stage G3 (2026-05-04) — Almgren-Chriss 2000 sqrt-impact transaction
    # cost. Adds a market-impact term to the objective:
    #     impact_i = impact_coef · σ_i · |Δw_i|^1.5 · sqrt(NAV / V_dollar_i)
    # where V_dollar_i is the per-asset average daily dollar volume (ADV),
    # and σ_i is taken from sqrt(diag(Σ)). The sqrt(participation) factor
    # is the empirical Gatheral form. Defaults to zero impact (preserves
    # legacy linear-only κ behaviour).
    impact_coef:    float = 0.0,            # b ≥ 0 (Gatheral)
    v_daily_dollar: Sequence[float] | None = None,  # n-vector of ADV in $; if None, impact=0
    nav_dollar:     float = 0.0,            # current NAV in dollars (for participation rate)
    # Stage G4 (2026-05-04) — Smoothed fixed cost per trade. Approximates
    # the discontinuous "if you trade at all, pay a fixed fee" with a
    # smooth tanh penalty:
    #     fixed_i = fixed_cost_per_trade · tanh(fixed_cost_beta · |Δw_i|)
    # Beta controls the steepness; large β → step. We avoid the literal
    # indicator because it breaks SLSQP gradients. Defaults to zero.
    fixed_cost_per_trade: float = 0.0,      # c_fix ≥ 0, NAV-fraction
    fixed_cost_beta:      float = 100.0,    # β > 0 (saturates around |Δw| ≈ 1/β)
    # 2026-05-05 — budget constraint mode. "inequality" (default, legacy)
    # = `Σw ≤ 1 − cash_reserve` (LE); "equality" = `Σw == 1 − cash_reserve`
    # which forces full deployment (modulo reserve). EQ is the textbook
    # Markowitz formulation but breaks SLSQP feasibility on empty-
    # portfolio start (Positive directional derivative for linesearch
    # at w_current=0). PREFER min_invested_pct below for cash-drag fix.
    budget_mode: str = "inequality",
    # 2026-05-05 cash-drag P0 (replaces equality experiment): impose a
    # SOFT floor on total deployment via a two-sided box constraint.
    # When min_invested_pct > 0, adds a second LinearConstraint:
    #     Σw ≥ min_invested_pct   (i.e. 1' Δw ≥ min_invested_pct − 1'w_current)
    # combined with the existing 1' Δw ≤ (1 − cash_reserve − 1'w_current).
    # Default 0.0 → no floor (legacy parity). Setting 0.7 forces
    # 70–100% deployed. Easier for SLSQP than equality (feasibility
    # region is non-degenerate).
    min_invested_pct:     float = 0.0,
) -> QPSolution:
    """Solve the single-period Markowitz QP with linear-cost transaction.

    All weights are fractions of total portfolio value. `Sigma` is the
    forecast covariance matrix. If only `sigma` (the per-asset σ vector)
    is given, the solver falls back to a diagonal Σ — which discards
    cross-asset correlation but stays well-defined.

    `wash_sale_mask` (optional, n-bool): True for tickers recently sold
    (within wash-sale window) — Δw_i ≤ 0 is enforced so they cannot be
    re-bought. Selling further is still permitted.

    Returns a QPSolution. Raises ValueError on shape mismatch.
    """
    w_current = np.asarray(w_current, dtype=float)
    mu        = np.asarray(mu,        dtype=float)
    n         = len(w_current)
    if len(mu) != n:
        raise ValueError(f"len(mu)={len(mu)} != len(w_current)={n}")

    # Resolve Σ
    if Sigma is None:
        if sigma is None:
            raise ValueError("must provide either Sigma (n×n) or sigma (n-vector)")
        sigma_arr = np.asarray(sigma, dtype=float)
        if len(sigma_arr) != n:
            raise ValueError(f"len(sigma)={len(sigma_arr)} != n={n}")
        # 2026-05-04 audit Issue 32 fix: NaN sigma slipped through
        # `np.clip(arr, 1e-6, None)` (np.clip preserves NaN), then
        # `arr**2 = NaN` → `np.diag(NaN)` poisoned Σ → QP objective
        # `post_w @ Σ @ post_w` returned NaN → SLSQP behavior on NaN
        # objective is undefined. Sanitize NaN/inf to a sane default
        # (5% — a typical equity vol) BEFORE clipping.
        sigma_arr = np.where(np.isfinite(sigma_arr), sigma_arr, 0.05)
        # Floor σ at 1e-6 to keep Σ positive definite even for stale rows
        sigma_arr = np.clip(sigma_arr, 1e-6, None)
        Sigma_mat = np.diag(sigma_arr ** 2)
    else:
        Sigma_mat = np.asarray(Sigma, dtype=float)
        if Sigma_mat.shape != (n, n):
            raise ValueError(
                f"Sigma shape {Sigma_mat.shape} != (n={n}, n={n})",
            )
        # Issue 32 (full-Σ path): same NaN-slip class. If caller's
        # correlation matrix had NaN cells, Σ_ij = NaN propagates.
        if not np.isfinite(Sigma_mat).all():
            n_bad = int(np.sum(~np.isfinite(Sigma_mat)))
            log.warning(
                "QP: full Σ has %d non-finite cell(s) — replacing with 0 "
                "(may degrade solution quality; check correlation artifact)",
                n_bad,
            )
            Sigma_mat = np.where(np.isfinite(Sigma_mat), Sigma_mat, 0.0)
            # Re-add diagonal floor to preserve PSD
            Sigma_mat += 1e-8 * np.eye(n)

    # Broadcast caps
    if np.isscalar(w_upper):
        w_upper_arr = np.full(n, float(w_upper))
    else:
        w_upper_arr = np.asarray(w_upper, dtype=float)
    if np.isscalar(w_lower):
        w_lower_arr = np.full(n, float(w_lower))
    else:
        w_lower_arr = np.asarray(w_lower, dtype=float)
    if np.isscalar(dw_max):
        dw_max_arr = np.full(n, float(dw_max))
    else:
        dw_max_arr = np.asarray(dw_max, dtype=float)

    # Sanitise NaN/inf in mu — drop rows by setting μ=0 (no signal)
    finite_mu = np.isfinite(mu)
    mu_clean  = np.where(finite_mu, mu, 0.0)

    # Stage 2 — Garleanu-Pedersen 2013: pre-discount the signal by
    # (1 - φ) where φ is the per-bar autocorrelation. Persistent
    # signals (φ→1) shrink slowly and amortize cost; one-shot
    # signals (φ=0) keep full magnitude (myopic Markowitz). The
    # scale `1/(1-φ_decay)` reflects the cumulative future value of
    # the signal under exponential decay. Keep φ < 0.99 to avoid
    # numerical blow-up; clamp here defensively.
    sd = float(signal_decay)
    if sd > 0.0:
        sd = min(sd, 0.99)
        mu_clean = mu_clean * (1.0 / (1.0 - sd))

    # Stage 5 — Garlappi-Uppal-Wang 2007 robust μ: subtract κ·σ_i to
    # represent worst-case under uncertainty ellipsoid. Scaled Sharpe
    # penalty per asset; with κ=1 this is equivalent to a 1-σ
    # confidence band on the mean estimate.
    if robust_mu_kappa != 0.0:
        # Use diagonal of Σ as σ_i² → σ_i
        sigma_diag = np.sqrt(np.maximum(np.diag(Sigma_mat), 0.0))
        mu_clean = mu_clean - float(robust_mu_kappa) * sigma_diag

    # Stage 4 — Grossman-Zhou 1993 DD scaler:
    # γ_eff = γ_base / max(eps, 1 - DD/α)
    # When DD → α, γ_eff → ∞ (forces Δw → 0; risk shrinkage).
    dd = float(max(0.0, drawdown))
    dd_lim = float(max(1e-6, drawdown_limit))
    dd_factor = max(1e-3, 1.0 - dd / dd_lim)
    gamma_eff = float(risk_aversion) / dd_factor

    # Stage 7 — Rockafellar-Uryasev 2002 CVaR tail-risk multiplier.
    # For Gaussian returns r ~ N(μ, σ²): CVaR_α(-r) = -μ + (φ(z_α)/α)·σ.
    # Tail multiplier z_α = φ(z_α)/α; e.g. α=0.05 → z_α ≈ 2.063, α=0.01 → 2.665.
    # We add (cvar_lambda · z_α) to gamma_eff's variance multiplier so
    # that higher α-tail risk gets penalised proportionally to σ. This
    # is a simplification of the full 2-stage LP formulation but
    # captures the same qualitative behaviour (heavier tail penalty).
    if cvar_lambda > 0.0:
        from scipy.stats import norm  # noqa: PLC0415
        # phi(z_α) / α is the conditional Sharpe shortfall multiplier
        z_alpha = float(norm.ppf(1.0 - cvar_alpha))
        phi_z   = float(norm.pdf(z_alpha))
        cvar_mult = phi_z / max(cvar_alpha, 1e-6)
        gamma_eff = gamma_eff + cvar_lambda * cvar_mult

    # Decision variable: Δw ∈ ℝⁿ
    # Bounds: max(w_lower - w_current, -dw_max) ≤ Δw ≤ min(w_upper - w_current, +dw_max)
    lo_bounds = np.maximum(w_lower_arr - w_current, -dw_max_arr)
    hi_bounds = np.minimum(w_upper_arr - w_current,  dw_max_arr)
    if wash_sale_mask is not None:
        # Wash-sale: recently-sold tickers cannot be BOUGHT (can still
        # be sold further). Cap Δw_i ≤ 0 for masked positions.
        wsm = np.asarray(wash_sale_mask, dtype=bool)
        hi_bounds = np.where(wsm, np.minimum(hi_bounds, 0.0), hi_bounds)
    # Sanity — feasibility check (audit fix QP-INFEASIBLE-WARN, 2026-04-26):
    # When per-asset bounds are inconsistent (e.g. current weight already
    # outside the cap, AND slippage band can't reach back inside in one
    # bar), we clip rather than fail — pick the tightest bound. Pre-fix,
    # this happened silently. Now log a warning so the operator notices.
    bad = lo_bounds > hi_bounds + 1e-12
    if bad.any():
        n_bad = int(bad.sum())
        log.warning(
            "QP: %d/%d asset bound(s) clipped due to infeasibility — "
            "current weight may be outside cap by more than dw_max",
            n_bad, n,
        )
        lo_bounds[bad] = hi_bounds[bad]
    bounds = list(zip(lo_bounds.tolist(), hi_bounds.tolist()))

    # Linear constraint: 1' (w_current + Δw) [≤ or =] 1 - cash_reserve
    #                ⇔ 1' Δw [≤ or =] (1 - cash_reserve - 1' w_current)
    # 2026-05-05 cash-drag fix: budget_mode="equality" (caller-set) forces
    # `Σw == 1 − cash_reserve` instead of `≤`. Equality removes the
    # "cash is free" loophole — when μ is small / non-positive across
    # the board, the LE form leaves cash idle (cash drag). EQ deploys
    # everything except the reserve; risk is still controlled by
    # γ·wᵀΣw + per-asset w_upper. Default LE preserves legacy parity.
    # References: Markowitz 1952 §III; Best & Grauer 1991 §4.
    cash_slack = (1.0 - cash_reserve) - float(np.sum(w_current))
    # 2026-05-05 cash-drag fix: replace inequality / equality with a
    # two-sided box. lb is min_invested_floor (default -inf = legacy
    # behavior); ub is the existing cash_slack. SLSQP handles boxes
    # cleanly; equality often fails Positive-directional-derivative.
    min_invested_slack = float("-inf")
    if min_invested_pct > 0:
        min_invested_slack = (
            float(min_invested_pct) - float(np.sum(w_current))
        )
        # Sanity 1: if floor > ceiling (e.g. min=0.9, cash_reserve=0.2 →
        # ceiling=0.8), clamp floor to ceiling minus epsilon to keep
        # feasible.
        if min_invested_slack > cash_slack:
            min_invested_slack = cash_slack
        # Sanity 2 (2026-05-05 — Track B'' debug): floor INFEASIBILITY
        # by capacity. With small per-asset caps (e.g. max_position_pct
        # × conf_mult = 0.075) and few candidates (e.g. 8), max possible
        # ΣΔw = 8×0.075 = 0.60. If floor=0.70, no Δw can satisfy LB →
        # SLSQP fails "Positive directional derivative for linesearch".
        # Auto-clamp by per-asset hi_bounds total. Conservative: leave
        # 1% slack so SLSQP has interior feasibility.
        max_capacity = float(np.sum(hi_bounds))
        if min_invested_slack > max_capacity - 0.01:
            min_invested_slack = max(-np.inf, max_capacity - 0.01)
    if budget_mode == "equality":
        cash_constraint = LinearConstraint(
            A=np.ones((1, n)),
            lb=cash_slack,    # lb == ub → equality
            ub=cash_slack,
        )
    else:   # default "inequality" — legacy + optional floor
        cash_constraint = LinearConstraint(
            A=np.ones((1, n)),
            lb=min_invested_slack,
            ub=cash_slack,
        )

    # Audit fix QP-MATMUL-WARN (2026-04-26): Apple Silicon Accelerate
    # BLAS emits spurious 'divide by zero' / 'overflow' RuntimeWarnings
    # during matmul when matrices have many zero entries (which is
    # exactly our case — post_w starts at zero, Σ_mat is diagonal).
    # The warnings don't reflect numerical issues — the math is fine.
    # Suppress them locally to keep logs clean.
    # Tax-basis cost vector — only applies to sells (max(0, -Δw_i)).
    # If not supplied, zero (preserves Stage-1 behaviour).
    if tax_cost_per_sell is not None:
        tax_arr = np.asarray(tax_cost_per_sell, dtype=float)
        if len(tax_arr) != n:
            raise ValueError(
                f"tax_cost_per_sell length {len(tax_arr)} != n={n}",
            )
        # 2026-05-04 v4: removed `np.maximum(tax_arr, 0.0)` clamp.
        # Berkin-Jeffrey (1990) tax-loss harvesting: selling losers can
        # OFFSET prior YTD realized gains, which is a NEGATIVE tax cost
        # (= reward). The clamp killed this entire alpha source. Caller
        # (JointPortfolioQPTask) computes the negative coefficient when
        # ctx.ytd_realized_gain_dollar is positive and the position has
        # an unrealized loss to harvest. NaN/inf entries still get
        # zeroed — those are bad data, not legitimate negative costs.
        tax_arr = np.where(np.isfinite(tax_arr), tax_arr, 0.0)
    else:
        tax_arr = np.zeros(n)

    # Stage G3 — Almgren-Chriss participation factor.
    # impact_per_unit_pow_1.5_i = impact_coef · σ_i · sqrt(NAV / V_i)
    # We pre-multiply σ and the sqrt(NAV/V) into a single per-asset
    # coefficient so _obj only does |Δw|^1.5.
    b_impact = float(max(0.0, impact_coef))
    if (b_impact > 0.0
            and v_daily_dollar is not None
            and float(nav_dollar) > 0.0):
        v_arr = np.asarray(v_daily_dollar, dtype=float)
        if len(v_arr) != n:
            raise ValueError(
                f"v_daily_dollar length {len(v_arr)} != n={n}",
            )
        # Sanitise: any non-finite or non-positive ADV → no impact for
        # that asset (we don't want NaN poisoning the objective; missing
        # ADV signals "data unavailable", not "ADV is zero").
        v_safe = np.where((np.isfinite(v_arr)) & (v_arr > 0.0), v_arr, np.inf)
        sigma_diag_g3 = np.sqrt(np.maximum(np.diag(Sigma_mat), 0.0))
        # NAV / V → participation; sqrt for Gatheral.
        impact_coef_arr = b_impact * sigma_diag_g3 * np.sqrt(
            float(nav_dollar) / v_safe,
        )
        # v_safe = inf → coef = 0 cleanly.
        impact_coef_arr = np.where(
            np.isfinite(impact_coef_arr), impact_coef_arr, 0.0,
        )
    else:
        impact_coef_arr = np.zeros(n)

    # Stage G4 — Smoothed fixed cost. c_fix · tanh(β·|Δw|).
    # tanh(0) = 0 (no trade → no cost), tanh(β·|Δw|) → 1 as |Δw| >> 1/β.
    # β large → step-like; β small → smooth quadratic-ish near zero.
    c_fix = float(max(0.0, fixed_cost_per_trade))
    beta_fix = float(max(1e-6, fixed_cost_beta))

    def _obj(dw: np.ndarray) -> float:
        post_w = w_current + dw
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            ret  = float(np.dot(mu_clean, post_w))
            var  = float(post_w @ Sigma_mat @ post_w)
        abs_dw = np.abs(dw)
        cost = float(cost_kappa * np.sum(abs_dw))
        # Tax cost only on sells: cost_i × max(0, -Δw_i)
        sell_amt = np.maximum(-dw, 0.0)
        tax_cost = float(np.sum(tax_arr * sell_amt))
        # G3 sqrt-impact (Almgren-Chriss / Gatheral)
        impact_cost = (
            float(np.sum(impact_coef_arr * np.power(abs_dw, 1.5)))
            if b_impact > 0 else 0.0
        )
        # G4 smoothed fixed cost (tanh)
        fixed = (
            c_fix * float(np.sum(np.tanh(beta_fix * abs_dw)))
            if c_fix > 0 else 0.0
        )
        return -(ret - gamma_eff * var - cost - tax_cost
                  - impact_cost - fixed)  # minimize -obj

    def _grad(dw: np.ndarray) -> np.ndarray:
        post_w = w_current + dw
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            d_var = 2.0 * gamma_eff * (Sigma_mat @ post_w)
        # Minimizing  f(Δw) = -ret + γ·var + κ|Δw| + tax · max(0,-Δw)
        #                       + b·σ·|Δw|^1.5·sqrt(NAV/V) + c_fix·tanh(β|Δw|)
        #   d_ret      = -μ
        #   d_var      = 2γ·Σ·(w + Δw)
        #   d_cost     = κ·sign(Δw)
        #   d_tax      = -tax_i · 1[Δw_i < 0]
        #   d_impact   = 1.5 · b·σ·sqrt(NAV/V) · sqrt(|Δw|) · sign(Δw)
        #   d_fixed    = c_fix · β · sech²(β|Δw|) · sign(Δw)
        d_ret  = -mu_clean
        d_cost = cost_kappa * np.sign(dw)
        d_tax  = -tax_arr * (dw < 0).astype(float)
        abs_dw = np.abs(dw)
        sgn    = np.sign(dw)
        if b_impact > 0:
            d_impact = 1.5 * impact_coef_arr * np.sqrt(abs_dw) * sgn
        else:
            d_impact = np.zeros(n)
        if c_fix > 0:
            sech2 = 1.0 / np.cosh(beta_fix * abs_dw) ** 2
            d_fixed = c_fix * beta_fix * sech2 * sgn
        else:
            d_fixed = np.zeros(n)
        return d_ret + d_var + d_cost + d_tax + d_impact + d_fixed

    # Initial guess: zero trade is OUTSIDE the feasible region when
    # min_invested_pct > current_invested. SLSQP fails with "Positive
    # directional derivative for linesearch". Warm-start with a feasible
    # uniform allocation that satisfies both LB (min_invested_slack) and
    # UB (cash_slack) when applicable. The QP itself will optimize away
    # from this baseline; we only need a starting point inside the
    # feasibility region.
    dw0 = np.zeros(n)
    if min_invested_pct > 0 and min_invested_slack > 0:
        # Spread min_invested_slack uniformly across all assets; clamp
        # by per-asset upper bound so the warm-start respects w_upper.
        per_asset = min_invested_slack / max(1, n)
        # If upper bound is finite and binding, use it; else use uniform.
        for i in range(n):
            ub_i = hi_bounds[i] if i < len(hi_bounds) else per_asset
            dw0[i] = min(per_asset, max(0.0, ub_i))
        # Re-balance if total still below LB (some assets capped):
        deficit = min_invested_slack - float(np.sum(dw0))
        if deficit > 1e-9:
            # distribute remaining deficit to non-capped assets
            for i in range(n):
                if dw0[i] < hi_bounds[i] - 1e-9:
                    headroom = hi_bounds[i] - dw0[i]
                    take = min(headroom, deficit)
                    dw0[i] += take
                    deficit -= take
                    if deficit <= 1e-9:
                        break

    constraints: list = [cash_constraint]
    # Stage 9: turnover hard constraint Σ|Δw| ≤ τ_max
    if turnover_max is not None and float(turnover_max) > 0:
        tau = float(turnover_max)
        turnover_constraint = NonlinearConstraint(
            fun=lambda dw: float(np.sum(np.abs(dw))),
            lb=-np.inf,
            ub=tau,
            jac=lambda dw: np.sign(dw),
        )
        constraints.append(turnover_constraint)

    res = minimize(
        _obj, dw0,
        method="SLSQP",
        jac=_grad,
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 200},
    )

    delta_w = np.asarray(res.x, dtype=float)

    # ── Phase A' (2026-05-06): cvxpy fallback when SLSQP fails ────────────
    # SLSQP returns "Positive directional derivative for linesearch" when
    # the warm-start lands at a degenerate point. cvxpy + interior-point
    # solvers (CLARABEL/OSQP) handle these cleanly. We pay the 5× speed
    # penalty only on the failure path. See `doc/research/qp-cvxportfolio-
    # refactor-plan.md` Phase A'.
    if not res.success and ("Positive directional derivative" in (res.message or "")
                             or "Inequality constraints incompatible" in (res.message or "")):
        try:
            cvx_dw = _solve_via_cvxpy_fallback(
                w_current=w_current, mu=mu_clean, Sigma=Sigma_mat,
                risk_aversion=gamma_eff, cost_kappa=cost_kappa,
                cash_reserve=cash_reserve,
                w_lower_arr=w_lower_arr, w_upper_arr=w_upper_arr,
                dw_max_arr=dw_max_arr,
                min_invested_pct=min_invested_pct,
                turnover_max=turnover_max,
            )
            if cvx_dw is not None:
                log.warning(
                    "qp_solver: SLSQP infeasible (%s) → cvxpy fallback succeeded",
                    (res.message or "")[:60],
                )
                delta_w = cvx_dw
                # Re-evaluate objective at new delta_w
                res = type("CvxpyRes", (), {
                    "success": True,
                    "x": cvx_dw,
                    "fun": _obj(cvx_dw),
                    "nit": -1,
                    "message": "cvxpy_fallback",
                })()
        except Exception as exc:
            log.warning("qp_solver: cvxpy fallback also failed: %s", exc)

    target_w = w_current + delta_w
    # 2026-05-04 audit Issue 22 fix: when μ is all zeros (all candidates
    # failed scoring → mu=0 fallback in JointPortfolioQPTask), the
    # objective `-(0 - γ·var - κ|Δw| - tax)` is minimized at Δw = 0,
    # so SLSQP returns success=True with delta_w ≈ 0 and the caller
    # silently emits zero buys / zero sells. The pipeline status looks
    # "optimal" but it's a sentinel for missing signal, not a real
    # decision. Tag the status so the caller can branch on it (e.g.
    # log.warning + fall through to greedy or Kelly defaults).
    finite_mu_count = int(np.sum(np.isfinite(mu)))
    nonzero_mu_count = int(np.sum(np.abs(mu_clean) > 1e-12))
    if res.success and nonzero_mu_count == 0:
        status = "optimal_no_signal"
    else:
        status = "optimal" if res.success else f"failed:{res.message[:50]}"
    return QPSolution(
        delta_w=delta_w,
        target_w=target_w,
        objective=-float(res.fun),
        n_iter=int(res.nit) if hasattr(res, "nit") else 0,
        status=status,
        diagnostics={
            "n_assets":        n,
            "risk_aversion":   risk_aversion,
            "gamma_effective": gamma_eff,
            "dd_factor":       dd_factor,
            "signal_decay":    sd,
            "robust_kappa":    float(robust_mu_kappa),
            "cvar_lambda":     float(cvar_lambda),
            "cvar_alpha":      float(cvar_alpha),
            "cost_kappa":      cost_kappa,
            "cash_reserve":    cash_reserve,
            "n_finite_mu":     int(finite_mu.sum()),
            "n_wash_blocked":  (int(np.asarray(wash_sale_mask).sum())
                                  if wash_sale_mask is not None else 0),
            "tax_cost_max":    float(tax_arr.max()) if tax_arr.size else 0.0,
            "tax_cost_mean":   float(tax_arr.mean()) if tax_arr.size else 0.0,
            "turnover_max":    float(turnover_max) if turnover_max is not None else None,
            "actual_turnover": float(np.sum(np.abs(delta_w))),
            "sigma_off_diag_nonzero": int((np.abs(Sigma_mat) > 1e-12).sum() - np.count_nonzero(np.diag(Sigma_mat))),
            "impact_coef":     b_impact,
            "impact_cost_max": float(impact_coef_arr.max()) if impact_coef_arr.size else 0.0,
            "fixed_cost":      c_fix,
            "fixed_cost_beta": beta_fix if c_fix > 0 else 0.0,
        },
    )
