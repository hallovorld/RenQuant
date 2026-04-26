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
from scipy.optimize import LinearConstraint, minimize

log = logging.getLogger("kernel.portfolio_qp.qp_solver")


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
        # Floor σ at 1e-6 to keep Σ positive definite even for stale rows
        sigma_arr = np.clip(sigma_arr, 1e-6, None)
        Sigma_mat = np.diag(sigma_arr ** 2)
    else:
        Sigma_mat = np.asarray(Sigma, dtype=float)
        if Sigma_mat.shape != (n, n):
            raise ValueError(
                f"Sigma shape {Sigma_mat.shape} != (n={n}, n={n})",
            )

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

    # Linear constraint: 1' (w_current + Δw) ≤ 1 - cash_reserve
    #                ⇔ 1' Δw ≤ (1 - cash_reserve - 1' w_current)
    cash_slack = (1.0 - cash_reserve) - float(np.sum(w_current))
    cash_constraint = LinearConstraint(
        A=np.ones((1, n)),
        lb=-np.inf,
        ub=cash_slack,
    )

    def _obj(dw: np.ndarray) -> float:
        post_w = w_current + dw
        ret    = float(np.dot(mu_clean, post_w))
        var    = float(post_w @ Sigma_mat @ post_w)
        cost   = float(cost_kappa * np.sum(np.abs(dw)))
        return -(ret - gamma_eff * var - cost)  # minimize -obj

    def _grad(dw: np.ndarray) -> np.ndarray:
        post_w = w_current + dw
        # d/d(Δw) of: -(μ'(w+Δw) - γ_eff(w+Δw)'Σ(w+Δw) - κ|Δw|)
        d_ret  = -mu_clean
        d_var  =  2.0 * gamma_eff * (Sigma_mat @ post_w)
        d_cost = cost_kappa * np.sign(dw)
        return d_ret + d_var + d_cost

    # Initial guess: zero trade
    dw0 = np.zeros(n)

    res = minimize(
        _obj, dw0,
        method="SLSQP",
        jac=_grad,
        bounds=bounds,
        constraints=[cash_constraint],
        options={"ftol": 1e-9, "maxiter": 200},
    )

    delta_w = np.asarray(res.x, dtype=float)
    target_w = w_current + delta_w
    return QPSolution(
        delta_w=delta_w,
        target_w=target_w,
        objective=-float(res.fun),
        n_iter=int(res.nit) if hasattr(res, "nit") else 0,
        status="optimal" if res.success else f"failed:{res.message[:50]}",
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
        },
    )
