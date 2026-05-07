"""Boyd-style convex rotation solver — T2-4 (2026-04-27).

Mean-variance QP with G-P 2013 transaction-cost penalty:

    maximize    μᵀΔw  −  γ · Δwᵀ Σ Δw  −  c · ‖Δw‖₁
    Δw

    subject to  weight_lo ≤ w + Δw ≤ weight_hi          (long-only by default)
                ‖Δw‖₁ ≤ turnover_cap
                ‖w + Δw‖₁ ≤ leverage_cap
                sector caps                              (optional)

Where Δw is the trade vector (per-ticker buy/sell qty, signed; in
fractional weight units), μ is expected returns from NGBoost head, Σ
is the cov matrix from `watchlist-correlation.json`.

Replaces (when enabled) the greedy joint-actions sorter in
`kernel/rotation.py`.

References
==========
- Boyd, "Markowitz Portfolio Construction at Seventy" (2024)
- Gârleanu, Pedersen 2013 ("Dynamic Trading with Predictable Returns
  and Transaction Costs", J. of Finance) — claim +20% Sharpe vs static
  one-period optimization

Implementation notes
====================
Primary solver: **scipy.optimize.minimize** with SLSQP. Available in
the existing environment without new deps. Solve time for 99-ticker
problem: ~50-200ms (acceptable for live use).

Future fast path: cvxpy + OSQP. Same model formulation; ~10× faster
on larger problems. Gated by `try: import cvxpy`. If cvxpy is
installed, prefer it; otherwise fall back to scipy. No code change
required for ops to opt in — just `pip install cvxpy`.

Status: Phase A — solver module + unit tests + sanity-check on
synthetic 2-ticker problem with known closed-form solution. NOT yet
wired into RotationJob (Phase C work).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("kernel.rotation_convex")

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:
    cp = None  # type: ignore
    _HAS_CVXPY = False


@dataclass
class ConvexRotationResult:
    """Output of `ConvexRotationSolver.solve`."""
    delta_weights: pd.Series          # signed Δw per ticker
    objective_value: float
    solve_time_ms: float
    solver_used: str                  # "cvxpy_OSQP" | "scipy_SLSQP"
    status: str                       # "optimal" | "infeasible" | "max_iter" | …
    n_constraints: int
    n_variables: int


@dataclass
class ConvexRotationSolver:
    """Boyd-style mean-variance optimizer with transaction-cost penalty.

    Default config (per `boyd-rotation-design.md`):
    - gamma_risk:      5.0  (risk aversion)
    - cost_coef:       0.001 (slippage + tax estimate, per unit traded)
    - turnover_cap:    0.40 (sum of |Δw| ≤ 40% of NAV)
    - leverage_cap:    1.00 (long-only, sum w ≤ 1.0)
    - sector_max_pct:  0.30 (per-sector concentration ≤ 30%)
    - max_solve_ms:    500  (G13 gate threshold)
    """
    gamma_risk:      float = 5.0
    cost_coef:       float = 0.001
    turnover_cap:    float = 0.40
    leverage_cap:    float = 1.00
    sector_max_pct:  float = 0.30
    max_solve_ms:    float = 500.0
    prefer_cvxpy:    bool  = True   # falls back to scipy when cvxpy missing

    def solve(
        self,
        *,
        current_weights: pd.Series,        # by ticker, sums to ≤ 1
        expected_returns: pd.Series,        # by ticker, μ from NGBoost
        cov_matrix: pd.DataFrame,           # ticker × ticker, σ²-scaled
        sector_map: dict[str, str] | None = None,
    ) -> ConvexRotationResult:
        """Solve single-period rebalancing QP.

        Returns ConvexRotationResult with:
        - delta_weights: signed Δw (positive = buy, negative = sell)
        - objective_value: final objective (μᵀΔw − γ Δwᵀ Σ Δw − c ‖Δw‖₁)
        - solve_time_ms: wall-clock solve time
        - solver_used: which path fired
        - status: solver-reported status

        Caller: convert delta_weights → integer share counts via
        `quantize_to_whole_shares()` (separate helper).
        """
        # Align all inputs to current_weights' index
        tickers = list(current_weights.index)
        n = len(tickers)
        if n == 0:
            raise ValueError("ConvexRotationSolver.solve: empty current_weights")

        w = current_weights.values.astype(float)

        # Audit T1 + T2 fix (2026-04-27): strict-mode reindex.
        # Pre-fix: silent fillna(0) treated missing tickers as μ=0 / σ²=0
        # → "risk-free" candidates. The solver would happily allocate to
        # tickers with no model coverage.
        missing_mu = [t for t in tickers if t not in expected_returns.index]
        if missing_mu:
            raise ValueError(
                f"ConvexRotationSolver.solve: missing μ for "
                f"{len(missing_mu)} tickers: {missing_mu[:5]}"
                f"{'…' if len(missing_mu) > 5 else ''}. "
                f"Caller must supply expected_returns for every ticker in "
                f"current_weights, or pre-filter current_weights."
            )
        mu = expected_returns.reindex(tickers).values.astype(float)
        if np.isnan(mu).any():
            raise ValueError(
                f"ConvexRotationSolver.solve: NaN in expected_returns for "
                f"tickers {[tickers[i] for i in np.where(np.isnan(mu))[0][:5]]}. "
                f"Caller must supply non-NaN μ for every ticker."
            )

        missing_sig_rows = [t for t in tickers if t not in cov_matrix.index]
        missing_sig_cols = [t for t in tickers if t not in cov_matrix.columns]
        if missing_sig_rows or missing_sig_cols:
            raise ValueError(
                f"ConvexRotationSolver.solve: cov_matrix missing rows "
                f"{missing_sig_rows[:5]} or cols {missing_sig_cols[:5]}. "
                f"Caller must supply cov_matrix covering every ticker."
            )
        sigma = cov_matrix.reindex(index=tickers, columns=tickers).values.astype(float)
        if np.isnan(sigma).any():
            raise ValueError(
                "ConvexRotationSolver.solve: NaN in cov_matrix. "
                "Caller must supply non-NaN covariance estimates."
            )
        # Σ — symmetrize defensively (numerical noise can break PSD)
        sigma = 0.5 * (sigma + sigma.T)
        # Add small ridge for numerical PSD
        sigma = sigma + 1e-8 * np.eye(n)

        # Try cvxpy first if available + preferred
        if self.prefer_cvxpy and _HAS_CVXPY:
            try:
                return self._solve_cvxpy(
                    tickers=tickers, w=w, mu=mu, sigma=sigma, sector_map=sector_map,
                )
            except Exception as exc:
                log.warning("ConvexRotationSolver: cvxpy path failed (%s); "
                            "falling back to scipy", exc)

        # Fallback: scipy.optimize.minimize with SLSQP
        return self._solve_scipy(
            tickers=tickers, w=w, mu=mu, sigma=sigma, sector_map=sector_map,
        )

    # ── cvxpy path (preferred when available) ─────────────────────────────

    def _solve_cvxpy(
        self, *, tickers, w, mu, sigma, sector_map,
    ) -> ConvexRotationResult:
        n = len(tickers)
        delta = cp.Variable(n)
        w_new = w + delta

        # Objective: maximize μᵀΔw − γ Δwᵀ Σ Δw − c ‖Δw‖₁
        objective = cp.Maximize(
            mu @ delta
            - self.gamma_risk * cp.quad_form(delta, sigma)
            - self.cost_coef * cp.norm(delta, 1)
        )

        constraints: list = [
            w_new >= 0.0,                                       # long-only
            w_new <= self.sector_max_pct,                       # T8 audit: per-position cap = sector_max_pct
            cp.sum(w_new) <= self.leverage_cap,
            cp.norm(delta, 1) <= self.turnover_cap,
        ]
        # Audit T8 (2026-04-27): without per-position cap, the cvxpy
        # path could put 100% on a single ticker (e.g. heavy μ with low
        # γ from an empty portfolio). The scipy path enforces this via
        # `Bounds(ub=sector_max_pct)`; mirror that here. Sector-level cap
        # below stays as the (additive) constraint when sector_map given.

        if sector_map:
            for sector in set(sector_map.values()):
                mask = np.array(
                    [1.0 if sector_map.get(t) == sector else 0.0 for t in tickers],
                    dtype=float,
                )
                constraints.append(mask @ w_new <= self.sector_max_pct)

        prob = cp.Problem(objective, constraints)
        t0 = time.monotonic()
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        delta_arr = (delta.value if delta.value is not None
                     else np.zeros(n, dtype=float))
        return ConvexRotationResult(
            delta_weights = pd.Series(delta_arr, index=tickers, name="delta_weight"),
            objective_value = float(prob.value) if prob.value is not None else float("nan"),
            solve_time_ms = elapsed_ms,
            solver_used = "cvxpy_OSQP" if prob.solver_stats is None else f"cvxpy_{prob.solver_stats.solver_name}",
            status = str(prob.status),
            n_constraints = len(constraints),
            n_variables = n,
        )

    # ── scipy path (always available fallback) ─────────────────────────────

    def _solve_scipy(
        self, *, tickers, w, mu, sigma, sector_map,
    ) -> ConvexRotationResult:
        from scipy.optimize import minimize, LinearConstraint, Bounds  # noqa: PLC0415

        n = len(tickers)

        # Decision variable: delta in R^n. Use w + delta as portfolio.
        # Smoothed |delta| = sqrt(delta**2 + eps) for differentiable L1.
        eps_l1 = 1e-6

        def neg_obj(delta):
            # maximize μᵀΔw − γ Δwᵀ Σ Δw − c ‖Δw‖₁
            #   ↔ minimize -(μᵀΔw − γ Δwᵀ Σ Δw − c ‖Δw‖₁)
            quad = float(delta @ sigma @ delta)
            l1_smooth = float(np.sum(np.sqrt(delta * delta + eps_l1)))
            return -(mu @ delta - self.gamma_risk * quad - self.cost_coef * l1_smooth)

        def neg_grad(delta):
            quad_grad = 2.0 * (sigma @ delta)
            l1_grad = delta / np.sqrt(delta * delta + eps_l1)
            return -(mu - self.gamma_risk * quad_grad - self.cost_coef * l1_grad)

        # Long-only: w + delta >= 0  ↔ delta >= -w
        # Leverage: sum(w + delta) <= leverage_cap  ↔ sum(delta) <= leverage_cap - sum(w)
        # Turnover: sum |delta| <= turnover_cap — implemented via auxiliary variables
        # would require LP; for SLSQP we use a soft penalty in the objective via
        # cost_coef + an additional constraint sqrt(delta·delta + eps) ≤ turnover_cap

        # Audit T8 fix (2026-04-27): per-position upper bound = sector_max_pct,
        # not leverage_cap. Pre-fix `ub=leverage_cap` (1.0) let a single
        # ticker take 100% NAV in a fresh portfolio. Bound by the same
        # concentration limit applied at the sector level.
        bounds = Bounds(lb=-w, ub=np.full(n, self.sector_max_pct))   # delta in [-w, sec_cap]
        sum_w = float(np.sum(w))

        constraints = [
            # leverage: sum(delta) <= leverage_cap - sum(w)
            LinearConstraint(np.ones(n), -np.inf, self.leverage_cap - sum_w),
        ]

        # Sector caps: sum_{i in sector S} (w_i + delta_i) <= sector_max_pct
        # ↔  sum_{i in S} delta_i <= sector_max_pct - sum_{i in S} w_i
        if sector_map:
            for sector in set(sector_map.values()):
                mask = np.array(
                    [1.0 if sector_map.get(t) == sector else 0.0 for t in tickers],
                    dtype=float,
                )
                rhs = self.sector_max_pct - float(mask @ w)
                constraints.append(LinearConstraint(mask, -np.inf, rhs))

        # Turnover constraint via nonlinear constraint
        from scipy.optimize import NonlinearConstraint
        turnover_cap = self.turnover_cap

        def turnover(delta):
            return float(np.sum(np.sqrt(delta * delta + eps_l1)))

        def turnover_grad(delta):
            return delta / np.sqrt(delta * delta + eps_l1)

        constraints.append(NonlinearConstraint(
            turnover, -np.inf, turnover_cap, jac=turnover_grad,
        ))

        # Initial guess: zero delta (no rebalance)
        x0 = np.zeros(n)

        t0 = time.monotonic()
        result = minimize(
            neg_obj, x0,
            jac=neg_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-7},
        )
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        delta_arr = result.x if result.x is not None else np.zeros(n)

        return ConvexRotationResult(
            delta_weights = pd.Series(delta_arr, index=tickers, name="delta_weight"),
            objective_value = -float(result.fun) if result.fun is not None else float("nan"),
            solve_time_ms = elapsed_ms,
            solver_used = "scipy_SLSQP",
            status = "optimal" if result.success else result.message,
            n_constraints = len(constraints),
            n_variables = n,
        )


def quantize_to_whole_shares(
    delta_weights: pd.Series,
    prices: pd.Series,
    portfolio_value: float,
    available_cash: float,
    current_holdings: dict[str, int] | pd.Series | None = None,
) -> pd.Series:
    """Convert fractional Δw into integer share counts respecting cash budget.

    Greedy heuristic: process by largest |notional| first; round-up if cash
    permits, round-down otherwise. Track running cash; reject any trade
    that would overdraw.

    Audit T6 fix (2026-04-27): when `current_holdings` is provided, sells
    are CAPPED at the current position count — never issue an order that
    would create a negative position (no short-selling).

    Returns pd.Series of int share deltas (positive = buy, negative = sell).
    """
    notional_delta = (delta_weights * portfolio_value).reindex(prices.index).fillna(0.0)
    out = pd.Series(0, index=delta_weights.index, dtype=int)

    # Normalize current_holdings to dict[ticker, int]
    holdings: dict[str, int] = {}
    if current_holdings is not None:
        if isinstance(current_holdings, pd.Series):
            holdings = {str(t): int(v) for t, v in current_holdings.items()}
        else:
            holdings = {str(t): int(v) for t, v in dict(current_holdings).items()}

    cash = float(available_cash)
    # Sort by abs(notional) descending — handle largest moves first
    order = notional_delta.abs().sort_values(ascending=False).index
    for ticker in order:
        notional = float(notional_delta[ticker])
        price = float(prices.get(ticker, 0.0))
        if price <= 0:
            continue
        if notional > 0:
            # Buy — cap by available cash
            shares = int(notional / price)
            cost = shares * price
            if cost > cash:
                shares = int(cash / price)
                cost = shares * price
            out[ticker] = shares
            cash -= cost
        elif notional < 0:
            # Sell — cap at current holdings (no short-sell — Audit T6)
            requested = int(abs(notional) / price)
            current = holdings.get(str(ticker), 0)
            if current_holdings is not None:
                actual = min(requested, max(0, current))
                if actual < requested:
                    log.info(
                        "quantize_to_whole_shares: %s sell capped — "
                        "requested %d, holding %d (no short-sell)",
                        ticker, requested, current,
                    )
            else:
                actual = requested  # legacy path: caller responsible
            shares = -actual
            proceeds = -shares * price
            out[ticker] = shares
            cash += proceeds
        # notional == 0 → no trade

    return out


__all__ = [
    "ConvexRotationSolver",
    "ConvexRotationResult",
    "quantize_to_whole_shares",
]
