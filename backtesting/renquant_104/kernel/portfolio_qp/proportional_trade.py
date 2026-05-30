"""Gârleanu-Pedersen partial-rebalance (research recommendation B).

The current QP architecture is "one-shot": solve for the optimal target w*,
then either trade to w* in this bar (modulo turnover_max + no-trade band)
or skip. Per Gârleanu-Pedersen 2013 *J.Fin.* 68(6):2309, the optimal
multi-period policy under proportional cost is to **trade partially
toward the target at a rate** that depends on cost / risk-aversion::

    x_t  =  (I - Λ_rate) · x_{t-1}  +  Λ_rate · aim_t

For the single-rate case (one ``rate`` per portfolio), this collapses to
the operationally simple form used by ``cvxportfolio.ProportionalTradeToTargets``::

    actual_w_t  =  current_w  +  (target_w - current_w) / N

where ``N`` is the number of trading days to traverse the gap. ``N=1`` is
the legacy "all-or-nothing" behavior; ``N=20`` smooths a 60-day rebalance
over a month; ``N → ∞`` never trades.

The trade-rate ``1/N`` is the GP-2013 optimal rate-matrix's scalar form.
``N`` is a per-regime knob (PRIME DIRECTIVE): BULL_CALM rebalances slowly
(N=20), BEAR / CHOPPY rebalance quickly (N=3-5) because alpha decays fast
in those regimes.

References:
  * Gârleanu, N. & Pedersen, L.H. (2013) "Dynamic Trading with Predictable
    Returns and Transaction Costs," *Journal of Finance* 68(6):2309-2340.
    https://nbgarleanu.github.io/DynTrad.pdf
  * Boyd, S. et al. (2017) "Multi-Period Trading via Convex Optimization,"
    *Foundations and Trends in Optimization* 3(1):1-76.
  * Verbatim reference impl: ``cvxgrp/cvxportfolio/cvxportfolio/policies.py
    ::ProportionalTradeToTargets``
    (https://github.com/cvxgrp/cvxportfolio/blob/master/cvxportfolio/policies.py)

The function is intentionally side-effect-free and vector-typed: pass
current_w + target_w as 1-D numpy arrays, get the partial target back.
Wiring into the QP pipeline lives in a new Task (sibling to
``EmitOrdersFromQPSolutionTask``) so this module can be tested in
isolation.
"""
from __future__ import annotations

import numpy as np


def proportional_trade_target(
    *,
    current_w: np.ndarray,
    target_w: np.ndarray,
    n_days: int | float,
) -> np.ndarray:
    """Compute the per-bar partial target weights (GP-2013 partial rebalance).

    Parameters
    ----------
    current_w : np.ndarray
        Current portfolio weights (1-D, len = n_assets).
    target_w : np.ndarray
        QP-computed target weights (same shape as ``current_w``).
    n_days : int | float
        Trading days to traverse the gap. ``1`` = legacy all-or-nothing,
        ``> 1`` = partial. Values ``≤ 0`` are coerced to 1 (defensive).

    Returns
    -------
    np.ndarray
        Partial target weights ``current_w + (target_w - current_w) / N``.
        Shape matches the inputs; never modifies them.
    """
    current = np.asarray(current_w, dtype=float)
    target = np.asarray(target_w, dtype=float)
    if current.shape != target.shape:
        raise ValueError(
            f"current_w {current.shape} and target_w {target.shape} must match"
        )
    n = max(float(n_days), 1.0)
    return current + (target - current) / n


def resolve_trade_horizon_days(
    *,
    regime: str | None,
    regime_params: dict,
    default_days: int | float | None,
) -> float:
    """Look up the per-regime partial-trade horizon, falling back to default.

    The PRIME DIRECTIVE says every numeric knob lives under
    ``regime_params.<REGIME>.<knob>``. This helper does that lookup with a
    graceful default. Returns ``1.0`` when nothing is configured (preserves
    legacy all-or-nothing).
    """
    if regime and isinstance(regime_params, dict):
        rp = regime_params.get(str(regime)) or {}
        if isinstance(rp, dict):
            val = rp.get("qp_partial_trade_horizon_days")
            if val is not None:
                return float(val)
    if default_days is not None:
        return float(default_days)
    return 1.0
