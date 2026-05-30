"""Davis-Norman closed-form no-trade band (canonical implementation).

The current QP band is parameterised by three ad-hoc knobs::

    eff_band = max(qp_min_dw_pct, min(qp_no_trade_band_cap,
                                       qp_no_trade_band_factor × σ_h))

with defaults ``(0.02, 0.05, 1.0)``. These are hand-tuned; the literature
gives a closed-form derived from (cost, σ, γ, π*).

This module implements the Davis-Norman 1990 asymptotic with the
Janeček-Shreve 2004 refinement::

    δ*(cost, σ, γ, π*)  =  ( 1.5/γ · ε · π*·(1-π*)² · σ² )^(1/3)

where:
  ε       = proportional one-way transaction cost (fee + slippage)
  σ       = annualised volatility of the asset
  γ       = risk aversion coefficient
  π*      = frictionless target weight (a.k.a. "Merton fraction")

Derivation references:
  * Davis, M.H.A. & Norman, A.R. (1990) "Portfolio Selection with Transaction
    Costs," *Math.Op.Res.* 15(4):676. https://doi.org/10.1287/moor.15.4.676
  * Janeček, K. & Shreve, S.E. (2004) "Asymptotic Analysis for Optimal
    Investment and Consumption with Transaction Costs," *F&S* 8:181.
  * Guasoni, P. & Muhle-Karbe, J. (2013) "Portfolio Choice with Transaction
    Costs: A User's Guide," *Annual Review of Financial Economics* 5:75.
  * Whalley, A.E. & Wilmott, P. (1997) "An Asymptotic Analysis of an Optimal
    Hedging Model for Option Pricing with Transaction Costs," *Math.Fin.* 7:307.

The 1/3 exponent is universal under proportional cost; under fixed cost the
exponent is 1/4 (Atkinson-Wilmott 1995). RenQuant uses proportional costs
(fee_pct + slippage_pct), so 1/3 applies.

Usage::

    band = davis_norman_band(eps_oneway=0.001, sigma=0.20, gamma=3.0, pi_star=0.07)
    # → ~0.011 (≈ 1.1%), the literature-correct floor for those params.

Compare to the hand-tuned 2% ``qp_min_dw_pct``: DN gives ~half. The 2% is
about 2× the optimum for typical mid-cap names with σ≈0.20 at γ=3.0.
"""
from __future__ import annotations

import math


def davis_norman_band(
    *,
    eps_oneway: float,
    sigma: float,
    gamma: float,
    pi_star: float,
) -> float:
    """Closed-form Davis-Norman no-trade band half-width (proportional cost).

    All inputs are positive scalars; the function clamps inputs into sensible
    ranges and returns 0 if any input is non-positive (preserves "no band"
    semantics when the inputs are missing).

    Parameters
    ----------
    eps_oneway : float
        One-way proportional transaction cost (fee_pct + slippage_pct). For
        round-trip pass ``cost_round_trip / 2``.
    sigma : float
        Annualised volatility of the asset (or portfolio average for an
        aggregate band).
    gamma : float
        Risk-aversion coefficient. Higher γ → tighter band (more aggressive
        rebalancing).
    pi_star : float
        Frictionless target weight (e.g. Merton or post-QP target). The DN
        asymptotic was derived in the single-asset Merton setting; applied
        per-asset in a portfolio, ``π*`` is the asset's optimum weight.

    Returns
    -------
    float
        Band half-width as a fraction (e.g. ``0.011`` = 1.1%). The full
        no-trade region is then ``[π* - δ, π* + δ]``.
    """
    if eps_oneway <= 0 or sigma <= 0 or gamma <= 0 or pi_star <= 0:
        return 0.0
    # Numerical safety: ensure pi_star ∈ (0, 1) for the (1-π*)² term.
    pi = max(min(float(pi_star), 0.999), 0.001)
    inside = (1.5 / float(gamma)) * float(eps_oneway) * pi * (1.0 - pi) ** 2 * float(sigma) ** 2
    if inside <= 0:
        return 0.0
    return float(inside ** (1.0 / 3.0))


def davis_norman_band_clamped(
    *,
    eps_oneway: float,
    sigma: float,
    gamma: float,
    pi_star: float,
    floor: float = 0.0,
    ceiling: float = 1.0,
) -> float:
    """Clamped wrapper for production use.

    The raw DN formula can produce very tight bands (<10 bps) for low-σ /
    low-cost regimes. Operators may still want a minimum band to avoid
    nuisance turnover; ``floor`` provides that. ``ceiling`` caps extreme
    outputs at the natural per-asset weight cap.
    """
    band = davis_norman_band(
        eps_oneway=eps_oneway, sigma=sigma, gamma=gamma, pi_star=pi_star,
    )
    return float(max(floor, min(ceiling, band)))


def round_trip_to_one_way(round_trip_cost: float) -> float:
    """Helper: convert a round-trip cost (e.g. ``qp_cost_kappa=0.002``) to
    the one-way ε the DN formula expects."""
    return float(round_trip_cost) / 2.0 if round_trip_cost > 0 else 0.0
