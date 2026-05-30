"""Davis-Norman closed-form band — unit tests.

Pins the 1/3-power scaling and basic invariants. Includes a sanity-anchor:
the formula evaluated at typical RenQuant params (cost=0.001, σ=0.20,
γ=3.0, π*=0.07) lands at ≈ 1.1% — matching the back-of-envelope in the
research report (Davis-Norman 1990 / Janeček-Shreve 2004 / Guasoni-Muhle-
Karbe 2013).
"""
from __future__ import annotations

import math

import pytest

from backtesting.renquant_104.kernel.portfolio_qp.davis_norman import (
    davis_norman_band,
    davis_norman_band_clamped,
    round_trip_to_one_way,
)


def test_typical_renquant_params_around_one_percent():
    """The literature-anchored sanity case from the 2026-05-30 research report."""
    band = davis_norman_band(
        eps_oneway=0.001, sigma=0.20, gamma=3.0, pi_star=0.07,
    )
    # Research report: ~0.011 (≈ 1.1%). Allow ±20% headroom for floating-point
    # / sensitivity to pi*(1-π*)² rounding.
    assert 0.008 <= band <= 0.014, f"expected ~1.1%, got {band:.4f}"


def test_zero_inputs_return_zero_band():
    """Defensive: any non-positive input → 0 band (no constraint)."""
    assert davis_norman_band(eps_oneway=0, sigma=0.2, gamma=3, pi_star=0.07) == 0
    assert davis_norman_band(eps_oneway=0.001, sigma=0, gamma=3, pi_star=0.07) == 0
    assert davis_norman_band(eps_oneway=0.001, sigma=0.2, gamma=0, pi_star=0.07) == 0
    assert davis_norman_band(eps_oneway=0.001, sigma=0.2, gamma=3, pi_star=0) == 0


def test_band_scales_as_cube_root_of_cost():
    """δ ∝ ε^(1/3): a 27× cost increase → 3× band."""
    base = davis_norman_band(eps_oneway=0.001, sigma=0.2, gamma=3, pi_star=0.07)
    higher = davis_norman_band(eps_oneway=0.027, sigma=0.2, gamma=3, pi_star=0.07)
    ratio = higher / base
    assert 2.85 < ratio < 3.15, f"expected ratio ≈ 3.0, got {ratio:.2f}"


def test_band_scales_as_two_thirds_of_volatility():
    """δ ∝ σ^(2/3): an 8× σ → 4× band."""
    base = davis_norman_band(eps_oneway=0.001, sigma=0.10, gamma=3, pi_star=0.07)
    higher = davis_norman_band(eps_oneway=0.001, sigma=0.80, gamma=3, pi_star=0.07)
    ratio = higher / base
    assert 3.85 < ratio < 4.15, f"expected ratio ≈ 4.0, got {ratio:.2f}"


def test_band_scales_as_inverse_cube_root_of_gamma():
    """δ ∝ γ^(-1/3): an 8× γ → 0.5× band."""
    base = davis_norman_band(eps_oneway=0.001, sigma=0.2, gamma=1.0, pi_star=0.07)
    higher = davis_norman_band(eps_oneway=0.001, sigma=0.2, gamma=8.0, pi_star=0.07)
    ratio = higher / base
    assert 0.45 < ratio < 0.55, f"expected ratio ≈ 0.5, got {ratio:.2f}"


def test_pi_star_clamped_in_unit_interval():
    """π* outside (0, 1) is clamped to avoid (1-π*)² blowing up or going negative."""
    # π* = 1 (fully invested in one name) would give (1-π*)² = 0 → band=0
    band = davis_norman_band(eps_oneway=0.001, sigma=0.2, gamma=3, pi_star=0.999)
    assert 0 < band < 0.005  # very small, but not zero
    # π* clamped from 1.5 to 0.999
    band2 = davis_norman_band(eps_oneway=0.001, sigma=0.2, gamma=3, pi_star=1.5)
    assert band2 == pytest.approx(band, rel=1e-9)


def test_band_clamped_floor_and_ceiling():
    """Clamped wrapper respects floor + ceiling bounds."""
    # raw DN → ~0.011; clamp floor to 0.02 (the current qp_min_dw_pct)
    clamped = davis_norman_band_clamped(
        eps_oneway=0.001, sigma=0.20, gamma=3.0, pi_star=0.07,
        floor=0.02, ceiling=0.10,
    )
    assert clamped == pytest.approx(0.02, abs=1e-9)
    # Very small raw → floor binds
    clamped_floor = davis_norman_band_clamped(
        eps_oneway=0.0001, sigma=0.05, gamma=10.0, pi_star=0.05,
        floor=0.005, ceiling=0.10,
    )
    assert clamped_floor >= 0.005
    # Very large raw → ceiling binds (force with extreme inputs)
    clamped_ceil = davis_norman_band_clamped(
        eps_oneway=0.1, sigma=2.0, gamma=0.1, pi_star=0.30,
        floor=0.0, ceiling=0.05,
    )
    assert clamped_ceil == pytest.approx(0.05, abs=1e-9)


def test_round_trip_to_one_way_halves():
    assert round_trip_to_one_way(0.002) == pytest.approx(0.001)
    assert round_trip_to_one_way(0.0) == 0.0
    assert round_trip_to_one_way(-0.001) == 0.0


def test_band_for_renquant_meta_today():
    """The 2026-05-30 META scenario: σ=0.196, π*≈0.057, γ=3.0, cost=0.001.

    Current eff_band was 0.0500 (band_cap binding). META wanted to move Δw=-0.0376
    but got skipped because 0.0376 < 0.0500. With DN-derived band ≈ 0.01, the trade
    would have passed through.
    """
    band = davis_norman_band(
        eps_oneway=0.001, sigma=0.196, gamma=3.0, pi_star=0.057,
    )
    assert band < 0.038, (
        f"DN band {band:.4f} should be below today's META Δw=0.0376 so the trade "
        f"passes through; if this fails the DN scaling drifted"
    )
    # And the band IS in the ballpark of the round-figure 0.01
    assert 0.005 < band < 0.020
