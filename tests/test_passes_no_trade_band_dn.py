"""Tests for `_passes_no_trade_band` with the Davis-Norman closed-form path.

Pins:
  * legacy path unchanged when band_method='legacy' (default)
  * DN path uses the closed-form threshold when band_method='davis_norman'
  * The 2026-05-30 META scenario: Δw=0.0376 passes DN band but skipped by legacy
"""
from __future__ import annotations

import sys
from pathlib import Path

# tasks.py needs the strategy dir on sys.path so `kernel.*` imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backtesting/renquant_104"))

from kernel.portfolio_qp.tasks import _passes_no_trade_band


def test_legacy_default_path_unchanged():
    """No mode → legacy. min_dw=2%, factor=1.0, cap=5%. σ=0.10 → band=5% (cap)."""
    # Δw=3% — legacy band=max(2%, min(5%, 1.0×0.10))=5% (cap) → skipped
    passes, in_band = _passes_no_trade_band(
        dw=0.03, sig_i=0.10, min_dw=0.02, no_trade_factor=1.0, band_cap=0.05,
    )
    assert passes is False
    assert in_band is True  # |dw|=0.03 >= 0.02 min_dw


def test_legacy_at_high_sigma_band_capped():
    """High σ would create 24% band; cap at 5%. Δw=6% passes."""
    passes, _ = _passes_no_trade_band(
        dw=0.06, sig_i=0.24, min_dw=0.02, no_trade_factor=1.0, band_cap=0.05,
    )
    assert passes is True


def test_dn_path_passes_meta_scenario():
    """The 5/30 META scenario: σ=0.196, π*=0.057, ε=0.001, γ=3.0, Δw=0.0376.

    Legacy band: max(0.02, min(0.05, 1.0×0.196)) = 0.05 → SKIP (0.0376 < 0.05).
    DN band:    ≈ 0.011 → PASS (0.0376 > 0.011).
    """
    passes_legacy, _ = _passes_no_trade_band(
        dw=0.0376, sig_i=0.196, min_dw=0.02, no_trade_factor=1.0, band_cap=0.05,
    )
    assert passes_legacy is False, "legacy path should reject (band_cap binds)"

    passes_dn, _ = _passes_no_trade_band(
        dw=0.0376, sig_i=0.196, min_dw=0.02, no_trade_factor=1.0, band_cap=0.05,
        band_method="davis_norman",
        dn_eps_oneway=0.001, dn_gamma=3.0, dn_pi_star=0.057,
    )
    # DN raw ~0.011 but min_dw floor of 0.02 is still applied → effective ~0.02
    # 0.0376 > 0.02 → passes
    assert passes_dn is True, "DN path should let META Δw=0.0376 trade through"


def test_dn_path_still_respects_min_dw_floor():
    """DN can produce <min_dw values; min_dw is still a floor on the threshold.

    With ε=0.0001, σ=0.10, γ=10 → very small DN band (~0.003), but min_dw=0.02
    keeps the floor at 0.02. Δw=0.01 should be rejected.
    """
    passes, _ = _passes_no_trade_band(
        dw=0.01, sig_i=0.10, min_dw=0.02, no_trade_factor=1.0, band_cap=0.05,
        band_method="davis_norman",
        dn_eps_oneway=0.0001, dn_gamma=10.0, dn_pi_star=0.05,
    )
    assert passes is False


def test_dn_ceiling_clamps_extreme_band():
    """With pathologically high σ + cost, the DN raw output is capped at dn_ceiling."""
    # eps=0.1, σ=2.0, γ=0.1 → DN raw very large; ceiling clamps at 0.05
    passes, _ = _passes_no_trade_band(
        dw=0.06, sig_i=2.0, min_dw=0.02, no_trade_factor=1.0, band_cap=0.05,
        band_method="davis_norman",
        dn_eps_oneway=0.1, dn_gamma=0.1, dn_pi_star=0.3,
        dn_ceiling=0.05,
    )
    # ceiling=0.05; 0.06 > 0.05 → passes
    assert passes is True
