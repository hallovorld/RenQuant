"""BUG #7 regression tests — σ-derived no-trade band cap.

Pre-fix: `threshold = max(min_dw, no_trade_factor × σ)` produced 24%+
bands for high-σ holdings (BA σ̂=0.24 → 24% band) → 6.7% needed Δw
suppressed → BA could never be sold despite μ̂=-0.12 (12% expected
underperformance over 60d).

Post-fix: σ-derived band capped at `band_cap` (default 5%). Hard floor
at min_dw remains. Result: high-σ holdings can be rebalanced when the
needed Δw exceeds 5%.

Reference: Davis-Norman (1990) "Portfolio Selection with Transaction
Costs"; Constantinides (1979) "Multiperiod Consumption and Investment".
The bandwidth in those papers shrinks with σ² (precision-weighted),
not grows linearly — proper cvxportfolio-style implementation needed
later. This patch is the conservative quick-fix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.portfolio_qp.tasks import _passes_no_trade_band


class TestNoTradeBandCap:
    """Cap on σ-derived no-trade band — BUG #7 regression."""

    MIN_DW = 0.02
    FACTOR = 1.0     # 1σ band
    CAP = 0.05       # 5% hard cap

    def test_low_sigma_ticker_uses_min_dw(self):
        """σ < min_dw → band = min_dw."""
        # σ=0.01 → 1.0σ = 1%, but min_dw=2% wins
        assert _passes_no_trade_band(0.025, 0.01, 0.02, 1.0, self.CAP)[0]   # passes (Δw=2.5% > 2% threshold)
        assert not _passes_no_trade_band(0.015, 0.01, 0.02, 1.0, self.CAP)[0]   # fails (Δw=1.5% < 2%)

    def test_high_sigma_ticker_capped_at_5pct(self):
        """BUG #7 reproducer: BA-shaped σ=0.24 should produce 5% band, not 24%."""
        sigma = 0.24
        # Pre-fix: threshold = max(0.02, 1.0 × 0.24) = 0.24
        # Post-fix: threshold = max(0.02, min(0.05, 1.0 × 0.24)) = 0.05
        # Δw = 0.067 (6.7% — BA's actual rebalance need)
        ok, _ = _passes_no_trade_band(0.067, sigma, self.MIN_DW, self.FACTOR, self.CAP)
        assert ok, "BUG #7: σ=0.24 holding's 6.7% Δw must clear the 5%-capped band"

    def test_extreme_sigma_still_capped(self):
        """Even σ=1.0 (100% vol) doesn't produce > 5% band."""
        ok, _ = _passes_no_trade_band(0.06, 1.0, self.MIN_DW, self.FACTOR, self.CAP)
        assert ok, "Δw=6% > 5% cap must clear regardless of σ"

    def test_below_cap_band_still_blocks_small_dw(self):
        """If σ × factor < cap, threshold is σ-driven (uncapped behavior)."""
        # σ=0.03, factor=1.0 → σ-band = 3% < 5% cap → threshold = 3%
        ok, _ = _passes_no_trade_band(0.025, 0.03, self.MIN_DW, self.FACTOR, self.CAP)
        assert not ok, "Δw=2.5% < 3% σ-band must be suppressed"
        ok, _ = _passes_no_trade_band(0.035, 0.03, self.MIN_DW, self.FACTOR, self.CAP)
        assert ok, "Δw=3.5% > 3% σ-band must pass"

    def test_default_band_cap_5pct(self):
        """Default band_cap is 5% when not specified."""
        # Pre-fix would have suppressed; post-fix cap defaults to 5%
        ok, _ = _passes_no_trade_band(0.06, 0.30, self.MIN_DW, self.FACTOR)   # no band_cap kwarg
        assert ok, "Default band_cap=5% should let 6% Δw pass at high σ"

    def test_min_dw_floor_still_holds(self):
        """min_dw is the absolute lower bound — even σ=0 doesn't allow below it.
        in_band = (Δw is between min_dw and threshold) — only True when σ-band
        suppressed the trade. Δw < min_dw is "too small to bother" not σ-suppressed."""
        ok, in_band = _passes_no_trade_band(0.01, 0.0, self.MIN_DW, self.FACTOR, self.CAP)
        assert not ok    # blocked
        assert not in_band, "Δw=1% < 2% min_dw → not σ-band suppression"

    def test_zero_sigma_uses_min_dw_only(self):
        ok, _ = _passes_no_trade_band(0.025, 0.0, self.MIN_DW, self.FACTOR, self.CAP)
        assert ok, "Δw=2.5% > 2% min_dw at σ=0 must pass"

    def test_factor_zero_disables_sigma_scaling(self):
        """no_trade_factor=0 → only min_dw matters."""
        ok, _ = _passes_no_trade_band(0.025, 0.50, self.MIN_DW, 0.0, self.CAP)
        assert ok, "factor=0 disables σ-scaling regardless of σ"


class TestBABugReproduction:
    """Reproduces the exact BA scenario that triggered the discovery."""

    def test_ba_scenario_passes_post_fix(self):
        """BA: σ̂=0.2434, weight=6.73%, Kelly target=0%, Δw needed=-6.73%.
        Pre-fix: 24.34% band suppressed it. Post-fix: 5% cap → 6.73% > 5% → SELL."""
        ok, _ = _passes_no_trade_band(
            dw=-0.0673, sig_i=0.2434, min_dw=0.02, no_trade_factor=1.0,
            band_cap=0.05,
        )
        assert ok, ("BA must be sellable post BUG #7 fix: |Δw|=6.73% > "
                    "5% cap (was suppressed at 24.34% pre-fix)")

    def test_ba_scenario_pre_fix_would_suppress(self):
        """Sanity: confirm that without the cap, BA WAS suppressed."""
        # Simulate pre-fix by passing band_cap=999 (effectively no cap)
        ok, _ = _passes_no_trade_band(
            dw=-0.0673, sig_i=0.2434, min_dw=0.02, no_trade_factor=1.0,
            band_cap=999.0,
        )
        assert not ok, "Without cap, BA's 6.73% Δw is suppressed by 24.34% σ-band"
