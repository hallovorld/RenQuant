"""Regression tests for the execution-fee model.

Pins the Alpaca / FINRA / SEC Q4 2025 commission schedule so a stray
edit to ``kernel.execution.fees`` immediately breaks CI rather than
silently inflating sim P&L.

Per CLAUDE.md §5.13.3 — every fix names the invariant. Per §5.13.5 —
single source of truth (every callsite imports from this module).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.execution.fees import (  # noqa: E402
    FeeConfig,
    compute_buy_fees,
    compute_sell_fees,
)


class TestFeeRegressionAlpacaSchedule:
    """AUDIT REGRESSION GUARD — pin Alpaca Q4 2025 commission schedule.

    Reverting these defaults silently inflates sim APY. The numbers in
    this class are the load-bearing invariants.
    """

    def test_sec_fee_on_100k_sell_matches_alpaca_schedule(self):
        # $100,000 notional sell → SEC §31 fee at $27 per $1M = $2.70
        fees = compute_sell_fees(shares=1000.0, price=100.0, cfg=FeeConfig())
        assert fees["sec_fee"] == pytest.approx(100_000 * 27.0e-6, rel=1e-6)
        # Exact dollar figure: $2.70
        assert fees["sec_fee"] == pytest.approx(2.70, abs=0.001)

    def test_taf_on_1000_shares_sold_matches_finra(self):
        # 1000 shares × $0.000119/share = $0.119
        fees = compute_sell_fees(shares=1000.0, price=50.0, cfg=FeeConfig())
        assert fees["taf"] == pytest.approx(1000 * 1.19e-4, rel=1e-6)
        # Exact: ~$0.12
        assert fees["taf"] == pytest.approx(0.119, abs=0.001)

    def test_buy_fees_zero_by_default_alpaca_zero_commission(self):
        # Alpaca / IBKR-Lite: zero commission on buys, no SEC/TAF on buys.
        fees = compute_buy_fees(shares=10_000.0, price=200.0, cfg=FeeConfig())
        assert fees["sec_fee"] == 0.0
        assert fees["taf"] == 0.0
        assert fees["custom"] == 0.0
        assert fees["total"] == 0.0

    def test_custom_bps_applies_to_both_sides(self):
        # IBKR-Pro Tiered example: 5 bps commission.
        cfg = FeeConfig(custom_bps=5.0)
        buy = compute_buy_fees(shares=100.0, price=100.0, cfg=cfg)
        sell = compute_sell_fees(shares=100.0, price=100.0, cfg=cfg)
        # 5 bps on $10k notional = $5.00
        assert buy["custom"] == pytest.approx(10_000 * 5.0 * 1e-4)
        assert sell["custom"] == pytest.approx(10_000 * 5.0 * 1e-4)


class TestFeeNaNGuards:
    """Per CLAUDE.md §5.13.11 — every monetary `>` / `<` is finite-guarded.

    A NaN or inf upstream price MUST collapse to zero fees, NOT propagate
    into total. Pre-guard a single bad bar poisoned every subsequent
    fee computation through `total += NaN`.
    """

    def test_nan_price_returns_zero_total(self):
        fees = compute_sell_fees(shares=100.0, price=float("nan"), cfg=FeeConfig())
        assert fees["total"] == 0.0
        assert math.isfinite(fees["total"])

    def test_nan_shares_returns_zero_total(self):
        fees = compute_sell_fees(shares=float("nan"), price=100.0, cfg=FeeConfig())
        assert fees["total"] == 0.0
        assert math.isfinite(fees["total"])

    def test_inf_price_returns_zero(self):
        fees = compute_sell_fees(shares=100.0, price=float("inf"), cfg=FeeConfig())
        assert fees["total"] == 0.0

    def test_negative_shares_treated_as_zero(self):
        # Defensive: negative quantity should not generate negative fees.
        fees = compute_sell_fees(shares=-100.0, price=100.0, cfg=FeeConfig())
        assert fees["sec_fee"] == 0.0
        assert fees["taf"] == 0.0
        assert fees["total"] == 0.0


class TestFeeBreakdownStructure:
    """Schema invariants: every output dict has the same 4 keys."""

    def test_sell_dict_has_required_keys(self):
        fees = compute_sell_fees(shares=10.0, price=50.0, cfg=FeeConfig())
        assert set(fees.keys()) == {"sec_fee", "taf", "custom", "total"}

    def test_buy_dict_has_required_keys(self):
        fees = compute_buy_fees(shares=10.0, price=50.0, cfg=FeeConfig())
        assert set(fees.keys()) == {"sec_fee", "taf", "custom", "total"}

    def test_total_equals_sum_of_components(self):
        cfg = FeeConfig(custom_bps=3.0)
        fees = compute_sell_fees(shares=500.0, price=120.0, cfg=cfg)
        assert fees["total"] == pytest.approx(
            fees["sec_fee"] + fees["taf"] + fees["custom"],
            rel=1e-9,
        )
