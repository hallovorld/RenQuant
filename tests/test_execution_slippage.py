"""Regression tests for the slippage model.

Pins:
- half-spread directionality (buy pays more, sell receives less)
- linear-impact scaling with ADV ratio
- defensive zero-ADV behavior
- §5.13.12 hard-clip on bps (50 bps each side max)

Per CLAUDE.md §5.13.3 — names the invariant. Per §5.13.11 — every
numeric branch finite-guarded.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.execution.slippage import (  # noqa: E402
    SlippageConfig,
    slip_fill_price,
)


class TestHalfSpreadDirectionality:
    """AUDIT REGRESSION GUARD — buy pays more, sell receives less.

    Inverting either side silently inflates sim APY by ~0.5-1.0%/yr on
    a 100-trade/year strategy. This is the directionality invariant.
    """

    def test_buy_fill_above_market_with_default_half_spread(self):
        cfg = SlippageConfig()  # 2 bps half-spread default
        market = 100.0
        fill = slip_fill_price(market, "buy", shares=100.0, adv_shares=None, cfg=cfg)
        assert fill > market
        # 2 bps = 0.02% — expect 100 * (1 + 0.0002) = 100.02
        assert fill == pytest.approx(100.02, rel=1e-6)

    def test_sell_fill_below_market_with_default_half_spread(self):
        cfg = SlippageConfig()
        fill = slip_fill_price(100.0, "sell", shares=100.0, adv_shares=None, cfg=cfg)
        assert fill < 100.0
        assert fill == pytest.approx(99.98, rel=1e-6)

    def test_zero_half_spread_returns_market(self):
        cfg = SlippageConfig(half_spread_bps=0.0)
        fill_b = slip_fill_price(50.0, "buy", 100, None, cfg)
        fill_s = slip_fill_price(50.0, "sell", 100, None, cfg)
        assert fill_b == pytest.approx(50.0)
        assert fill_s == pytest.approx(50.0)


class TestImpactScalesWithADV:
    """Impact factor is linear in shares/ADV ratio (Almgren-Chriss)."""

    def test_impact_doubles_when_shares_double(self):
        cfg = SlippageConfig(half_spread_bps=0.0,
                             impact_bps_per_pct_adv=100.0)
        # 1% ADV → 100 bps × 0.01 = 1 bp; doubling shares → 2 bps
        fill_1 = slip_fill_price(100.0, "buy", shares=100, adv_shares=10_000, cfg=cfg)
        fill_2 = slip_fill_price(100.0, "buy", shares=200, adv_shares=10_000, cfg=cfg)
        # fill_1 = 100 * (1 + 1e-4); fill_2 = 100 * (1 + 2e-4)
        assert fill_1 == pytest.approx(100.01, rel=1e-6)
        assert fill_2 == pytest.approx(100.02, rel=1e-6)

    def test_zero_adv_yields_zero_impact_defensive(self):
        """Per spec: missing ADV must NOT block the fill — return 0 impact."""
        cfg = SlippageConfig(half_spread_bps=0.0,
                             impact_bps_per_pct_adv=1000.0)
        # ADV=0 should yield zero impact, not divide-by-zero
        fill = slip_fill_price(100.0, "buy", shares=100, adv_shares=0, cfg=cfg)
        assert fill == pytest.approx(100.0)
        # ADV=None same
        fill_none = slip_fill_price(100.0, "buy", shares=100, adv_shares=None, cfg=cfg)
        assert fill_none == pytest.approx(100.0)

    def test_negative_adv_yields_zero_impact(self):
        cfg = SlippageConfig(half_spread_bps=0.0,
                             impact_bps_per_pct_adv=1000.0)
        fill = slip_fill_price(100.0, "buy", 100, -500.0, cfg)
        assert fill == pytest.approx(100.0)


class TestHardClipAt50Bps:
    """§5.13.12 — fat-finger config (200 bps typo) clipped to 50 bps."""

    def test_excessive_half_spread_clipped(self):
        cfg = SlippageConfig(half_spread_bps=200.0)  # typo for "2"
        fill = slip_fill_price(100.0, "buy", shares=100, adv_shares=None, cfg=cfg)
        # Clipped to 50 bps = 0.5% → 100.50, NOT 102.00
        assert fill == pytest.approx(100.50, rel=1e-6)

    def test_excessive_impact_clipped(self):
        cfg = SlippageConfig(half_spread_bps=0.0,
                             impact_bps_per_pct_adv=999_999.0)
        # 1% ADV × 999,999 bps = 9999 bps raw → clipped to 50 bps
        fill = slip_fill_price(100.0, "buy", shares=100, adv_shares=10_000, cfg=cfg)
        # 50 bps = 0.5% → 100.50
        assert fill == pytest.approx(100.50, rel=1e-6)


class TestSlippageNaNGuards:
    """§5.13.11 — non-finite market price must not propagate NaN."""

    def test_nan_market_returns_unchanged(self):
        cfg = SlippageConfig()
        fill = slip_fill_price(float("nan"), "buy", 100, None, cfg)
        # Returns the input as-is (caller responsible for upstream reject)
        assert math.isnan(fill)

    def test_zero_market_returns_unchanged(self):
        fill = slip_fill_price(0.0, "buy", 100, None, SlippageConfig())
        assert fill == 0.0

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError):
            slip_fill_price(100.0, "HOLD", 100, None, SlippageConfig())
