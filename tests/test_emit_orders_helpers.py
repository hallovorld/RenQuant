"""Unit tests for helper functions extracted from EmitOrdersFromQPSolutionTask
during the §1c split (2026-05-06).

Each helper has a single responsibility; testing them in isolation proves
the split preserved the original gate semantics."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


class TestPassesNoTradeBand:
    """Davis-Norman 1990 / Constantinides 1979 — no-trade band."""

    def test_passes_above_threshold(self):
        from kernel.portfolio_qp.tasks import _passes_no_trade_band
        # min_dw=0.005, no factor → trade if |dw| ≥ 0.005
        ok, in_band = _passes_no_trade_band(0.01, 0.0, 0.005, 0.0)
        assert ok is True
        assert in_band is False

    def test_blocks_below_min_dw(self):
        from kernel.portfolio_qp.tasks import _passes_no_trade_band
        ok, in_band = _passes_no_trade_band(0.001, 0.0, 0.005, 0.0)
        assert ok is False
        assert in_band is False    # below min_dw → not "in band"

    def test_in_band_above_min_dw_below_factor(self):
        from kernel.portfolio_qp.tasks import _passes_no_trade_band
        # min_dw=0.005, factor=2.0, σ=0.02 → threshold=max(0.005, 0.04)=0.04
        # dw=0.01 → above min but inside vol-band
        ok, in_band = _passes_no_trade_band(0.01, 0.02, 0.005, 2.0)
        assert ok is False
        assert in_band is True   # was above min_dw; in band due to high σ

    def test_factor_zero_is_legacy_uniform_min_dw(self):
        from kernel.portfolio_qp.tasks import _passes_no_trade_band
        # factor=0 → only min_dw matters
        ok, _ = _passes_no_trade_band(0.005, 1e6, 0.005, 0.0)
        assert ok is True   # exactly at threshold ⇒ pass


class TestGateBuyOrBlock:
    def test_sell_returns_none(self):
        from kernel.portfolio_qp.tasks import _gate_buy_or_block
        import datetime
        # dw < 0 → not a buy → no gate
        assert _gate_buy_or_block(
            "AAPL", -0.05, datetime.date(2026, 5, 6), {}, 3, False,
        ) is None

    def test_buys_gated(self):
        from kernel.portfolio_qp.tasks import _gate_buy_or_block
        import datetime
        assert _gate_buy_or_block(
            "AAPL", 0.05, datetime.date(2026, 5, 6), {}, 3, True,
        ) == "buys_gated"

    def test_earnings_blackout(self):
        from kernel.portfolio_qp.tasks import _gate_buy_or_block
        import datetime
        cal = {"AAPL": ["2026-05-08"]}   # is_earnings_blocked expects ISO strings
        # 2 days before earnings, buffer=3 → blocked
        assert _gate_buy_or_block(
            "AAPL", 0.05, datetime.date(2026, 5, 6), cal, 3, False,
        ) == "earnings"

    def test_passes_when_no_gates_fire(self):
        from kernel.portfolio_qp.tasks import _gate_buy_or_block
        import datetime
        cal = {"AAPL": ["2027-01-01"]}   # earnings far away
        assert _gate_buy_or_block(
            "AAPL", 0.05, datetime.date(2026, 5, 6), cal, 3, False,
        ) is None


class TestSharesFromDw:
    def test_basic_calculation(self):
        from kernel.portfolio_qp.tasks import _shares_from_dw
        # dw=0.05, nav=$10,000, px=$50 → 0.05 * 10000 / 50 = 10 shares
        assert _shares_from_dw(0.05, 10000.0, 50.0) == 10

    def test_floor_to_int(self):
        from kernel.portfolio_qp.tasks import _shares_from_dw
        # 0.03 * 1000 / 50 = 0.6 → floor to 0
        assert _shares_from_dw(0.03, 1000.0, 50.0) == 0

    def test_nan_dw_returns_zero(self):
        from kernel.portfolio_qp.tasks import _shares_from_dw
        import math
        assert _shares_from_dw(math.nan, 10000.0, 50.0) == 0

    def test_inf_inputs_return_zero(self):
        from kernel.portfolio_qp.tasks import _shares_from_dw
        import math
        assert _shares_from_dw(math.inf, 10000.0, 50.0) == 0
        assert _shares_from_dw(0.05, math.inf, 50.0) == 0
        assert _shares_from_dw(0.05, 10000.0, math.inf) == 0

    def test_zero_or_negative_inputs_return_zero(self):
        from kernel.portfolio_qp.tasks import _shares_from_dw
        assert _shares_from_dw(0.05, 0.0, 50.0) == 0
        assert _shares_from_dw(0.05, -1.0, 50.0) == 0
        assert _shares_from_dw(0.05, 10000.0, 0.0) == 0
        assert _shares_from_dw(0.05, 10000.0, -50.0) == 0
