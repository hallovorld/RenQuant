"""make_context decomposition — compute_broker_mark_prices tests (RU-PRICE-1)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from adapters.runner_prices import compute_broker_mark_prices  # noqa: E402


class TestComputeBrokerMarkPrices:
    def test_normal_mark(self):
        prices, marks = compute_broker_mark_prices(
            {"MU": {"qty": 10, "market_value": 1000.0}},
            sell_only=True, use_intraday_prices=False)
        assert marks["MU"] == 100.0 and prices["MU"] == 100.0

    def test_full_run_marks_not_in_prices(self):
        # full daily run: broker_mark_prices populated, prices empty
        prices, marks = compute_broker_mark_prices(
            {"MU": {"qty": 10, "market_value": 1000.0}},
            sell_only=False, use_intraday_prices=False)
        assert marks["MU"] == 100.0 and "MU" not in prices

    def test_dust_qty_rejected_ru_price_1(self):
        # 1e-6 shares, $100 mkt → would be $100M/share; must be rejected
        prices, marks = compute_broker_mark_prices(
            {"MU": {"qty": 1e-6, "market_value": 100.0}},
            sell_only=True, use_intraday_prices=False)
        assert "MU" not in marks and "MU" not in prices

    def test_nonfinite_rejected(self):
        prices, marks = compute_broker_mark_prices(
            {"MU": {"qty": float("nan"), "market_value": 100.0}},
            sell_only=True, use_intraday_prices=False)
        assert marks == {}

    def test_zero_mkt_rejected(self):
        prices, marks = compute_broker_mark_prices(
            {"MU": {"qty": 10, "market_value": 0.0}},
            sell_only=True, use_intraday_prices=False)
        assert marks == {}

    def test_insane_price_capped(self):
        # price >= 1e6 rejected as a sanity cap
        prices, marks = compute_broker_mark_prices(
            {"X": {"qty": 1, "market_value": 2e6}},
            sell_only=True, use_intraday_prices=False)
        assert marks == {}
