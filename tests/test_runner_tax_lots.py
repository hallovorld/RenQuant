"""runner.py decomposition slice 6 — runner_tax_lots pure-function tests."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from adapters.runner_tax_lots import (  # noqa: E402
    reconstruct_live_tax_lots_from_fills,
    sell_event_price,
)


class TestSellEventPrice:
    def test_uses_sig_sell_price(self):
        sig = SimpleNamespace(sell_price=12.5)
        assert sell_event_price(sig, fallback_price=10.0) == 12.5

    def test_falls_back_when_missing(self):
        sig = SimpleNamespace(sell_price=None)
        assert sell_event_price(sig, fallback_price=10.0) == 10.0

    def test_nonfinite_falls_back(self):
        sig = SimpleNamespace(sell_price=float("nan"))
        assert sell_event_price(sig, fallback_price=9.0) == 9.0

    def test_both_invalid_returns_zero(self):
        sig = SimpleNamespace(sell_price=None)
        assert sell_event_price(sig, fallback_price=None) == 0.0


class TestReconstructTaxLots:
    def test_returns_dict_keyed_by_ticker(self):
        fills = [
            {"symbol": "MU", "action": "BUY", "qty": 10, "avg_price": 100.0,
             "filled_at": "2026-05-01T15:00:00Z"},
            {"symbol": "MU", "action": "SELL", "qty": 4, "avg_price": 110.0,
             "filled_at": "2026-05-10T15:00:00Z"},
        ]
        lots = reconstruct_live_tax_lots_from_fills(fills)
        assert isinstance(lots, dict)
        # MU has remaining long lots after a partial sell
        assert "MU" in lots

    def test_none_fills_empty_dict(self):
        assert reconstruct_live_tax_lots_from_fills(None) == {}

    def test_empty_fills_empty_dict(self):
        assert reconstruct_live_tax_lots_from_fills([]) == {}

    def test_multiple_symbols_separated(self):
        fills = [
            {"symbol": "MU", "action": "BUY", "qty": 5, "avg_price": 100.0,
             "filled_at": "2026-05-01T15:00:00Z"},
            {"symbol": "GE", "action": "BUY", "qty": 3, "avg_price": 50.0,
             "filled_at": "2026-05-01T15:00:00Z"},
        ]
        lots = reconstruct_live_tax_lots_from_fills(fills)
        assert set(lots) == {"MU", "GE"}
