"""runner.py decomposition slice 8 — runner_ext_sell pure/parameterized tests."""
from __future__ import annotations

import datetime
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from adapters.runner_ext_sell import (  # noqa: E402
    attribute_ext_sell,
    bar_date,
    lookup_ext_sell_fills,
    normalize_fill_record,
)


class TestNormalizeFillRecord:
    def test_umbrella_schema(self):
        n = normalize_fill_record({"action": "SELL", "avg_price": 10.0,
                                   "qty": 5, "order_id": "o1", "filled_at": "t"})
        assert n == {"order_id": "o1", "side": "sell", "price": 10.0,
                     "qty": 5.0, "filled_at": "t"}

    def test_execution_schema_no_side(self):
        # execution subrepo: no side field, filled_avg_price/filled_qty/id
        n = normalize_fill_record({"filled_avg_price": 7.0, "filled_qty": 3,
                                   "id": "x", "filled_at": "t"})
        assert n["side"] == "" and n["price"] == 7.0 and n["qty"] == 3.0

    def test_zero_price_skipped(self):
        assert normalize_fill_record({"avg_price": 0, "action": "buy"})["price"] is None


class TestBarDate:
    def test_date_passthrough(self):
        assert bar_date(SimpleNamespace(today=datetime.date(2026, 6, 12))) == \
            datetime.date(2026, 6, 12)

    def test_datetime_normalized_to_date(self):
        dt = datetime.datetime(2026, 6, 12, 15, 30)
        assert bar_date(SimpleNamespace(today=dt)) == datetime.date(2026, 6, 12)


class TestLookupExtSellFills:
    def test_no_broker_api(self):
        assert lookup_ext_sell_fills(object(), SimpleNamespace(today=datetime.date(2026, 6, 12)), ["MU"]) == {}

    def test_empty_disappeared(self):
        assert lookup_ext_sell_fills(object(), SimpleNamespace(today=datetime.date(2026, 6, 12)), []) == {}

    def test_picks_latest_sell_for_ticker(self):
        class _B:
            def get_filled_orders(self, after):
                return [
                    {"symbol": "MU", "action": "SELL", "avg_price": 10, "qty": 2,
                     "order_id": "o1", "filled_at": "2026-06-10T15:00:00"},
                    {"symbol": "MU", "action": "SELL", "avg_price": 11, "qty": 2,
                     "order_id": "o2", "filled_at": "2026-06-11T15:00:00"},
                ]
        out = lookup_ext_sell_fills(_B(), SimpleNamespace(today=datetime.date(2026, 6, 12)), ["MU"])
        assert out["MU"]["order_id"] == "o2"  # most recent


class TestAttributeExtSell:
    def test_z9_stop(self):
        s = attribute_ext_sell({"MU": {"order_id": "z9"}}, {},
                               "MU", {"MU": {"order_id": "z9"}})
        assert "source=z9_stop" in s

    def test_runner_exit(self):
        s = attribute_ext_sell({}, {"o1": {"exit_type": "stop_loss"}},
                               "MU", {"MU": {"order_id": "o1"}})
        assert "source=runner_stop_loss" in s

    def test_external_or_manual(self):
        s = attribute_ext_sell({}, {}, "MU", {"MU": {"order_id": "unknown"}})
        assert "source=external_or_manual" in s

    def test_no_fill_record(self):
        assert attribute_ext_sell({}, {}, "MU", {}) == "no_broker_fill_record"
