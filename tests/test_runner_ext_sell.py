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
    ext_sell_fill_date,
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


class TestExtSellFillDate:
    """2026-07-01 fix: STATE-EXT-SELL must stamp the wash-sale clock from
    the ACTUAL broker SELL fill date it already looked up via
    ``lookup_ext_sell_fills``, not from ``today_str`` (the date the
    reconciliation code happens to run).

    Confirmed live: META's ``last_sell_dates`` was wrongly stamped with
    the reconciliation RUN date (2026-06-26) instead of the real broker
    SELL fill date (2026-06-02) — a 24-day wash-sale over-extension.
    """

    def test_extracts_date_from_normalized_fill(self):
        fill = {"order_id": "o1", "side": "sell", "price": 590.0,
                "qty": 5.0, "filled_at": "2026-06-02T14:31:00-04:00"}
        assert ext_sell_fill_date(fill) == datetime.date(2026, 6, 2)

    def test_none_fill_returns_none(self):
        assert ext_sell_fill_date(None) is None

    def test_empty_dict_returns_none(self):
        assert ext_sell_fill_date({}) is None

    def test_missing_filled_at_returns_none(self):
        assert ext_sell_fill_date({"order_id": "o1", "side": "sell"}) is None

    def test_unparseable_filled_at_returns_none(self):
        assert ext_sell_fill_date({"filled_at": "not-a-date"}) is None

    def test_reconciliation_delay_scenario_uses_real_fill_date_not_today(self):
        """The exact scenario behind the live META incident, reproduced at
        the function-composition level ``adapters/runner.py``'s
        STATE-EXT-SELL block uses (``lookup_ext_sell_fills`` then
        ``ext_sell_fill_date`` on its result): a broker SELL fill really
        happened on D1, but the reconciliation/GC step that would normally
        catch the disappearance was skipped for several bars (e.g. an
        unrelated pipeline failure), so ``ctx.today`` — and the
        ``disappeared`` check — doesn't fire until a LATER date D2 > D1.

        The stamped wash-sale date must be D1 (the real fill), never D2
        (today, when reconciliation happened to catch it)."""
        d1 = datetime.date(2026, 6, 2)   # real broker SELL fill date
        d2 = datetime.date(2026, 6, 5)   # date reconciliation actually runs

        class _Broker:
            def get_filled_orders(self, after):
                return [{
                    "symbol": "META", "action": "SELL", "avg_price": 590.0,
                    "qty": 10, "order_id": "real-fill-1",
                    "filled_at": d1.isoformat() + "T14:31:00-04:00",
                }]

        ctx = SimpleNamespace(today=d2)
        ext_sell_fills = lookup_ext_sell_fills(_Broker(), ctx, ["META"])

        stamped = ext_sell_fill_date(ext_sell_fills.get("META"))

        assert stamped == d1, (
            f"stamped={stamped}, expected the REAL fill date {d1}, not the "
            f"reconciliation-run date {d2} (META incident regression)"
        )
        assert stamped != d2

    def test_no_broker_fill_found_yields_none_not_a_fabricated_date(self):
        """Genuine unknown-cause disappearance: the broker has no matching
        SELL fill for the ticker at all (corporate action / account
        transfer / a disposition the broker API can't attribute to a dated
        fill). The composed lookup must yield ``None`` so the caller in
        ``runner.py`` takes the explicit NO-FILL-FOUND fallback path
        (stamps ``today_str`` with a distinct log marker) instead of
        inventing a non-today date."""
        class _Broker:
            def get_filled_orders(self, after):
                return []   # no fills at all

        ctx = SimpleNamespace(today=datetime.date(2026, 6, 5))
        ext_sell_fills = lookup_ext_sell_fills(_Broker(), ctx, ["ZZZZ"])

        assert ext_sell_fill_date(ext_sell_fills.get("ZZZZ")) is None
