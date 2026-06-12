"""P0.2a protective-order census tests (intraday roadmap §4 P0.2 gate)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from protective_census import _alert, census  # noqa: E402


def _pos(symbol, qty):
    return {"symbol": symbol, "qty": qty}


def _order(symbol, side="sell", type="stop", qty=10):
    return {"symbol": symbol, "side": side, "type": type, "qty": qty}


class TestCensus:

    def test_long_with_full_stop_protected(self):
        r = census([_pos("MU", 10)], [_order("MU", qty=10)])
        assert [e["symbol"] for e in r["protected"]] == ["MU"]
        assert r["naked"] == []

    def test_long_without_orders_naked(self):
        r = census([_pos("MU", 10)], [])
        assert [e["symbol"] for e in r["naked"]] == ["MU"]

    def test_partial_coverage_is_naked(self):
        # The G1 invariant is FULL coverage; a half-covered position is
        # still a gap.
        r = census([_pos("MU", 10)], [_order("MU", qty=4)])
        assert r["naked"][0]["covered_qty"] == 4

    def test_multiple_stops_sum(self):
        r = census([_pos("MU", 10)],
                   [_order("MU", qty=6), _order("MU", type="trailing_stop", qty=4)])
        assert r["naked"] == []

    def test_plain_limit_sell_is_not_protection(self):
        # A take-profit limit does not protect the downside.
        r = census([_pos("MU", 10)], [_order("MU", type="limit", qty=10)])
        assert [e["symbol"] for e in r["naked"]] == ["MU"]

    def test_short_needs_protective_buy(self):
        r = census([_pos("XYZ", -5)], [_order("XYZ", side="buy", qty=5)])
        assert r["naked"] == []
        r2 = census([_pos("XYZ", -5)], [_order("XYZ", side="sell", qty=5)])
        assert [e["symbol"] for e in r2["naked"]] == ["XYZ"]

    def test_orphan_protective_orders_reported(self):
        r = census([], [_order("GE", qty=3)])
        assert r["orphan_orders"] == ["GE"]

    def test_zero_qty_position_ignored(self):
        r = census([_pos("MU", 0)], [])
        assert r["protected"] == [] and r["naked"] == []


def test_alert_uses_ntfy_url_and_event(monkeypatch):
    import live.alerts as alerts

    seen = {}

    def _fake_post(url, event, *, logger=None, state_path=None):
        seen["url"] = url
        seen["event"] = event
        seen["logger"] = logger
        return True

    monkeypatch.setattr(alerts, "post_ntfy_alert", _fake_post)
    monkeypatch.setenv("RENQUANT_NTFY_TOPIC", "ops-test")

    _alert("Title", "Body", ("naked", "2026-06-12"), priority="urgent")

    assert seen["url"] == "https://ntfy.sh/ops-test"
    assert seen["event"].taxonomy == "census.protective_orders"
    assert seen["event"].title == "Title"
    assert seen["event"].priority == "urgent"
