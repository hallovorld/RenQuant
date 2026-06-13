"""runner.py decomposition slice 4 — runner_trace pure-function tests."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from adapters.runner_trace import (  # noqa: E402
    live_execution_attempt_events,
    live_trace_selection_maps,
)


def _ctx(**kw):
    base = dict(today="2026-06-12", regime="BULL_CALM", confidence=0.7,
                orders_pending=[], orders_skipped=[], exits_pending=[],
                exits_failed=[], holdings={}, prices={})
    base.update(kw)
    return SimpleNamespace(**base)


class TestSelectionMaps:
    def test_pending_marked_blocked(self):
        sel, blocked, pending = live_trace_selection_maps(
            trade_events=[], pending_orders=[{"ticker": "MU"}], blocked_map={})
        assert pending == {"MU"}
        assert blocked["MU"] == "broker_pending_submitted"

    def test_existing_blocked_preserved(self):
        _, blocked, _ = live_trace_selection_maps(
            trade_events=[], pending_orders=[{"ticker": "MU"}],
            blocked_map={"MU": "wash_sale"})
        assert blocked["MU"] == "wash_sale"  # setdefault never overwrites


class TestAttemptEvents:
    def test_buy_pending_event(self):
        ctx = _ctx(orders_pending=[{"ticker": "MU", "order_id": "o1",
                                    "status": "new"}])
        events = live_execution_attempt_events(ctx)
        assert len(events) == 1
        e = events[0]
        assert e["action"] == "buy_pending"
        assert e["blocked_by"] == "broker_pending_submitted"
        assert e["date"] == "2026-06-12" and e["regime"] == "BULL_CALM"

    def test_buy_skipped_carries_reason(self):
        ctx = _ctx(orders_skipped=[{"ticker": "GE", "skip_reason": "cash"}])
        e = live_execution_attempt_events(ctx)[0]
        assert e["action"] == "buy_skipped"
        assert "broker_skip:cash" in e["blocked_by"]

    def test_sell_rejected_event(self):
        ctx = _ctx(exits_failed=[{"ticker": "MU", "qty": 5,
                                  "exit_type": "stop_loss", "error": "rejected"}])
        e = live_execution_attempt_events(ctx)[0]
        assert e["action"] == "sell_rejected"
        assert e["blocked_by"] == "rejected"
        assert e["attribution_version"] == "live_execution_attempt_v1"

    def test_non_dict_items_ignored(self):
        ctx = _ctx(orders_pending=["not-a-dict", {"ticker": "MU"}])
        assert len(live_execution_attempt_events(ctx)) == 1
