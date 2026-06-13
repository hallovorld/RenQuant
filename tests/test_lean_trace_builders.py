"""Unit tests for the extracted LEAN execution-trace builders.

Pins the behavior of adapters/lean_trace.py (lean.py decomposition slice 1,
eng plan S2 item 5) at the module boundary, independent of the LeanAdapter
end-to-end persistence tests in test_lean_trace_persistence.py. These are the
pure functions that turn LEAN order attempts into decision-trace audit events.

REGRESSION GUARD: the builders MUST remain importable from BOTH
adapters.lean_trace (canonical home) and adapters.lean (back-compat
re-export) as the *same* object, so existing callers keep working after the
move.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from adapters import lean as _lean  # noqa: E402
from adapters import lean_trace as _lt  # noqa: E402

D = datetime.date(2026, 6, 13)


def _ctx(**kw):
    kw.setdefault("today", D)
    kw.setdefault("regime", "BULL_VOLATILE")
    kw.setdefault("confidence", 0.72)
    return SimpleNamespace(**kw)


class TestReexportIdentity:
    """The move must be transparent: lean.<fn> is lean_trace.<fn>."""

    def test_attempt_action_is_same_object(self):
        assert _lean._lean_attempt_action is _lt._lean_attempt_action

    def test_buy_event_is_same_object(self):
        assert _lean._lean_buy_attempt_event is _lt._lean_buy_attempt_event

    def test_sell_event_is_same_object(self):
        assert _lean._lean_sell_attempt_event is _lt._lean_sell_attempt_event


class TestAttemptAction:
    """status → terminal/non-terminal classification is keyword-driven and
    case-insensitive; anything not clearly terminal is treated as pending so
    a still-working order is never mislabeled as a hard rejection."""

    def test_rejected_keywords_map_to_rejected(self):
        for status in (
            "Rejected",
            "insufficient buying power — INVALID",
            "Canceled by user",
            "broker ERROR 500",
            "symbol missing from universe",
        ):
            assert _lt._lean_attempt_action("buy", status) == "buy_rejected", status

    def test_non_terminal_status_is_pending(self):
        for status in ("Submitted", "PartiallyFilled", "New", "working", ""):
            assert _lt._lean_attempt_action("sell", status) == "sell_pending", status

    def test_none_status_is_pending(self):
        assert _lt._lean_attempt_action("buy", None) == "buy_pending"

    def test_side_is_preserved_verbatim(self):
        assert _lt._lean_attempt_action("cover", "rejected") == "cover_rejected"


class TestBuyAttemptEvent:

    def _order(self, **kw):
        base = {"ticker": "AAPL", "shares": 10, "price": 100.0}
        base.update(kw)
        return base

    def test_action_defaults_from_status(self):
        row = _lt._lean_buy_attempt_event(
            self._order(), ctx=_ctx(), status="Rejected: buying power",
            blocked_by="insufficient_bp",
        )
        assert row["action"] == "buy_rejected"
        assert row["blocked_by"] == "insufficient_bp"
        assert row["exit_reason"] == "Rejected: buying power"

    def test_explicit_action_overrides_status(self):
        row = _lt._lean_buy_attempt_event(
            self._order(), ctx=_ctx(), status="Submitted",
            blocked_by="risk_cap", action="buy_held",
        )
        assert row["action"] == "buy_held"

    def test_status_threaded_into_inputs_and_snapshot(self):
        row = _lt._lean_buy_attempt_event(
            self._order(), ctx=_ctx(), status="New", blocked_by="pdt_guard",
        )
        for bag in (row["decision_inputs"], row["score_snapshot"]):
            assert bag["status"] == "New"
            assert bag["blocked_by"] == "pdt_guard"
            assert bag["attempt_status"] == "buy_pending"
        # acceptance_reason is seeded from blocked_by but not clobbered if
        # build_buy_trade_event already supplied one.
        assert row["decision_inputs"]["acceptance_reason"] == "pdt_guard"

    def test_attribution_version_is_attempt_marker(self):
        row = _lt._lean_buy_attempt_event(
            self._order(), ctx=_ctx(), status="Rejected", blocked_by="x",
        )
        assert row["attribution_version"] == "lean_execution_attempt_v1"
        assert row["date"] == D
        assert row["regime"] == "BULL_VOLATILE"


class TestSellAttemptEvent:

    def _sig(self, **kw):
        base = {"exit_type": "stop_loss", "reason": "sigma_breach"}
        base.update(kw)
        return SimpleNamespace(**base)

    def _holding(self, **kw):
        base = {
            "rank_score": 0.3, "panel_score": 0.1, "mu": -0.02,
            "model_type": "patchtst", "sector": "tech",
        }
        base.update(kw)
        return SimpleNamespace(**base)

    def test_core_fields_populated(self):
        row = _lt._lean_sell_attempt_event(
            ticker="NVDA", sig=self._sig(), holding=self._holding(),
            ctx=_ctx(), requested_shares=5.0, price=120.0, status="Rejected",
        )
        assert row["ticker"] == "NVDA"
        assert row["action"] == "sell_rejected"
        assert row["shares"] == 5.0
        assert row["price"] == 120.0
        assert row["date"] == D
        assert row["order_type"] == "SELL_ATTEMPT_stop_loss"
        assert row["attribution_version"] == "lean_execution_attempt_v1"

    def test_blocked_by_falls_back_to_status_then_action(self):
        # no blocked_by, no action → blocked = status
        row = _lt._lean_sell_attempt_event(
            ticker="NVDA", sig=self._sig(), holding=self._holding(),
            ctx=_ctx(), requested_shares=5.0, price=120.0, status="Canceled",
        )
        assert row["blocked_by"] == "Canceled"
        assert row["score_snapshot"]["blocked_by"] == "Canceled"
        assert row["decision_inputs"]["blocked_by"] == "Canceled"

    def test_source_routing_defaults_when_sig_silent(self):
        row = _lt._lean_sell_attempt_event(
            ticker="NVDA", sig=self._sig(), holding=self._holding(),
            ctx=_ctx(), requested_shares=5.0, price=120.0, status="New",
        )
        assert row["source"] == "LeanOrderTicket"
        assert row["source_job"] == "LeanOrderTicket"
        assert row["source_task"] == "sell_pending"
        assert row["order_source"] == "LeanOrderTicket.sell_pending"

    def test_source_routing_honors_sig_overrides(self):
        sig = self._sig(source_job="IntradayGovernor", source_task="forced_cover",
                        order_source="IntradayGovernor.forced_cover")
        row = _lt._lean_sell_attempt_event(
            ticker="NVDA", sig=sig, holding=self._holding(),
            ctx=_ctx(), requested_shares=5.0, price=120.0, status="Rejected",
        )
        assert row["source_job"] == "IntradayGovernor"
        assert row["source_task"] == "forced_cover"
        assert row["order_source"] == "IntradayGovernor.forced_cover"

    def test_snapshot_pulls_from_holding_and_ctx(self):
        row = _lt._lean_sell_attempt_event(
            ticker="NVDA", sig=self._sig(), holding=self._holding(mu=-0.05),
            ctx=_ctx(confidence=0.9), requested_shares=5.0, price=120.0,
            status="Rejected",
        )
        snap = row["score_snapshot"]
        assert snap["rank_score"] == 0.3
        assert snap["mu"] == -0.05
        assert snap["confidence"] == 0.9
        assert snap["model_type"] == "patchtst"
        assert snap["sector"] == "tech"
