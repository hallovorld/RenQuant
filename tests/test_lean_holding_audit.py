"""Unit tests for the extracted LEAN holding audit-field stamping helpers.

Pins adapters/lean_holding_audit.py (lean.py decomposition slice 5). These
pure functions pull decision-trace fields from an order (top-level, else from
its score_snapshot / decision_inputs payloads) and stamp them onto the live
holding so LEAN audit rows carry the same attribution sim/live emit.

REGRESSION GUARD: importable from BOTH adapters.lean_holding_audit (canonical)
and adapters.lean (back-compat re-export) as the SAME object.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from adapters import lean as _lean  # noqa: E402
from adapters import lean_holding_audit as _lh  # noqa: E402


class TestReexportIdentity:
    def test_same_objects(self):
        assert _lean._order_payload is _lh._order_payload
        assert _lean._stamp_holding_audit_fields is _lh._stamp_holding_audit_fields


class TestOrderPayload:
    def test_top_level_value_wins(self):
        assert _lh._order_payload({"mu": 0.5, "score_snapshot": {"mu": 9}}, "mu") == 0.5

    def test_falls_back_to_score_snapshot(self):
        assert _lh._order_payload({"score_snapshot": {"sector": "tech"}}, "sector") == "tech"

    def test_falls_back_to_decision_inputs(self):
        assert _lh._order_payload({"decision_inputs": {"blocked_by": "pdt"}}, "blocked_by") == "pdt"

    def test_score_snapshot_preferred_over_decision_inputs(self):
        order = {"score_snapshot": {"mu": 1.0}, "decision_inputs": {"mu": 2.0}}
        assert _lh._order_payload(order, "mu") == 1.0

    def test_none_when_absent_everywhere(self):
        assert _lh._order_payload({"other": 1}, "mu") is None

    def test_non_dict_payload_is_skipped(self):
        # score_snapshot present but not a dict → ignored, no crash.
        assert _lh._order_payload({"score_snapshot": "oops"}, "mu") is None


class TestStampHoldingAuditFields:
    def test_stamps_all_present_fields(self):
        holding = NS()
        order = {
            "model_type": "patchtst", "sector": "tech", "mu": -0.02,
            "rank_score": 0.3, "kelly_target_pct": 0.05,
        }
        _lh._stamp_holding_audit_fields(holding, order)
        assert holding.model_type == "patchtst"
        assert holding.sector == "tech"
        assert holding.mu == -0.02
        assert holding.rank_score == 0.3
        assert holding.kelly_target_pct == 0.05

    def test_stamps_from_nested_payloads(self):
        holding = NS()
        order = {"score_snapshot": {"sigma": 0.1}, "decision_inputs": {"blocked_by": "veto"}}
        _lh._stamp_holding_audit_fields(holding, order)
        assert holding.sigma == 0.1
        assert holding.blocked_by == "veto"

    def test_absent_fields_not_set(self):
        holding = NS()
        _lh._stamp_holding_audit_fields(holding, {"mu": 1.0})
        assert holding.mu == 1.0
        assert not hasattr(holding, "sector")

    def test_none_holding_is_noop(self):
        # must not raise
        _lh._stamp_holding_audit_fields(None, {"mu": 1.0})

    def test_non_dict_order_is_noop(self):
        holding = NS()
        _lh._stamp_holding_audit_fields(holding, "not-a-dict")
        assert not hasattr(holding, "mu")
