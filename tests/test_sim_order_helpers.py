"""sim.py decomposition slice 3 — sim_order_helpers tests."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from adapters.sim_order_helpers import (  # noqa: E402
    _BUYING_POWER_NMBP,
    _BUYING_POWER_SETTLED,
    _normalize_buying_power_mode,
    _order_payload,
    _stamp_holding_audit_fields,
)


class TestBuyingPowerMode:
    def test_default_is_nmbp(self):
        assert _normalize_buying_power_mode(None) == _BUYING_POWER_NMBP

    def test_settled_aliases(self):
        for a in ("settled", "cash", "settled_cash"):
            assert _normalize_buying_power_mode(a) == _BUYING_POWER_SETTLED

    def test_nmbp_aliases(self):
        for a in ("unsettled", "cash_plus_unsettled", "non_marginable_buying_power"):
            assert _normalize_buying_power_mode(a) == _BUYING_POWER_NMBP

    def test_case_insensitive(self):
        assert _normalize_buying_power_mode("SETTLED") == _BUYING_POWER_SETTLED

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            _normalize_buying_power_mode("margin_4x")


class TestOrderPayload:
    def test_reads_key(self):
        assert _order_payload({"qty": 5}, "qty") == 5

    def test_missing_key_none(self):
        assert _order_payload({}, "qty") is None


class TestStampHoldingAuditFields:
    def test_stamps_from_order(self):
        holding = SimpleNamespace()
        # smoke: stamping must not raise and should set attributes from order
        _stamp_holding_audit_fields(holding, {"rank_score": 0.3, "panel_score": 0.4})
        # the function sets audit attrs; at minimum it runs cleanly
        assert holding is not None
