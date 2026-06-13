"""runner.py decomposition slice 3 — z9_stops (broker-side stop emit) tests.

Test-gated (sim replay does not cover the live adapter). Pins the Z9
invariants: never-loosen replacement, NaN/non-finite guards, the G1
pct-override catastrophe line, idempotent cancel.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from adapters.z9_stops import (  # noqa: E402
    cancel_stop,
    place_or_replace_stop,
    z9_enabled,
    z9_stop_pct,
)


class _Broker:
    def __init__(self, supports=True):
        self._supports = supports
        self.placed = []
        self.cancelled = []
        self._seq = 0

    def supports_broker_side_stops(self):
        return self._supports

    def place_stop_order(self, ticker, qty, stop):
        self._seq += 1
        oid = f"o{self._seq}"
        self.placed.append((ticker, qty, stop, oid))
        return {"order_id": oid}

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)


def _ctx(**kw):
    base = dict(config={}, regime="BULL_CALM")
    base.update(kw)
    return SimpleNamespace(**base)


class TestEnabled:
    def test_disabled_when_config_absent(self):
        assert z9_enabled(_Broker(), _ctx()) is False

    def test_disabled_when_broker_unsupported(self):
        ctx = _ctx(config={"live": {"broker_side_stops": {"enabled": True}}})
        assert z9_enabled(_Broker(supports=False), ctx) is False

    def test_enabled(self):
        ctx = _ctx(config={"live": {"broker_side_stops": {"enabled": True}}})
        assert z9_enabled(_Broker(), ctx) is True


class TestStopPct:
    def test_g1_pct_override(self):
        ctx = _ctx(config={"live": {"broker_side_stops": {"pct": 0.20}}})
        assert z9_stop_pct(ctx) == 0.20

    def test_invalid_pct_falls_back_to_regime(self):
        ctx = _ctx(config={"live": {"broker_side_stops": {"pct": 1.5}},
                           "regime_params": {"BULL_CALM": {"max_single_day_loss_pct": 0.06}}})
        assert z9_stop_pct(ctx) == 0.06

    def test_absent_pct_uses_regime_default(self):
        ctx = _ctx(config={"regime_params": {"BULL_CALM": {"max_single_day_loss_pct": 0.08}}})
        assert z9_stop_pct(ctx) == 0.08

    def test_no_config_default_6pct(self):
        assert z9_stop_pct(_ctx()) == 0.06


class TestPlaceOrReplace:
    def test_places_at_reference_minus_pct(self):
        b, store = _Broker(), {}
        place_or_replace_stop(b, store, "MU", 10, 100.0, "2026-06-12", ctx_pct=0.20)
        assert store["MU"]["stop_price"] == 80.0
        assert b.placed == [("MU", 10, 80.0, "o1")]

    def test_never_loosens_replacement(self):
        b, store = _Broker(), {}
        place_or_replace_stop(b, store, "MU", 10, 100.0, "d1", ctx_pct=0.20)  # 80
        # higher reference would imply a looser stop (88) — must keep 80
        place_or_replace_stop(b, store, "MU", 10, 110.0, "d2", ctx_pct=0.20)  # 88 proposed
        assert store["MU"]["stop_price"] == 80.0
        assert "o1" in b.cancelled  # old cancelled before re-place

    def test_nan_qty_skipped(self):
        b, store = _Broker(), {}
        place_or_replace_stop(b, store, "MU", float("nan"), 100.0, "d1")
        assert store == {} and b.placed == []

    def test_nonfinite_reference_skipped(self):
        b, store = _Broker(), {}
        place_or_replace_stop(b, store, "MU", 10, float("inf"), "d1")
        assert store == {} and b.placed == []

    def test_invalid_ctx_pct_coerced_to_6pct(self):
        b, store = _Broker(), {}
        place_or_replace_stop(b, store, "MU", 10, 100.0, "d1", ctx_pct=2.0)
        assert store["MU"]["stop_price"] == 94.0  # 100 * (1 - 0.06)

    def test_place_failure_leaves_store_clean(self):
        class _Boom(_Broker):
            def place_stop_order(self, *a):
                raise RuntimeError("rejected")
        b, store = _Boom(), {}
        place_or_replace_stop(b, store, "MU", 10, 100.0, "d1")
        assert "MU" not in store


class TestCancel:
    def test_cancel_existing(self):
        b, store = _Broker(), {"MU": {"order_id": "o9", "stop_price": 80.0}}
        cancel_stop(b, store, "MU", reason="liquidation")
        assert "MU" not in store and b.cancelled == ["o9"]

    def test_cancel_absent_is_noop(self):
        b, store = _Broker(), {}
        cancel_stop(b, store, "MU")
        assert b.cancelled == []
