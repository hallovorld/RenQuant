"""Tests for the intraday protective-action governor (roadmap #26).

The governor is a pure, deterministic primitive: ``now_epoch`` + ``session_date``
are injected, so every case below is exact (no wall-clock, no sleeps). These
tests pin the safety contract — especially that the default (disabled) governor
is a no-op — so the later live-wiring change can be reviewed against them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.intraday_governor import GovernorDecision, IntradayGovernor  # noqa: E402

DAY = "2026-06-16"


def _gov(**cfg) -> IntradayGovernor:
    base = {"enabled": True}
    base.update(cfg)
    return IntradayGovernor.from_config(base)


# ── default-off contract (the safety-critical invariant) ─────────────────────
class TestDisabledIsNoop:
    def test_disabled_always_allows(self):
        g = IntradayGovernor.from_config(None)  # default: disabled
        assert g.enabled is False
        for _ in range(5):
            d = g.decide("MU", now_epoch=1000.0, session_date=DAY)
            assert d.allowed is True
            assert d.governor == ""

    def test_disabled_record_is_noop(self):
        g = IntradayGovernor.from_config({"enabled": False,
                                          "per_symbol_session_cap": 1})
        g.record("MU", now_epoch=1000.0, session_date=DAY)
        # nothing accumulated, so even a 1-cap governor would still be empty
        assert g.global_count == 0
        assert g.symbol_counts == {}
        assert g.last_global_action_epoch is None

    def test_empty_config_disabled(self):
        assert IntradayGovernor.from_config({}).enabled is False


# ── per-symbol cooldown ──────────────────────────────────────────────────────
class TestPerSymbolCooldown:
    def test_blocks_within_window_allows_after(self):
        g = _gov(per_symbol_cooldown_seconds=300)
        assert g.decide("MU", 1000.0, DAY).allowed is True
        g.record("MU", 1000.0, DAY)
        # 299s later: still cooling down
        d = g.decide("MU", 1299.0, DAY)
        assert d.allowed is False
        assert d.governor == "per_symbol_cooldown"
        # exactly at the boundary (300s): allowed (>= window)
        assert g.decide("MU", 1300.0, DAY).allowed is True

    def test_cooldown_is_per_symbol(self):
        g = _gov(per_symbol_cooldown_seconds=300)
        g.record("MU", 1000.0, DAY)
        # a different symbol is unaffected by MU's cooldown
        assert g.decide("EQIX", 1100.0, DAY).allowed is True
        assert g.decide("MU", 1100.0, DAY).allowed is False

    def test_zero_disables_this_governor(self):
        g = _gov(per_symbol_cooldown_seconds=0)
        g.record("MU", 1000.0, DAY)
        assert g.decide("MU", 1000.0, DAY).allowed is True


# ── global cooldown ──────────────────────────────────────────────────────────
class TestGlobalCooldown:
    def test_throttles_across_symbols(self):
        g = _gov(global_cooldown_seconds=60)
        g.record("MU", 1000.0, DAY)
        d = g.decide("EQIX", 1030.0, DAY)  # different symbol, still throttled
        assert d.allowed is False
        assert d.governor == "global_cooldown"
        assert g.decide("EQIX", 1060.0, DAY).allowed is True


# ── session caps ─────────────────────────────────────────────────────────────
class TestSessionCaps:
    def test_per_symbol_cap(self):
        g = _gov(per_symbol_session_cap=2)
        for t in (1000.0, 2000.0):
            assert g.decide("MU", t, DAY).allowed is True
            g.record("MU", t, DAY)
        d = g.decide("MU", 3000.0, DAY)
        assert d.allowed is False
        assert d.governor == "per_symbol_session_cap"
        # a different symbol still has its own budget
        assert g.decide("EQIX", 3000.0, DAY).allowed is True

    def test_global_cap(self):
        g = _gov(global_session_cap=2)
        g.record("MU", 1000.0, DAY)
        g.record("EQIX", 1000.0, DAY)
        d = g.decide("AAPL", 1000.0, DAY)
        assert d.allowed is False
        assert d.governor == "global_session_cap"

    def test_zero_caps_unlimited(self):
        g = _gov(per_symbol_session_cap=0, global_session_cap=0)
        for t in range(10):
            g.record("MU", float(t), DAY)
        assert g.decide("MU", 100.0, DAY).allowed is True


# ── session rollover ─────────────────────────────────────────────────────────
class TestSessionRollover:
    def test_new_session_resets_counts(self):
        g = _gov(per_symbol_session_cap=1, per_symbol_cooldown_seconds=999999)
        g.record("MU", 1000.0, DAY)
        assert g.decide("MU", 1001.0, DAY).allowed is False  # capped + cooling
        # next trading day: fresh budget, cooldown cleared
        assert g.decide("MU", 1001.0, "2026-06-17").allowed is True

    def test_record_on_new_session_rolls_counters(self):
        g = _gov(global_session_cap=5)
        g.record("MU", 1000.0, DAY)
        assert g.global_count == 1
        g.record("MU", 1000.0, "2026-06-17")
        assert g.session_date == "2026-06-17"
        assert g.global_count == 1  # rolled, not 2

    def test_stale_snapshot_does_not_throttle_today(self):
        # decide() must treat a mismatched session_date as fresh WITHOUT mutating
        g = _gov(global_cooldown_seconds=600)
        g.load_state({"session_date": DAY, "last_global_action_epoch": 5000.0,
                      "global_count": 3})
        d = g.decide("MU", 5001.0, "2026-06-17")
        assert d.allowed is True
        assert d.reason == "ok (new session)"
        # decide did not mutate the loaded (stale) state
        assert g.session_date == DAY
        assert g.global_count == 3


# ── purity ───────────────────────────────────────────────────────────────────
class TestDecideIsPure:
    def test_decide_does_not_mutate(self):
        g = _gov(per_symbol_cooldown_seconds=300, global_session_cap=2)
        before = g.to_state()
        for _ in range(3):
            g.decide("MU", 1000.0, DAY)
        assert g.to_state() == before  # no state change from decide


# ── persistence round-trip ───────────────────────────────────────────────────
class TestPersistence:
    def test_state_round_trip(self):
        g = _gov(per_symbol_cooldown_seconds=300, global_session_cap=5)
        g.record("MU", 1000.0, DAY)
        g.record("EQIX", 1100.0, DAY)
        snap = g.to_state()

        g2 = _gov(per_symbol_cooldown_seconds=300, global_session_cap=5)
        g2.load_state(snap)
        assert g2.to_state() == snap
        # rehydrated cooldown is honoured
        assert g2.decide("MU", 1200.0, DAY).allowed is False
        assert g2.decide("MU", 1400.0, DAY).allowed is True

    def test_load_none_is_clean(self):
        g = _gov().load_state(None)
        assert g.to_state() == {
            "session_date": "", "last_action_epoch": {}, "symbol_counts": {},
            "global_count": 0, "last_global_action_epoch": None,
        }

    def test_negative_state_counters_rejected(self):
        with pytest.raises(ValueError, match="global_count"):
            _gov(global_session_cap=1).load_state({
                "session_date": DAY,
                "global_count": -7,
            })
        with pytest.raises(ValueError, match="symbol_counts"):
            _gov(per_symbol_session_cap=1).load_state({
                "session_date": DAY,
                "symbol_counts": {"MU": -1},
            })

    def test_negative_state_epochs_rejected(self):
        with pytest.raises(ValueError, match="last_action_epoch"):
            _gov(per_symbol_cooldown_seconds=60).load_state({
                "session_date": DAY,
                "last_action_epoch": {"MU": -1.0},
            })
        with pytest.raises(ValueError, match="last_global_action_epoch"):
            _gov(global_cooldown_seconds=60).load_state({
                "session_date": DAY,
                "last_global_action_epoch": -1.0,
            })


# ── from_config parsing ──────────────────────────────────────────────────────
class TestFromConfig:
    def test_parses_and_coerces(self):
        g = IntradayGovernor.from_config({
            "enabled": True,
            "per_symbol_cooldown_seconds": "300",
            "global_cooldown_seconds": 60,
            "per_symbol_session_cap": "2",
            "global_session_cap": 8,
        })
        assert g.enabled is True
        assert g.per_symbol_cooldown_seconds == 300.0
        assert g.per_symbol_session_cap == 2
        assert g.global_session_cap == 8

    def test_none_config_is_disabled_defaults(self):
        g = IntradayGovernor.from_config(None)
        assert g == IntradayGovernor()

    def test_negative_config_values_rejected(self):
        for key in (
            "per_symbol_cooldown_seconds",
            "global_cooldown_seconds",
            "per_symbol_session_cap",
            "global_session_cap",
        ):
            with pytest.raises(ValueError, match=key):
                IntradayGovernor.from_config({"enabled": True, key: -1})

    def test_fractional_caps_rejected(self):
        for key in ("per_symbol_session_cap", "global_session_cap"):
            with pytest.raises(ValueError, match=key):
                IntradayGovernor.from_config({"enabled": True, key: 1.5})


def test_decision_dataclass_is_frozen():
    d = GovernorDecision(True, "ok")
    try:
        d.allowed = False  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("GovernorDecision should be frozen")
