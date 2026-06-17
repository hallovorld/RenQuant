"""s1-wire-live-state-v2 — typed live-state model + opt-in wiring.

Proves the two contracts the roadmap item rests on:

  1. LOSSLESS round-trip: ``LiveStateV2.parse(v1).to_v1_dict()`` reproduces a
     v1 dict byte-identical to the input under ``json.dumps(..., indent=2)``
     — for a representative live_state dict AND the real committed
     ``live_state.alpaca.json`` snapshot.
  2. FLAG-OFF is unchanged: with the opt-in flag off (default),
     ``save_live_state_atomic`` / ``load_live_state`` produce exactly the
     same bytes / dict as before. Flag-ON yields the SAME on-disk bytes.

Self-contained: no DB, no broker, no network. Inserts the strategy package
onto sys.path the same way the other root-level adapter tests do.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STRAT = REPO / "backtesting" / "renquant_104"
sys.path.insert(0, str(STRAT))

import adapters.state_store as state_store  # noqa: E402
from kernel.live_state_v2 import HoldingV2, LiveStateV2  # noqa: E402


# A representative live_state dict: every top-level key the runner writes,
# in the order it writes them, including unmodeled blobs (monitor_state,
# regime_state, entry_signals, stop_orders, recent_sell_orders) and the
# per-holding column that is the one-line-extensibility example
# (protection_breaches).
REPRESENTATIVE_STATE: dict = {
    "regime": "CHOPPY",
    "regime_confidence": 0.5,
    "high_water_mark": 11079.22,
    "entry_dates": {"MU": "2026-04-27", "EQIX": "2026-05-17"},
    "sell_streaks": {"MU": 0, "EQIX": 1},
    "protection_breaches": {"EQIX": 2, "MU": 0},
    "last_sell_dates": {"GE": "2026-06-15", "TXN": "2026-05-20"},
    "position_hwm": {"MU": 1106.34, "EQIX": 1104.855},
    "monitor_state": {
        "no_trade_streak": 3,
        "last_activity_date": "2026-05-27",
        "last_check_date": "2026-06-16",
    },
    "regime_state": {
        "regime": "CHOPPY",
        "confidence": 0.5,
        "in_transition": True,
        "countdown": 1,
        "cusum_pos": 0.0,
        "cusum_neg": 0.0,
        "cooldown_start": "2026-06-16T00:00:00",
    },
    "entry_signals": {
        "MU": {"rank_score": 0.32513821903305495,
               "panel_score": None, "kelly_target_pct": None},
        "EQIX": {"rank_score": 0.8199780461031834,
                 "panel_score": None, "kelly_target_pct": None},
    },
    "stop_orders": {},
    "last_stop_exit_dates": {"BA": "2026-05-15", "VRT": "2026-05-18"},
    "skip_buys": False,
    "recent_sell_orders": {},
}


def _committed_live_state() -> dict:
    p = STRAT / "live_state.alpaca.json"
    return json.loads(p.read_text())


# ── 1. Lossless round-trip ────────────────────────────────────────────────

def test_roundtrip_dict_equal_representative():
    out = LiveStateV2.parse(REPRESENTATIVE_STATE).to_v1_dict()
    assert out == REPRESENTATIVE_STATE


def test_roundtrip_byte_identical_indent2_representative():
    """The on-disk serialiser is json.dumps(state, indent=2). Bytes match."""
    before = json.dumps(REPRESENTATIVE_STATE, indent=2)
    after = json.dumps(
        LiveStateV2.parse(REPRESENTATIVE_STATE).to_v1_dict(), indent=2
    )
    assert before == after


def test_roundtrip_byte_identical_real_committed_snapshot():
    """Round-trips the REAL committed live_state.alpaca.json byte-for-byte."""
    raw_text = (STRAT / "live_state.alpaca.json").read_text()
    state = json.loads(raw_text)
    rt = LiveStateV2.parse(state).to_v1_dict()
    # dict equality + serialised-byte equality at the production indent.
    assert rt == state
    assert json.dumps(rt, indent=2) == json.dumps(state, indent=2)


def test_top_level_key_order_preserved():
    out = LiveStateV2.parse(REPRESENTATIVE_STATE).to_v1_dict()
    assert list(out.keys()) == list(REPRESENTATIVE_STATE.keys())


def test_empty_state_roundtrips():
    assert LiveStateV2.parse({}).to_v1_dict() == {}
    assert LiveStateV2.parse(None).to_v1_dict() == {}


def test_unmodeled_keys_pass_through_verbatim():
    """Keys the schema does not model ride along unchanged (lossless)."""
    out = LiveStateV2.parse(REPRESENTATIVE_STATE).to_v1_dict()
    assert out["monitor_state"] == REPRESENTATIVE_STATE["monitor_state"]
    assert out["regime_state"] == REPRESENTATIVE_STATE["regime_state"]
    assert out["entry_signals"] == REPRESENTATIVE_STATE["entry_signals"]
    assert out["recent_sell_orders"] == {}
    assert out["stop_orders"] == {}


# ── Typed per-holding view (the one-line-extensibility ergonomics) ────────

def test_typed_holdings_view():
    holdings = LiveStateV2.parse(REPRESENTATIVE_STATE).holdings
    assert set(holdings) == {"MU", "EQIX"}
    assert isinstance(holdings["MU"], HoldingV2)
    assert holdings["MU"].entry_date == "2026-04-27"
    assert holdings["EQIX"].sell_streak == 1
    assert holdings["EQIX"].protection_breaches == 2  # the new one-line field
    assert holdings["MU"].position_hwm == 1106.34


def test_typed_view_does_not_mutate_roundtrip():
    """Accessing the typed view must not perturb the lossless inverse."""
    model = LiveStateV2.parse(REPRESENTATIVE_STATE)
    _ = model.holdings  # force the derived view
    assert model.to_v1_dict() == REPRESENTATIVE_STATE


# ── 2. Flag-OFF is unchanged; flag-ON keeps bytes identical ───────────────

def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("RQ_LIVE_STATE_V2", raising=False)
    assert state_store._typed_v2_enabled(None) is False
    assert state_store._typed_v2_enabled({}) is False
    assert state_store._typed_v2_enabled({"live_state": {}}) is False


def test_flag_on_via_env(monkeypatch):
    monkeypatch.setenv("RQ_LIVE_STATE_V2", "1")
    assert state_store._typed_v2_enabled(None) is True


def test_flag_on_via_config(monkeypatch):
    monkeypatch.delenv("RQ_LIVE_STATE_V2", raising=False)
    assert state_store._typed_v2_enabled({"live_state": {"typed_v2": True}}) is True


def test_save_flag_off_unchanged(tmp_path, monkeypatch):
    """Flag OFF: save writes exactly json.dumps(state, indent=2)."""
    monkeypatch.delenv("RQ_LIVE_STATE_V2", raising=False)
    f = tmp_path / "live_state.alpaca.json"
    state_store.save_live_state_atomic(f, REPRESENTATIVE_STATE)
    assert f.read_text() == json.dumps(REPRESENTATIVE_STATE, indent=2)


def test_save_flag_on_byte_identical_to_off(tmp_path, monkeypatch):
    """Flag ON produces the SAME bytes as flag OFF (lossless guarantee)."""
    monkeypatch.delenv("RQ_LIVE_STATE_V2", raising=False)
    off = tmp_path / "off.json"
    state_store.save_live_state_atomic(off, REPRESENTATIVE_STATE)

    monkeypatch.setenv("RQ_LIVE_STATE_V2", "1")
    on = tmp_path / "on.json"
    state_store.save_live_state_atomic(on, REPRESENTATIVE_STATE)

    assert on.read_text() == off.read_text()


def test_load_flag_off_vs_on_same_dict(tmp_path, monkeypatch):
    """Loading the same file yields an equal dict whether flag on or off."""
    f = tmp_path / "live_state.alpaca.json"
    f.write_text(json.dumps(REPRESENTATIVE_STATE, indent=2))

    monkeypatch.delenv("RQ_LIVE_STATE_V2", raising=False)
    off = state_store.load_live_state(f, {}, STRAT)

    monkeypatch.setenv("RQ_LIVE_STATE_V2", "1")
    on = state_store.load_live_state(f, {}, STRAT)

    assert off == on == REPRESENTATIVE_STATE


def test_save_flag_on_config_byte_identical(tmp_path, monkeypatch):
    """Flag via config (not env) also keeps bytes identical."""
    monkeypatch.delenv("RQ_LIVE_STATE_V2", raising=False)
    cfg = {"live_state": {"typed_v2": True}}
    f = tmp_path / "live_state.alpaca.json"
    state_store.save_live_state_atomic(f, REPRESENTATIVE_STATE, cfg)
    assert f.read_text() == json.dumps(REPRESENTATIVE_STATE, indent=2)
