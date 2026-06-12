"""P0.1 stream watchdog tests (intraday roadmap §4).

The core is pure (no network): a fake clock + tmp data root exercise
anchor/level/alert semantics, the one-shot-per-level rule, the heartbeat
cadence, the event log (P0.4 seed), and the read-only invariant.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from live.stream_watchdog import WatchdogCore  # noqa: E402


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _core(tmp_path, clock=None, **kw):
    kw.setdefault("held", {"MU", "GE"})
    return WatchdogCore(data_root=tmp_path, clock=clock or FakeClock(), **kw)


class TestAlertSemantics:

    def test_first_trade_sets_anchor_no_alert(self, tmp_path):
        core = _core(tmp_path)
        assert core.on_trade("MU", 100.0) == []
        assert core.state["MU"].anchor == 100.0

    def test_drop_below_threshold_silent(self, tmp_path):
        core = _core(tmp_path)
        core.on_trade("MU", 100.0)
        assert core.on_trade("MU", 96.0) == []  # −4% < 5%

    def test_drop_crossing_threshold_alerts_once(self, tmp_path):
        core = _core(tmp_path)
        core.on_trade("MU", 100.0)
        alerts = core.on_trade("MU", 94.0)  # −6%
        assert len(alerts) == 1
        assert alerts[0]["level"] == 1 and alerts[0]["held"] is True
        # same level does NOT re-alert
        assert core.on_trade("MU", 94.5) == []
        assert core.on_trade("MU", 93.0) == []

    def test_deeper_level_alerts_again(self, tmp_path):
        core = _core(tmp_path)
        core.on_trade("MU", 100.0)
        core.on_trade("MU", 94.0)            # level 1
        alerts = core.on_trade("MU", 89.0)   # −11% → level 2
        assert len(alerts) == 1 and alerts[0]["level"] == 2

    def test_spy_uses_market_threshold(self, tmp_path):
        core = _core(tmp_path)
        core.on_trade("SPY", 500.0)
        assert core.on_trade("SPY", 487.0) == []          # −2.6% < 3%
        alerts = core.on_trade("SPY", 484.0)              # −3.2%
        assert len(alerts) == 1 and alerts[0]["held"] is False

    def test_non_positive_price_ignored(self, tmp_path):
        core = _core(tmp_path)
        core.on_trade("MU", 100.0)
        assert core.on_trade("MU", 0.0) == []
        assert core.state["MU"].last == 100.0


class TestEventLogAndHeartbeat:

    def test_event_log_is_replayable_jsonl(self, tmp_path):
        clock = FakeClock()
        core = _core(tmp_path, clock=clock)
        core.on_trade("MU", 100.0)
        clock.t += 10
        core.on_trade("MU", 90.0)
        events = [json.loads(line) for line in
                  core._event_log.read_text().splitlines()]
        assert [e["kind"] for e in events] == ["anchor", "alert"]
        assert events[1]["symbol"] == "MU"

    def test_heartbeat_cadence(self, tmp_path):
        clock = FakeClock()
        core = _core(tmp_path, clock=clock)
        core.on_trade("MU", 100.0)
        core.on_trade("MU", 99.0)
        hb1 = core._heartbeat_file.read_text()
        clock.t += 5
        core.on_trade("MU", 99.5)   # < 30s — no rewrite
        assert core._heartbeat_file.read_text() == hb1
        clock.t += 31
        core.on_trade("MU", 99.4)
        assert core._heartbeat_file.read_text() != hb1

    def test_staleness_measure(self, tmp_path):
        clock = FakeClock()
        core = _core(tmp_path, clock=clock)
        core.on_trade("MU", 100.0)
        core.on_trade("MU", 100.5)
        clock.t += 120
        assert core.staleness_seconds("MU") == 120
        assert core.staleness_seconds("UNSEEN") is None


class TestReadOnlyInvariant:

    def test_module_never_imports_trading_client(self):
        import ast

        src = (REPO / "live" / "stream_watchdog.py").read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "alpaca.trading" not in (node.module or ""), \
                    "watchdog must never import the trading API"
            if isinstance(node, ast.Attribute):
                assert node.attr not in ("submit_order", "place_order",
                                         "cancel_order"), \
                    f"order-authority call {node.attr} in read-only watchdog"


class TestHeartbeatDecoupling:
    """codex review #323: heartbeat must advance with ZERO trades."""

    def test_write_heartbeat_unconditional(self, tmp_path):
        clock = FakeClock()
        core = _core(tmp_path, clock=clock)
        core.write_heartbeat()           # no trade ever seen
        hb1 = core._heartbeat_file.read_text()
        clock.t += 30
        core.write_heartbeat()
        assert core._heartbeat_file.read_text() != hb1

    def test_timer_thread_beats_without_trades(self, tmp_path, monkeypatch):
        # Drive the daemon's beat loop body directly: simulate the timer
        # thread's behavior across two intervals with no on_trade calls.
        clock = FakeClock()
        core = _core(tmp_path, clock=clock)
        for _ in range(2):
            core.write_heartbeat()
            clock.t += 30
        assert float(core._heartbeat_file.read_text()) == clock.t - 30

    def test_quiet_market_vs_dead_process_distinguishable(self, tmp_path):
        # Quiet market: heartbeat fresh (timer), data stale (no trades) —
        # the two signals must be independently readable.
        clock = FakeClock()
        core = _core(tmp_path, clock=clock)
        core.on_trade("MU", 100.0)
        clock.t += 600                   # ten quiet minutes
        core.write_heartbeat()           # timer keeps beating
        assert float(core._heartbeat_file.read_text()) == clock.t
        assert core.staleness_seconds("MU") == 600
