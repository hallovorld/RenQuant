"""S-FRAC stage 3 (core, sprint D2) — software-stop layer tests.

Design: renquant-orchestrator doc/design/2026-07-02-s-frac-fractional-v2.md
§3.2 (registry + sell-only-loop delta) / §3.3 (failure modes) / §3.4
(staleness watchdog). Coverage demanded by the sprint-D2 task:

  1. Ratchet-only invariant (loosening refused without the explicit
     rewrite path)                                → TestRatchetOnlyInvariant
  2. Trigger correctness incl. fractional qty     → TestTriggerCorrectness
  3. Gap-through pricing (slippage logged)        → TestGapThroughPricing
  4. Flag-off byte-inertness on the sell-only loop → TestFlagOffInert
  5. Registry round-trip / corruption fail-closed
     (corrupt registry blocks new fractional
     entries via the stage-0 capability gate)     → TestRegistryRoundTrip,
                                                    TestCorruptRegistryFailClosed
  6. Staleness watchdog arithmetic                → TestStalenessWatchdog
  7. Commit-path wiring through the stage-0 seam
     (register at entry, deregister at full exit,
     GC on external disposition, never-loosen
     through the Z9 route)                        → TestCommitWiringE2E,
                                                    TestSellOnlyLoopWiring
"""
from __future__ import annotations

import datetime
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY = REPO_ROOT / "backtesting" / "renquant_104"
for _p in (str(REPO_ROOT), str(_STRATEGY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from adapters.software_stops import (  # noqa: E402
    DEFAULT_MAX_STALENESS_MINUTES,
    SoftwareStopRegistry,
    SoftwareStopRegistryCorrupt,
    compute_staleness,
    registry_path_for,
)

# Reuse the stage-0 active-path harness (FakeBroker drives the REAL
# RunnerAdapter.commit; the seam consumed here is the one stage 0 built).
from tests.test_s_frac_stage0_commit_contract import (  # noqa: E402
    FRACTIONAL_QTY,
    TODAY,
    FakeBroker,
    _config,
    _make_adapter,
    _make_ctx,
    _saved_state,
)

NOW = datetime.datetime(2026, 7, 3, 11, 0, tzinfo=datetime.timezone.utc)


def _registry(tmp_path, **kwargs) -> SoftwareStopRegistry:
    return SoftwareStopRegistry(
        tmp_path / "data" / "rq105" / "software_stops.json", **kwargs
    )


def _armed_with(tmp_path, symbol="BLK", qty=FRACTIONAL_QTY, stop=760.0,
                source="z9") -> SoftwareStopRegistry:
    reg = _registry(tmp_path)
    reg.register(symbol, qty, stop, source=source, today_str="2026-07-03")
    return reg


def _load_watchdog():
    spec = importlib.util.spec_from_file_location(
        "check_software_stops_liveness",
        REPO_ROOT / "scripts" / "check_software_stops_liveness.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ═════════════════════════════════════════════════════════════════════════════
# Registry schema + round-trip
# ═════════════════════════════════════════════════════════════════════════════

class TestRegistryRoundTrip:
    def test_register_persists_schema_and_reloads(self, tmp_path):
        reg = _armed_with(tmp_path)
        assert reg.is_armed() is True

        raw = json.loads(reg.path.read_text())
        assert raw["version"] == 1
        assert raw["contract"] == "software-stops-v1"
        assert "max_staleness_minutes" in raw
        entry = raw["stops"]["BLK"]
        assert entry["symbol"] == "BLK"
        assert entry["qty"] == FRACTIONAL_QTY          # float verbatim
        assert entry["stop_price"] == 760.0
        assert entry["armed_at"] == "2026-07-03"
        assert entry["source"] == "z9"
        assert entry["history"][0]["action"] == "register"

        # Fresh instance loads the identical protection surface.
        reloaded = SoftwareStopRegistry(reg.path)
        assert reloaded.is_armed() is True
        assert reloaded.get("BLK")["stop_price"] == 760.0
        assert reloaded.get("BLK")["qty"] == FRACTIONAL_QTY

    def test_no_file_until_first_write(self, tmp_path):
        reg = _registry(tmp_path)
        assert reg.is_armed() is True   # empty registry is armed
        assert not reg.path.exists()    # …but writes nothing until used

    def test_invalid_source_rejected(self, tmp_path):
        reg = _registry(tmp_path)
        with pytest.raises(ValueError, match="source"):
            reg.register("BLK", 1.0, 80.0, source="cosmic-ray")

    def test_from_config_flag_off_returns_none(self):
        assert SoftwareStopRegistry.from_config(None) is None
        assert SoftwareStopRegistry.from_config({}) is None
        cfg = {"execution": {"software_stops": {"enabled": False}}}
        assert SoftwareStopRegistry.from_config(cfg) is None

    def test_from_config_broker_tagged_path(self, tmp_path):
        cfg = {"execution": {"software_stops": {
            "enabled": True,
            "registry_path": "data/rq105/software_stops.json",
        }}}
        reg = SoftwareStopRegistry.from_config(
            cfg, broker_name="alpaca", repo_root=tmp_path,
        )
        assert reg is not None
        assert reg.path == (tmp_path / "data" / "rq105"
                            / "software_stops.alpaca.json")
        # Idempotent tagging + sim/test passthrough.
        assert registry_path_for(reg.path, "alpaca") == reg.path
        assert registry_path_for("x/software_stops.json", None) == Path(
            "x/software_stops.json")


# ═════════════════════════════════════════════════════════════════════════════
# Never-loosen: ratchet-only invariant
# ═════════════════════════════════════════════════════════════════════════════

class TestRatchetOnlyInvariant:
    def test_lower_stop_refused(self, tmp_path, caplog):
        reg = _armed_with(tmp_path, stop=80.0)
        with caplog.at_level("WARNING", logger="live.runner"):
            reg.register("BLK", FRACTIONAL_QTY, 72.0, source="z9")
        entry = reg.get("BLK")
        assert entry["stop_price"] == 80.0             # unchanged
        assert entry["history"][-1]["action"] == "ratchet_refused"
        assert entry["history"][-1]["proposed_stop_price"] == 72.0
        assert "never-loosen" in caplog.text
        # And it survives a reload — the refusal was persisted as history,
        # not applied to the stop.
        assert SoftwareStopRegistry(reg.path).get("BLK")["stop_price"] == 80.0

    def test_higher_stop_ratchets_up(self, tmp_path):
        reg = _armed_with(tmp_path, stop=80.0)
        reg.register("BLK", FRACTIONAL_QTY, 96.0, source="z9")
        entry = reg.get("BLK")
        assert entry["stop_price"] == 96.0
        assert entry["history"][-1]["action"] == "ratchet_up"

    def test_loosening_requires_explicit_rewrite_with_reason(self, tmp_path, caplog):
        reg = _armed_with(tmp_path, stop=80.0)
        with pytest.raises(ValueError, match="reason"):
            reg.rewrite_stop("BLK", 60.0, reason="")
        with pytest.raises(ValueError, match="reason"):
            reg.rewrite_stop("BLK", 60.0, reason="   ")
        assert reg.get("BLK")["stop_price"] == 80.0

        with caplog.at_level("WARNING", logger="live.runner"):
            reg.rewrite_stop(
                "BLK", 60.0, reason="operator: post-earnings vol reset",
            )
        entry = reg.get("BLK")
        assert entry["stop_price"] == 60.0
        last = entry["history"][-1]
        assert last["action"] == "explicit_rewrite"
        assert last["previous_stop_price"] == 80.0
        assert last["reason"] == "operator: post-earnings vol reset"
        assert "explicit rewrite" in caplog.text

    def test_rewrite_unknown_symbol_raises(self, tmp_path):
        reg = _registry(tmp_path)
        with pytest.raises(KeyError):
            reg.rewrite_stop("GHOST", 10.0, reason="typo")

    def test_topup_refreshes_qty_but_not_stop_direction(self, tmp_path):
        reg = _armed_with(tmp_path, qty=0.4, stop=80.0)
        reg.register("BLK", 0.9, 80.0, source="z9")    # top-up, same stop
        entry = reg.get("BLK")
        assert entry["qty"] == 0.9                     # protected qty grows
        assert entry["stop_price"] == 80.0
        assert entry["history"][-1]["action"] == "refresh"


# ═════════════════════════════════════════════════════════════════════════════
# Trigger correctness (fractional qty)
# ═════════════════════════════════════════════════════════════════════════════

class TestTriggerCorrectness:
    def test_breach_fires_full_fractional_qty(self, tmp_path):
        reg = _armed_with(tmp_path, qty=FRACTIONAL_QTY, stop=760.0)
        intents = reg.evaluate({"BLK": 760.0})         # price == stop ⇒ fires
        assert len(intents) == 1
        intent = intents[0]
        assert intent["symbol"] == "BLK"
        assert intent["qty"] == FRACTIONAL_QTY          # FULL registered qty
        assert intent["stop_price"] == 760.0
        assert intent["trigger_price"] == 760.0
        assert intent["gap_pct"] == 0.0
        assert "software_stop breach" in intent["reason"]

    def test_above_stop_does_not_fire(self, tmp_path):
        reg = _armed_with(tmp_path, stop=760.0)
        assert reg.evaluate({"BLK": 760.01}) == []

    def test_missing_or_bad_quote_stays_armed(self, tmp_path, caplog):
        reg = _armed_with(tmp_path, stop=760.0)
        with caplog.at_level("WARNING", logger="live.runner"):
            assert reg.evaluate({}) == []
            assert reg.evaluate({"BLK": float("nan")}) == []
        assert "NOT evaluated" in caplog.text
        assert reg.get("BLK") is not None              # still protected

    def test_refires_until_deregistered(self, tmp_path):
        """A breached stop stays registered until the exit is broker-
        confirmed (commit deregisters on full liquidation) — a failed
        SELL re-fires next pass instead of silently unprotecting."""
        reg = _armed_with(tmp_path, stop=760.0)
        assert len(reg.evaluate({"BLK": 700.0})) == 1
        assert len(reg.evaluate({"BLK": 700.0})) == 1
        reg.deregister("BLK", reason="full liquidation")
        assert reg.evaluate({"BLK": 700.0}) == []


# ═════════════════════════════════════════════════════════════════════════════
# Gap-down-through-stop (design §3.3)
# ═════════════════════════════════════════════════════════════════════════════

class TestGapThroughPricing:
    def test_gap_size_measured_logged_and_carried(self, tmp_path, caplog):
        reg = _armed_with(tmp_path, stop=760.0)
        with caplog.at_level("WARNING", logger="live.runner"):
            intents = reg.evaluate({"BLK": 700.0})     # gapped 7.89% through
        assert len(intents) == 1
        intent = intents[0]
        assert intent["trigger_price"] == 700.0
        assert intent["gap_pct"] == pytest.approx((760.0 - 700.0) / 760.0)
        assert "gap" in intent["reason"]
        # Slippage accepted + logged with the gap size (§3.3).
        assert "gap=7.89%" in caplog.text
        assert "slippage accepted" in caplog.text

    def test_exit_intent_is_market_exit_for_full_qty_regardless_of_gap(
            self, tmp_path):
        reg = _armed_with(tmp_path, qty=FRACTIONAL_QTY, stop=760.0)
        deep = reg.evaluate({"BLK": 380.0})            # 50% through the stop
        assert deep[0]["qty"] == FRACTIONAL_QTY
        assert deep[0]["gap_pct"] == pytest.approx(0.5)


# ═════════════════════════════════════════════════════════════════════════════
# Sell-only loop wiring (SoftwareStopExitTask)
# ═════════════════════════════════════════════════════════════════════════════

class TestSellOnlyLoopWiring:
    def _ctx(self, registry, prices):
        return SimpleNamespace(
            software_stops=registry, prices=dict(prices), exits=[],
        )

    def test_breach_appends_software_stop_exit(self, tmp_path):
        from kernel.pipeline.task_software_stops import SoftwareStopExitTask

        reg = _armed_with(tmp_path, qty=FRACTIONAL_QTY, stop=760.0)
        ctx = self._ctx(reg, {"BLK": 700.0})
        SoftwareStopExitTask().run(ctx)
        assert len(ctx.exits) == 1
        ticker, sig = ctx.exits[0]
        assert ticker == "BLK"
        assert sig.should_exit is True
        assert sig.exit_type == "software_stop"
        assert sig.quantity == FRACTIONAL_QTY          # FULL registered qty
        assert "software_stop breach" in sig.reason
        # The pass stamped the liveness heartbeat.
        raw = json.loads(reg.path.read_text())
        assert raw["last_evaluated_at"] is not None

    def test_no_breach_no_exit_but_heartbeat_stamped(self, tmp_path):
        from kernel.pipeline.task_software_stops import SoftwareStopExitTask

        reg = _armed_with(tmp_path, stop=760.0)
        ctx = self._ctx(reg, {"BLK": 800.0})
        SoftwareStopExitTask().run(ctx)
        assert ctx.exits == []
        assert json.loads(reg.path.read_text())["last_evaluated_at"] is not None

    def test_unarmed_registry_is_loud_noop(self, tmp_path, caplog):
        from kernel.pipeline.task_software_stops import SoftwareStopExitTask

        bad = tmp_path / "software_stops.json"
        bad.write_text("{ this is not json")
        reg = SoftwareStopRegistry(bad)
        ctx = self._ctx(reg, {"BLK": 1.0})
        with caplog.at_level("ERROR", logger="kernel.pipeline"):
            SoftwareStopExitTask().run(ctx)
        assert ctx.exits == []
        assert "NOT armed" in caplog.text

    def test_taxonomy_membership(self):
        """A software stop is a stop: bypasses panel veto + per-bar cap,
        triggers the post-stop re-entry blackout, and is NOT meta-label
        vetoable (only canonical core types are)."""
        from kernel.exit_types import (
            META_LABEL_VETO_ELIGIBLE,
            PANEL_VETO_BYPASS,
            PER_BAR_CAP_EXEMPT,
            POST_STOP_COOLDOWN_TRIGGERS,
        )
        assert "software_stop" in PANEL_VETO_BYPASS
        assert "software_stop" in PER_BAR_CAP_EXEMPT
        assert "software_stop" in POST_STOP_COOLDOWN_TRIGGERS
        assert "software_stop" not in META_LABEL_VETO_ELIGIBLE

    def test_sell_only_pipeline_runs_task_after_veto_and_cap(self):
        """Source-order pin: the software-stop pass runs AFTER the
        meta-label veto and the per-bar sell cap (a broker-resident stop
        can't be vetoed or capped; nor can its software mirror)."""
        from kernel.pipeline.pp_inference import SellOnlyPipeline

        src = inspect.getsource(SellOnlyPipeline.run)
        i_veto = src.index("MetaLabelVetoTask().run")
        i_cap = src.index("LimitSellsPerBarTask().run")
        i_sw = src.index("SoftwareStopExitTask().run")
        assert i_veto < i_sw
        assert i_cap < i_sw


class TestFlagOffInert:
    def test_task_noop_without_registry(self, tmp_path):
        """Flag-off byte-inertness on the sell-only loop: no registry on
        ctx ⇒ no exits appended, nothing written anywhere."""
        from kernel.pipeline.task_software_stops import SoftwareStopExitTask

        for ctx in (
            SimpleNamespace(prices={"BLK": 1.0}, exits=[]),           # attr absent
            SimpleNamespace(software_stops=None,
                            prices={"BLK": 1.0}, exits=[]),           # attr None
        ):
            task = SoftwareStopExitTask()
            assert task.should_skip(ctx) is True
            task.run(ctx)
            assert ctx.exits == []
        assert list(tmp_path.iterdir()) == []          # no state file created

    def test_commit_flag_off_no_registry_interaction(self, tmp_path):
        """Whole-share flag-off commit never touches a registry file —
        the stage-0 byte-identical regression extended to stage 3."""
        config = _config(fractional=False)
        broker = FakeBroker(fills={
            "OXY": {"status": "filled", "order_id": "o1",
                    "filled_qty": 5.0, "filled_avg_price": 50.0},
        })
        ra = _make_adapter(tmp_path, config=config, broker=broker)
        assert ra._software_stops is None
        ctx = _make_ctx(
            config,
            orders=[{"ticker": "OXY", "shares": 5, "price": 50.0}],
            prices={"OXY": 50.0}, cash=1_000.0,
        )
        ra.commit(ctx)
        assert [o["ticker"] for o in ctx.orders_placed] == ["OXY"]
        assert not list(tmp_path.rglob("software_stops*.json"))


# ═════════════════════════════════════════════════════════════════════════════
# Corruption fail-closed (blocks NEW fractional entries, never silent)
# ═════════════════════════════════════════════════════════════════════════════

class TestCorruptRegistryFailClosed:
    def _corrupt_registry(self, tmp_path, payload="{ not json"):
        path = tmp_path / "software_stops.json"
        path.write_text(payload)
        return SoftwareStopRegistry(path), path

    def test_corrupt_is_not_armed(self, tmp_path):
        reg, _ = self._corrupt_registry(tmp_path)
        assert reg.corrupt is True
        assert reg.is_armed() is False

    def test_schema_violation_is_corrupt(self, tmp_path):
        bad = {"version": 1, "stops": {"BLK": {
            "symbol": "BLK", "qty": -1, "stop_price": 80.0, "source": "z9",
        }}}
        reg, _ = self._corrupt_registry(tmp_path, json.dumps(bad))
        assert reg.is_armed() is False

    def test_corrupt_blocks_new_fractional_entry_via_stage0_gate(self, tmp_path):
        """The sprint-D2 requirement verbatim: a corrupt registry BLOCKS
        new fractional entries via the stage-0 capability gate — proven
        on the REAL commit path (no order reaches the broker)."""
        from adapters.commit_contract import (
            fractional_capability_gate,
            software_stops_armed,
        )

        reg, _ = self._corrupt_registry(tmp_path)
        assert software_stops_armed(reg) is False

        config = _config(fractional=True)
        broker = FakeBroker(
            fills={"BLK": {"status": "filled", "order_id": "x",
                           "filled_qty": FRACTIONAL_QTY,
                           "filled_avg_price": 100.0}},
            fractional_contract=True,
        )
        gate = fractional_capability_gate(config, broker, reg)
        assert gate["ok"] is False
        assert "software_stop_layer" in gate["missing"]

        ra = _make_adapter(tmp_path, config=config, broker=broker,
                           software_stops=reg)
        ctx = _make_ctx(
            config,
            orders=[{"ticker": "BLK", "shares": FRACTIONAL_QTY, "price": 100.0}],
            prices={"BLK": 100.0}, cash=1_000.0,
        )
        ra.commit(ctx)
        assert broker.place_order_calls == []          # nothing submitted
        assert ctx.orders_placed == []
        assert len(ctx.orders_skipped) == 1
        assert ctx.orders_skipped[0]["skip_reason"] == (
            "fractional_capability_gate_failed:software_stop_layer"
        )

    def test_corrupt_refuses_writes_and_preserves_bytes(self, tmp_path, caplog):
        reg, path = self._corrupt_registry(tmp_path)
        original = path.read_bytes()
        with pytest.raises(SoftwareStopRegistryCorrupt):
            reg.register("BLK", 1.0, 80.0, source="z9")
        with pytest.raises(SoftwareStopRegistryCorrupt):
            reg.rewrite_stop("BLK", 60.0, reason="attempt")
        with caplog.at_level("ERROR", logger="live.runner"):
            assert reg.evaluate({"BLK": 1.0}) == []    # loud, no exits invented
        assert "CORRUPT" in caplog.text
        assert path.read_bytes() == original           # evidence untouched

    def test_corrupt_watchdog_state(self, tmp_path):
        reg, _ = self._corrupt_registry(tmp_path)
        state = reg.staleness_state(now=NOW)
        assert state["corrupt"] is True
        assert state["stale"] is True


# ═════════════════════════════════════════════════════════════════════════════
# Staleness watchdog (design §3.4)
# ═════════════════════════════════════════════════════════════════════════════

class TestStalenessWatchdog:
    def _snapshot(self, *, n_stops=1, heartbeat, budget=30.0):
        stops = {
            f"S{i}": {"symbol": f"S{i}", "qty": 0.5, "stop_price": 10.0,
                      "armed_at": "2026-07-03", "source": "z9", "history": []}
            for i in range(n_stops)
        }
        return {
            "version": 1, "contract": "software-stops-v1",
            "max_staleness_minutes": budget,
            "last_evaluated_at": heartbeat, "stops": stops,
        }

    def test_arithmetic(self):
        fresh = (NOW - datetime.timedelta(minutes=10)).isoformat()
        old = (NOW - datetime.timedelta(minutes=31)).isoformat()

        s = compute_staleness(self._snapshot(heartbeat=fresh), now=NOW)
        assert s["stale"] is False
        assert s["age_minutes"] == pytest.approx(10.0)

        s = compute_staleness(self._snapshot(heartbeat=old), now=NOW)
        assert s["stale"] is True
        assert s["age_minutes"] == pytest.approx(31.0)

        # Armed entries with NO heartbeat ever ⇒ stale.
        s = compute_staleness(self._snapshot(heartbeat=None), now=NOW)
        assert s["stale"] is True

        # No armed entries ⇒ never stale (nothing unprotected).
        s = compute_staleness(
            self._snapshot(n_stops=0, heartbeat=old), now=NOW)
        assert s["stale"] is False

        # No registry file at all ⇒ ok.
        s = compute_staleness(None, now=NOW)
        assert s["exists"] is False
        assert s["stale"] is False

        # Budget honored from the file.
        s = compute_staleness(
            self._snapshot(heartbeat=old, budget=60.0), now=NOW)
        assert s["stale"] is False

    def test_default_budget_matches_loop_cadence(self):
        # 12-minute loop cadence × 2 missed passes + slack.
        assert DEFAULT_MAX_STALENESS_MINUTES == 30.0

    def test_evaluate_stamps_fresh_heartbeat(self, tmp_path):
        reg = _armed_with(tmp_path)
        reg.evaluate({"BLK": 900.0}, now=NOW)
        state = reg.staleness_state(now=NOW)
        assert state["stale"] is False
        assert state["age_minutes"] == pytest.approx(0.0, abs=1e-6)

    def test_cli_exit_codes(self, tmp_path):
        wd = _load_watchdog()

        # No file ⇒ OK.
        code, msg = wd.check(tmp_path / "missing.json", now=NOW,
                             force_session=True)
        assert code == wd.OK and "never armed" in msg

        # Corrupt ⇒ 2.
        bad = tmp_path / "corrupt.json"
        bad.write_text("{ nope")
        code, msg = wd.check(bad, now=NOW, force_session=True)
        assert code == wd.CORRUPT and "OPERATOR ACTION" in msg

        # Armed + stale in-session ⇒ 1.
        reg = _armed_with(tmp_path)
        reg.evaluate({"BLK": 900.0},
                     now=NOW - datetime.timedelta(minutes=45))
        code, msg = wd.check(reg.path, now=NOW, force_session=True)
        assert code == wd.STALE and "UNPROTECTED" in msg

        # Armed + fresh ⇒ 0.
        reg.evaluate({"BLK": 900.0}, now=NOW)
        code, msg = wd.check(reg.path, now=NOW, force_session=True)
        assert code == wd.OK

        # Armed + stale but market closed (Saturday) ⇒ 0 with a note.
        saturday = datetime.datetime(
            2026, 7, 4, 15, 0, tzinfo=datetime.timezone.utc)
        reg.evaluate({"BLK": 900.0},
                     now=saturday - datetime.timedelta(hours=20))
        code, msg = wd.check(reg.path, now=saturday, force_session=False)
        assert code == wd.OK and "session closed" in msg

        # Zero armed stops, ancient heartbeat ⇒ 0.
        reg.deregister("BLK", reason="flat")
        code, msg = wd.check(reg.path, now=NOW, force_session=True)
        assert code == wd.OK and "0 armed stops" in msg


# ═════════════════════════════════════════════════════════════════════════════
# Commit-path wiring through the stage-0 seam (the ACTIVE path)
# ═════════════════════════════════════════════════════════════════════════════

def _frac_config(tmp_path):
    config = _config(fractional=True)   # z9 enabled, pct=0.2
    config["execution"]["software_stops"] = {
        "enabled": True,
        "registry_path": str(tmp_path / "data" / "rq105"
                             / "software_stops.json"),
    }
    return config


class TestCommitWiringE2E:
    def test_fractional_buy_registers_software_stop(self, tmp_path, caplog):
        """Entry commit: broker can't stop-protect 0.435578 ⇒ the Z9
        router REGISTERS a software stop (stage-0 seam consumed, not
        reimplemented): stop = fill × (1 − z9 pct), source=z9, float qty
        verbatim; never a truncated broker stop."""
        config = _frac_config(tmp_path)
        registry = SoftwareStopRegistry.from_config(config)
        broker = FakeBroker(
            fills={"BLK": {"status": "filled", "order_id": "ord-1",
                           "filled_qty": FRACTIONAL_QTY,
                           "filled_avg_price": 100.0}},
            fractional_contract=True,
        )
        ra = _make_adapter(tmp_path, config=config, broker=broker,
                           software_stops=registry)
        ctx = _make_ctx(
            config,
            orders=[{"ticker": "BLK", "shares": FRACTIONAL_QTY, "price": 100.0}],
            prices={"BLK": 100.0}, cash=1_000.0,
        )
        with caplog.at_level("INFO", logger="live.runner"):
            ra.commit(ctx)

        assert [o["ticker"] for o in ctx.orders_placed] == ["BLK"]
        entry = registry.get("BLK")
        assert entry is not None
        assert entry["qty"] == FRACTIONAL_QTY
        assert entry["stop_price"] == pytest.approx(100.0 * (1 - 0.2))
        assert entry["source"] == "z9"
        assert "software stop registered" in caplog.text
        # Broker-side: no stop order, no Z9 bookkeeping.
        assert broker.place_stop_calls == []
        assert _saved_state(tmp_path)["stop_orders"] == {}

    def test_breach_to_full_exit_deregisters(self, tmp_path):
        """The full loop: armed registry + intraday breach ⇒ the sell-only
        task queues the exit ⇒ the REAL commit sells the FULL fractional
        qty, stamps wash-sale + post-stop blackout, and DISARMS the
        registry entry — zero residual protection ghosts."""
        from kernel.exits import HoldingState
        from kernel.pipeline.task_software_stops import SoftwareStopExitTask

        config = _frac_config(tmp_path)
        registry = SoftwareStopRegistry.from_config(config)
        registry.register("BLK", FRACTIONAL_QTY, 80.0, source="z9",
                          today_str="2026-06-20")
        broker = FakeBroker(
            fills={"BLK": {"status": "filled", "order_id": "sell-1",
                           "filled_qty": FRACTIONAL_QTY,
                           "filled_avg_price": 78.5}},
            positions={"BLK": FRACTIONAL_QTY},
            fractional_contract=True,
        )
        ra = _make_adapter(
            tmp_path, config=config, broker=broker, software_stops=registry,
            positions={"BLK": {"qty": FRACTIONAL_QTY,
                               "qty_available": FRACTIONAL_QTY,
                               "avg_entry_price": 100.0}},
            entry_dates={"BLK": "2026-06-20"},
            position_hwm={"BLK": 105.0},
        )
        hs = HoldingState(entry_price=100.0,
                          entry_date=datetime.date(2026, 6, 20),
                          high_watermark=105.0)
        ctx = _make_ctx(config, holdings={"BLK": hs},
                        prices={"BLK": 78.5})
        ctx.software_stops = registry

        # The sell-only loop pass (gap-down through the $80 stop).
        SoftwareStopExitTask().run(ctx)
        assert [t for t, _ in ctx.exits] == ["BLK"]
        assert ctx.exits[0][1].quantity == FRACTIONAL_QTY

        ra.commit(ctx)
        assert [t for t, _ in ctx.exits_placed] == ["BLK"]
        # Full liquidation of the fractional qty ⇒ registry disarmed.
        assert registry.get("BLK") is None
        assert registry.symbols() == []
        state = _saved_state(tmp_path)
        assert state["last_sell_dates"]["BLK"] == TODAY.isoformat()
        # software_stop is a stop ⇒ post-stop re-entry blackout stamped.
        assert state["last_stop_exit_dates"]["BLK"] == TODAY.isoformat()
        assert broker.get_position("BLK") == 0.0       # zero residual

    def test_topup_never_loosens_through_z9_route(self, tmp_path):
        """The Z9 router consumes the registry's ratchet: a top-up at a
        LOWER reference can never widen the stop; a higher reference
        tightens it."""
        from adapters.z9_stops import place_or_replace_stop

        registry = _registry(tmp_path)
        broker = FakeBroker()
        stop_orders: dict = {}

        place_or_replace_stop(broker, stop_orders, "BLK", FRACTIONAL_QTY,
                              100.0, "2026-07-03", ctx_pct=0.2,
                              software_stops=registry)
        assert registry.get("BLK")["stop_price"] == pytest.approx(80.0)

        # Top-up at a lower reference ⇒ proposed 72 REFUSED, stop stays 80.
        place_or_replace_stop(broker, stop_orders, "BLK", 0.9,
                              90.0, "2026-07-04", ctx_pct=0.2,
                              software_stops=registry)
        entry = registry.get("BLK")
        assert entry["stop_price"] == pytest.approx(80.0)
        assert entry["qty"] == 0.9                     # qty refreshed
        assert entry["history"][-1]["action"] == "ratchet_refused"

        # Higher reference ⇒ ratchets up to 96.
        place_or_replace_stop(broker, stop_orders, "BLK", 0.9,
                              120.0, "2026-07-05", ctx_pct=0.2,
                              software_stops=registry)
        assert registry.get("BLK")["stop_price"] == pytest.approx(96.0)

        # Never a broker-side order for the fractional qty, ever.
        assert broker.place_stop_calls == []
        assert stop_orders == {}

    def test_external_disposition_gc_disarms(self, tmp_path):
        """A position sold outside the runner (external/manual) leaves a
        registry ghost — STATE-GC disarms it on the next commit."""
        config = _frac_config(tmp_path)
        registry = SoftwareStopRegistry.from_config(config)
        registry.register("GONE", 0.5, 40.0, source="z9")
        broker = FakeBroker(fractional_contract=True)
        ra = _make_adapter(tmp_path, config=config, broker=broker,
                           software_stops=registry)
        ctx = _make_ctx(config, prices={}, cash=100.0)
        ra.commit(ctx)
        assert registry.get("GONE") is None
        assert registry.symbols() == []
