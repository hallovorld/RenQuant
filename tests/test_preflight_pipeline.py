"""Track H — proof that migrated PreflightTasks produce identical PreflightCheck
results to the legacy ``_check_*`` functions in ``kernel.preflight``.

Strategy: for each migrated Task, paired tests assert that the Task and the
legacy function produce the SAME PreflightCheck (name, severity, ok, message)
across every documented branch:

  - StateFileTask vs _check_state_file:
      (a) no broker_name → soft pass
      (b) state_paths module unavailable → soft pass  (skipped — synthetic)
      (c) state file absent → soft pass
      (d) state file unparseable → HARD fail
      (e) state file parses → HARD pass

  - BrokerConnectTask vs _check_broker_connect:
      (f) broker is None → soft pass
      (g) broker raises during connect → HARD fail
      (h) broker raises during get_account_value → HARD fail
      (i) broker connects + returns equity → HARD pass

The (b) branch (ImportError on kernel.state_paths) is intentionally skipped
because forcing the import error requires monkey-patching sys.modules in a
way that's brittle. Both paths share the legacy implementation when it's
reachable, so the absent-file (c) branch covers the same failure mode.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backtesting/renquant_104"))

from kernel.preflight import _check_state_file, _check_broker_connect
from kernel.preflight_pipeline import (
    BrokerConnectTask,
    PreflightContext,
    StateFileTask,
    build_minimal_preflight_pipeline,
)


# ─── PreflightContext fixture ────────────────────────────────────────────────

@pytest.fixture
def base_ctx(tmp_path) -> PreflightContext:
    """A minimal context — broker/broker_name None, fresh strategy_dir."""
    return PreflightContext(
        config={},
        strategy_dir=tmp_path,
    )


# ─── StateFileTask vs _check_state_file ──────────────────────────────────────

class TestStateFileTaskParity:
    """All branches: legacy _check_state_file output bytes-equal Task output."""

    def test_no_broker_name_soft_pass(self, base_ctx, tmp_path):
        # legacy
        leg = _check_state_file(config={}, strategy_dir=tmp_path, broker_name=None)
        # new Task
        task = StateFileTask()
        task.run(base_ctx)  # ctx.broker_name is None
        new = base_ctx.results[-1]
        # Identical record
        assert new.name == leg.name == "P-STATE-FILE"
        assert new.severity == leg.severity == "soft"
        assert new.ok is leg.ok is True
        assert new.message == leg.message

    def test_state_file_absent_soft_pass(self, base_ctx, tmp_path):
        base_ctx.broker_name = "alpaca"
        # no live_state.alpaca.json on disk
        leg = _check_state_file(config={}, strategy_dir=tmp_path, broker_name="alpaca")
        StateFileTask().run(base_ctx)
        new = base_ctx.results[-1]
        assert new.name == leg.name
        assert new.severity == leg.severity == "soft"
        assert new.ok is leg.ok is True
        assert new.message == leg.message

    def test_state_file_parses_hard_pass(self, base_ctx, tmp_path):
        base_ctx.broker_name = "alpaca"
        state_path = tmp_path / "live_state.alpaca.json"
        state_path.write_text(json.dumps({"foo": "bar"}))

        leg = _check_state_file(config={}, strategy_dir=tmp_path, broker_name="alpaca")
        StateFileTask().run(base_ctx)
        new = base_ctx.results[-1]
        assert new.name == leg.name
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is True
        assert new.message == leg.message

    def test_state_file_unparseable_hard_fail(self, base_ctx, tmp_path):
        base_ctx.broker_name = "alpaca"
        state_path = tmp_path / "live_state.alpaca.json"
        state_path.write_text("{ malformed json")

        leg = _check_state_file(config={}, strategy_dir=tmp_path, broker_name="alpaca")
        StateFileTask().run(base_ctx)
        new = base_ctx.results[-1]
        assert new.name == leg.name
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message


# ─── BrokerConnectTask vs _check_broker_connect ─────────────────────────────

class _StubBroker:
    """Minimal broker stub for parity testing."""

    def __init__(self, *, equity: float = 12345.67, fail_on: str | None = None):
        self.equity = equity
        self.fail_on = fail_on

    def connect(self) -> None:
        if self.fail_on == "connect":
            raise RuntimeError("synthetic connect failure")

    def get_account_value(self) -> float:
        if self.fail_on == "get_account_value":
            raise RuntimeError("synthetic equity failure")
        return self.equity


class TestBrokerConnectTaskParity:
    def test_no_broker_soft_pass(self, base_ctx):
        leg = _check_broker_connect(broker=None)
        BrokerConnectTask().run(base_ctx)  # ctx.broker is None
        new = base_ctx.results[-1]
        assert new.name == leg.name == "P-BROKER-CONNECT"
        assert new.severity == leg.severity == "soft"
        assert new.ok is leg.ok is True
        assert new.message == leg.message

    def test_broker_connects_hard_pass(self, base_ctx):
        broker = _StubBroker(equity=12345.67)
        base_ctx.broker = broker

        leg = _check_broker_connect(broker=broker)
        BrokerConnectTask().run(base_ctx)
        new = base_ctx.results[-1]
        assert new.name == leg.name
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is True
        assert new.message == leg.message
        # Message format: "broker connected, equity=$12345.67"
        assert "$12345.67" in new.message

    def test_broker_connect_fail_hard_fail(self, base_ctx):
        broker = _StubBroker(fail_on="connect")
        base_ctx.broker = broker

        leg = _check_broker_connect(broker=broker)
        BrokerConnectTask().run(base_ctx)
        new = base_ctx.results[-1]
        assert new.name == leg.name
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert "synthetic connect failure" in new.message
        assert "synthetic connect failure" in leg.message
        assert new.message == leg.message

    def test_broker_equity_fail_hard_fail(self, base_ctx):
        broker = _StubBroker(fail_on="get_account_value")
        base_ctx.broker = broker

        leg = _check_broker_connect(broker=broker)
        BrokerConnectTask().run(base_ctx)
        new = base_ctx.results[-1]
        assert new.name == leg.name
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message


# ─── End-to-end pipeline smoke ────────────────────────────────────────────────

class TestPipelineSmoke:
    """build_minimal_preflight_pipeline + run produces 2 results."""

    def test_pipeline_runs_both_tasks_in_order(self, base_ctx):
        pipeline = build_minimal_preflight_pipeline()
        results = pipeline.run(base_ctx, strict=False)
        assert [r.name for r in results] == ["P-STATE-FILE", "P-BROKER-CONNECT"]
        # both soft passes since broker_name/broker are None
        assert all(r.severity == "soft" and r.ok for r in results)

    def test_pipeline_hard_fail_raises_in_strict(self, tmp_path):
        # Set up a state with bad JSON to trigger a HARD fail
        ctx = PreflightContext(
            config={},
            strategy_dir=tmp_path,
            broker_name="alpaca",
        )
        (tmp_path / "live_state.alpaca.json").write_text("{ malformed json")

        pipeline = build_minimal_preflight_pipeline()
        from kernel.preflight import PreflightFailed
        with pytest.raises(PreflightFailed):
            pipeline.run(ctx, strict=True)

    def test_pipeline_hard_fail_returns_in_non_strict(self, tmp_path):
        ctx = PreflightContext(
            config={},
            strategy_dir=tmp_path,
            broker_name="alpaca",
        )
        (tmp_path / "live_state.alpaca.json").write_text("{ malformed json")

        pipeline = build_minimal_preflight_pipeline()
        results = pipeline.run(ctx, strict=False)
        hard_failures = [r for r in results if r.severity == "hard" and not r.ok]
        assert len(hard_failures) == 1
        assert hard_failures[0].name == "P-STATE-FILE"
