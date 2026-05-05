"""Job-layer acceptance tests — Job in isolation, ctx contract verified.

Each Job has a documented contract: which ctx fields it READS and which
it WRITES. These tests build a minimal stub ctx, run the Job, and
verify the writes match the contract. Catches regressions where:
  * A Job stops populating a ctx field that downstream Jobs expect
  * A Job produces output of the wrong shape (e.g. dict instead of list)
  * A Job's "should_skip" returns True when it shouldn't (silent no-op)

User mandate (2026-05-04).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[3]
STRATEGY = REPO / "backtesting" / "renquant_104"
if str(STRATEGY) not in sys.path:
    sys.path.insert(0, str(STRATEGY))


# ── ScanTrainingDataTask (preflight) ────────────────────────────────────────

class TestScanTrainingDataTaskAcceptance:
    """Contract: writes ctx.training_data_scan as a non-empty dict."""

    def test_writes_training_data_scan_dict(self, tmp_path):
        from training_panel.pp_panel_training import (
            ScanTrainingDataTask, PanelTrainingContext,
        )
        # Lay out a minimal repo
        sd = tmp_path / "backtesting" / "renquant_104"
        sd.mkdir(parents=True)
        ctx = PanelTrainingContext(
            config={
                "_strategy_dir": str(sd),
                "panel_ltr": {
                    "data_scan": {"enabled": True, "strict": False},
                    "hourly":    {"enabled": False},
                    "minute":    {"enabled": False},
                },
            },
            watchlist=["AAPL", "MSFT"],
        )
        ScanTrainingDataTask().run(ctx)
        # Contract: non-empty dict
        assert isinstance(ctx.training_data_scan, dict)
        assert ctx.training_data_scan, "training_data_scan must be populated"
        # Schema verification via protocol
        sys.path.insert(0, str(REPO / "tests"))
        from acceptance.protocol import assert_data_scan_report
        # Persist and verify via path-level assertion
        artifact_path = sd / "artifacts" / "training_data_scan.json"
        assert artifact_path.exists(), "Scan task must persist report to artifacts/"
        assert_data_scan_report(artifact_path)


# ── BuyGatesJob acceptance: sets ctx.buy_blocked correctly ──────────────────

class TestBuyGatesJobAcceptance:
    """Contract: BuyGatesJob writes ctx.buy_blocked when SPY missing /
    drawdown halt / velocity crash."""

    def _make_ctx(self, **overrides):
        # Minimal stub for InferenceContext-like
        from kernel.regime import RegimeState
        ctx = SimpleNamespace(
            regime="BULL_CALM",
            confidence=0.5,
            buy_blocked=False,
            bear_only=False,
            skip_buys=False,
            spy_returns=[0.001] * 100,
            ohlcv={},   # SPY missing
            counters={},
            regime_state=RegimeState(),
            config={"regime_params": {"BULL_CALM": {}}, "regime": {}},
        )
        for k, v in overrides.items():
            setattr(ctx, k, v)
        return ctx

    def test_ema50gate_missing_spy_blocks_buys(self):
        from kernel.pipeline.task_gates import EMA50GateTask
        ctx = self._make_ctx()
        EMA50GateTask().run(ctx)
        assert ctx.buy_blocked is True, \
            "BuyGates contract: missing SPY OHLCV must set buy_blocked=True"


# ── DataFreshnessGate acceptance ────────────────────────────────────────────

class TestDataFreshnessGateAcceptance:
    """Contract: empty ctx.ohlcv warns + returns True; stale ohlcv raises."""

    def test_empty_ohlcv_warns_no_raise(self):
        from kernel.pipeline.task_data_freshness import DataFreshnessGateTask
        import datetime as _dt
        ctx = SimpleNamespace(
            today=_dt.date(2026, 5, 1),
            ohlcv={},
            config={"data_freshness": {"enabled": True}},
        )
        # Must not raise on empty ohlcv (test stub friendly)
        result = DataFreshnessGateTask().run(ctx)
        assert result is True

    def test_disabled_short_circuits(self):
        from kernel.pipeline.task_data_freshness import DataFreshnessGateTask
        import datetime as _dt
        ctx = SimpleNamespace(
            today=_dt.date(2026, 5, 1),
            ohlcv={"SPY": "ANYTHING"},
            config={"data_freshness": {"enabled": False}},
        )
        # Disabled flag must short-circuit cleanly
        DataFreshnessGateTask().run(ctx)


# ── PanelDataJob acceptance: ScanTrainingDataTask runs FIRST ────────────────

class TestPanelDataJobAcceptance:
    """Contract: PanelDataJob.tasks[0] is ScanTrainingDataTask (preflight
    runs before any expensive load)."""

    def test_scan_runs_first(self):
        from training_panel.pp_panel_training import (
            PanelDataJob, ScanTrainingDataTask,
        )
        tasks = PanelDataJob().tasks
        assert isinstance(tasks[0], ScanTrainingDataTask), (
            "PanelDataJob contract: ScanTrainingDataTask must be the "
            "first task (data preflight before any expensive load)"
        )
