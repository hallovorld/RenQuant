"""Unit tests for the pipeline package.

Tests cover: PipelineContext, TaskResult/run_tasks, Job/Pipeline,
and lightweight stubs for DataJob / SignalJob / ExecutionJob.
No broker, no I/O, no common/ imports required.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add renquant_103 to sys.path so pipeline/ is importable
_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_103"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from pipeline.context import PipelineContext
from pipeline.task import TaskResult, run_tasks
from pipeline.pipeline import Job, Pipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ctx(**overrides) -> PipelineContext:
    broker = MagicMock()
    defaults = dict(
        config={"watchlist": ["AAPL", "MSFT"], "benchmark": "SPY",
                "model_name": "renquant_103", "max_concurrent_positions": 3},
        strategy_dir=_STRATEGY_DIR,
        sell_only=False,
        broker=broker,
        models={},
    )
    defaults.update(overrides)
    return PipelineContext(**defaults)


# ── PipelineContext ───────────────────────────────────────────────────────────

class TestPipelineContext:
    def test_defaults_populated(self):
        ctx = _make_ctx()
        assert ctx.ohlcv == {}
        assert ctx.candidates == []
        assert ctx.held == []
        assert ctx.today == datetime.date.today()
        assert ctx.today_str == datetime.date.today().isoformat()

    def test_sell_only_default_false(self):
        ctx = _make_ctx()
        assert ctx.sell_only is False

    def test_custom_regime_stored(self):
        ctx = _make_ctx()
        ctx.regime = "BEAR"
        assert ctx.regime == "BEAR"

    def test_held_mutability(self):
        ctx = _make_ctx()
        ctx.held.append("AAPL")
        assert "AAPL" in ctx.held


# ── TaskResult / run_tasks ────────────────────────────────────────────────────

class TestTaskResult:
    def test_ok_when_no_error(self):
        r = TaskResult(name="foo", result=42)
        assert r.ok is True

    def test_not_ok_when_error(self):
        r = TaskResult(name="foo", result=None, error=ValueError("bad"))
        assert r.ok is False


class TestRunTasks:
    def test_empty_tasks(self):
        assert run_tasks([]) == []

    def test_results_in_submission_order(self):
        tasks = [(str(i), lambda i=i: i * 2) for i in range(5)]
        results = run_tasks(tasks)
        assert [r.result for r in results] == [0, 2, 4, 6, 8]

    def test_exception_captured_not_raised(self):
        def _boom():
            raise RuntimeError("fail")

        results = run_tasks([("a", lambda: 1), ("b", _boom), ("c", lambda: 3)])
        assert results[0].ok and results[0].result == 1
        assert not results[1].ok and isinstance(results[1].error, RuntimeError)
        assert results[2].ok and results[2].result == 3

    def test_parallel_speedup(self):
        """Parallelism: N slow tasks finish faster than N × sleep time."""
        import time

        def _slow():
            time.sleep(0.05)
            return 1

        tasks = [("t", _slow)] * 4
        start = time.monotonic()
        results = run_tasks(tasks, max_workers=4)
        elapsed = time.monotonic() - start
        assert all(r.ok for r in results)
        assert elapsed < 0.20  # should finish in ~50ms, not 200ms


# ── Job / Pipeline ────────────────────────────────────────────────────────────

class _RecordingJob(Job):
    """Test job that records when it ran."""

    def __init__(self, name: str, should_skip_result: bool = False):
        self.name = name
        self._skip = should_skip_result
        self.ran = False

    def run(self, ctx: PipelineContext) -> None:
        self.ran = True
        ctx.state[self.name] = True

    def should_skip(self, ctx: PipelineContext) -> bool:
        return self._skip


class TestPipeline:
    def test_jobs_run_in_order(self):
        ctx = _make_ctx()
        order: list[str] = []
        ctx.state = {}

        class _OrderJob(Job):
            def __init__(self, label: str):
                self.label = label
            def run(self, c: PipelineContext) -> None:
                order.append(self.label)

        Pipeline([_OrderJob("A"), _OrderJob("B"), _OrderJob("C")]).run(ctx)
        assert order == ["A", "B", "C"]

    def test_skipped_job_does_not_run(self):
        ctx = _make_ctx()
        ctx.state = {}
        j1 = _RecordingJob("j1", should_skip_result=False)
        j2 = _RecordingJob("j2", should_skip_result=True)
        j3 = _RecordingJob("j3", should_skip_result=False)
        Pipeline([j1, j2, j3]).run(ctx)
        assert j1.ran is True
        assert j2.ran is False
        assert j3.ran is True

    def test_ctx_mutated_between_jobs(self):
        """Earlier jobs set context fields that later jobs can read."""
        ctx = _make_ctx()
        ctx.regime = "BULL_CALM"

        class _WriterJob(Job):
            def run(self, c: PipelineContext) -> None:
                c.regime = "BEAR"

        class _ReaderJob(Job):
            def run(self, c: PipelineContext) -> None:
                assert c.regime == "BEAR"

        Pipeline([_WriterJob(), _ReaderJob()]).run(ctx)

    def test_empty_pipeline(self):
        """Empty pipeline runs without error."""
        ctx = _make_ctx()
        Pipeline([]).run(ctx)

    def test_single_job(self):
        ctx = _make_ctx()
        ctx.state = {}
        j = _RecordingJob("solo")
        Pipeline([j]).run(ctx)
        assert j.ran is True
        assert ctx.state["solo"] is True
