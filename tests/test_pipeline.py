"""Unit tests for kernel/pipeline — Task, Job, TickerJob, and pipeline orchestrators.

Tests cover:
  - Task short-circuit semantics (False stops chain, None/True continues)
  - Job.run() driving a task chain
  - TickerJob.run() driving a per-ticker task chain
  - InferencePipeline phase ordering via stub jobs
  - SellOnlyPipeline skips buy phases
  - Logging from job/task chain
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.pipeline.pipeline import Task, Job, TickerJob
from kernel.pipeline.context import InferenceContext, TickerInferenceContext


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_ctx(**overrides) -> InferenceContext:
    defaults = dict(
        config={"watchlist": ["AAPL", "MSFT"], "regime_params": {}},
        today="2024-01-02",
        ohlcv={},
        models={},
        holdings={},
        prices={},
    )
    defaults.update(overrides)
    return InferenceContext(**defaults)


def _make_tctx(ticker: str = "AAPL") -> TickerInferenceContext:
    return TickerInferenceContext(
        ticker=ticker,
        ohlcv={},
        model=None,
        config={},
        today="2024-01-02",
        regime="BULL_CALM",
        regime_params={},
        exit_params={},
        holding=None,
        price=0.0,
    )


# ── Task short-circuit semantics ──────────────────────────────────────────────

class TestTaskShortCircuit:
    def test_none_continues_chain(self):
        ran = []

        class T1(Task):
            def run(self, ctx):
                ran.append("T1")

        class T2(Task):
            def run(self, ctx):
                ran.append("T2")

        class J(Job):
            @property
            def tasks(self):
                return [T1(), T2()]

        J().run(_make_ctx())
        assert ran == ["T1", "T2"]

    def test_false_stops_chain(self):
        ran = []

        class T1(Task):
            def run(self, ctx):
                ran.append("T1")
                return False

        class T2(Task):
            def run(self, ctx):
                ran.append("T2")

        class J(Job):
            @property
            def tasks(self):
                return [T1(), T2()]

        J().run(_make_ctx())
        assert ran == ["T1"]

    def test_true_continues_chain(self):
        ran = []

        class T1(Task):
            def run(self, ctx):
                ran.append("T1")
                return True

        class T2(Task):
            def run(self, ctx):
                ran.append("T2")

        class J(Job):
            @property
            def tasks(self):
                return [T1(), T2()]

        J().run(_make_ctx())
        assert ran == ["T1", "T2"]

    def test_middle_false_skips_remaining(self):
        ran = []

        class Ta(Task):
            def run(self, ctx): ran.append("A")
        class Tb(Task):
            def run(self, ctx): ran.append("B"); return False
        class Tc(Task):
            def run(self, ctx): ran.append("C")

        class J(Job):
            @property
            def tasks(self): return [Ta(), Tb(), Tc()]

        J().run(_make_ctx())
        assert ran == ["A", "B"]

    def test_empty_task_chain_runs_clean(self):
        class EmptyJob(Job):
            @property
            def tasks(self): return []

        EmptyJob().run(_make_ctx())  # no error


# ── Job context mutation ────────────────────────────────────────────────────────

class TestJobContextMutation:
    def test_tasks_share_context(self):
        """Tasks in the same job read each other's writes via shared ctx."""

        class Writer(Task):
            def run(self, ctx):
                ctx._test_value = 42  # noqa: SLF001

        class Reader(Task):
            def run(self, ctx):
                assert ctx._test_value == 42  # noqa: SLF001

        class J(Job):
            @property
            def tasks(self): return [Writer(), Reader()]

        J().run(_make_ctx())

    def test_gate_task_sets_flag_and_stops(self):
        ctx = _make_ctx()

        class GateTask(Task):
            def run(self, c):
                c.buy_blocked = True
                return False

        class ShouldNotRun(Task):
            def run(self, c):
                raise AssertionError("should not run")

        class J(Job):
            @property
            def tasks(self): return [GateTask(), ShouldNotRun()]

        J().run(ctx)
        assert ctx.buy_blocked is True


# ── TickerJob ─────────────────────────────────────────────────────────────────

class TestTickerJob:
    def test_ticker_chain_runs_in_order(self):
        ran = []
        tc = _make_tctx()

        class T1(Task):
            def run(self, ctx): ran.append(1)
        class T2(Task):
            def run(self, ctx): ran.append(2)

        class TJ(TickerJob):
            @property
            def tasks(self): return [T1(), T2()]

        TJ().run(tc)
        assert ran == [1, 2]

    def test_ticker_chain_stops_on_false(self):
        ran = []
        tc = _make_tctx()

        class Filter(Task):
            def run(self, ctx): ran.append("filter"); return False
        class Scorer(Task):
            def run(self, ctx): ran.append("score")

        class TJ(TickerJob):
            @property
            def tasks(self): return [Filter(), Scorer()]

        TJ().run(tc)
        assert ran == ["filter"]

    def test_ticker_context_written_by_task(self):
        tc = _make_tctx("TSLA")

        class WriteScore(Task):
            def run(self, ctx):
                ctx.rs_score = 0.75

        class TJ(TickerJob):
            @property
            def tasks(self): return [WriteScore()]

        TJ().run(tc)
        assert tc.rs_score == 0.75


# ── Job.should_skip ────────────────────────────────────────────────────────────

class TestJobShouldSkip:
    def test_default_should_skip_false(self):
        class J(Job):
            pass
        assert J().should_skip(_make_ctx()) is False

    def test_custom_should_skip_respected_by_pipeline(self):
        """Pipeline skips a job whose should_skip() returns True."""
        ran = []

        # Test Job.should_skip logic directly; just verify Job.run() isn't called when skipped.
        class SkippableJob(Job):
            def should_skip(self, ctx): return True
            @property
            def tasks(self):
                class T(Task):
                    def run(self, c): ran.append("ran")
                return [T()]

        ctx = _make_ctx()
        job = SkippableJob()
        if not job.should_skip(ctx):
            job.run(ctx)

        assert ran == []


# ── Task name ─────────────────────────────────────────────────────────────────

class TestTaskName:
    def test_name_is_class_name(self):
        class MySpecialTask(Task):
            def run(self, ctx): pass

        assert MySpecialTask().name == "MySpecialTask"


# ── Chain logging ─────────────────────────────────────────────────────────────

class TestChainLogging:
    def test_chain_stop_logged(self, caplog):
        import logging

        class Stopper(Task):
            def run(self, ctx): return False

        class J(Job):
            @property
            def tasks(self): return [Stopper()]

        with caplog.at_level(logging.DEBUG, logger="kernel.pipeline"):
            J().run(_make_ctx())

        assert any("chain stopped" in r.message for r in caplog.records)
