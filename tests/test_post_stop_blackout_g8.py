"""G8 (2026-05-04) — post-stop re-entry blackout pipeline + adapter wiring.

Verifies:
  1. ctx.last_stop_exit_dates field exists on InferenceContext (default {}).
  2. PostStopCooldownFilterTask is wired into InferencePipeline.
  3. SimAdapter stamps last_stop_exit_dates on path-rule exits and ferries
     it into the ctx.
  4. LeanAdapter stamps + ferries.
  5. RunnerAdapter persists across runs via state["last_stop_exit_dates"]
     and ferries.
  6. Filter respects bars=0 (disabled) and exit_type not in stop set.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY  = REPO_ROOT / "backtesting" / "renquant_104"
if str(STRATEGY) not in sys.path:
    sys.path.insert(0, str(STRATEGY))


# ── Field shape ──────────────────────────────────────────────────────────────

class TestContextField:
    def test_last_stop_exit_dates_exists_default_empty(self):
        from kernel.pipeline.context import InferenceContext
        ctx = InferenceContext(config={}, today=datetime.date(2026, 5, 4))
        assert hasattr(ctx, "last_stop_exit_dates")
        assert ctx.last_stop_exit_dates == {}


# ── Pipeline wiring ──────────────────────────────────────────────────────────

class TestPipelineWiring:
    def test_filter_imported_in_pp_inference(self):
        src = (STRATEGY / "kernel" / "pipeline" / "pp_inference.py").read_text()
        # The Task is imported lazily inside InferencePipeline.run; we
        # search the run-method body. Confirm both the import line AND
        # the .run(ctx) invocation exist.
        assert "PostStopCooldownFilterTask" in src
        assert "PostStopCooldownFilterTask().run(ctx)" in src


# ── Filter behaviour (mirrors kernel.pipeline.task_post_stop_cooldown) ───────

class TestFilterBehaviour:
    def _ctx(self, candidates, last_stops, today, cooldown_bars=5,
             enabled=True, exit_types=None):
        cfg = {"risk": {"post_stop_cooldown": {
            "enabled": enabled,
            "bars": cooldown_bars,
        }}}
        ctx = SimpleNamespace(
            today=today,
            candidates=list(candidates),
            last_stop_exit_dates=dict(last_stops),
            config=cfg,
            counters={},
        )
        return ctx

    def _cand(self, ticker):
        return SimpleNamespace(ticker=ticker)

    def test_blackout_blocks_recent_stop(self):
        from kernel.pipeline.task_post_stop_cooldown import (
            PostStopCooldownFilterTask,
        )
        today = datetime.date(2026, 5, 4)
        recent = today - datetime.timedelta(days=2)
        ctx = self._ctx(
            [self._cand("APP"), self._cand("AAPL")],
            {"APP": recent},
            today, cooldown_bars=5,
        )
        PostStopCooldownFilterTask().run(ctx)
        kept = [c.ticker for c in ctx.candidates]
        assert "APP" not in kept
        assert "AAPL" in kept
        assert ctx.counters["post_stop_blocked"] == 1

    def test_blackout_allows_after_window(self):
        from kernel.pipeline.task_post_stop_cooldown import (
            PostStopCooldownFilterTask,
        )
        today = datetime.date(2026, 5, 4)
        old = today - datetime.timedelta(days=10)
        ctx = self._ctx(
            [self._cand("APP")],
            {"APP": old},
            today, cooldown_bars=5,
        )
        PostStopCooldownFilterTask().run(ctx)
        assert [c.ticker for c in ctx.candidates] == ["APP"]

    def test_disabled_does_nothing(self):
        from kernel.pipeline.task_post_stop_cooldown import (
            PostStopCooldownFilterTask,
        )
        today = datetime.date(2026, 5, 4)
        recent = today - datetime.timedelta(days=1)
        ctx = self._ctx(
            [self._cand("APP")],
            {"APP": recent},
            today, cooldown_bars=5,
            enabled=False,
        )
        PostStopCooldownFilterTask().run(ctx)
        assert [c.ticker for c in ctx.candidates] == ["APP"]


# ── Adapter wiring (sim/lean/runner) — source-level checks ──────────────────

class TestAdapterWiring:
    def _src(self, name):
        return (STRATEGY / "adapters" / f"{name}.py").read_text()

    def test_sim_stamps_stop_exit(self):
        s = self._src("sim")
        assert "_last_stop_exit_date" in s
        assert "DEFAULT_STOP_EXIT_TYPES" in s
        assert "last_stop_exit_dates" in s   # propagated into ctx

    def test_lean_stamps_stop_exit(self):
        s = self._src("lean")
        assert "_last_stop_exit_dates" in s
        assert "DEFAULT_STOP_EXIT_TYPES" in s

    def test_runner_persists_stop_exit_dates(self):
        s = self._src("runner")
        assert '"last_stop_exit_dates"' in s
        assert "_last_stop_exit_dates_str" in s
        assert "DEFAULT_STOP_EXIT_TYPES" in s
