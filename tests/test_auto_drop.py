"""Auto-drop watchlist feature: ticker filtered for N days → drop from universe.

User feature 2026-04-24: ticker that gets filtered out (no candidate
emerging) for `monitoring.auto_drop_filter_days` consecutive days is
removed from the universe (LoadUniverseJob unloads its model).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


class TestStreakIncrement:
    def test_streak_increments_on_filtered_ticker(self):
        from kernel.pipeline.task_monitor import MonitorIdleStreakTask
        from kernel.pipeline.context import InferenceContext
        import datetime

        ctx = InferenceContext(
            config={
                "watchlist": ["NVDA", "MSFT"],
                "monitoring": {"auto_drop_filter_days": 5},
            },
            today=datetime.date(2025, 6, 1),
        )
        # NVDA in candidates, MSFT filtered
        ctx.candidates = [SimpleNamespace(ticker="NVDA")]

        MonitorIdleStreakTask().run(ctx)

        streaks = ctx.monitor_state["filter_streaks"]
        assert streaks["NVDA"] == 0, "NVDA should reset (in candidates)"
        assert streaks["MSFT"] == 1, "MSFT should increment (not in candidates)"

    def test_streak_resumes_from_state(self):
        from kernel.pipeline.task_monitor import MonitorIdleStreakTask
        from kernel.pipeline.context import InferenceContext
        import datetime

        ctx = InferenceContext(
            config={
                "watchlist": ["MSFT"],
                "monitoring": {"auto_drop_filter_days": 5},
            },
            today=datetime.date(2025, 6, 1),
        )
        ctx.monitor_state = {"filter_streaks": {"MSFT": 4}}
        ctx.candidates = []

        MonitorIdleStreakTask().run(ctx)
        assert ctx.monitor_state["filter_streaks"]["MSFT"] == 5

    def test_disabled_no_streak_tracking(self):
        """auto_drop_filter_days=0 → no filter_streaks key written."""
        from kernel.pipeline.task_monitor import MonitorIdleStreakTask
        from kernel.pipeline.context import InferenceContext
        import datetime

        ctx = InferenceContext(
            config={
                "watchlist": ["MSFT"],
                "monitoring": {"auto_drop_filter_days": 0},
            },
            today=datetime.date(2025, 6, 1),
        )
        ctx.candidates = []
        MonitorIdleStreakTask().run(ctx)
        assert "filter_streaks" not in ctx.monitor_state


class TestAutoDropTask:
    def test_drops_above_threshold(self, tmp_path):
        from kernel.pipeline.job_universe import (
            FilterAutoDropTask, UniverseContext,
        )
        # Seed live_state.json with a streak for HOOD = 70 days
        ls = tmp_path / "live_state.json"
        ls.write_text(json.dumps({
            "monitor_state": {"filter_streaks": {"HOOD": 70, "NVDA": 5}}
        }))

        uctx = UniverseContext(
            config={
                "monitoring": {"auto_drop_filter_days": 63},
                "defensive_tickers": [],
            },
            strategy_dir=tmp_path,
            loaded_models={
                "HOOD": {"_metadata": {"sharpe": 1.5}},
                "NVDA": {"_metadata": {"sharpe": 1.5}},
            },
            rejections=[],
        )
        FilterAutoDropTask().run(uctx)

        # HOOD dropped (70 >= 63), NVDA kept (5 < 63)
        assert "HOOD" not in uctx.loaded_models
        assert "NVDA" in uctx.loaded_models
        assert any("auto_drop" in r[1] for r in uctx.rejections
                    if r[0] == "HOOD")

    def test_disabled_short_circuits(self, tmp_path):
        from kernel.pipeline.job_universe import (
            FilterAutoDropTask, UniverseContext,
        )
        uctx = UniverseContext(
            config={"monitoring": {"auto_drop_filter_days": 0}},
            strategy_dir=tmp_path,
            loaded_models={"HOOD": {"_metadata": {"sharpe": 1.5}}},
            rejections=[],
        )
        # Even with a high streak in state, if flag=0 → never run
        assert FilterAutoDropTask().should_skip(uctx) is True

    def test_defensive_tickers_exempt(self, tmp_path):
        from kernel.pipeline.job_universe import (
            FilterAutoDropTask, UniverseContext,
        )
        ls = tmp_path / "live_state.json"
        ls.write_text(json.dumps({
            "monitor_state": {"filter_streaks": {"GLD": 100}}
        }))

        uctx = UniverseContext(
            config={
                "monitoring": {"auto_drop_filter_days": 63},
                "defensive_tickers": ["GLD", "TLT"],
            },
            strategy_dir=tmp_path,
            loaded_models={"GLD": {"_metadata": {"sharpe": 1.5}}},
            rejections=[],
        )
        FilterAutoDropTask().run(uctx)
        # GLD is defensive — never drop
        assert "GLD" in uctx.loaded_models

    def test_held_tickers_exempt(self, tmp_path):
        from kernel.pipeline.job_universe import (
            FilterAutoDropTask, UniverseContext,
        )
        ls = tmp_path / "live_state.json"
        ls.write_text(json.dumps({
            "monitor_state": {
                "filter_streaks": {"HELD": 100, "DEAD": 100},
            }
        }))

        uctx = UniverseContext(
            config={
                "monitoring": {"auto_drop_filter_days": 63},
                "defensive_tickers": [],
            },
            strategy_dir=tmp_path,
            loaded_models={
                "HELD": {"_metadata": {"sharpe": 0.2}},
                "DEAD": {"_metadata": {"sharpe": 1.5}},
            },
            held_tickers={"HELD"},
            rejections=[],
        )
        FilterAutoDropTask().run(uctx)
        assert "HELD" in uctx.loaded_models
        assert "DEAD" not in uctx.loaded_models

    def test_no_state_file_no_drop(self, tmp_path):
        """When live_state.json doesn't exist (fresh sim), no streaks → no drops."""
        from kernel.pipeline.job_universe import (
            FilterAutoDropTask, UniverseContext,
        )
        uctx = UniverseContext(
            config={
                "monitoring": {"auto_drop_filter_days": 63},
                "defensive_tickers": [],
            },
            strategy_dir=tmp_path,
            loaded_models={"HOOD": {"_metadata": {"sharpe": 1.5}}},
            rejections=[],
        )
        FilterAutoDropTask().run(uctx)
        assert "HOOD" in uctx.loaded_models   # no state → no drop
