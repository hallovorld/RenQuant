"""Tests for Phase 1B: LoadMacroFactorsTask + PanelTrainingContext field.

Per macro design doc Phase 1B (kernel/pp_panel_training.py).

Verifies:
1. ctx.macro_factor_frame is None by default (off-by-default safety).
2. LoadMacroFactorsTask is a no-op when panel_ltr.macro.enabled=false.
3. When enabled + cache populated, ctx.macro_factor_frame becomes a
   non-empty DataFrame.
4. F1/F2/F4/F5/F9 safety paths bubble up: any exception leaves
   ctx.macro_factor_frame as None (pipeline proceeds in no-macro mode).
5. Task is wired into PanelDataJob's task chain.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.macro import MacroFactorStore   # noqa: E402
from training_panel.context import PanelTrainingContext   # noqa: E402
from training_panel.pp_panel_training import (   # noqa: E402
    LoadMacroFactorsTask,
    PanelDataJob,
)


def _synth(n: int = 500, start: str = "2024-01-01", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    rets = rng.normal(0, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": 1e6,
    }, index=dates)


def _ctx(*, enabled: bool = False, cache_dir: str = "data/macro",
          spy: pd.DataFrame | None = None) -> PanelTrainingContext:
    return PanelTrainingContext(
        config={
            "panel_ltr": {
                "macro": {
                    "enabled":  enabled,
                    "cache_dir": cache_dir,
                },
            },
            "benchmark": "SPY",
            "_strategy_dir": "/tmp/test_macro",
        },
        ohlcv={"SPY": spy} if spy is not None else {},
    )


# ── Context field default ──────────────────────────────────────────────────────

class TestContextDefaults:
    def test_macro_factor_frame_defaults_to_none(self):
        ctx = PanelTrainingContext(config={})
        assert ctx.macro_factor_frame is None

    def test_macro_metadata_defaults_to_empty_dict(self):
        ctx = PanelTrainingContext(config={})
        assert ctx.macro_metadata == {}


# ── LoadMacroFactorsTask — flag respect ───────────────────────────────────────

class TestLoadMacroFactorsTaskFlag:
    def test_disabled_flag_no_op(self, tmp_path):
        """Flag off → ctx.macro_factor_frame stays None."""
        ctx = _ctx(enabled=False)
        result = LoadMacroFactorsTask().run(ctx)
        assert result is True
        assert ctx.macro_factor_frame is None

    def test_pre_populated_skipped(self, tmp_path):
        """If macro_factor_frame already set, task doesn't overwrite."""
        ctx = _ctx(enabled=True)
        sentinel = pd.DataFrame({"a_z": [1.0]}, index=[pd.Timestamp("2024-01-01")])
        ctx.macro_factor_frame = sentinel
        LoadMacroFactorsTask().run(ctx)
        assert ctx.macro_factor_frame is sentinel


# ── LoadMacroFactorsTask — happy path with cached macros ──────────────────────

class TestLoadMacroFactorsHappyPath:
    def test_loads_cached_macros(self, tmp_path):
        # Populate cache with 1000 days (>252 + warmup), ensures F4 coverage passes
        store = MacroFactorStore(data_dir=tmp_path)
        store.save(_synth(1000), "VXX")

        ctx = _ctx(enabled=True, cache_dir=str(tmp_path),
                    spy=_synth(1000))
        ctx.config["panel_ltr"]["macro"]["symbols"] = ["VXX"]
        ctx.config["panel_ltr"]["macro"]["cache_dir"] = str(tmp_path)
        LoadMacroFactorsTask().run(ctx)
        assert ctx.macro_factor_frame is not None
        assert not ctx.macro_factor_frame.empty
        assert "vxx_level_z" in ctx.macro_factor_frame.columns

    def test_metadata_populated(self, tmp_path):
        store = MacroFactorStore(data_dir=tmp_path)
        store.save(_synth(1000), "VXX")
        store.save(_synth(1000, seed=1), "HYG")

        ctx = _ctx(enabled=True, spy=_synth(1000))
        ctx.config["panel_ltr"]["macro"]["symbols"] = ["VXX", "HYG"]
        ctx.config["panel_ltr"]["macro"]["cache_dir"] = str(tmp_path)
        LoadMacroFactorsTask().run(ctx)
        assert ctx.macro_metadata.get("n_features") == 6   # 2 syms × 3 transforms
        assert "VXX" in ctx.macro_metadata.get("symbols_used", [])
        assert "HYG" in ctx.macro_metadata.get("symbols_used", [])


# ── LoadMacroFactorsTask — safety paths leave None ────────────────────────────

class TestLoadMacroFactorsSafety:
    def test_no_cache_dir_proceeds_in_no_macro_mode(self, tmp_path):
        """Cache dir doesn't exist → build_macro_frame returns empty
        frame; ctx.macro_factor_frame is set to empty (not None)."""
        ctx = _ctx(enabled=True)
        ctx.config["panel_ltr"]["macro"]["symbols"] = ["VXX"]
        ctx.config["panel_ltr"]["macro"]["cache_dir"] = str(tmp_path / "missing")
        LoadMacroFactorsTask().run(ctx)
        # With no symbols cached, build_macro_frame returns empty frame
        assert ctx.macro_factor_frame is not None
        assert ctx.macro_factor_frame.empty   # empty is OK; pipeline checks
        assert ctx.macro_metadata.get("n_features") == 0

    def test_task_level_exception_leaves_none(self, tmp_path, monkeypatch):
        """If something inside the task raises (e.g., import fails,
        config malformed), ctx.macro_factor_frame stays None."""
        ctx = _ctx(enabled=True, spy=_synth(100))
        # Force build_macro_frame to raise
        from training_panel import pp_panel_training as ppt
        original = ppt.LoadMacroFactorsTask.run
        # Directly patch one of the inner imports to fail
        import kernel.macro as macro_mod
        def boom(*a, **kw):
            raise RuntimeError("simulated failure")
        monkeypatch.setattr(macro_mod, "build_macro_frame", boom)
        LoadMacroFactorsTask().run(ctx)
        # Task-level except sets ctx.macro_factor_frame to None
        assert ctx.macro_factor_frame is None


# ── Wiring into PanelDataJob ──────────────────────────────────────────────────

class TestPanelDataJobWiring:
    def test_load_macro_in_task_chain(self):
        job = PanelDataJob()
        task_names = [type(t).__name__ for t in job.tasks]
        assert "LoadMacroFactorsTask" in task_names

    def test_load_macro_after_load_minute_bars(self):
        """Order matters: macro depends on no upstream data, so it's
        last in the chain (cleanest dependency)."""
        job = PanelDataJob()
        names = [type(t).__name__ for t in job.tasks]
        i_minute = names.index("LoadMinuteBarsTask")
        i_macro = names.index("LoadMacroFactorsTask")
        assert i_macro > i_minute, "LoadMacroFactorsTask must come after LoadMinuteBarsTask"
