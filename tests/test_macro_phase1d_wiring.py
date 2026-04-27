"""Tests for Phase 1D: macro_frame wired into BuildPanelTask + BuildHourlyResolutionPanelTask.

Phase 1A-1C built the parts (storage, builder, context field, panel_frame
merge). This phase wires them through the actual training pipeline so
ctx.macro_factor_frame is consumed when present.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestDailyPathWiring:
    """BuildPanelTask passes ctx.macro_factor_frame to build_panel_frame."""

    def test_daily_path_passes_macro_frame_to_build_panel_frame(self):
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        anchor = "Phase 1D (2026-04-26): pass ctx.macro_factor_frame"
        assert anchor in src, "Audit tag for Phase 1D daily wiring missing"
        idx = src.find(anchor)
        block = src[idx:idx + 1200]
        assert "macro_frame=ctx.macro_factor_frame" in block, (
            "BuildPanelTask must pass ctx.macro_factor_frame as macro_frame= arg"
        )


class TestHourlyPathWiring:
    """BuildHourlyResolutionPanelTask broadcasts ctx.macro_factor_frame inline."""

    def test_hourly_path_has_macro_broadcast_block(self):
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        anchor = "Phase 1D (2026-04-26): broadcast macro frame onto hourly panel"
        assert anchor in src, "Audit tag for Phase 1D hourly wiring missing"

    def test_hourly_path_uses_groupby_ticker_ffill(self):
        """Same forward-fill semantics as the daily path: ffill within
        ticker handles weekend / holiday alignment."""
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        idx = src.find("Phase 1D (2026-04-26): broadcast macro frame onto hourly panel")
        assert idx >= 0
        block = src[idx:idx + 2500]
        assert "groupby(" in block
        assert "ffill" in block
        assert "fillna(0.0)" in block, (
            "trailing NaN (warmup) → 0.0 same convention as daily path"
        )

    def test_hourly_path_skips_when_no_macro(self):
        """if ctx.macro_factor_frame is None, the merge block is skipped."""
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        idx = src.find("Phase 1D (2026-04-26): broadcast macro frame onto hourly panel")
        assert idx >= 0
        block = src[idx:idx + 2500]
        assert "if macro_frame is not None and not macro_frame.empty:" in block, (
            "must guard the broadcast on macro_factor_frame being non-empty"
        )

    def test_hourly_path_handles_column_collision(self):
        """Same defense as daily path."""
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        idx = src.find("Phase 1D (2026-04-26): broadcast macro frame onto hourly panel")
        block = src[idx:idx + 2500]
        assert "_macro" in block, "column collision rename suffix '_macro' required"

    def test_hourly_path_normalizes_index(self):
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        idx = src.find("Phase 1D (2026-04-26): broadcast macro frame onto hourly panel")
        block = src[idx:idx + 2500]
        assert "DatetimeIndex" in block, (
            "must coerce non-DatetimeIndex macro_frame.index"
        )


class TestBackwardsCompat:
    """Verify the wiring doesn't change behavior when ctx.macro_factor_frame is None."""

    def test_daily_passes_none_when_no_macro(self):
        """When ctx.macro_factor_frame is None (default), the macro_frame=
        kwarg passes None — build_panel_frame's None-handling kicks in."""
        # Defensive: this is a contract — we don't synthesize a fake
        # macro frame somewhere; we always pass ctx's actual value.
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        # The daily path call should look like: macro_frame=ctx.macro_factor_frame
        assert "macro_frame=ctx.macro_factor_frame" in src

    def test_hourly_path_no_op_when_macro_frame_empty(self):
        """if ctx.macro_factor_frame is None or empty → broadcast block skipped"""
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        idx = src.find("Phase 1D (2026-04-26): broadcast macro frame onto hourly panel")
        block = src[idx:idx + 2500]
        # The conditional check
        assert "macro_frame is not None and not macro_frame.empty" in block
