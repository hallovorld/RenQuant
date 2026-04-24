"""BC regression tests — Kelly-delta rotation gate in task_rotation.

Parallels the existing panel_rotation_advantage gate: after
find_rotation_pairs emits ER-based pairs, filter out any pair where
cand.kelly_target_pct - held.kelly_target_pct < rotation_advantage.
Unifies rotation math with Kelly sizing (selection, top-up, trim).

Gate is config-controlled: `ranking.kelly_sizing.rotation_advantage`
(default 0.0 = off). Pairs with missing Kelly target on either side
skip the gate (fallback to prior behaviour).
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.pipeline.task_rotation import BuildPairsTask  # noqa: E402
from kernel.rotation import RotationPair  # noqa: E402


def _hs(entry_date, entry_price=100.0, rank=0.3, er=0.02,
        panel_score=0.3, kelly_target=0.10):
    from kernel.exits import HoldingState
    h = HoldingState(
        entry_price=entry_price, entry_date=entry_date,
        shares=100, high_watermark=entry_price,
    )
    h.rank_score = rank
    h.expected_return = er
    h.panel_score = panel_score
    h.kelly_target_pct = kelly_target
    return h


def _cand(ticker, rank=0.4, er=0.08, panel_score=0.4, kelly_target=0.25):
    return SimpleNamespace(
        ticker=ticker, rank_score=rank, expected_return=er,
        panel_score=panel_score, kelly_target_pct=kelly_target,
        rs_score=0.0, raw_score=4.0, detail="",
    )


def _ctx(holdings, ranked, *, kelly_rot_advantage=0.0,
         panel_rot_advantage=0.0):
    today = datetime.date(2026, 4, 24)
    prices = {t: 100.0 for t in holdings}
    return SimpleNamespace(
        today      = today,
        holdings   = holdings,
        ranked     = ranked,
        exits      = [],
        rotations  = [],
        bear_only  = False,
        prices     = prices,
        counters   = {},
        config     = {
            "rotation": {
                "enabled": True,
                "min_expected_advantage_pct": 0.01,
                "target_horizon_days":        20,
                "transaction_cost_pct":       0.0,
                "min_rotation_hold_days":     0,     # test with no hold guard
                "lt_protection_days":         0,
                "max_rotations_per_bar":      5,
            },
            "ranking": {
                "panel_scoring": {"rotation_advantage": panel_rot_advantage},
                "kelly_sizing":  {"rotation_advantage": kelly_rot_advantage},
            },
            "tax": {"short_term_rate": 0.37, "long_term_rate": 0.20,
                    "long_term_threshold_days": 365},
        },
    )


class TestKellyRotationGateDefault:
    def test_off_by_default_no_filtering(self):
        """kelly_rot_advantage=0.0 → gate is a no-op, ER-only rule wins."""
        entry_d = datetime.date(2026, 1, 10)
        holdings = {"NVDA": _hs(entry_d, kelly_target=0.20)}
        ranked = [_cand("TSLA", er=0.10, kelly_target=0.22)]   # +0.02 Kelly
        ctx = _ctx(holdings, ranked, kelly_rot_advantage=0.0)
        BuildPairsTask().run(ctx)
        assert len(ctx.rotations) == 1
        assert ctx.rotations[0].buy_ticker == "TSLA"


class TestKellyRotationGateOn:
    def test_candidate_kelly_beats_held_by_threshold(self):
        """Kelly delta 0.25 - 0.10 = 0.15 > 0.10 threshold → pair kept."""
        entry_d = datetime.date(2026, 1, 10)
        holdings = {"NVDA": _hs(entry_d, kelly_target=0.10)}
        ranked = [_cand("TSLA", er=0.10, kelly_target=0.25)]
        ctx = _ctx(holdings, ranked, kelly_rot_advantage=0.10)
        BuildPairsTask().run(ctx)
        assert len(ctx.rotations) == 1
        assert ctx.rotations[0].buy_ticker == "TSLA"

    def test_candidate_kelly_below_threshold_filters(self):
        """Kelly delta 0.12 - 0.10 = 0.02 < 0.10 threshold → pair dropped."""
        entry_d = datetime.date(2026, 1, 10)
        holdings = {"NVDA": _hs(entry_d, kelly_target=0.10)}
        ranked = [_cand("TSLA", er=0.10, kelly_target=0.12)]
        ctx = _ctx(holdings, ranked, kelly_rot_advantage=0.10)
        BuildPairsTask().run(ctx)
        assert ctx.rotations == []
        assert ctx.counters.get("kelly_rotation_rejects", 0) == 1

    def test_candidate_kelly_below_held_filters(self):
        """Candidate has LOWER Kelly target — should be filtered."""
        entry_d = datetime.date(2026, 1, 10)
        holdings = {"NVDA": _hs(entry_d, kelly_target=0.25)}
        ranked = [_cand("TSLA", er=0.10, kelly_target=0.10)]   # lower
        ctx = _ctx(holdings, ranked, kelly_rot_advantage=0.05)
        BuildPairsTask().run(ctx)
        assert ctx.rotations == []


class TestMissingKellyFalsBack:
    """If either side's kelly_target_pct is None, gate skips (keeps the pair)."""

    def test_candidate_kelly_none_keeps_pair(self):
        entry_d = datetime.date(2026, 1, 10)
        holdings = {"NVDA": _hs(entry_d, kelly_target=0.10)}
        ranked = [_cand("TSLA", er=0.10, kelly_target=None)]
        ctx = _ctx(holdings, ranked, kelly_rot_advantage=0.10)
        BuildPairsTask().run(ctx)
        assert len(ctx.rotations) == 1

    def test_held_kelly_none_keeps_pair(self):
        entry_d = datetime.date(2026, 1, 10)
        holdings = {"NVDA": _hs(entry_d, kelly_target=None)}
        ranked = [_cand("TSLA", er=0.10, kelly_target=0.25)]
        ctx = _ctx(holdings, ranked, kelly_rot_advantage=0.10)
        BuildPairsTask().run(ctx)
        assert len(ctx.rotations) == 1


class TestPanelAndKellyGatesCombine:
    """Both gates can run in sequence — each filters independently."""

    def test_panel_gate_rejects_before_kelly(self):
        """Panel delta 0.30-0.25=0.05 < 0.10 threshold → panel drops it.
        Kelly gate never sees it."""
        entry_d = datetime.date(2026, 1, 10)
        holdings = {"NVDA": _hs(entry_d, panel_score=0.25, kelly_target=0.10)}
        ranked = [_cand("TSLA", er=0.10, panel_score=0.30, kelly_target=0.30)]
        ctx = _ctx(holdings, ranked,
                    panel_rot_advantage=0.10,
                    kelly_rot_advantage=0.10)
        BuildPairsTask().run(ctx)
        assert ctx.rotations == []
        # Counted against panel, not kelly, since panel runs first
        assert ctx.counters.get("panel_rotation_rejects", 0) == 1
        assert ctx.counters.get("kelly_rotation_rejects", 0) == 0

    def test_panel_passes_kelly_rejects(self):
        """Panel delta 0.20 > 0.10 threshold → kept by panel gate.
        Kelly delta 0.02 < 0.10 threshold → dropped by Kelly gate."""
        entry_d = datetime.date(2026, 1, 10)
        holdings = {"NVDA": _hs(entry_d, panel_score=0.25, kelly_target=0.10)}
        ranked = [_cand("TSLA", er=0.10, panel_score=0.45, kelly_target=0.12)]
        ctx = _ctx(holdings, ranked,
                    panel_rot_advantage=0.10,
                    kelly_rot_advantage=0.10)
        BuildPairsTask().run(ctx)
        assert ctx.rotations == []
        assert ctx.counters.get("kelly_rotation_rejects", 0) == 1
