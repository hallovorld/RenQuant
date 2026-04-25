"""Rotation V4 — full 4-point thesis-symmetric mode.

User's spec (2026-04-24):
  "买进 A 当天 AB 的 decision factor (from DB) 和今天 AB 的 scores"

Compares A at entry + today, AND B at entry (via DB lookup) + today.
Three thresholds:
  * a_velocity  = A_today − A_entry    ≤ −max_a_velocity
  * b_velocity  = B_today − B_entry    ≥ +min_b_velocity
  * cross_flip  = gap_today − gap_entry ≥ +min_cross_flip

Fires only when ALL three hold. Missing B_entry (no DB row on that
date) → skip pair (don't fire).
"""
from __future__ import annotations

import datetime
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


@dataclass
class _Cand:
    ticker: str
    rank_score: float


def _meta(entry_date="2025-01-01", entry_price=100.0, current_price=100.0):
    return {
        "entry_date":    datetime.date.fromisoformat(entry_date),
        "entry_price":   entry_price,
        "current_price": current_price,
    }


def _cfg(**overrides):
    cfg = {
        "enabled":                True,
        "min_rotation_hold_days": 0,
        "lt_protection_days":     0,
        "max_rotations_per_bar":  2,
        "transaction_cost_pct":   0.0,
        "target_horizon_days":    20,
    }
    cfg.update(overrides)
    return cfg


class TestThresholds:
    def test_all_three_pass_fires(self):
        from kernel.rotation import find_thesis_symmetric_pairs

        # A entered at 0.40, today 0.20 → a_velocity = −0.20 (ok vs -0.10)
        # B entered at 0.20, today 0.45 → b_velocity = +0.25 (ok vs +0.05)
        # gap_today = 0.45-0.20 = +0.25; gap_entry = 0.20-0.40 = -0.20
        # cross_flip = 0.25 - (-0.20) = +0.45 (ok vs 0.15)
        pairs = find_thesis_symmetric_pairs(
            held_entry_scores = {"A": 0.40},
            held_today_scores = {"A": 0.20},
            held_meta         = {"A": _meta()},
            candidates        = [_Cand("B", rank_score=0.45)],
            entry_day_lookup  = {("B", datetime.date(2025, 1, 1)): 0.20},
            today             = datetime.date(2025, 6, 1),
            rotation_cfg      = _cfg(thesis_symmetric={
                "max_a_velocity": 0.10, "min_b_velocity": 0.05,
                "min_cross_flip": 0.15,
            }),
            tax_cfg           = {},
        )
        assert len(pairs) == 1
        assert pairs[0].sell_ticker == "A"
        assert pairs[0].buy_ticker == "B"

    def test_a_velocity_fails_blocks(self):
        """A hasn't decayed enough."""
        from kernel.rotation import find_thesis_symmetric_pairs

        # A: entry 0.40, today 0.38 → vel = -0.02 (fails -0.10)
        pairs = find_thesis_symmetric_pairs(
            held_entry_scores = {"A": 0.40},
            held_today_scores = {"A": 0.38},
            held_meta         = {"A": _meta()},
            candidates        = [_Cand("B", rank_score=0.50)],
            entry_day_lookup  = {("B", datetime.date(2025, 1, 1)): 0.20},
            today             = datetime.date(2025, 6, 1),
            rotation_cfg      = _cfg(thesis_symmetric={
                "max_a_velocity": 0.10, "min_b_velocity": 0.05,
                "min_cross_flip": 0.15,
            }),
            tax_cfg           = {},
        )
        assert pairs == []

    def test_b_velocity_fails_blocks(self):
        """B hasn't gained enough."""
        from kernel.rotation import find_thesis_symmetric_pairs

        # B: entry 0.45, today 0.47 → vel = +0.02 (fails +0.05)
        pairs = find_thesis_symmetric_pairs(
            held_entry_scores = {"A": 0.40},
            held_today_scores = {"A": 0.20},
            held_meta         = {"A": _meta()},
            candidates        = [_Cand("B", rank_score=0.47)],
            entry_day_lookup  = {("B", datetime.date(2025, 1, 1)): 0.45},
            today             = datetime.date(2025, 6, 1),
            rotation_cfg      = _cfg(thesis_symmetric={
                "max_a_velocity": 0.10, "min_b_velocity": 0.05,
                "min_cross_flip": 0.15,
            }),
            tax_cfg           = {},
        )
        assert pairs == []

    def test_cross_flip_fails_blocks(self):
        """A and B both moved the same direction — gap unchanged."""
        from kernel.rotation import find_thesis_symmetric_pairs

        # A: 0.40 → 0.30 (vel -0.10, just passes)
        # B: 0.50 → 0.60 (vel +0.10, passes)
        # gap_entry = 0.50 - 0.40 = +0.10
        # gap_today = 0.60 - 0.30 = +0.30
        # cross_flip = 0.20 — passes 0.15
        # But let's construct failing case: both move same way
        # A: 0.40 → 0.20 (-0.20) ok
        # B: 0.25 → 0.30 (+0.05) ok
        # gap_entry = 0.25 - 0.40 = -0.15
        # gap_today = 0.30 - 0.20 = +0.10
        # cross_flip = 0.10 - (-0.15) = +0.25 → still passes
        # Really hard to construct. Instead: small changes on both, flip=0.
        # A: 0.40 → 0.28 (vel -0.12) ok
        # B: 0.38 → 0.44 (vel +0.06) ok
        # gap_entry = 0.38 - 0.40 = -0.02
        # gap_today = 0.44 - 0.28 = +0.16
        # cross_flip = 0.18 (passes 0.15)
        # To fail: A's today fall + B's entry rise matter less
        # A: 0.40 → 0.25 (vel -0.15) ok
        # B: 0.30 → 0.40 (vel +0.10) ok
        # gap_entry = 0.30 - 0.40 = -0.10
        # gap_today = 0.40 - 0.25 = +0.15
        # cross_flip = 0.15 - (-0.10) = 0.25 → still passes
        # Hmm. Let me try smaller b improvement:
        # A: 0.40 → 0.25 (vel -0.15) ok
        # B: 0.20 → 0.27 (vel +0.07) ok
        # gap_entry = 0.20 - 0.40 = -0.20
        # gap_today = 0.27 - 0.25 = +0.02
        # cross_flip = +0.02 - (-0.20) = 0.22 → passes
        # Let me try tight failure:
        # A: 0.40 → 0.30 (vel -0.10) ok (barely)
        # B: 0.30 → 0.36 (vel +0.06) ok
        # gap_entry = -0.10
        # gap_today = +0.06
        # cross_flip = 0.06 - (-0.10) = 0.16 → passes 0.15
        # Let's just raise threshold high:
        pairs = find_thesis_symmetric_pairs(
            held_entry_scores = {"A": 0.40},
            held_today_scores = {"A": 0.30},
            held_meta         = {"A": _meta()},
            candidates        = [_Cand("B", rank_score=0.36)],
            entry_day_lookup  = {("B", datetime.date(2025, 1, 1)): 0.30},
            today             = datetime.date(2025, 6, 1),
            rotation_cfg      = _cfg(thesis_symmetric={
                "max_a_velocity": 0.05, "min_b_velocity": 0.05,
                "min_cross_flip": 0.25,   # HIGH cross_flip req
            }),
            tax_cfg           = {},
        )
        assert pairs == []


class TestMissingEntryLookup:
    def test_no_entry_score_skips_pair(self):
        """If B was NOT in watchlist on A's entry date (no DB row),
        skip the pair rather than fire or error."""
        from kernel.rotation import find_thesis_symmetric_pairs

        pairs = find_thesis_symmetric_pairs(
            held_entry_scores = {"A": 0.40},
            held_today_scores = {"A": 0.20},
            held_meta         = {"A": _meta()},
            candidates        = [_Cand("B", rank_score=0.50)],
            entry_day_lookup  = {},   # EMPTY — no entry data for B
            today             = datetime.date(2025, 6, 1),
            rotation_cfg      = _cfg(thesis_symmetric={
                "max_a_velocity": 0.10, "min_b_velocity": 0.05,
                "min_cross_flip": 0.15,
            }),
            tax_cfg           = {},
        )
        assert pairs == []


class TestGreedyPairing:
    def test_picks_biggest_cross_flip(self):
        """When multiple holds are swappable for one cand, pick the
        one with biggest cross_flip (strongest overtaking signal)."""
        from kernel.rotation import find_thesis_symmetric_pairs

        # Two holds, same entry_date
        # Pair 1 (A1→B): cross_flip modest
        # Pair 2 (A2→B): cross_flip bigger
        # Cand B: entry 0.20, today 0.55 → vel +0.35
        # Held A1: entry 0.40, today 0.20 → vel -0.20, gap_entry=-0.20, gap_today=0.35, flip=0.55
        # Held A2: entry 0.60, today 0.25 → vel -0.35, gap_entry=-0.40, gap_today=0.30, flip=0.70
        pairs = find_thesis_symmetric_pairs(
            held_entry_scores = {"A1": 0.40, "A2": 0.60},
            held_today_scores = {"A1": 0.20, "A2": 0.25},
            held_meta         = {"A1": _meta(), "A2": _meta()},
            candidates        = [_Cand("B", rank_score=0.55)],
            entry_day_lookup  = {("B", datetime.date(2025, 1, 1)): 0.20},
            today             = datetime.date(2025, 6, 1),
            rotation_cfg      = _cfg(thesis_symmetric={
                "max_a_velocity": 0.10, "min_b_velocity": 0.05,
                "min_cross_flip": 0.15,
            }),
            tax_cfg           = {},
        )
        assert len(pairs) == 1
        assert pairs[0].sell_ticker == "A2"   # bigger flip
