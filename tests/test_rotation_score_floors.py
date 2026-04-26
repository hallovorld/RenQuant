"""Phase 1 — panel_buy_floor + panel_sell_floor double-gate.

User spec (2026-04-25):
  "被替换的 portfolio 里的 stock 的 score 要低于一个值 (panel_sell_floor),
   进到 portfolio 的 stock 的 score 要高于一个值 (panel_buy_floor),
   这个值可以就是 calibrate score"

Both floors apply to the calibrated rank_score (post ApplyGlobalCalibrationTask).
- panel_buy_floor:  candidate.rank_score >= floor    (None = disabled)
- panel_sell_floor: held(today).rank_score <= floor  (None = disabled)

When either is None, behaviour matches pre-Phase-1.
"""
from __future__ import annotations

import datetime
import json
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
    expected_return: float


def _meta(entry_date="2025-01-01", entry_price=100.0, current_price=100.0):
    return {
        "entry_date":    datetime.date.fromisoformat(entry_date),
        "entry_price":   entry_price,
        "current_price": current_price,
    }


def _base_cfg(**overrides):
    cfg = {
        "enabled":                      True,
        "min_expected_advantage_pct":   0.01,
        "target_horizon_days":          20,
        "transaction_cost_pct":         0.0,
        "min_rotation_hold_days":       0,
        "lt_protection_days":           0,
        "max_rotations_per_bar":        2,
    }
    cfg.update(overrides)
    return cfg


# ── find_rotation_pairs (ER-mode) ─────────────────────────────────────────────


class TestFindRotationPairsScoreFloors:
    def test_no_floors_preserves_existing_behaviour(self):
        """Both None → identical to pre-Phase-1 behaviour."""
        from kernel.rotation import find_rotation_pairs

        held_scores = {"NVDA": 0.30}
        held_er     = {"NVDA": -0.02}
        held_meta   = {"NVDA": _meta()}
        cand = _Cand("AMD", rank_score=0.40, expected_return=0.03)

        pairs = find_rotation_pairs(
            held_scores=held_scores, held_er=held_er, held_meta=held_meta,
            candidates=[cand], today=datetime.date(2025, 6, 1),
            rotation_cfg=_base_cfg(), tax_cfg={},
            # explicitly None — current behaviour
            panel_buy_floor=None, panel_sell_floor=None,
        )
        assert len(pairs) == 1
        assert pairs[0].sell_ticker == "NVDA"
        assert pairs[0].buy_ticker == "AMD"

    def test_buy_floor_blocks_weak_candidate(self):
        """cand.rank_score=0.30 vs buy_floor=0.45 → blocked."""
        from kernel.rotation import find_rotation_pairs

        held_scores = {"NVDA": 0.10}
        held_er     = {"NVDA": -0.05}
        held_meta   = {"NVDA": _meta()}
        cand = _Cand("AMD", rank_score=0.30, expected_return=0.05)  # weak

        pairs = find_rotation_pairs(
            held_scores=held_scores, held_er=held_er, held_meta=held_meta,
            candidates=[cand], today=datetime.date(2025, 6, 1),
            rotation_cfg=_base_cfg(), tax_cfg={},
            panel_buy_floor=0.45, panel_sell_floor=None,
        )
        assert pairs == []

    def test_buy_floor_admits_strong_candidate(self):
        """cand.rank_score=0.50 vs buy_floor=0.45 → admitted."""
        from kernel.rotation import find_rotation_pairs

        held_scores = {"NVDA": 0.10}
        held_er     = {"NVDA": -0.05}
        held_meta   = {"NVDA": _meta()}
        cand = _Cand("AMD", rank_score=0.50, expected_return=0.05)  # strong

        pairs = find_rotation_pairs(
            held_scores=held_scores, held_er=held_er, held_meta=held_meta,
            candidates=[cand], today=datetime.date(2025, 6, 1),
            rotation_cfg=_base_cfg(), tax_cfg={},
            panel_buy_floor=0.45, panel_sell_floor=None,
        )
        assert len(pairs) == 1
        assert pairs[0].buy_ticker == "AMD"

    def test_sell_floor_blocks_strong_held(self):
        """held(today).rank_score=0.50 vs sell_floor=0.20 → held protected."""
        from kernel.rotation import find_rotation_pairs

        held_scores = {"NVDA": 0.50}     # strong
        held_er     = {"NVDA": -0.05}
        held_meta   = {"NVDA": _meta()}
        cand = _Cand("AMD", rank_score=0.60, expected_return=0.05)

        pairs = find_rotation_pairs(
            held_scores=held_scores, held_er=held_er, held_meta=held_meta,
            candidates=[cand], today=datetime.date(2025, 6, 1),
            rotation_cfg=_base_cfg(), tax_cfg={},
            panel_buy_floor=None, panel_sell_floor=0.20,
        )
        assert pairs == []

    def test_sell_floor_admits_weak_held(self):
        """held.rank_score=0.10 vs sell_floor=0.20 → eligible to swap out."""
        from kernel.rotation import find_rotation_pairs

        held_scores = {"NVDA": 0.10}     # weak
        held_er     = {"NVDA": -0.05}
        held_meta   = {"NVDA": _meta()}
        cand = _Cand("AMD", rank_score=0.60, expected_return=0.05)

        pairs = find_rotation_pairs(
            held_scores=held_scores, held_er=held_er, held_meta=held_meta,
            candidates=[cand], today=datetime.date(2025, 6, 1),
            rotation_cfg=_base_cfg(), tax_cfg={},
            panel_buy_floor=None, panel_sell_floor=0.20,
        )
        assert len(pairs) == 1
        assert pairs[0].sell_ticker == "NVDA"

    def test_both_floors_pass(self):
        """cand=0.50 >= 0.45 AND held=0.10 <= 0.20 → rotation fires."""
        from kernel.rotation import find_rotation_pairs

        held_scores = {"NVDA": 0.10}
        held_er     = {"NVDA": -0.05}
        held_meta   = {"NVDA": _meta()}
        cand = _Cand("AMD", rank_score=0.50, expected_return=0.05)

        pairs = find_rotation_pairs(
            held_scores=held_scores, held_er=held_er, held_meta=held_meta,
            candidates=[cand], today=datetime.date(2025, 6, 1),
            rotation_cfg=_base_cfg(), tax_cfg={},
            panel_buy_floor=0.45, panel_sell_floor=0.20,
        )
        assert len(pairs) == 1

    def test_buy_floor_at_boundary_admits(self):
        """cand.rank_score == buy_floor → admitted (>= comparison)."""
        from kernel.rotation import find_rotation_pairs

        held_scores = {"NVDA": 0.10}
        held_er     = {"NVDA": -0.05}
        held_meta   = {"NVDA": _meta()}
        cand = _Cand("AMD", rank_score=0.45, expected_return=0.05)

        pairs = find_rotation_pairs(
            held_scores=held_scores, held_er=held_er, held_meta=held_meta,
            candidates=[cand], today=datetime.date(2025, 6, 1),
            rotation_cfg=_base_cfg(), tax_cfg={},
            panel_buy_floor=0.45, panel_sell_floor=None,
        )
        assert len(pairs) == 1

    def test_sell_floor_at_boundary_admits(self):
        """held.rank_score == sell_floor → still eligible (<= comparison)."""
        from kernel.rotation import find_rotation_pairs

        held_scores = {"NVDA": 0.20}
        held_er     = {"NVDA": -0.05}
        held_meta   = {"NVDA": _meta()}
        cand = _Cand("AMD", rank_score=0.50, expected_return=0.05)

        pairs = find_rotation_pairs(
            held_scores=held_scores, held_er=held_er, held_meta=held_meta,
            candidates=[cand], today=datetime.date(2025, 6, 1),
            rotation_cfg=_base_cfg(), tax_cfg={},
            panel_buy_floor=None, panel_sell_floor=0.20,
        )
        assert len(pairs) == 1

    def test_buy_floor_filters_one_of_two_candidates(self):
        """Mixed candidate strengths: only the strong one survives."""
        from kernel.rotation import find_rotation_pairs

        held_scores = {"NVDA": 0.10, "TSLA": 0.05}
        held_er     = {"NVDA": -0.05, "TSLA": -0.07}
        held_meta   = {"NVDA": _meta(), "TSLA": _meta()}
        cands = [
            _Cand("AMD",  rank_score=0.30, expected_return=0.05),  # weak
            _Cand("MSFT", rank_score=0.55, expected_return=0.04),  # strong
        ]

        pairs = find_rotation_pairs(
            held_scores=held_scores, held_er=held_er, held_meta=held_meta,
            candidates=cands, today=datetime.date(2025, 6, 1),
            rotation_cfg=_base_cfg(), tax_cfg={},
            panel_buy_floor=0.45, panel_sell_floor=None,
        )
        # AMD blocked by buy_floor; MSFT proceeds
        assert len(pairs) == 1
        assert pairs[0].buy_ticker == "MSFT"


# ── find_thesis_primary_pairs ─────────────────────────────────────────────────


class TestFindThesisPrimaryPairsScoreFloors:
    def _setup(self, held_today=0.10, cand_score=0.50):
        held_entry  = {"NVDA": 0.50}
        held_today_d = {"NVDA": held_today}
        held_meta   = {"NVDA": _meta()}
        cand = _Cand("AMD", rank_score=cand_score, expected_return=0.0)
        return held_entry, held_today_d, held_meta, cand

    def _cfg(self):
        # Loose thresholds — degradation/uplift won't block
        cfg = _base_cfg()
        cfg["thesis"] = {"degradation_pct": 0.10, "uplift_pct": 0.0}
        return cfg

    def test_no_floors_preserves_existing(self):
        from kernel.rotation import find_thesis_primary_pairs
        held_entry, held_today_d, held_meta, cand = self._setup()
        pairs = find_thesis_primary_pairs(
            held_entry_scores=held_entry, held_today_scores=held_today_d,
            held_meta=held_meta, candidates=[cand],
            today=datetime.date(2025, 6, 1),
            rotation_cfg=self._cfg(), tax_cfg={},
        )
        assert len(pairs) == 1

    def test_buy_floor_blocks_weak_candidate(self):
        from kernel.rotation import find_thesis_primary_pairs
        held_entry, held_today_d, held_meta, cand = self._setup(cand_score=0.30)
        pairs = find_thesis_primary_pairs(
            held_entry_scores=held_entry, held_today_scores=held_today_d,
            held_meta=held_meta, candidates=[cand],
            today=datetime.date(2025, 6, 1),
            rotation_cfg=self._cfg(), tax_cfg={},
            panel_buy_floor=0.45,
        )
        assert pairs == []

    def test_sell_floor_blocks_strong_held(self):
        from kernel.rotation import find_thesis_primary_pairs
        held_entry, held_today_d, held_meta, cand = self._setup(held_today=0.50)
        # held degraded (0.50 entry → 0.50 today is no degradation but use lower entry)
        held_entry["NVDA"] = 0.80  # degradation = (0.80-0.50)/0.80 = 0.375 >= 0.10
        pairs = find_thesis_primary_pairs(
            held_entry_scores=held_entry, held_today_scores=held_today_d,
            held_meta=held_meta, candidates=[cand],
            today=datetime.date(2025, 6, 1),
            rotation_cfg=self._cfg(), tax_cfg={},
            panel_sell_floor=0.20,
        )
        assert pairs == []  # held today=0.50 > floor=0.20

    def test_both_floors_pass(self):
        from kernel.rotation import find_thesis_primary_pairs
        held_entry, held_today_d, held_meta, cand = self._setup(
            held_today=0.10, cand_score=0.50,
        )
        pairs = find_thesis_primary_pairs(
            held_entry_scores=held_entry, held_today_scores=held_today_d,
            held_meta=held_meta, candidates=[cand],
            today=datetime.date(2025, 6, 1),
            rotation_cfg=self._cfg(), tax_cfg={},
            panel_buy_floor=0.45, panel_sell_floor=0.20,
        )
        assert len(pairs) == 1


# ── find_thesis_symmetric_pairs ───────────────────────────────────────────────


class TestFindThesisSymmetricPairsScoreFloors:
    def _setup(self, held_today=0.10, cand_score=0.50):
        held_entry  = {"NVDA": 0.50}
        held_today_d = {"NVDA": held_today}
        held_meta   = {"NVDA": _meta()}
        cand = _Cand("AMD", rank_score=cand_score, expected_return=0.0)
        # B's score on A's entry date — set low so b_velocity is large
        entry_lookup = {("AMD", datetime.date.fromisoformat("2025-01-01")): 0.10}
        return held_entry, held_today_d, held_meta, cand, entry_lookup

    def _cfg(self):
        cfg = _base_cfg()
        cfg["thesis_symmetric"] = {
            "max_a_velocity":  0.10,   # A must drop >= 0.10
            "min_b_velocity":  0.05,   # B must rise >= 0.05
            "min_cross_flip":  0.05,   # gap must widen >= 0.05
        }
        return cfg

    def test_no_floors_preserves_existing(self):
        from kernel.rotation import find_thesis_symmetric_pairs
        he, hd, hm, cand, lookup = self._setup()
        pairs = find_thesis_symmetric_pairs(
            held_entry_scores=he, held_today_scores=hd,
            held_meta=hm, candidates=[cand],
            entry_day_lookup=lookup,
            today=datetime.date(2025, 6, 1),
            rotation_cfg=self._cfg(), tax_cfg={},
        )
        assert len(pairs) == 1

    def test_buy_floor_blocks(self):
        from kernel.rotation import find_thesis_symmetric_pairs
        he, hd, hm, cand, lookup = self._setup(cand_score=0.30)
        pairs = find_thesis_symmetric_pairs(
            held_entry_scores=he, held_today_scores=hd,
            held_meta=hm, candidates=[cand],
            entry_day_lookup=lookup,
            today=datetime.date(2025, 6, 1),
            rotation_cfg=self._cfg(), tax_cfg={},
            panel_buy_floor=0.45,
        )
        assert pairs == []

    def test_sell_floor_blocks(self):
        from kernel.rotation import find_thesis_symmetric_pairs
        # held today=0.50 (still strong) — but a_velocity = 0.50-0.50 = 0 fails
        # max_a_velocity=0.10. Make a_velocity meet the gate first by lowering today.
        he = {"NVDA": 0.50}
        hd = {"NVDA": 0.30}             # a_velocity = -0.20, passes
        hm = {"NVDA": _meta()}
        cand = _Cand("AMD", rank_score=0.50, expected_return=0.0)
        lookup = {("AMD", datetime.date.fromisoformat("2025-01-01")): 0.10}
        pairs = find_thesis_symmetric_pairs(
            held_entry_scores=he, held_today_scores=hd,
            held_meta=hm, candidates=[cand],
            entry_day_lookup=lookup,
            today=datetime.date(2025, 6, 1),
            rotation_cfg=self._cfg(), tax_cfg={},
            panel_sell_floor=0.20,    # held today=0.30 > 0.20 → blocked
        )
        assert pairs == []

    def test_both_floors_pass(self):
        from kernel.rotation import find_thesis_symmetric_pairs
        he, hd, hm, cand, lookup = self._setup(
            held_today=0.10, cand_score=0.50,
        )
        pairs = find_thesis_symmetric_pairs(
            held_entry_scores=he, held_today_scores=hd,
            held_meta=hm, candidates=[cand],
            entry_day_lookup=lookup,
            today=datetime.date(2025, 6, 1),
            rotation_cfg=self._cfg(), tax_cfg={},
            panel_buy_floor=0.45, panel_sell_floor=0.20,
        )
        assert len(pairs) == 1


# ── BuildPairsTask wiring ─────────────────────────────────────────────────────


class TestBuildPairsTaskFloorsWired:
    """Integration-ish: BuildPairsTask reads rotation.panel_buy_floor /
    rotation.panel_sell_floor and forwards them to find_rotation_pairs."""

    def test_floors_forwarded_to_kernel(self):
        from unittest.mock import patch
        from kernel.pipeline.task_rotation import BuildPairsTask

        # Build a minimal InferenceContext-like object via SimpleNamespace
        from types import SimpleNamespace as NS

        cfg = {
            "rotation": {
                "enabled": True,
                "min_expected_advantage_pct": 0.01,
                "min_rotation_hold_days": 0,
                "lt_protection_days": 0,
                "max_rotations_per_bar": 2,
                "panel_buy_floor": 0.55,
                "panel_sell_floor": 0.15,
            },
            "tax": {},
            "ranking": {"panel_scoring": {}, "kelly_sizing": {}, "thesis_rotation": {}},
        }
        held = NS(
            entry_date=datetime.date.fromisoformat("2024-01-01"),
            entry_price=100.0,
            entry_rank_score=0.10,
            rank_score=0.10,
            expected_return=-0.05,
            mu=None, sigma=None,
        )
        ranked = [NS(
            ticker="AMD", rank_score=0.60, expected_return=0.05,
            mu=None, sigma=None, panel_score=None,
        )]
        ctx = NS(
            today=datetime.date(2025, 6, 1),
            holdings={"NVDA": held},
            prices={"NVDA": 100.0, "AMD": 50.0},
            ranked=ranked,
            exits=[],
            bear_only=False,
            regime="BULL_CALM",
            config=cfg,
            counters={},
            rotations=[],
            rotations_blocked=[],
        )

        with patch("kernel.rotation.find_rotation_pairs") as mock_find:
            mock_find.return_value = []
            BuildPairsTask().run(ctx)
            assert mock_find.called
            kwargs = mock_find.call_args.kwargs
            assert kwargs["panel_buy_floor"] == 0.55
            assert kwargs["panel_sell_floor"] == 0.15

    def test_floors_default_none_when_omitted(self):
        from unittest.mock import patch
        from kernel.pipeline.task_rotation import BuildPairsTask
        from types import SimpleNamespace as NS

        cfg = {
            "rotation": {
                "enabled": True,
                "min_expected_advantage_pct": 0.01,
                "min_rotation_hold_days": 0,
                "lt_protection_days": 0,
                "max_rotations_per_bar": 2,
                # no panel_buy_floor / panel_sell_floor
            },
            "tax": {},
            "ranking": {"panel_scoring": {}, "kelly_sizing": {}, "thesis_rotation": {}},
        }
        held = NS(
            entry_date=datetime.date.fromisoformat("2024-01-01"),
            entry_price=100.0,
            entry_rank_score=0.10,
            rank_score=0.10,
            expected_return=-0.05,
            mu=None, sigma=None,
        )
        ranked = [NS(
            ticker="AMD", rank_score=0.60, expected_return=0.05,
            mu=None, sigma=None, panel_score=None,
        )]
        ctx = NS(
            today=datetime.date(2025, 6, 1),
            holdings={"NVDA": held},
            prices={"NVDA": 100.0, "AMD": 50.0},
            ranked=ranked,
            exits=[],
            bear_only=False,
            regime="BULL_CALM",
            config=cfg,
            counters={},
            rotations=[],
            rotations_blocked=[],
        )

        with patch("kernel.rotation.find_rotation_pairs") as mock_find:
            mock_find.return_value = []
            BuildPairsTask().run(ctx)
            assert mock_find.called
            kwargs = mock_find.call_args.kwargs
            assert kwargs["panel_buy_floor"] is None
            assert kwargs["panel_sell_floor"] is None


# ── Default-config integration ───────────────────────────────────────────────


class TestDefaultConfigCarriesFloors:
    """Both strategy_config.json and strategy_config.golden.json must carry
    the new floor keys so the drift check stays clean."""

    def _load(self, name):
        path = _STRATEGY_DIR / name
        with open(path, "r") as f:
            return json.load(f)

    def test_strategy_config_has_panel_buy_floor(self):
        cfg = self._load("strategy_config.json")
        assert "panel_buy_floor" in cfg["rotation"]
        assert isinstance(cfg["rotation"]["panel_buy_floor"], (int, float))

    def test_strategy_config_has_panel_sell_floor(self):
        cfg = self._load("strategy_config.json")
        assert "panel_sell_floor" in cfg["rotation"]
        assert isinstance(cfg["rotation"]["panel_sell_floor"], (int, float))

    def test_golden_config_has_panel_buy_floor(self):
        cfg = self._load("strategy_config.golden.json")
        assert "panel_buy_floor" in cfg["rotation"]

    def test_golden_config_has_panel_sell_floor(self):
        cfg = self._load("strategy_config.golden.json")
        assert "panel_sell_floor" in cfg["rotation"]

    def test_floors_match_between_configs(self):
        """Drift-check: both configs must agree on the floor values."""
        live   = self._load("strategy_config.json")
        golden = self._load("strategy_config.golden.json")
        assert live["rotation"]["panel_buy_floor"]  == golden["rotation"]["panel_buy_floor"]
        assert live["rotation"]["panel_sell_floor"] == golden["rotation"]["panel_sell_floor"]
