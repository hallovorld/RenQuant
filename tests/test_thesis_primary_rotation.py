"""Route B — rotation_mode=thesis_primary.

Direct thesis-degradation as primary rotation gate (bypasses
ER-based find_rotation_pairs). Useful when ER magnitudes are
smaller than min_expected_advantage_pct, making ER-mode emit 0 pairs.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.rotation import find_thesis_primary_pairs  # noqa: E402
from kernel.pipeline.task_rotation import BuildPairsTask  # noqa: E402


def _mk_cand(ticker, rank, er=0.0):
    return SimpleNamespace(
        ticker=ticker, rank_score=rank, expected_return=er,
        panel_score=rank, kelly_target_pct=0.15,
        rs_score=0.0, raw_score=rank * 10, detail="",
    )


# ── find_thesis_primary_pairs unit ───────────────────────────────────────────

class TestThesisPrimaryUnit:
    def _common_cfg(self):
        return {
            "rotation": {
                "enabled": True,
                "mode": "thesis_primary",
                "target_horizon_days": 20,
                "transaction_cost_pct": 0.0,
                "min_rotation_hold_days": 30,
                "lt_protection_days": 0,
                "max_rotations_per_bar": 5,
                "thesis": {"degradation_pct": 0.30, "uplift_pct": 0.10},
            },
            "tax": {"short_term_rate": 0.37, "long_term_rate": 0.20,
                    "long_term_threshold_days": 365},
        }

    def test_emit_when_degradation_and_uplift_met(self):
        """Entry 0.50, today 0.30 → degradation 40% ≥ 30%.
        Cand today 0.62 → uplift = 0.62 - 0.50 = 0.12 ≥ 0.10."""
        cfg = self._common_cfg()
        entry_d = datetime.date(2026, 1, 1)
        held_entry = {"NVDA": 0.50}
        held_today = {"NVDA": 0.30}
        held_meta  = {"NVDA": {"entry_date": entry_d,
                                "entry_price": 100.0,
                                "current_price": 110.0}}
        cand = [_mk_cand("TSLA", 0.62)]
        pairs = find_thesis_primary_pairs(
            held_entry, held_today, held_meta, cand,
            today=datetime.date(2026, 4, 24),
            rotation_cfg=cfg["rotation"], tax_cfg=cfg["tax"],
        )
        assert len(pairs) == 1
        assert pairs[0].sell_ticker == "NVDA"
        assert pairs[0].buy_ticker  == "TSLA"

    def test_skip_when_thesis_intact(self):
        """Entry 0.50, today 0.45 → degradation 10% < 30% → no emit."""
        cfg = self._common_cfg()
        held_entry = {"NVDA": 0.50}
        held_today = {"NVDA": 0.45}
        held_meta  = {"NVDA": {"entry_date": datetime.date(2026, 1, 1),
                                "entry_price": 100.0, "current_price": 110.0}}
        cand = [_mk_cand("TSLA", 0.70)]
        pairs = find_thesis_primary_pairs(
            held_entry, held_today, held_meta, cand,
            today=datetime.date(2026, 4, 24),
            rotation_cfg=cfg["rotation"], tax_cfg=cfg["tax"],
        )
        assert pairs == []

    def test_skip_when_cand_doesnt_beat_baseline(self):
        """Entry 0.50, today 0.30 → degraded. Cand 0.55 → uplift 0.05 < 0.10."""
        cfg = self._common_cfg()
        held_entry = {"NVDA": 0.50}
        held_today = {"NVDA": 0.30}
        held_meta  = {"NVDA": {"entry_date": datetime.date(2026, 1, 1),
                                "entry_price": 100.0, "current_price": 110.0}}
        cand = [_mk_cand("TSLA", 0.55)]
        pairs = find_thesis_primary_pairs(
            held_entry, held_today, held_meta, cand,
            today=datetime.date(2026, 4, 24),
            rotation_cfg=cfg["rotation"], tax_cfg=cfg["tax"],
        )
        assert pairs == []

    def test_respects_min_hold_days(self):
        """Held only 10 days < 30 → not eligible for rotation."""
        cfg = self._common_cfg()
        held_entry = {"NVDA": 0.50}
        held_today = {"NVDA": 0.30}
        held_meta  = {"NVDA": {"entry_date": datetime.date(2026, 4, 14),  # 10 days
                                "entry_price": 100.0, "current_price": 110.0}}
        cand = [_mk_cand("TSLA", 0.65)]
        pairs = find_thesis_primary_pairs(
            held_entry, held_today, held_meta, cand,
            today=datetime.date(2026, 4, 24),
            rotation_cfg=cfg["rotation"], tax_cfg=cfg["tax"],
        )
        assert pairs == []

    def test_disabled_rotation_returns_empty(self):
        cfg = self._common_cfg()
        cfg["rotation"]["enabled"] = False
        pairs = find_thesis_primary_pairs(
            {"NVDA": 0.50}, {"NVDA": 0.30},
            {"NVDA": {"entry_date": datetime.date(2026, 1, 1),
                       "entry_price": 100.0, "current_price": 110.0}},
            [_mk_cand("TSLA", 0.65)],
            today=datetime.date(2026, 4, 24),
            rotation_cfg=cfg["rotation"], tax_cfg=cfg["tax"],
        )
        assert pairs == []

    def test_missing_entry_score_skips_ticker(self):
        """Legacy holding without stamped baseline — skip."""
        cfg = self._common_cfg()
        pairs = find_thesis_primary_pairs(
            {"NVDA": None}, {"NVDA": 0.30},
            {"NVDA": {"entry_date": datetime.date(2026, 1, 1),
                       "entry_price": 100.0, "current_price": 110.0}},
            [_mk_cand("TSLA", 0.70)],
            today=datetime.date(2026, 4, 24),
            rotation_cfg=cfg["rotation"], tax_cfg=cfg["tax"],
        )
        assert pairs == []

    def test_multi_holdings_picks_most_degraded(self):
        """Two degraded holdings, cand beats both baselines — pick the
        MOST degraded one (highest degradation %)."""
        cfg = self._common_cfg()
        held_entry = {"NVDA": 0.50, "AAPL": 0.50}
        held_today = {"NVDA": 0.40, "AAPL": 0.20}   # AAPL more degraded
        held_meta  = {
            "NVDA": {"entry_date": datetime.date(2026, 1, 1),
                     "entry_price": 100.0, "current_price": 110.0},
            "AAPL": {"entry_date": datetime.date(2026, 1, 1),
                     "entry_price": 100.0, "current_price": 110.0},
        }
        cand = [_mk_cand("TSLA", 0.65)]
        pairs = find_thesis_primary_pairs(
            held_entry, held_today, held_meta, cand,
            today=datetime.date(2026, 4, 24),
            rotation_cfg=cfg["rotation"], tax_cfg=cfg["tax"],
        )
        # NVDA degradation = (0.50-0.40)/0.50 = 20% < 30% → not eligible
        # AAPL degradation = (0.50-0.20)/0.50 = 60% > 30% → eligible
        assert len(pairs) == 1
        assert pairs[0].sell_ticker == "AAPL"


# ── BuildPairsTask mode dispatch ─────────────────────────────────────────────

class TestBuildPairsTaskDispatch:
    def _setup_ctx(self, mode: str):
        from kernel.exits import HoldingState

        entry_d = datetime.date(2026, 1, 1)
        hs = HoldingState(
            entry_price=100.0, entry_date=entry_d,
            shares=100, high_watermark=100.0,
        )
        hs.rank_score        = 0.30
        hs.entry_rank_score  = 0.50
        hs.expected_return   = 0.004   # small ER — can't clear 0.03 ER threshold
        hs.panel_score       = 0.30
        hs.kelly_target_pct  = 0.15

        ctx = SimpleNamespace(
            today      = datetime.date(2026, 4, 24),
            holdings   = {"NVDA": hs},
            ranked     = [_mk_cand("TSLA", 0.65, er=0.005)],
            exits      = [],
            rotations  = [],
            bear_only  = False,
            prices     = {"NVDA": 110.0, "TSLA": 100.0},
            counters   = {},
            config     = {
                "rotation": {
                    "enabled": True,
                    "mode": mode,
                    "min_expected_advantage_pct": 0.03,  # tight — blocks ER mode
                    "target_horizon_days": 20,
                    "transaction_cost_pct": 0.0,
                    "min_rotation_hold_days": 0,   # ignore hold guard for test
                    "lt_protection_days": 0,
                    "max_rotations_per_bar": 5,
                    "thesis": {"degradation_pct": 0.30, "uplift_pct": 0.10},
                },
                "ranking": {
                    "panel_scoring": {"rotation_advantage": 0.0},
                    "kelly_sizing": {"rotation_advantage": 0.0},
                    "thesis_rotation": {
                        "enabled": False,   # off so we test Route B directly
                        "degradation_pct": 0.30, "uplift_pct": 0.10,
                    },
                },
                "tax": {"short_term_rate": 0.37, "long_term_rate": 0.20,
                        "long_term_threshold_days": 365},
            },
        )
        return ctx

    def test_er_mode_returns_zero_with_tight_threshold(self):
        """Legacy ER mode can't clear 0.03 threshold with ER ~0.005."""
        ctx = self._setup_ctx(mode="er")
        BuildPairsTask().run(ctx)
        assert len(ctx.rotations) == 0

    def test_thesis_primary_mode_emits_pair_despite_small_er(self):
        """Thesis-primary uses entry baseline — ER doesn't matter."""
        ctx = self._setup_ctx(mode="thesis_primary")
        BuildPairsTask().run(ctx)
        assert len(ctx.rotations) == 1
        assert ctx.rotations[0].sell_ticker == "NVDA"
        assert ctx.rotations[0].buy_ticker  == "TSLA"

    def test_thesis_primary_excludes_holdings_with_existing_exit(self):
        """A same-bar hard/soft exit owns the sell leg; rotation must not add one."""
        ctx = self._setup_ctx(mode="thesis_primary")
        ctx.exits = [("NVDA", SimpleNamespace(reason="panel_conviction"))]

        BuildPairsTask().run(ctx)

        assert ctx.rotations == []

    def test_default_mode_is_er(self):
        """No mode specified → ER mode (backward compat)."""
        ctx = self._setup_ctx(mode="er")
        # Remove mode key
        ctx.config["rotation"].pop("mode", None)
        BuildPairsTask().run(ctx)
        # Should still run ER path (0 pairs due to tight threshold)
        assert len(ctx.rotations) == 0
