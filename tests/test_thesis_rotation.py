"""Approach A — thesis-degradation rotation gate.

User insight 2026-04-24: comparing today's cand.kelly vs today's
held.kelly is noise-on-noise. Compare instead against the held
position's FIXED entry baseline (rank_score at buy time). Swap
only when:
  (1) held's thesis has DEGRADED (entry_score - today_score >= 30%)
  (2) cand beats that original baseline (cand_today - held_entry >= 10%)

Flag-gated: `ranking.thesis_rotation.enabled` defaults False.
Graceful fallback when entry_rank_score is None (legacy positions).
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

from kernel.pipeline.task_rotation import BuildPairsTask  # noqa: E402


def _hs(entry_date, *, today_score=0.30, entry_score=None, mu=0.02, er=0.02,
        entry_price=100.0):
    """HoldingState factory with flexible scoring fields."""
    from kernel.exits import HoldingState
    h = HoldingState(
        entry_price=entry_price, entry_date=entry_date,
        shares=100, high_watermark=entry_price,
    )
    h.rank_score        = today_score
    h.entry_rank_score  = entry_score   # None = no baseline stamped
    h.mu                = mu
    h.expected_return   = er
    h.panel_score       = today_score
    h.kelly_target_pct  = 0.15
    return h


def _cand(ticker, today_score, er=0.08):
    return SimpleNamespace(
        ticker=ticker, rank_score=today_score, expected_return=er,
        panel_score=today_score, kelly_target_pct=0.20,
        rs_score=0.0, raw_score=today_score * 10, detail="",
    )


def _ctx(holdings, ranked, *, thesis_enabled=True,
         degradation_pct=0.30, uplift_pct=0.10):
    today = datetime.date(2026, 4, 24)
    prices = {t: 100.0 for t in holdings}
    return SimpleNamespace(
        today       = today,
        holdings    = holdings,
        ranked      = ranked,
        exits       = [],
        rotations   = [],
        bear_only   = False,
        prices      = prices,
        counters    = {},
        config      = {
            "rotation": {
                "enabled": True,
                "min_expected_advantage_pct": 0.01,
                "target_horizon_days":        20,
                "transaction_cost_pct":       0.0,
                "min_rotation_hold_days":     0,
                "lt_protection_days":         0,
                "max_rotations_per_bar":      5,
            },
            "ranking": {
                "panel_scoring":   {"rotation_advantage": 0.0},
                "kelly_sizing":    {"rotation_advantage": 0.0},
                "thesis_rotation": {
                    "enabled":         thesis_enabled,
                    "degradation_pct": degradation_pct,
                    "uplift_pct":      uplift_pct,
                },
            },
            "tax": {"short_term_rate": 0.37, "long_term_rate": 0.20,
                    "long_term_threshold_days": 365},
        },
    )


# ── Flag gating ───────────────────────────────────────────────────────────────

class TestFlagGating:
    def test_disabled_is_pass_through(self):
        """thesis_enabled=False → ER-based pair survives untouched."""
        entry_d = datetime.date(2026, 1, 10)
        holdings = {"NVDA": _hs(entry_d, today_score=0.40, entry_score=0.40,
                                 er=0.02)}
        ranked = [_cand("TSLA", today_score=0.30, er=0.10)]
        ctx = _ctx(holdings, ranked, thesis_enabled=False)
        BuildPairsTask().run(ctx)
        assert len(ctx.rotations) == 1


# ── Core thesis logic ─────────────────────────────────────────────────────────

class TestThesisLogic:
    def test_swap_when_degradation_and_uplift(self):
        """Held bought at 0.50, now 0.30 → degradation 40%. Cand at 0.62 → uplift
        0.12. Both thresholds (30%, 0.10) cleared → swap."""
        entry_d = datetime.date(2026, 1, 10)
        holdings = {"NVDA": _hs(entry_d,
                                 today_score=0.30, entry_score=0.50, er=0.02)}
        ranked = [_cand("TSLA", today_score=0.62, er=0.10)]
        ctx = _ctx(holdings, ranked)
        BuildPairsTask().run(ctx)
        assert len(ctx.rotations) == 1

    def test_skip_when_no_degradation(self):
        """Held today still strong (0.48 vs entry 0.50) — thesis intact, skip."""
        entry_d = datetime.date(2026, 1, 10)
        holdings = {"NVDA": _hs(entry_d,
                                 today_score=0.48, entry_score=0.50, er=0.02)}
        ranked = [_cand("TSLA", today_score=0.70, er=0.10)]
        ctx = _ctx(holdings, ranked)
        BuildPairsTask().run(ctx)
        assert len(ctx.rotations) == 0
        assert ctx.counters.get("thesis_rotation_rejects", 0) == 1

    def test_skip_when_no_uplift(self):
        """Held degraded (entry 0.50 → today 0.25) but cand only 0.55 — just
        barely above baseline (0.05 < 0.10 uplift threshold) → skip."""
        entry_d = datetime.date(2026, 1, 10)
        holdings = {"NVDA": _hs(entry_d,
                                 today_score=0.25, entry_score=0.50, er=0.02)}
        ranked = [_cand("TSLA", today_score=0.55, er=0.10)]
        ctx = _ctx(holdings, ranked)
        BuildPairsTask().run(ctx)
        assert len(ctx.rotations) == 0

    def test_fallback_when_entry_score_missing(self):
        """Legacy position with no baseline → keep pair (no thesis filter)."""
        entry_d = datetime.date(2026, 1, 10)
        holdings = {"NVDA": _hs(entry_d,
                                 today_score=0.30, entry_score=None, er=0.02)}
        ranked = [_cand("TSLA", today_score=0.40, er=0.10)]
        ctx = _ctx(holdings, ranked)
        BuildPairsTask().run(ctx)
        # With entry_score=None, the thesis gate keeps the pair (fallback).
        assert len(ctx.rotations) == 1

    def test_threshold_tunability(self):
        """Lower degradation + uplift thresholds accept more swaps."""
        entry_d = datetime.date(2026, 1, 10)
        holdings = {"NVDA": _hs(entry_d,
                                 today_score=0.45, entry_score=0.50, er=0.02)}
        ranked = [_cand("TSLA", today_score=0.52, er=0.10)]
        # Tight: 30% degradation and 0.10 uplift — this case fails both
        ctx_tight = _ctx(holdings, ranked,
                          degradation_pct=0.30, uplift_pct=0.10)
        BuildPairsTask().run(ctx_tight)
        assert len(ctx_tight.rotations) == 0

        # Loose: 5% degradation + 0.01 uplift — this case passes both
        holdings_fresh = {"NVDA": _hs(entry_d,
                                       today_score=0.45, entry_score=0.50,
                                       er=0.02)}
        ranked_fresh = [_cand("TSLA", today_score=0.52, er=0.10)]
        ctx_loose = _ctx(holdings_fresh, ranked_fresh,
                          degradation_pct=0.05, uplift_pct=0.01)
        BuildPairsTask().run(ctx_loose)
        assert len(ctx_loose.rotations) == 1


# ── HoldingState carries the new fields ──────────────────────────────────────

class TestHoldingStateFields:
    def test_fields_default_none(self):
        from kernel.exits import HoldingState
        h = HoldingState(
            entry_price=100.0, entry_date=datetime.date(2026, 1, 15),
            shares=10, high_watermark=100.0,
        )
        assert h.entry_rank_score       is None
        assert h.entry_panel_score      is None
        assert h.entry_kelly_target_pct is None

    def test_fields_round_trip(self):
        from kernel.exits import HoldingState
        h = HoldingState(
            entry_price=100.0, entry_date=datetime.date(2026, 1, 15),
            shares=10, high_watermark=100.0,
            entry_rank_score=0.45,
            entry_panel_score=0.42,
            entry_kelly_target_pct=0.20,
        )
        assert h.entry_rank_score == 0.45
        assert h.entry_panel_score == 0.42
        assert h.entry_kelly_target_pct == 0.20
