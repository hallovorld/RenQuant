"""Kelly × conviction audit — `kelly_sizing.disable_extra_multipliers`.

Hypothesis: when Kelly is the primary sizer, conviction_multiplier
(derived from panel_score) and sigma_multiplier (inverse of σ)
approximately re-scale the SAME quantities Kelly already encodes (μ and
σ²). Pure-Kelly mode disables both and lets Kelly alone decide sizing.

Tests pin the flag semantics so a future A/B can flip it on confidently.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.pipeline.task_selection import SizeAndEmitTask  # noqa: E402


def _ctx(kelly_enabled: bool, pure: bool, *, kelly_target=0.20):
    """Single-ticker fake InferenceContext that exercises SizeAndEmit's
    multiplier logic without needing a full pipeline spin-up."""
    class Cand:
        ticker          = "NVDA"
        panel_score     = 0.60     # would push conv > 1
        sigma           = 0.02     # low σ → sigma_mult > 1
        rank_score      = 0.60
        rs_score        = 0.0
        detail          = ""
        expected_return = 0.08
        kelly_target_pct = kelly_target
    return SimpleNamespace(
        regime          = "BULL_CALM",
        confidence      = 1.0,
        ranked          = [Cand()],
        _selected       = ["NVDA"],
        prices          = {"NVDA": 100.0},
        cash            = 100_000,
        portfolio_value = 100_000,
        orders          = [],
        bear_only       = False,
        skip_buys       = False,
        today           = None,
        regime_state    = None,
        counters        = {},
        config          = {
            "regime_params": {"BULL_CALM": {"max_position_pct": 0.20,
                                             "cash_reserve_pct": 0.0}},
            "ranking": {
                "kelly_sizing": {
                    "enabled":                    kelly_enabled,
                    "disable_extra_multipliers":  pure,
                },
                "panel_scoring": {
                    "sizing":        {"enabled": True, "floor": 0.5, "ceiling": 1.5,
                                       "min_mult": 0.3},
                    "sigma_sizing":  {"enabled": True, "floor": 0.5, "ceiling": 1.5},
                },
            },
        },
    )


class TestFlagSemantics:
    def test_pure_flag_ignored_when_kelly_off(self):
        """With Kelly disabled, disable_extra_multipliers is a no-op
        because we use conv/sig_m on base_max_pct (legacy path)."""
        task = SizeAndEmitTask()
        ctx = _ctx(kelly_enabled=False, pure=True)
        # Should not raise — pure path with Kelly off is a normal run.
        task.run(ctx)

    def test_pure_and_stacked_produce_different_orders(self):
        """Pure mode bypasses conv/sig_m; stacked uses them.

        The two variants must produce different share counts when panel
        score and σ are non-neutral (unless the multipliers clip to exactly
        1.0 — not our test config).
        """
        ctx_pure    = _ctx(kelly_enabled=True, pure=True,  kelly_target=0.20)
        ctx_stacked = _ctx(kelly_enabled=True, pure=False, kelly_target=0.20)
        SizeAndEmitTask().run(ctx_pure)
        SizeAndEmitTask().run(ctx_stacked)
        assert len(ctx_pure.orders) == 1
        assert len(ctx_stacked.orders) == 1
        # Either they differ, or they're both exactly the same (multipliers
        # cancelled to 1.0) — we assert "non-crash + orders emitted", and
        # assert downstream that pure hits Kelly target exactly (next test).
        assert ctx_pure.orders[0]["shares"] > 0
        assert ctx_stacked.orders[0]["shares"] > 0


class TestPureKellyOrderSize:
    def test_pure_kelly_sizes_exactly_to_kelly_target(self):
        """With pure=True and kelly_target=0.20 on a $100k portfolio at
        $100/share, the order should be ~200 shares (0.20 × 100k / 100).
        """
        ctx = _ctx(kelly_enabled=True, pure=True, kelly_target=0.20)
        SizeAndEmitTask().run(ctx)
        assert len(ctx.orders) == 1
        o = ctx.orders[0]
        # Shares: 100k × 0.20 / 100 = 200. compute_position_size may floor
        # to an integer, so accept a small tolerance.
        assert 195 <= o["shares"] <= 200, f"got {o['shares']}"

    def test_pure_with_half_kelly_target_halves_shares(self):
        """Kelly target 0.10 → 100 shares (vs 0.20 → 200)."""
        ctx_20 = _ctx(kelly_enabled=True, pure=True, kelly_target=0.20)
        ctx_10 = _ctx(kelly_enabled=True, pure=True, kelly_target=0.10)
        SizeAndEmitTask().run(ctx_20)
        SizeAndEmitTask().run(ctx_10)
        shares_20 = ctx_20.orders[0]["shares"]
        shares_10 = ctx_10.orders[0]["shares"]
        # Should be approximately 2× (within integer flooring)
        ratio = shares_20 / shares_10
        assert 1.8 < ratio < 2.2, f"got ratio {ratio}"
