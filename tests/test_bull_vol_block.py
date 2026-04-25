"""BULL_VOL-reversal regression tests (AA-surfaced IC = -0.172 in BULL_VOL).

Guards the new BullVolOffensiveBlockTask:
  * Default-off preserves current behaviour.
  * `bull_vol_block_offensive=true` + BULL_VOLATILE regime →
    ctx.bear_only flips, counter fires, downstream tasks short-circuit.
  * Other regimes (BULL_CALM / CHOPPY / BEAR) are untouched.
  * `bull_vol_defensives_too=true` blocks defensives too (cash).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.config import BEAR, BULL_CALM, BULL_VOLATILE, CHOPPY  # noqa: E402
from kernel.pipeline.task_gates import BullVolOffensiveBlockTask  # noqa: E402
from kernel.pipeline.job_gates import BuyGatesJob  # noqa: E402


def _ctx(regime: str, config: dict | None = None) -> SimpleNamespace:
    """Minimal fake InferenceContext for gate tasks."""
    return SimpleNamespace(
        regime      = regime,
        config      = config or {},
        counters    = {},
        bear_only   = False,
        buy_blocked = False,
    )


class TestFlagOff:
    def test_default_no_op_in_bull_vol(self):
        """Flag absent → task is a no-op, preserves v4 behaviour."""
        task = BullVolOffensiveBlockTask()
        ctx = _ctx(BULL_VOLATILE, config={})
        assert task.run(ctx) is None
        assert ctx.bear_only   is False
        assert ctx.buy_blocked is False
        assert ctx.counters.get("bull_vol_blocks", 0) == 0

    def test_flag_off_noop(self):
        task = BullVolOffensiveBlockTask()
        ctx = _ctx(BULL_VOLATILE,
                   config={"regime": {"bull_vol_block_offensive": False}})
        assert task.run(ctx) is None
        assert ctx.bear_only is False


class TestDefensivesOnly:
    """bull_vol_block_offensive=true → bear_only=True (defensives still allowed)."""

    def test_flips_bear_only_in_bull_vol(self):
        task = BullVolOffensiveBlockTask()
        ctx = _ctx(BULL_VOLATILE,
                   config={"regime": {"bull_vol_block_offensive": True}})
        # Defensives-only branch returns None (continue chain) so the
        # downstream VelocityCrash + EMA50 macros can still set buy_blocked.
        # Pre-2026-04-24, this returned False which short-circuited those
        # gates — see audit #15 / fix in task_gates.py.
        assert task.run(ctx) is None
        assert ctx.bear_only   is True
        assert ctx.buy_blocked is False
        assert ctx.counters["bull_vol_blocks"] == 1

    def test_bull_calm_untouched(self):
        task = BullVolOffensiveBlockTask()
        ctx = _ctx(BULL_CALM,
                   config={"regime": {"bull_vol_block_offensive": True}})
        assert task.run(ctx) is None
        assert ctx.bear_only is False

    def test_choppy_untouched(self):
        task = BullVolOffensiveBlockTask()
        ctx = _ctx(CHOPPY,
                   config={"regime": {"bull_vol_block_offensive": True}})
        assert task.run(ctx) is None
        assert ctx.bear_only is False

    def test_bear_untouched(self):
        """BEAR path is handled by BEARBranchTask — this task stays hands-off."""
        task = BullVolOffensiveBlockTask()
        ctx = _ctx(BEAR,
                   config={"regime": {"bull_vol_block_offensive": True}})
        assert task.run(ctx) is None
        assert ctx.bear_only is False


class TestFullBlock:
    """bull_vol_defensives_too=true → buy_blocked=True (pure cash)."""

    def test_full_block_in_bull_vol(self):
        task = BullVolOffensiveBlockTask()
        ctx = _ctx(BULL_VOLATILE, config={"regime": {
            "bull_vol_block_offensive": True,
            "bull_vol_defensives_too":  True,
        }})
        assert task.run(ctx) is False
        assert ctx.buy_blocked is True
        assert ctx.counters["bull_vol_blocks"] == 1

    def test_full_block_not_triggered_in_other_regimes(self):
        task = BullVolOffensiveBlockTask()
        for r in (BULL_CALM, CHOPPY, BEAR):
            ctx = _ctx(r, config={"regime": {
                "bull_vol_block_offensive": True,
                "bull_vol_defensives_too":  True,
            }})
            assert task.run(ctx) is None
            assert ctx.buy_blocked is False


class TestBuyGatesJobWiring:
    def test_task_is_in_job_at_correct_position(self):
        """BullVolOffensiveBlockTask sits AFTER ConfidenceVeto + BEFORE BEARBranch."""
        from kernel.pipeline.task_gates import (
            BEARBranchTask, ConfidenceVetoTask,
        )
        tasks = BuyGatesJob().tasks
        types = [type(t) for t in tasks]
        i_confveto = types.index(ConfidenceVetoTask)
        i_bullvol  = types.index(BullVolOffensiveBlockTask)
        i_bear     = types.index(BEARBranchTask)
        assert i_confveto < i_bullvol < i_bear, \
            f"expected ConfVeto < BullVol < BEAR, got {types}"
