"""EMA50GateTask flag-gating regression guard.

Audit P0 2026-05-13: EMA50GateTask was hardcoded with no config toggle.
This pins the new ``gates.ema50_gate.enabled`` flag (default True so
baseline is unchanged) and verifies the disable path skips the gate.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _ctx_with_spy_below_ema(config: dict):
    """Construct a minimal ctx where SPY close < EMA50 → gate WOULD fire."""
    n = 100
    # Steady downtrend so the most recent close is well below the 50-day EMA.
    close = pd.Series([200.0 - i for i in range(n)])
    df = pd.DataFrame({"close": close})
    return SimpleNamespace(
        ohlcv={"SPY": df},
        config=config,
        buy_blocked=False,
    )


class TestEMA50GateFlag:

    def test_default_enabled_blocks_buys_below_ema(self):
        """No config → gate ON → blocks buys (preserves baseline)."""
        from kernel.pipeline.task_gates import EMA50GateTask
        task = EMA50GateTask()
        ctx = _ctx_with_spy_below_ema(config={})
        result = task.run(ctx)
        assert result is False
        assert ctx.buy_blocked is True

    def test_explicit_enabled_true_blocks_buys(self):
        """Explicit enabled=True → gate ON → blocks buys."""
        from kernel.pipeline.task_gates import EMA50GateTask
        task = EMA50GateTask()
        ctx = _ctx_with_spy_below_ema(
            config={"gates": {"ema50_gate": {"enabled": True}}}
        )
        result = task.run(ctx)
        assert result is False
        assert ctx.buy_blocked is True

    def test_disabled_skips_gate(self):
        """enabled=False → gate SKIPS → buys not blocked even when SPY below EMA50."""
        from kernel.pipeline.task_gates import EMA50GateTask
        task = EMA50GateTask()
        ctx = _ctx_with_spy_below_ema(
            config={"gates": {"ema50_gate": {"enabled": False}}}
        )
        result = task.run(ctx)
        assert result is None
        assert ctx.buy_blocked is False
