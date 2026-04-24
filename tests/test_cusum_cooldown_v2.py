"""CUSUM-v2 Design C — confidence-scaled sizing, no hard block.

User-locked 2026-04-24:
  "我建议 C。理由：已经装了 Kelly sizing —— 让 size 自己根据 μ/σ² 决定,
   CUSUM 只提供一个软信号（乘数）"

Implementation contract:
  * RegimeState.cooldown_start is stamped at the same moment
    countdown is armed (regime switch detected).
  * cusum_cooldown_progress(now, start, days) returns a multiplier
    in [0, 1]: 0 just after switch, 1.0 after `days` elapsed.
  * SizeAndEmitTask multiplies max_position_pct by this progress when
    `regime.cusum_cooldown_mode == "wall_time"`.
  * TransitionWindowTask is a no-op in wall_time mode (soft cooldown,
    no hard block).
  * Default mode "bar_count" preserves v4 behaviour.
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

from kernel.regime import RegimeState, cusum_cooldown_progress  # noqa: E402
from kernel.pipeline.task_gates import TransitionWindowTask  # noqa: E402


class TestProgressHelper:
    def test_none_start_returns_one(self):
        """No cooldown active → no penalty."""
        now = datetime.datetime(2026, 4, 24, 10, 0, 0)
        assert cusum_cooldown_progress(now, None, 3.0) == 1.0

    def test_zero_days_returns_one(self):
        """Disabled window → no penalty."""
        now = datetime.datetime(2026, 4, 24, 10, 0, 0)
        start = datetime.datetime(2026, 4, 24, 9, 0, 0)
        assert cusum_cooldown_progress(now, start, 0) == 1.0

    def test_just_started_zero(self):
        start = datetime.datetime(2026, 4, 24, 10, 0, 0)
        now = datetime.datetime(2026, 4, 24, 10, 0, 1)
        assert cusum_cooldown_progress(now, start, 3.0) < 0.001

    def test_halfway_half(self):
        start = datetime.datetime(2026, 4, 21, 12, 0, 0)
        now = datetime.datetime(2026, 4, 23, 0, 0, 0)   # 1.5 of 3 days
        p = cusum_cooldown_progress(now, start, 3.0)
        assert 0.4 < p < 0.6

    def test_fully_elapsed_one(self):
        start = datetime.datetime(2026, 4, 20, 10, 0, 0)
        now = datetime.datetime(2026, 4, 25, 10, 0, 0)   # 5 of 3 days
        assert cusum_cooldown_progress(now, start, 3.0) == 1.0

    def test_accepts_date_inputs(self):
        start = datetime.date(2026, 4, 21)
        now = datetime.date(2026, 4, 23)
        p = cusum_cooldown_progress(now, start, 3.0)
        assert 0.6 < p < 0.7


class TestRegimeStateField:
    def test_cooldown_start_default_is_none(self):
        rs = RegimeState()
        assert rs.cooldown_start is None

    def test_can_store_datetime(self):
        dt = datetime.datetime(2026, 4, 24, 10, 0, 0)
        rs = RegimeState(cooldown_start=dt)
        assert rs.cooldown_start == dt


class TestTransitionWindowTaskModes:
    def test_bar_count_default_blocks_in_transition(self):
        """Legacy behaviour — hard block when in_transition."""
        task = TransitionWindowTask()
        ctx = SimpleNamespace(
            regime_state = RegimeState(in_transition=True, countdown=2),
            counters     = {},
            buy_blocked  = False,
            config       = {"regime": {"cusum_cooldown_mode": "bar_count"}},
        )
        out = task.run(ctx)
        assert out is False
        assert ctx.buy_blocked is True
        assert ctx.counters["transition_blocks"] == 1

    def test_wall_time_mode_is_noop_even_in_transition(self):
        """Design C: no hard block — sizing handles the cooldown."""
        task = TransitionWindowTask()
        ctx = SimpleNamespace(
            regime_state = RegimeState(in_transition=True, countdown=2),
            counters     = {},
            buy_blocked  = False,
            config       = {"regime": {"cusum_cooldown_mode": "wall_time"}},
        )
        out = task.run(ctx)
        assert out is None
        assert ctx.buy_blocked is False
        assert ctx.counters.get("transition_blocks", 0) == 0

    def test_no_transition_either_mode(self):
        task = TransitionWindowTask()
        for mode in ("bar_count", "wall_time"):
            ctx = SimpleNamespace(
                regime_state = RegimeState(in_transition=False, countdown=0),
                counters     = {},
                buy_blocked  = False,
                config       = {"regime": {"cusum_cooldown_mode": mode}},
            )
            out = task.run(ctx)
            assert out is None
            assert ctx.buy_blocked is False


class TestISOParse:
    """Roundtrip: isoformat() → _parse_iso_dt restores the datetime."""

    def test_roundtrip(self):
        from adapters.runner import _parse_iso_dt
        original = datetime.datetime(2026, 4, 21, 15, 32, 7)
        parsed = _parse_iso_dt(original.isoformat())
        assert parsed == original

    def test_none_returns_none(self):
        from adapters.runner import _parse_iso_dt
        assert _parse_iso_dt(None) is None
        assert _parse_iso_dt("") is None

    def test_bad_string_returns_none(self):
        from adapters.runner import _parse_iso_dt
        assert _parse_iso_dt("not-a-date") is None
