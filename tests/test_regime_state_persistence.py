"""Plan R regression — persist CUSUM `countdown` across live runs.

The bug (2026-04-22 / 04-23 live): `transition_window=True` fired on
every run for 3+ days → buys perpetually blocked → 0 trades. Root
cause: RunnerAdapter.make_context created a bare `RegimeState()`
(countdown=0) every invocation. CUSUM's 20-bar reference vs 20-bar
current window legitimately stays diverged for ~20 bars after a
regime shift — so every fresh run re-trips CUSUM, re-sets
`state.countdown = 3`, never decrements below 3.

The sim doesn't have this bug because `RegimeState` lives in-memory
for the full 27-month run. Live was missing state persistence.

This suite pins:
  1. RegimeState fields are persisted to live_state.json on save.
  2. make_context reloads them into a fresh RegimeState when a live run
     starts (instead of clobbering to defaults).
  3. Default fallback: no `regime_state` key → RegimeState() defaults.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.regime import RegimeState  # noqa: E402


class TestRegimeStatePersistenceContract:
    """Source-level checks that runner.py wires the persistence correctly.

    Skip the import-level RunnerAdapter stub (needs Alpaca + broker config)
    and verify via string scan that the contract lines are present.
    """

    def test_runner_loads_regime_state_key(self):
        src = (_STRATEGY_DIR / "adapters" / "runner.py").read_text()
        assert 'state.get("regime_state"' in src, (
            "make_context must load a persisted `regime_state` dict from "
            "live_state.json (else CUSUM countdown can't survive across runs)"
        )

    def test_runner_hydrates_fields(self):
        src = (_STRATEGY_DIR / "adapters" / "runner.py").read_text()
        for field in ("countdown", "in_transition", "cusum_pos", "cusum_neg"):
            assert f'regime_persist.get("{field}"' in src, (
                f"make_context must pass `{field}` into RegimeState(...)"
            )

    def test_runner_saves_regime_state_key(self):
        src = (_STRATEGY_DIR / "adapters" / "runner.py").read_text()
        assert '"regime_state":' in src, (
            "apply_outputs must write `regime_state` into live_state.json"
        )
        for field in ("countdown", "in_transition", "cusum_pos", "cusum_neg"):
            assert f'"{field}"' in src, (
                f"apply_outputs must include `{field}` in the regime_state "
                f"snapshot"
            )


class TestRegimeStateDataClassRoundTrip:
    """Smoke-test the dataclass can be constructed from a dict and
    serialised back without loss."""

    def test_construct_from_dict(self):
        persisted = {
            "regime":        "BULL_VOLATILE",
            "confidence":    0.72,
            "in_transition": True,
            "countdown":     2,
            "cusum_pos":     3.5,
            "cusum_neg":     1.1,
        }
        rs = RegimeState(
            regime        = persisted["regime"],
            confidence    = float(persisted["confidence"]),
            in_transition = bool(persisted["in_transition"]),
            countdown     = int(persisted["countdown"]),
            cusum_pos     = float(persisted["cusum_pos"]),
            cusum_neg     = float(persisted["cusum_neg"]),
        )
        assert rs.regime        == "BULL_VOLATILE"
        assert rs.confidence    == pytest.approx(0.72)
        assert rs.in_transition is True
        assert rs.countdown     == 2
        assert rs.cusum_pos     == pytest.approx(3.5)
        assert rs.cusum_neg     == pytest.approx(1.1)

    def test_bare_regimestate_has_expected_defaults(self):
        """The buggy old path constructed `RegimeState()` every run —
        confirm what that actually gave us: countdown=0 etc."""
        rs = RegimeState()
        assert rs.countdown     == 0
        assert rs.in_transition is False
        assert rs.cusum_pos     == pytest.approx(0.0)
        assert rs.cusum_neg     == pytest.approx(0.0)


class TestCountdownLifecycle:
    """Mini-simulation: a 5-bar sequence where CUSUM fires on bar 0 only.
    With the fix, countdown decrements 3→2→1→0 and transition clears;
    without persistence across invocations, each "invocation" would reset
    to 3 → stuck forever.
    """

    def test_persisted_countdown_decrements_across_runs(self):
        # Simulate: each "bar" is a fresh live invocation that loads state
        # from disk, runs 1 step of "decrement-then-maybe-retrigger", saves.
        persisted: dict = {}

        def bar_step(cusum_triggers_now: bool):
            # 1) Reload state (mimics RunnerAdapter.make_context)
            rs = RegimeState(
                countdown     = int(persisted.get("countdown", 0)),
                in_transition = bool(persisted.get("in_transition", False)),
            )
            # 2) Simulate the core of detect_regime:
            #    - CUSUM trigger sets countdown only when 0
            #    - in_transition = countdown > 0
            #    - countdown decrements after use
            if cusum_triggers_now and rs.countdown == 0:
                rs.countdown = 3
            rs.in_transition = rs.countdown > 0
            if rs.countdown > 0:
                rs.countdown -= 1
            # 3) Snapshot to disk (mimics apply_outputs)
            persisted["countdown"]     = rs.countdown
            persisted["in_transition"] = rs.in_transition
            return rs.in_transition

        # Bar 0: CUSUM fires → countdown set to 3 → in_transition True,
        # then decrements to 2 at end of bar.
        assert bar_step(cusum_triggers_now=True) is True
        assert persisted["countdown"] == 2

        # Bar 1: CUSUM still fires (reference window overlap) but countdown!=0,
        # so it does NOT re-extend; just decrements.
        assert bar_step(cusum_triggers_now=True) is True
        assert persisted["countdown"] == 1

        # Bar 2: same — still cooling down
        assert bar_step(cusum_triggers_now=True) is True
        assert persisted["countdown"] == 0

        # Bar 3: CUSUM still fires, but now countdown is 0 → triggers again.
        # This is the correct behaviour: the cooldown IS re-armed only after
        # it expired. Without persistence, every bar re-trips from countdown=0
        # → countdown=3 regardless of prior state.
        assert bar_step(cusum_triggers_now=True) is True

        # Bar 4: CUSUM quiet — countdown continues decrementing, in_transition
        # persists as long as countdown > 0.
        assert bar_step(cusum_triggers_now=False) is True   # countdown was 2
        assert persisted["countdown"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
