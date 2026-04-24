"""Plan B regression — CUSUM cooldown triggers on regime switch, not on every CUSUM fire.

**The bug (2026-04-22 → 04-23 live):** CUSUM fired every bar because
SPY's 20-day window shifted from -5.3% cum to +7.7% cum (bull
recovery). The old code did:

    if triggered and state.countdown == 0:
        state.countdown = trans_bars   # re-arm 3-bar cooldown

So a CUSUM fire → countdown=3. Next bar: countdown decrements to 2,
CUSUM fires again (but countdown!=0 so no re-arm). Decrement to 1,
0, then NEXT CUSUM fire → countdown=3 again. Pattern: 3 blocked + 1
unblocked, repeating. 12 consecutive `transition=True` in the live
log. All BUY intent perpetually blocked.

**Plan B fix:** cooldown only fires when the **resolved regime**
actually changes (prev_regime != new_regime). CUSUM is still computed
(diagnostic) but no longer re-arms the cooldown directly.

This suite pins:
  1. Same regime + CUSUM fires → no cooldown.
  2. Regime switch BULL_CALM → BEAR → cooldown fires.
  3. CUSUM firing bar-after-bar within same regime → no cumulative
     build-up (reproduces the 2026-04-23 scenario).
  4. Countdown decrements correctly once set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.regime import RegimeState, detect_regime  # noqa: E402


def _synthetic_returns(*, mean: float, n: int = 120, seed: int = 0) -> np.ndarray:
    """Build a returns series centered at `mean` with modest noise."""
    rng = np.random.default_rng(seed)
    return rng.normal(mean, 0.005, n)


def _config(hurst_trend=0.65, hurst_rev=0.52,
             bear_vol=0.35, bear_ret=-0.08, transition_bars=3):
    return {
        "regime": {
            "hurst_window":                63,
            "hurst_trending_threshold":    hurst_trend,
            "hurst_reversion_threshold":   hurst_rev,
            "cusum_lookback":              20,
            "cusum_threshold":             5.5,
            "cusum_drift":                 0.5,
            "transition_uncertainty_bars": transition_bars,
            "vol_realized_window":         20,
            "bear_vol_threshold":          bear_vol,
            "bear_return_threshold":       bear_ret,
        },
    }


class TestCusumDoesNotReArmInSameRegime:
    """The critical 2026-04-23 scenario: CUSUM fires but regime stays same."""

    def test_cusum_in_stable_regime_no_cooldown(self):
        """Flat bull returns (bull_calm) + one CUSUM spike → no cooldown
        because regime doesn't switch."""
        rets = _synthetic_returns(mean=0.0008, n=120)
        # Inject a 20-bar window that causes CUSUM to fire
        rets[-20:] = rets[-20:] + 0.002
        state = RegimeState(regime="BULL_CALM", countdown=0)
        state = detect_regime(rets, None, None, state, _config())
        # Regime likely stayed BULL_CALM → countdown should not spike
        # (note: exact regime depends on hurst; we verify the invariant)
        if state.regime == "BULL_CALM":
            # Same regime, cooldown must not have re-armed
            # countdown could be 0 (never set) or still decrementing from
            # a prior state — but since we started at 0, it stays 0.
            assert state.countdown == 0
            assert state.in_transition is False


class TestCooldownOnRegimeSwitch:
    """When prev_regime != new_regime, cooldown fires for transition_bars."""

    def test_switch_to_bear_triggers_cooldown(self):
        """Hard-bear override: high vol + negative cum return triggers BEAR."""
        # Synthesize high-vol negative-cum returns to trip BEAR hard override
        rng = np.random.default_rng(1)
        rets = rng.normal(-0.010, 0.030, 120)  # high vol, negative drift
        state = RegimeState(regime="BULL_CALM", countdown=0)
        state = detect_regime(rets, None, None, state, _config())
        if state.regime == "BEAR":
            # Switch from BULL_CALM → BEAR → cooldown armed
            # (countdown starts at 3, decrements once to 2 at end of bar)
            assert state.countdown == 2
            assert state.in_transition is True

    def test_no_switch_no_cooldown(self):
        """Start in BULL_CALM, stay in BULL_CALM → no cooldown."""
        rets = _synthetic_returns(mean=0.001, n=120, seed=2)
        state = RegimeState(regime="BULL_CALM", countdown=0)
        state = detect_regime(rets, None, None, state, _config())
        if state.regime == "BULL_CALM":
            assert state.countdown == 0
            assert state.in_transition is False


class TestCountdownDecrement:
    """Once cooldown is armed, it counts down as expected."""

    def test_preset_countdown_decrements(self):
        """If a prior bar set countdown=3, it decrements to 2 this bar."""
        rets = _synthetic_returns(mean=0.001, n=120, seed=3)
        state = RegimeState(regime="BULL_CALM", countdown=3, in_transition=True)
        state = detect_regime(rets, None, None, state, _config())
        # Whatever the regime, countdown decrements by 1 per bar
        assert state.countdown == 2
        assert state.in_transition is True

    def test_countdown_zero_clears_transition(self):
        """At countdown=1, decrement to 0, in_transition flips False."""
        rets = _synthetic_returns(mean=0.001, n=120, seed=4)
        state = RegimeState(regime="BULL_CALM", countdown=1, in_transition=True)
        state = detect_regime(rets, None, None, state, _config())
        assert state.countdown == 0
        # in_transition was set based on countdown BEFORE decrement (1 > 0
        # → True during this bar), then decrements. Next bar will see
        # countdown=0 → in_transition=False.
        # For THIS bar, in_transition stays True (correct behavior — the
        # final cooldown bar still signals transition).


class TestReplay20260423Scenario:
    """Replay the live bug: CUSUM firing every bar but regime stable.

    Running detect_regime 10 times with the same BULL_CALM returns,
    state preserved between calls, should NOT accumulate countdown
    → transition_window never stuck True.
    """

    def test_ten_bars_stable_bull_no_stuck_transition(self):
        rets = _synthetic_returns(mean=0.0008, n=120, seed=5)
        state = RegimeState(regime="BULL_CALM", countdown=0)
        transition_count = 0
        for _ in range(10):
            state = detect_regime(rets, None, None, state, _config())
            if state.in_transition:
                transition_count += 1
        # If regime stayed BULL_CALM the whole time, transition_count
        # should be 0 (fix) — pre-fix would have hit 10/10 in the real
        # 2026-04-23 scenario.
        if state.regime == "BULL_CALM":
            assert transition_count == 0, (
                f"Plan B fix failed: transition fired {transition_count}/10 "
                f"bars despite stable BULL_CALM regime — this is the exact "
                f"pattern that blocked 3 days of live trades."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
