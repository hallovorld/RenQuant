"""Stale-HWM snap-down guard for RunnerAdapter (Plan A, 2026-04-23).

Bug context: `live_state.json` can end up with
`high_water_mark` set far above the real Alpaca account equity — from
a fresh install seeded at $100k, from a manual edit, or from a stored
state that got out of sync after a crash. `DrawdownCircuitTask` then
computes `(hwm - equity) / hwm` each bar, finds it above the 35%
halt threshold, and latches `skip_buys=True`. Result: zero offensive
orders placed every bar, as observed in the 2026-04-23 e2e
`daily_104.sh` run.

Fix: `resolve_hwm(stored, equity)` snaps the stored HWM down to
current equity when the stale ratio (default 1.5×) is exceeded.
Legitimate drawdowns up to ~33% (ratio 1.49) are preserved; genuine
stale-state (ratio 2× and up) is corrected.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from adapters.runner import resolve_hwm  # noqa: E402


class TestResolveHwm:
    def test_fresh_install_ratchets_up(self):
        """Empty state (hwm=0) starts ratcheting from current equity."""
        hwm, snapped = resolve_hwm(stored_hwm=0.0, account_value=50_000)
        assert hwm == 50_000
        assert snapped is False

    def test_equity_above_stored_ratchets_up(self):
        """Portfolio hitting a new high: ratchet HWM up to match."""
        hwm, snapped = resolve_hwm(stored_hwm=80_000, account_value=120_000)
        assert hwm == 120_000
        assert snapped is False

    def test_normal_drawdown_preserved(self):
        """A real 30% drawdown (hwm=100k, equity=70k → ratio 1.43)
        should NOT snap — DrawdownCircuitTask needs the real hwm to
        compute drawdown correctly.
        """
        hwm, snapped = resolve_hwm(stored_hwm=100_000, account_value=70_000)
        assert hwm == 100_000
        assert snapped is False

    def test_drawdown_at_boundary_preserved(self):
        """ratio = 1.49 (a 33% drawdown) is within the 1.5× threshold."""
        hwm, snapped = resolve_hwm(stored_hwm=149_000, account_value=100_000)
        assert hwm == 149_000
        assert snapped is False

    def test_stale_seed_is_snapped(self):
        """Classic regression: fresh install seeded HWM=100k, real
        Alpaca equity $10k. Ratio = 10× → SNAP to equity.
        """
        hwm, snapped = resolve_hwm(stored_hwm=100_000, account_value=10_000)
        assert hwm == 10_000
        assert snapped is True

    def test_just_above_threshold_snaps(self):
        """ratio = 1.51 — just over the 1.5× boundary → snap."""
        hwm, snapped = resolve_hwm(stored_hwm=151_001, account_value=100_000)
        assert hwm == 100_000
        assert snapped is True

    def test_zero_equity_falls_through_to_ratchet(self):
        """Broker returned equity=0 (no positions, maybe new account).
        Don't divide by zero; just use stored HWM as-is.
        """
        hwm, snapped = resolve_hwm(stored_hwm=50_000, account_value=0.0)
        assert hwm == 50_000
        assert snapped is False

    def test_negative_equity_falls_through_to_ratchet(self):
        """Paper accounts can briefly report negative values; guard against it."""
        hwm, snapped = resolve_hwm(stored_hwm=50_000, account_value=-100.0)
        assert hwm == 50_000
        assert snapped is False

    def test_custom_stale_ratio_respected(self):
        """Caller can tighten the threshold (e.g. 2× for a more forgiving
        policy that only snaps on truly extreme stale values).
        """
        # ratio 1.8 would snap at default 1.5 but NOT at custom 2.0:
        hwm_default, snapped_default = resolve_hwm(stored_hwm=180_000, account_value=100_000)
        hwm_custom,  snapped_custom  = resolve_hwm(
            stored_hwm=180_000, account_value=100_000, stale_ratio=2.0,
        )
        assert snapped_default is True
        assert hwm_default == 100_000
        assert snapped_custom is False
        assert hwm_custom == 180_000

    def test_snap_preserves_float_type(self):
        hwm, _ = resolve_hwm(stored_hwm=1_000_000, account_value=1_000)
        assert isinstance(hwm, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
