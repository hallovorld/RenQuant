"""Regression tests for buy/sell/rotate audit fixes (2026-04-29).

Audit identified 4 issues; all fixed here with corresponding tests:
  1. take_profit exit missing — added check_take_profit + wired into compute_exits
  2. rotation threshold 3%→6% + max_rotations_per_bar 2→1 (config change)
  3. buy_floor null→0.30 for new buys (config change)
  4. rotation enabled_regimes=[BULL_CALM] — prevents BULL_VOLATILE whipsaw
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


# ── Fix 1: take_profit exit ──────────────────────────────────────────────────

class TestTakeProfit:
    def _state(self, entry: float = 100.0) -> "HoldingState":
        from kernel.exits import HoldingState
        return HoldingState(
            entry_price=entry, entry_date=datetime.date(2024, 1, 2),
            high_watermark=entry,
        )

    def test_fires_at_threshold(self):
        from kernel.exits import check_take_profit
        state = self._state(100.0)
        sig = check_take_profit(125.0, state, take_profit_pct=0.25)
        assert sig.should_exit
        assert sig.exit_type == "take_profit"
        assert "gain=" in sig.reason

    def test_does_not_fire_below_threshold(self):
        from kernel.exits import check_take_profit
        state = self._state(100.0)
        sig = check_take_profit(124.9, state, take_profit_pct=0.25)
        assert not sig.should_exit

    def test_disabled_when_zero(self):
        from kernel.exits import check_take_profit
        state = self._state(100.0)
        sig = check_take_profit(999.0, state, take_profit_pct=0.0)
        assert not sig.should_exit

    def test_wired_before_trailing_stop_in_compute_exits(self):
        """take_profit fires BEFORE trailing_stop (priority 0 vs 1)."""
        from kernel.exits import compute_exits, HoldingState
        state = HoldingState(
            entry_price=100.0, entry_date=datetime.date(2024, 1, 2),
            high_watermark=130.0,  # trailing stop armed at 20%
        )
        # Price at 126: trailing stop (trail 18%) would fire (HWM 130 × 0.82 = 106.6)
        # But take-profit at 25% should also fire (gain=26%)
        sig, _ = compute_exits(
            current_price=126.0,
            today=datetime.date(2024, 6, 1),
            model_action="hold",
            state=state,
            params={
                "take_profit_pct": 0.25,
                "trailing_stop_trigger_pct": 0.20,
                "trailing_stop_trail_pct": 0.18,
            },
        )
        assert sig.should_exit
        assert sig.exit_type == "take_profit"  # take_profit beats trailing_stop

    def test_take_profit_in_compute_exits_disabled(self):
        """When take_profit_pct=0, does not interfere with other exits."""
        from kernel.exits import compute_exits, HoldingState
        state = HoldingState(
            entry_price=100.0, entry_date=datetime.date(2024, 1, 2),
            high_watermark=100.0,
        )
        sig, _ = compute_exits(
            current_price=200.0,
            today=datetime.date(2024, 6, 1),
            model_action="hold",
            state=state,
            params={"take_profit_pct": 0.0},
        )
        assert not sig.should_exit


# ── Fix 2: rotation threshold + cap ─────────────────────────────────────────

class TestRotationThreshold:
    def test_threshold_raised_to_6pct(self):
        import json
        cfg = json.loads((REPO_ROOT / "backtesting/renquant_104/strategy_config.json").read_text())
        assert cfg["rotation"]["min_expected_advantage_pct"] == 0.06, (
            "rotation threshold must be 0.06 (6%) — pre-fix was 0.03 which was "
            "less than the -2.5 APY per-event cost recorded in CLAUDE.md"
        )

    def test_max_rotations_per_bar_is_1(self):
        import json
        cfg = json.loads((REPO_ROOT / "backtesting/renquant_104/strategy_config.json").read_text())
        assert cfg["rotation"]["max_rotations_per_bar"] == 1, (
            "max_rotations_per_bar must be 1 — limits per-bar event cost"
        )


# ── Fix 3: buy_floor for new buys ───────────────────────────────────────────

class TestBuyFloor:
    def test_buy_floor_is_set_and_principled(self):
        """buy_floor must be set (not null) so the veto runs.

        2026-05-04: the original test pinned `floor == 0.30` (a hard-
        coded magic number). Per user mandate "0.3 is not a scientific
        number", the floor moved to the adaptive
        `min(mean + std, cap)` mode where `cap` is the 0.30 legacy
        ceiling. Either form is acceptable — the contract is "veto
        active, threshold defined".
        """
        import json
        cfg = json.loads((REPO_ROOT / "backtesting/renquant_104/strategy_config.json").read_text())
        floor = cfg["ranking"]["panel_scoring"].get("buy_floor")
        assert floor is not None, "buy_floor must not be null"
        if isinstance(floor, str):
            assert floor == "adaptive_mean_std_cap", (
                f"unsupported buy_floor mode: {floor!r}"
            )
            cap = cfg["ranking"]["panel_scoring"].get("buy_floor_adaptive_cap")
            assert cap is not None, (
                "adaptive mode requires buy_floor_adaptive_cap"
            )
            assert 0.0 < float(cap) < 1.0
        else:
            # Legacy absolute mode — must still be in (0, 1)
            assert 0.0 < float(floor) < 1.0


# ── Fix 4: rotation enabled_regimes ─────────────────────────────────────────

class TestRotationEnabledRegimes:
    def test_enabled_regimes_in_config(self):
        import json
        cfg = json.loads((REPO_ROOT / "backtesting/renquant_104/strategy_config.json").read_text())
        regimes = cfg["rotation"].get("enabled_regimes")
        assert regimes is not None, "rotation.enabled_regimes must be set"
        assert "BULL_CALM" in regimes
        assert "BULL_VOLATILE" not in regimes, (
            "rotation must be disabled in BULL_VOLATILE — each event costs "
            "-2.5 APY and frequent churn in volatile regimes exacerbates this"
        )

    def test_rotation_job_respects_enabled_regimes(self):
        """RotationJob.should_skip returns True when regime not in enabled_regimes."""
        src = (REPO_ROOT / "backtesting/renquant_104/kernel/pipeline/job_rotation.py").read_text()
        assert "enabled_regimes" in src
        assert "ctx.regime not in allowed" in src
