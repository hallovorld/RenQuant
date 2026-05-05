"""Functional + behavior tests for stop_loss / trailing_stop / single_day_loss
across all 4 regimes (BULL_CALM, BULL_VOLATILE, CHOPPY, BEAR).

Per user audit 2026-04-26 (SL-1, SL-2): pre-fix, only BULL_CALM had
trailing_stop + single_day_loss enabled. Other regimes had stop_loss
ONLY (cumulative-from-entry). This left held positions vulnerable to:

  - SL-1: gain 30% then drop 8% in BULL_VOLATILE — no trailing protection
          because trail_pct=0
  - SL-2: gap-down day in BULL_VOLATILE — no sdl gate

Post-fix:
  BULL_CALM:     trailing 20%/18%, sdl 10% (unchanged — was already enabled)
  BULL_VOLATILE: trailing 20%/10%, sdl 10% (NEW)
  CHOPPY:        trailing 15%/10%, sdl 10% (NEW)
  BEAR:          trailing 10%/5%,  sdl 0%  (NEW trailing; sdl skipped — stop_loss=5% covers)

Tests:
  - Behavior (unit): each check_* function fires correctly per regime threshold
  - Functional (config-level): strategy_config has all 4 regime params populated
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.exits import (  # noqa: E402
    HoldingState,
    check_stop_loss,
    check_single_day_loss,
    check_trailing_stop,
)


CONFIG_PATH = REPO_ROOT / "backtesting" / "renquant_104" / "strategy_config.json"
GOLDEN_PATH = REPO_ROOT / "backtesting" / "renquant_104" / "strategy_config.golden.json"


# ── Fixtures ────────────────────────────────────────────────────────────────

def _state(entry_price=100.0, prev_close=100.0, hwm=100.0,
           entry_date=None, sell_streak=0):
    return HoldingState(
        entry_price=entry_price,
        prev_close=prev_close,
        high_watermark=hwm,
        entry_date=entry_date or datetime.date(2026, 1, 1),
        sell_streak=sell_streak,
    )


def _load_regime_params(strategy_path: Path) -> dict:
    cfg = json.loads(strategy_path.read_text())
    return cfg["regime_params"]


# ── Functional: config-level invariants ─────────────────────────────────────

class TestRegimeConfigInvariants:
    """All 4 regimes must have stop_loss + trailing_stop + sdl populated."""

    @pytest.mark.parametrize("regime", ["BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR"])
    def test_strategy_config_has_stop_fields(self, regime):
        params = _load_regime_params(CONFIG_PATH)
        rp = params[regime]
        for key in (
            "stop_loss_pct",
            "max_single_day_loss_pct",
            "trailing_stop_trigger_pct",
            "trailing_stop_trail_pct",
        ):
            assert key in rp, f"{regime} missing {key}"

    @pytest.mark.parametrize("regime", ["BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR"])
    def test_golden_matches_strategy(self, regime):
        s = _load_regime_params(CONFIG_PATH)[regime]
        g = _load_regime_params(GOLDEN_PATH)[regime]
        for key in (
            "stop_loss_pct",
            "max_single_day_loss_pct",
            "trailing_stop_trigger_pct",
            "trailing_stop_trail_pct",
        ):
            assert s[key] == g[key], (
                f"{regime}.{key}: strategy={s[key]} != golden={g[key]}"
            )

    def test_all_offensive_regimes_have_trailing_enabled(self):
        """SL-1: BULL_CALM, BULL_VOLATILE, CHOPPY all have trailing > 0."""
        params = _load_regime_params(CONFIG_PATH)
        for regime in ("BULL_CALM", "BULL_VOLATILE", "CHOPPY"):
            rp = params[regime]
            assert rp["trailing_stop_trigger_pct"] > 0.0, (
                f"{regime} should have trailing_stop_trigger_pct > 0 (SL-1 fix)"
            )
            assert rp["trailing_stop_trail_pct"] > 0.0, (
                f"{regime} should have trailing_stop_trail_pct > 0 (SL-1 fix)"
            )

    def test_bear_has_trailing_for_defensives(self):
        """SL-1: BEAR has trailing for defensive holdings (GLD/TLT)."""
        params = _load_regime_params(CONFIG_PATH)
        rp = params["BEAR"]
        assert rp["trailing_stop_trigger_pct"] > 0.0, (
            "BEAR should trail defensives (SL-1 fix)"
        )

    def test_volatile_regimes_have_sdl(self):
        """SL-2: BULL_VOLATILE + CHOPPY have an SDL gate (absolute or σ-scaled).

        2026-05-04: BULL_CALM moved to σ-scaled (max_single_day_loss_pct=0,
        sdl_n_sigma=3.0) because the absolute 6% threshold tripped on
        noise for high-vol names — 20 SDL exits in B2 holdout had 40%
        win_rate / median pnl −5.2%. Either form (absolute > 0 OR
        σ-scaled > 0) counts as 'gate present'."""
        params = _load_regime_params(CONFIG_PATH)
        for regime in ("BULL_CALM", "BULL_VOLATILE", "CHOPPY"):
            rp = params[regime]
            abs_thr   = rp.get("max_single_day_loss_pct", 0)
            sigma_thr = rp.get("sdl_n_sigma", 0)
            assert (abs_thr > 0) or (sigma_thr > 0), (
                f"{regime} must have an SDL gate "
                f"(max_single_day_loss_pct OR sdl_n_sigma > 0)"
            )

    def test_volatile_trail_tighter_than_calm(self):
        """BULL_VOLATILE / CHOPPY trail tighter than BULL_CALM (faster exit)."""
        params = _load_regime_params(CONFIG_PATH)
        bc = params["BULL_CALM"]["trailing_stop_trail_pct"]
        bv = params["BULL_VOLATILE"]["trailing_stop_trail_pct"]
        ch = params["CHOPPY"]["trailing_stop_trail_pct"]
        assert bv <= bc, f"BULL_VOLATILE trail ({bv}) should be <= BULL_CALM ({bc})"
        assert ch <= bc, f"CHOPPY trail ({ch}) should be <= BULL_CALM ({bc})"

    def test_bear_most_defensive(self):
        """BEAR trail tightest of all regimes (most defensive)."""
        params = _load_regime_params(CONFIG_PATH)
        be = params["BEAR"]["trailing_stop_trail_pct"]
        for r in ("BULL_CALM", "BULL_VOLATILE", "CHOPPY"):
            assert be <= params[r]["trailing_stop_trail_pct"], (
                f"BEAR trail ({be}) should be <= {r} ({params[r]['trailing_stop_trail_pct']})"
            )


# ── Behavior: stop_loss per regime ──────────────────────────────────────────

class TestStopLossBehavior:
    @pytest.mark.parametrize("regime, stop_pct, current, expected_fire", [
        ("BULL_CALM",     0.15, 85.0,  True),    # 15% loss → fire
        ("BULL_CALM",     0.15, 86.0,  False),   # 14% loss → no fire
        ("BULL_VOLATILE", 0.05, 95.0,  True),    # 5% loss → fire (tighter)
        ("BULL_VOLATILE", 0.05, 96.0,  False),
        ("CHOPPY",        0.08, 92.0,  True),
        ("CHOPPY",        0.08, 93.0,  False),
        ("BEAR",          0.05, 95.0,  True),
        ("BEAR",          0.05, 96.0,  False),
    ])
    def test_stop_loss_fires_at_regime_threshold(self, regime, stop_pct, current, expected_fire):
        sig = check_stop_loss(current, _state(entry_price=100.0), stop_pct)
        assert sig.should_exit == expected_fire, (
            f"{regime} stop_loss={stop_pct} at price {current}: expected fire={expected_fire}"
        )
        if expected_fire:
            assert sig.exit_type == "stop_loss"


# ── Behavior: trailing_stop per regime ──────────────────────────────────────

class TestTrailingStopBehavior:
    @pytest.mark.parametrize("regime, trigger, trail, hwm, current, expected_fire", [
        # BULL_CALM: trigger 20%, trail 18%
        ("BULL_CALM",     0.20, 0.18, 120.0, 98.4,  True),    # HWM 120, trail floor 98.4 → fire at 98.4
        ("BULL_CALM",     0.20, 0.18, 120.0, 99.0,  False),   # 99 > 98.4 → no fire
        ("BULL_CALM",     0.20, 0.18, 119.0, 97.0,  False),   # gain only 19% < 20% trigger → no trail active
        # BULL_VOLATILE: trigger 20%, trail 10% (TIGHTER than CALM)
        ("BULL_VOLATILE", 0.20, 0.10, 120.0, 108.0, True),    # 12% drop from HWM > 10% trail
        ("BULL_VOLATILE", 0.20, 0.10, 120.0, 109.0, False),   # 9.2% drop < 10% trail
        # CHOPPY: trigger 15% (smaller), trail 10%
        ("CHOPPY",        0.15, 0.10, 115.0, 103.5, True),    # gain 15%, drop 10% → fire
        ("CHOPPY",        0.15, 0.10, 114.0, 100.0, False),   # gain only 14% < 15% trigger
        # BEAR: trigger 10% (very small), trail 5% (very tight)
        ("BEAR",          0.10, 0.05, 110.0, 104.4, True),    # gain 10%, drop 5% → fire
        ("BEAR",          0.10, 0.05, 109.0, 100.0, False),   # gain only 9% < 10% trigger
    ])
    def test_trailing_stop_fires_at_regime_threshold(
        self, regime, trigger, trail, hwm, current, expected_fire,
    ):
        state = _state(entry_price=100.0, hwm=hwm)
        sig = check_trailing_stop(current, state, trigger, trail)
        assert sig.should_exit == expected_fire, (
            f"{regime} trail trigger={trigger} trail={trail} hwm={hwm} cur={current}: "
            f"expected fire={expected_fire}, got {sig.should_exit}"
        )
        if expected_fire:
            assert sig.exit_type == "trailing_stop"


# ── Behavior: single_day_loss per regime ────────────────────────────────────

class TestSingleDayLossBehavior:
    @pytest.mark.parametrize("regime, sdl_pct, prev_close, current, expected_fire", [
        # BULL_CALM sdl=10%
        ("BULL_CALM",     0.10, 100.0, 89.0,  True),    # 11% drop → fire
        ("BULL_CALM",     0.10, 100.0, 91.0,  False),   # 9% drop → no fire
        # BULL_VOLATILE sdl=10% (NEW)
        ("BULL_VOLATILE", 0.10, 100.0, 89.0,  True),
        ("BULL_VOLATILE", 0.10, 100.0, 91.0,  False),
        # CHOPPY sdl=10% (NEW)
        ("CHOPPY",        0.10, 100.0, 89.0,  True),
        ("CHOPPY",        0.10, 100.0, 91.0,  False),
        # BEAR sdl=0 → never fires
        ("BEAR",          0.0,  100.0, 50.0,  False),   # even 50% drop → no sdl
    ])
    def test_sdl_fires_at_regime_threshold(
        self, regime, sdl_pct, prev_close, current, expected_fire,
    ):
        state = _state(entry_price=prev_close, prev_close=prev_close)
        sig = check_single_day_loss(current, state, sdl_pct)
        assert sig.should_exit == expected_fire, (
            f"{regime} sdl={sdl_pct} prev={prev_close} cur={current}: "
            f"expected fire={expected_fire}"
        )
        if expected_fire:
            assert sig.exit_type == "single_day_loss"


# 2026-05-04: σ-scaled SDL — adapts threshold to per-ticker volatility.
# B2 holdout post-mortem: 20 absolute-mode SDL exits had 40% win_rate /
# median pnl −5.2% — high-vol stocks tripped the 6% threshold on
# normal noise days. σ mode uses N × (NGBoost_σ / √5) as the threshold.

class TestSigmaScaledSingleDayLoss:
    def _state_with_sigma(self, sigma, prev_close=100.0):
        s = _state(entry_price=prev_close, prev_close=prev_close)
        s.sigma = sigma
        return s

    def test_sigma_mode_uses_sigma_over_sqrt5(self):
        """σ_5d=0.10 → daily_vol=10/√5 ≈ 4.47%; with N=3 → 13.4% threshold.
        12% drop should NOT fire (below threshold)."""
        state = self._state_with_sigma(sigma=0.10)
        sig = check_single_day_loss(
            current_price=88.0,    # 12% drop
            state=state,
            sdl_pct=0.0,           # absolute disabled
            sdl_n_sigma=3.0,
        )
        assert not sig.should_exit, "12% < 13.4% σ-threshold; should not fire"

    def test_sigma_mode_fires_above_threshold(self):
        """Same σ=0.10 (threshold 13.4%); 15% drop SHOULD fire."""
        state = self._state_with_sigma(sigma=0.10)
        sig = check_single_day_loss(
            current_price=85.0,    # 15% drop
            state=state,
            sdl_pct=0.0,
            sdl_n_sigma=3.0,
        )
        assert sig.should_exit
        assert "σN=" in sig.reason

    def test_low_vol_stock_low_threshold(self):
        """Low-vol stock σ=0.02 (daily ≈ 0.9%); N=3 → 2.7% threshold.
        4% drop fires; 2% drop does not."""
        state = self._state_with_sigma(sigma=0.02)
        # 2% drop — below threshold
        sig = check_single_day_loss(98.0, state, 0.0, 3.0)
        assert not sig.should_exit
        # 4% drop — above threshold
        sig = check_single_day_loss(96.0, state, 0.0, 3.0)
        assert sig.should_exit

    def test_combined_mode_uses_more_permissive_threshold(self):
        """When both abs and σ are set, threshold = max(abs, σ-derived).
        σ=0.10 → σ-threshold 13.4%; abs=0.06 → effective threshold = 13.4%
        (more permissive for high-vol). 8% drop should NOT fire."""
        state = self._state_with_sigma(sigma=0.10)
        sig = check_single_day_loss(
            current_price=92.0,    # 8% drop
            state=state,
            sdl_pct=0.06,
            sdl_n_sigma=3.0,
        )
        assert not sig.should_exit, (
            "8% < max(6%, 13.4%) — combined mode is more permissive"
        )

    def test_sigma_mode_no_op_when_sigma_missing(self):
        """If state.sigma is None or non-finite, σ-mode contributes 0 →
        falls back to absolute mode."""
        state = _state(entry_price=100.0, prev_close=100.0)
        # No sigma set; sdl_pct=0.06 still drives behavior
        sig = check_single_day_loss(93.0, state, 0.06, 3.0)
        assert sig.should_exit, (
            "7% drop should fire on absolute 6% threshold "
            "when σ is missing"
        )

    def test_both_disabled_never_fires(self):
        state = self._state_with_sigma(sigma=0.10)
        sig = check_single_day_loss(50.0, state, 0.0, 0.0)
        assert not sig.should_exit, (
            "with both thresholds disabled, even 50% drop must not fire"
        )


# ── Functional: priority ordering preserved ────────────────────────────────

class TestExitPriorityOrdering:
    """Stop-loss / trailing / sdl all present — priority order intact.

    Per kernel/exits.py docstring, order is:
      1. trailing_stop  (HWM-based)
      2. stop_loss      (cumulative)
      3. single_day_loss
      4. max_hold
      5. model_sell
    """

    def test_trailing_fires_before_stop_loss_when_both_would(self):
        """Held with HWM=130, current=104: trailing (10% drop from HWM) fires
        before stop_loss (4% loss from entry doesn't trigger 15% threshold)."""
        state = _state(entry_price=100.0, hwm=130.0)
        # HWM 130, trail 10% → trigger at 117
        # current 104 → drop from HWM = 26/130 = 20% > 10% trail
        # cumulative loss = 4% from entry < 15% stop
        trail_sig = check_trailing_stop(104.0, state, 0.20, 0.10)
        stop_sig = check_stop_loss(104.0, state, 0.15)
        assert trail_sig.should_exit
        assert not stop_sig.should_exit


# ── Behavior: edge cases ────────────────────────────────────────────────────

class TestStopLossEdgeCases:
    def test_stop_loss_zero_threshold_disabled(self):
        sig = check_stop_loss(50.0, _state(entry_price=100.0), 0.0)
        assert not sig.should_exit, "stop_pct=0 should disable"

    def test_stop_loss_negative_entry_price_skipped(self):
        sig = check_stop_loss(50.0, _state(entry_price=0.0), 0.15)
        assert not sig.should_exit

    def test_trailing_stop_below_trigger_no_fire(self):
        """Gain only 5% (below 20% trigger) — no trail floor active."""
        state = _state(entry_price=100.0, hwm=105.0)
        sig = check_trailing_stop(95.0, state, 0.20, 0.18)
        assert not sig.should_exit

    def test_trailing_stop_zero_trigger_skipped(self):
        sig = check_trailing_stop(100.0, _state(hwm=120.0), 0.0, 0.18)
        # trigger=0 means feature disabled
        assert not sig.should_exit

    def test_sdl_skips_on_nan_prev_close(self):
        state = _state(prev_close=float("nan"))
        sig = check_single_day_loss(50.0, state, 0.10)
        assert not sig.should_exit, "NaN prev_close should skip"

    def test_sdl_skips_on_zero_prev_close(self):
        state = _state(prev_close=0.0)
        sig = check_single_day_loss(50.0, state, 0.10)
        assert not sig.should_exit
