"""Regression tests for RC-MISMATCH + CONF-MULT audit fixes (2026-04-25).

Both bugs surfaced in the live Alpaca decision tree as: regime=BULL_CALM
but confidence=0.0041, causing max_position × confidence ≈ $6 → no buys
ever fire. User spec:

    "你那个sizing math靠谱吗？confidence也太低了"
    "直接*confidence不就是降档么这特么的在搞笑吧"
    "给的额度不够买一股的时候就买一股嘛"

Three layered fixes verified here:
  1. RC-MISMATCH — `compute_regime_confidence` now matches the regime
     decision source (Hurst vs GMM vs hard_bear) instead of always
     querying GMM.
  2. CONF-MULT — `confidence_to_size_multiplier` floors at 0.5 so low
     confidence dampens but doesn't degenerate sizing.
  3. MIN-1-SHARE — `compute_position_size` falls back to 1 share when
     conf-scaled cap and 25% fallback both produce 0 shares but cash
     is sufficient for one share.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

import pytest  # noqa: E402

from kernel.regime import (  # noqa: E402
    compute_regime_confidence,
    confidence_to_size_multiplier,
)
from kernel.sizing import compute_position_size  # noqa: E402


CONFIG = {
    "regime": {
        "hurst_trending_threshold": 0.65,
        "hurst_reversion_threshold": 0.52,
        "choppy_hurst_floor": 0.20,
    }
}


# ── compute_regime_confidence — RC-MISMATCH ───────────────────────────────────

class TestRegimeConfidenceMismatchFix:
    """The pre-fix scenario: Hurst forces BULL_CALM via MOMENTUM, but GMM gives
    BULL_CALM only 0.0041 probability. Pre-fix returned 0.0041; post-fix returns
    Hurst-distance based confidence (≥0.5 due to floor)."""

    def test_hurst_momentum_overrides_gmm_low_probability(self):
        gmm_probs = {"BULL_CALM": 0.0041, "BULL_VOLATILE": 0.50,
                     "CHOPPY": 0.49, "BEAR": 0.0059}
        # Hurst at 0.72 (well above threshold 0.65) → strong MOMENTUM
        conf = compute_regime_confidence(
            "BULL_CALM", hurst=0.72, gmm_probs=gmm_probs,
            in_transition=False, config=CONFIG,
            hurst_regime="MOMENTUM", hard_bear=False,
        )
        # Hurst (0.72-0.65)/(1-0.65) = 0.20 → +0.5 floor → 0.5 + 0.5*0.20 = 0.60
        assert conf == pytest.approx(0.60, abs=0.01)
        # Critical: NOT the raw GMM 0.0041
        assert conf > 0.5

    def test_hurst_at_threshold_floors_at_half(self):
        # Hurst exactly at threshold → conf = 0.5 + 0.5*0 = 0.5 (the floor)
        gmm_probs = {"BULL_CALM": 0.001}
        conf = compute_regime_confidence(
            "BULL_CALM", hurst=0.65, gmm_probs=gmm_probs,
            in_transition=False, config=CONFIG,
            hurst_regime="MOMENTUM", hard_bear=False,
        )
        assert conf == pytest.approx(0.5, abs=0.01)

    def test_hurst_max_returns_one(self):
        gmm_probs = {"BULL_CALM": 0.001}
        conf = compute_regime_confidence(
            "BULL_CALM", hurst=1.0, gmm_probs=gmm_probs,
            in_transition=False, config=CONFIG,
            hurst_regime="MOMENTUM", hard_bear=False,
        )
        assert conf == pytest.approx(1.0, abs=0.01)

    def test_bear_hard_override_returns_one(self):
        # When BEAR is decided definitively (hard_bear flag), confidence = 1.0
        gmm_probs = {"BEAR": 0.20, "BULL_CALM": 0.30, "CHOPPY": 0.50}
        conf = compute_regime_confidence(
            "BEAR", hurst=0.45, gmm_probs=gmm_probs,
            in_transition=False, config=CONFIG,
            hurst_regime="MIXED", hard_bear=True,
        )
        assert conf == 1.0

    def test_bear_via_gmm_dominant_returns_one(self):
        gmm_probs = {"BEAR": 0.60, "BULL_CALM": 0.30, "CHOPPY": 0.10}
        conf = compute_regime_confidence(
            "BEAR", hurst=0.45, gmm_probs=gmm_probs,
            in_transition=False, config=CONFIG,
            hurst_regime="MIXED", hard_bear=False,
        )
        assert conf == 1.0

    def test_choppy_branch_unchanged(self):
        # Existing CHOPPY logic must keep working: Hurst distance from 0.52
        conf = compute_regime_confidence(
            "CHOPPY", hurst=0.30, gmm_probs={"CHOPPY": 0.50},
            in_transition=False, config=CONFIG,
            hurst_regime="REVERSION", hard_bear=False,
        )
        # (0.52-0.30)/(0.52-0.20) = 0.6875
        assert conf == pytest.approx(0.6875, abs=0.001)

    def test_in_transition_returns_half(self):
        conf = compute_regime_confidence(
            "BULL_CALM", hurst=0.72, gmm_probs={"BULL_CALM": 0.0041},
            in_transition=True, config=CONFIG,
            hurst_regime="MOMENTUM", hard_bear=False,
        )
        assert conf == 0.5

    def test_dominant_gmm_route_falls_back_to_gmm(self):
        # When neither Hurst nor BEAR forces, the dominant-GMM route uses GMM posterior.
        gmm_probs = {"BULL_VOLATILE": 0.55, "BULL_CALM": 0.30}
        conf = compute_regime_confidence(
            "BULL_VOLATILE", hurst=0.55, gmm_probs=gmm_probs,
            in_transition=False, config=CONFIG,
            hurst_regime="MIXED", hard_bear=False,
        )
        assert conf == pytest.approx(0.55, abs=0.001)

    def test_backwards_compat_no_hurst_arg(self):
        # Old callsite that doesn't pass hurst_regime / hard_bear must still work.
        gmm_probs = {"BULL_CALM": 0.50}
        conf = compute_regime_confidence(
            "BULL_CALM", hurst=0.55, gmm_probs=gmm_probs,
            in_transition=False, config=CONFIG,
        )
        # Falls through to the legacy GMM-posterior return
        assert conf == 0.50


# ── confidence_to_size_multiplier — CONF-MULT ─────────────────────────────────

class TestConfidenceSizeMultiplier:
    """Pre-fix: max_position_pct × ctx.confidence directly. Range [0, 1] with no
    floor → 0.0041 confidence collapsed sizing to ~$6. Post-fix: floored at 0.5
    so low confidence dampens but doesn't degenerate."""

    def test_zero_confidence_floors_at_half(self):
        assert confidence_to_size_multiplier(0.0) == 0.5

    def test_extreme_low_confidence_floors_at_half(self):
        assert confidence_to_size_multiplier(0.0041) == 0.5

    def test_at_floor_stays_at_floor(self):
        assert confidence_to_size_multiplier(0.5) == 0.5

    def test_above_floor_passes_through(self):
        assert confidence_to_size_multiplier(0.7) == pytest.approx(0.7, abs=1e-9)

    def test_max_one(self):
        assert confidence_to_size_multiplier(1.0) == 1.0

    def test_above_one_clamps(self):
        # Defensive: if some bug pushes confidence > 1.0, we clamp at 1.0
        assert confidence_to_size_multiplier(1.5) == 1.0

    def test_nan_returns_floor(self):
        assert confidence_to_size_multiplier(float("nan")) == 0.5

    def test_custom_floor(self):
        assert confidence_to_size_multiplier(0.1, floor=0.7) == 0.7

    def test_pre_fix_scenario_yields_real_position(self):
        # Reproduction of the live bug: $10k portfolio, 15% max_position_pct,
        # confidence=0.0041 (Hurst MOMENTUM but GMM disagrees).
        portfolio = 10000.0
        max_pct = 0.15
        confidence = 0.0041
        # Pre-fix: max_pct * confidence * portfolio = $6.15 (broken)
        pre_fix_dollars = max_pct * confidence * portfolio
        assert pre_fix_dollars < 10  # broken state proof
        # Post-fix: max_pct * conf_mult(confidence) * portfolio = $750
        post_fix_dollars = max_pct * confidence_to_size_multiplier(confidence) * portfolio
        assert post_fix_dollars >= 700  # can afford a $700 NVDA share


# ── compute_position_size — MIN-1-SHARE ───────────────────────────────────────

class TestMinOneShareFallback:
    """Pre-fix: when both confidence-scaled cap AND 25% fallback produced 0 shares,
    compute_position_size returned (0.0, 0). Post-fix: if investable cash >= price,
    take 1 share. User spec: "给的额度不够买一股的时候就买一股嘛"."""

    def test_min_one_share_when_cap_too_small_but_cash_sufficient(self):
        # Setup: conf-scaled cap can't cover 1 share, but cash >= price.
        # max_position_pct passed already-scaled = $5 on $10k portfolio.
        # 25% fallback = $2500 but available_cash = $1000 only → 1 share at $700 fits.
        actual_pct, shares = compute_position_size(
            portfolio_value=10_000.0,
            available_cash=1_000.0,
            max_position_pct=0.0005,   # ridiculously low, $5 cap
            cash_reserve_pct=0.0,      # no reserve
            price=700.0,               # NVDA-like
        )
        assert shares == 1
        assert actual_pct == pytest.approx(700.0 / 10_000.0, abs=1e-6)

    def test_returns_zero_when_cash_insufficient_for_one_share(self):
        # Cash $100, price $700 → genuinely can't afford
        actual_pct, shares = compute_position_size(
            portfolio_value=10_000.0,
            available_cash=100.0,
            max_position_pct=0.0005,
            cash_reserve_pct=0.0,
            price=700.0,
        )
        assert shares == 0
        assert actual_pct == 0.0

    def test_normal_sizing_path_unchanged(self):
        # Normal case (cap covers many shares) — should NOT trigger min-1 fallback.
        actual_pct, shares = compute_position_size(
            portfolio_value=10_000.0,
            available_cash=5_000.0,
            max_position_pct=0.15,    # $1500 cap
            cash_reserve_pct=0.0,
            price=100.0,              # affords 15 shares
        )
        assert shares == 15
        assert actual_pct == pytest.approx(0.15, abs=1e-6)

    def test_25pct_fallback_still_fires_first(self):
        # max_pct=$1 cap, but 25% fallback ($2500) + $5000 cash + $100 price = 25 shares
        # via fallback BEFORE min-1 fires.
        actual_pct, shares = compute_position_size(
            portfolio_value=10_000.0,
            available_cash=5_000.0,
            max_position_pct=0.0001,   # $1 cap
            cash_reserve_pct=0.0,
            price=100.0,
        )
        # 25% fallback dominates → 25 shares, NOT just 1
        assert shares == 25
