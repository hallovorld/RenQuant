"""Tests for σ-aware stop_loss + realized-vol fallback (Fix #0a revive).

Reference: doc/AUDIT_2026-05-09.md §#1 — Fix #0a was rejected as "dead code"
because production NGBoost is OFF → state.sigma is always None → σ-aware
threshold never fires. The revive adds a `realized_sigma_daily` field
(20-day rolling stdev of daily returns, computed in PrepareHoldingTask)
that lets σ-aware exits work without NGB.

Industry references:
  Almgren-Chriss 2000 "Optimal Execution" — N-σ × √t cumulative band
  Edwards-Magee 1948 (Technical Analysis of Stock Trends) Ch. 28 — N-σ stops
  RiskMetrics 1996 (J.P. Morgan) — daily-σ as canonical risk unit

Audit regression guard (§5.13.3): pins that σ-aware stop_loss fires in
the absence of NGBoost, and that the legacy absolute stop_pct remains
honored as a floor (max-of-both semantics).
"""
from __future__ import annotations

import datetime
import math
import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = (
    Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
)
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.exits import (  # noqa: E402
    HoldingState, check_stop_loss, check_single_day_loss,
    _resolve_daily_sigma,
)


def _hs(today: datetime.date | None = None, entry_price: float = 100.0,
        *, sigma=None, realized=None, days_ago: int = 1) -> HoldingState:
    """Build a HoldingState whose entry_date is `days_ago` calendar days
    before `today`. Tests that pass `today` to check_stop_loss MUST pass
    the same `today` here so `days_held` is consistent."""
    if today is None:
        today = datetime.date(2024, 6, 1)
    entry_date = today - datetime.timedelta(days=days_ago)
    hs = HoldingState(
        entry_price=entry_price,
        entry_date=entry_date,
        high_watermark=entry_price,
    )
    if sigma is not None:
        hs.sigma = sigma
    if realized is not None:
        hs.realized_sigma_daily = realized
    return hs


# ─────────────────────────────────────────────────────────────────────────
# 1) _resolve_daily_sigma helper — priority chain
# ─────────────────────────────────────────────────────────────────────────

class TestResolveDailySigma:
    def test_ngb_sigma_preferred_over_realized(self):
        hs = _hs(sigma=0.10, realized=0.03)
        # NGBoost σ is fwd-5d → daily = 0.10 / √5 ≈ 0.0447
        result = _resolve_daily_sigma(hs)
        assert result is not None
        assert abs(result - 0.10 / math.sqrt(5.0)) < 1e-9

    def test_realized_fallback_when_ngb_none(self):
        hs = _hs(sigma=None, realized=0.025)
        assert _resolve_daily_sigma(hs) == pytest.approx(0.025)

    def test_realized_fallback_when_ngb_nan(self):
        hs = _hs(sigma=float("nan"), realized=0.030)
        assert _resolve_daily_sigma(hs) == pytest.approx(0.030)

    def test_realized_fallback_when_ngb_zero(self):
        hs = _hs(sigma=0.0, realized=0.020)
        assert _resolve_daily_sigma(hs) == pytest.approx(0.020)

    def test_returns_none_when_both_unavailable(self):
        hs = _hs(sigma=None, realized=None)
        assert _resolve_daily_sigma(hs) is None

    def test_returns_none_on_negative_realized(self):
        # Defensive — std cannot be negative; reject NaN-class
        hs = _hs(sigma=None, realized=-0.01)
        assert _resolve_daily_sigma(hs) is None


# ─────────────────────────────────────────────────────────────────────────
# 2) check_stop_loss — σ-aware threshold + backward compat
# ─────────────────────────────────────────────────────────────────────────

class TestStopLossBackwardCompat:
    """stop_n_sigma=0 (default) must produce identical behavior to legacy."""

    def test_legacy_stop_pct_fires(self):
        hs = _hs(entry_price=100.0)
        sig = check_stop_loss(current_price=85.0, state=hs, stop_pct=0.10)
        assert sig.should_exit is True
        assert sig.exit_type == "stop_loss"

    def test_legacy_stop_pct_no_fire(self):
        hs = _hs(entry_price=100.0)
        sig = check_stop_loss(current_price=95.0, state=hs, stop_pct=0.10)
        assert sig.should_exit is False

    def test_zero_stop_pct_disabled(self):
        hs = _hs(entry_price=100.0)
        sig = check_stop_loss(current_price=50.0, state=hs, stop_pct=0.0)
        assert sig.should_exit is False

    def test_nan_entry_price_no_fire(self):
        hs = _hs(entry_price=float("nan"))
        sig = check_stop_loss(current_price=80.0, state=hs, stop_pct=0.10)
        assert sig.should_exit is False


class TestStopLossSigmaAware:
    """σ-aware threshold: max(stop_pct, N × σ_daily × √hold_days)."""

    def test_sigma_aware_widens_stop_for_high_vol(self):
        # High-σ stock (3% daily): 2σ × √10 ≈ 19% threshold
        # Legacy 5% stop_pct alone would fire at -5%; σ-aware lifts to -19%
        today = datetime.date(2024, 6, 11)
        hs = _hs(today, entry_price=100.0, realized=0.03, days_ago=10)
        # Drop is -10% — legacy 5% would fire, σ-aware should NOT
        sig = check_stop_loss(
            current_price=90.0, state=hs,
            stop_pct=0.05, stop_n_sigma=2.0, today=today,
        )
        assert sig.should_exit is False, \
            "σ-aware stop should widen for high-σ (3% × 2 × √10 ≈ 19%) > 10% loss"

    def test_sigma_aware_still_fires_on_real_break(self):
        # Same high-σ stock — drop of -25% should fire (exceeds σ-band)
        today = datetime.date(2024, 6, 11)
        hs = _hs(today, entry_price=100.0, realized=0.03, days_ago=10)
        sig = check_stop_loss(
            current_price=75.0, state=hs,
            stop_pct=0.05, stop_n_sigma=2.0, today=today,
        )
        assert sig.should_exit is True
        # Threshold should reflect both modes in the reason text
        assert "σN=" in sig.reason
        assert "abs=" in sig.reason

    def test_legacy_stop_pct_acts_as_floor(self):
        # Low-σ stock (0.5% daily): 2σ × √5 ≈ 2.2%
        # σ-band is TIGHTER than legacy 5%, so max-of-both = legacy 5%
        # At -3% loss: below 5% legacy → no fire
        today = datetime.date(2024, 6, 6)
        hs = _hs(today, entry_price=100.0, realized=0.005, days_ago=5)
        sig = check_stop_loss(
            current_price=97.0, state=hs,   # -3% loss
            stop_pct=0.05, stop_n_sigma=2.0, today=today,
        )
        assert sig.should_exit is False
        # And the threshold should now reflect 5% legacy (not 2.2% σ)
        sig_fire = check_stop_loss(
            current_price=94.0, state=hs,   # -6% loss
            stop_pct=0.05, stop_n_sigma=2.0, today=today,
        )
        assert sig_fire.should_exit is True

    def test_sigma_aware_with_ngb_source(self):
        # NGB σ (5d) = 0.10 → daily = 0.0447, 2σ × √5 ≈ 20%
        today = datetime.date(2024, 6, 6)
        hs = _hs(today, entry_price=100.0, sigma=0.10, days_ago=5)
        sig = check_stop_loss(
            current_price=85.0, state=hs,   # -15% loss
            stop_pct=0.05, stop_n_sigma=2.0, today=today,
        )
        # σ-band ≈ 0.10/√5 × 2 × √5 = 0.20 → 20% threshold; -15% does NOT exceed
        assert sig.should_exit is False

    def test_hold_days_scaling_sigma_only(self):
        # Pure σ mode (stop_pct=0): 1 day held vs 25 days held
        # σ-band scales with √days_held
        today_old = datetime.date(2024, 6, 26)
        hs_old = _hs(today_old, entry_price=100.0, realized=0.02, days_ago=25)
        today_new = datetime.date(2024, 6, 2)
        hs_new = _hs(today_new, entry_price=100.0, realized=0.02, days_ago=1)
        # 2σ × 0.02 × √25 = 20% vs 2σ × 0.02 × √1 = 4%
        # At -10% loss:
        #   25-day position: σ-band 20%, -10% < 20% → no fire
        #   1-day position:  σ-band 4%,  -10% > 4%  → fire
        sig_old = check_stop_loss(
            current_price=90.0, state=hs_old,
            stop_pct=0.0, stop_n_sigma=2.0, today=today_old,
        )
        sig_new = check_stop_loss(
            current_price=90.0, state=hs_new,
            stop_pct=0.0, stop_n_sigma=2.0, today=today_new,
        )
        assert sig_old.should_exit is False, "25-day σ-band ≈ 20% > 10% loss"
        assert sig_new.should_exit is True, "1-day σ-band ≈ 4% < 10% loss"


# ─────────────────────────────────────────────────────────────────────────
# 3) check_single_day_loss now uses _resolve_daily_sigma — realized fallback
# ─────────────────────────────────────────────────────────────────────────

class TestSDLRealizedFallback:
    """SDL σ-aware path now works without NGB — realized vol fallback."""

    def test_realized_fallback_fires_when_ngb_off(self):
        # NGB OFF: state.sigma=None. Realized daily σ = 0.02.
        # SDL threshold = 2.5 × 0.02 = 5%. At -7% drop → fire.
        hs = _hs(entry_price=100.0, sigma=None, realized=0.02)
        hs.prev_close = 100.0
        sig = check_single_day_loss(
            current_price=93.0, state=hs,
            sdl_pct=0.0, sdl_n_sigma=2.5,
        )
        assert sig.should_exit is True
        assert sig.exit_type == "single_day_loss"

    def test_realized_fallback_no_fire_on_noise_day(self):
        # Same setup, noise drop of -3% < 5% threshold → no fire
        hs = _hs(entry_price=100.0, sigma=None, realized=0.02)
        hs.prev_close = 100.0
        sig = check_single_day_loss(
            current_price=97.0, state=hs,
            sdl_pct=0.0, sdl_n_sigma=2.5,
        )
        assert sig.should_exit is False


# ─────────────────────────────────────────────────────────────────────────
# 4) Audit regression guard — Fix #0a revive
# ─────────────────────────────────────────────────────────────────────────

class TestAuditP6aRegression:
    """AUDIT REGRESSION GUARD (§5.13.3) — Fix #0a revive.

    Pins the invariant that σ-aware stop_loss is NOT dead code in
    production. The realized-vol fallback must produce a non-None daily
    σ when NGBoost is OFF (the production config since AUDIT_2026-05-09).
    """

    def test_no_dead_code_without_ngb(self):
        """With NGB σ=None but realized_sigma_daily set, σ-aware threshold
        MUST compute (not silently revert to legacy stop_pct=0)."""
        today = datetime.date(2024, 6, 11)
        hs = _hs(today, entry_price=100.0, sigma=None, realized=0.025, days_ago=10)
        # stop_pct=0 (disabled), stop_n_sigma=2.0 — pure σ mode
        # Expected threshold: 2 × 0.025 × √10 ≈ 15.8%
        # Loss -20% should fire; -10% should NOT
        sig_fire = check_stop_loss(
            current_price=80.0, state=hs,
            stop_pct=0.0, stop_n_sigma=2.0, today=today,
        )
        sig_hold = check_stop_loss(
            current_price=90.0, state=hs,
            stop_pct=0.0, stop_n_sigma=2.0, today=today,
        )
        assert sig_fire.should_exit is True, \
            "σ-aware MUST fire in revived config — was dead code pre-2026-05-10"
        assert sig_hold.should_exit is False, \
            "σ-aware threshold correctly above moderate drops"

    def test_both_modes_disabled_no_fire(self):
        # Sanity: when both stop_pct=0 AND stop_n_sigma=0, no exit ever fires
        today = datetime.date(2024, 6, 6)
        hs = _hs(today, entry_price=100.0, realized=0.05, days_ago=5)
        sig = check_stop_loss(
            current_price=10.0, state=hs,  # -90% loss
            stop_pct=0.0, stop_n_sigma=0.0, today=today,
        )
        assert sig.should_exit is False
