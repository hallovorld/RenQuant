"""BEARBranchTask soft gate (Kaminski-Lo 2014 fix, 2026-05-14).

Pre-fix: ANY bar with regime=BEAR triggered ctx.bear_only=True → strategy
went defensives-only. With the 2026-05-14 detector improvements (MA50
direction-aware Hurst, HMM), bull windows get 5-11% transient BEAR
mis-labels — each one caused a full defensive switch and catastrophic
performance loss (Panel A Q11 BULL_STRONG −27pt, Q15 −25pt, Q10 −11pt).

Post-fix: bear_only only fires when:
  (a) NOT in_transition (regime has settled past the 3-bar cooldown)
  (b) confidence ≥ regime.bear_branch_min_confidence (default 0.60)

These tests pin the soft-gate invariant + legacy-mode escape hatch.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.pipeline.task_gates import BEARBranchTask  # noqa: E402


def _ctx(regime="BEAR", confidence=1.0, in_transition=False, *,
         legacy=False, min_conf=None):
    rs = SimpleNamespace(in_transition=in_transition)
    regime_cfg = {}
    if legacy:
        regime_cfg["bear_branch_legacy_mode"] = True
    if min_conf is not None:
        regime_cfg["bear_branch_min_confidence"] = min_conf
    return SimpleNamespace(
        regime=regime, confidence=confidence, regime_state=rs,
        config={"regime": regime_cfg}, counters={}, bear_only=False,
    )


class TestBEARBranchSoftGate:
    """The Panel A Q11 catastrophe fix: don't slam strategy into defensives
    on transient/low-confidence BEAR mis-labels."""

    def test_non_bear_regime_no_op(self):
        ctx = _ctx(regime="BULL_CALM")
        BEARBranchTask().run(ctx)
        assert ctx.bear_only is False, "non-BEAR regime must never set bear_only"

    def test_bear_high_confidence_fires(self):
        """Real BEAR (confidence 1.0) → defensives-only."""
        ctx = _ctx(regime="BEAR", confidence=1.0, in_transition=False)
        BEARBranchTask().run(ctx)
        assert ctx.bear_only is True

    def test_bear_in_transition_does_not_fire(self):
        """3-bar transition cooldown: don't react to a regime that JUST
        changed. CUSUM/HMM may be mid-noise — wait for settle."""
        ctx = _ctx(regime="BEAR", confidence=1.0, in_transition=True)
        BEARBranchTask().run(ctx)
        assert ctx.bear_only is False, (
            "in_transition=True must veto bear_only even with high confidence"
        )
        assert ctx.counters.get("bear_branch_skipped_transition") == 1

    def test_bear_low_confidence_does_not_fire(self):
        """Transient BEAR mis-label from direction-aware Hurst typically
        has confidence ≈ 0.5 (in_transition path). After cooldown ends,
        low confidence still blocks bear_only."""
        ctx = _ctx(regime="BEAR", confidence=0.50, in_transition=False)
        BEARBranchTask().run(ctx)
        assert ctx.bear_only is False
        assert ctx.counters.get("bear_branch_skipped_lowconf") == 1

    def test_bear_just_above_threshold_fires(self):
        ctx = _ctx(regime="BEAR", confidence=0.65, in_transition=False, min_conf=0.60)
        BEARBranchTask().run(ctx)
        assert ctx.bear_only is True

    def test_bear_just_below_threshold_blocked(self):
        ctx = _ctx(regime="BEAR", confidence=0.55, in_transition=False, min_conf=0.60)
        BEARBranchTask().run(ctx)
        assert ctx.bear_only is False

    def test_bear_nan_confidence_blocked(self):
        """Per audit fix G-1: non-finite confidence routes safely (don't
        slam defensives on a non-classifiable signal)."""
        ctx = _ctx(regime="BEAR", confidence=float("nan"), in_transition=False)
        BEARBranchTask().run(ctx)
        assert ctx.bear_only is False, "NaN confidence must not trigger bear_only"
        assert ctx.counters.get("bear_branch_skipped_lowconf") == 1

    def test_bear_none_confidence_blocked(self):
        ctx = _ctx(regime="BEAR", confidence=None, in_transition=False)
        BEARBranchTask().run(ctx)
        assert ctx.bear_only is False

    def test_legacy_mode_restores_hard_switch(self):
        """Escape hatch: bear_branch_legacy_mode=True restores pre-2026-05-14
        behavior — fire on ANY bar where regime=BEAR, no confidence gate."""
        ctx = _ctx(regime="BEAR", confidence=0.30, in_transition=True, legacy=True)
        BEARBranchTask().run(ctx)
        assert ctx.bear_only is True, (
            "legacy mode must restore hard-switch even on low conf + transition"
        )

    def test_custom_min_confidence_threshold(self):
        """High threshold — only the strongest BEAR signals fire."""
        ctx = _ctx(regime="BEAR", confidence=0.75, min_conf=0.90)
        BEARBranchTask().run(ctx)
        assert ctx.bear_only is False
        # And same conf with default threshold (0.60) would fire
        ctx2 = _ctx(regime="BEAR", confidence=0.75)
        BEARBranchTask().run(ctx2)
        assert ctx2.bear_only is True
