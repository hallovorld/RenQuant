"""Tests for the QP min_share_floor (2026-05-17 EQIX-class fix).

Pre-fix: any candidate whose share price exceeded the QP's dollar budget
(target_w × NAV) had `_shares_from_dw` return 0 → silently dropped at the
`if shares <= 0: continue` gate. For a $10k account this blocks EQIX
($1059/share), BKNG ($5k), NVR ($8k), etc. entirely — biasing the
strategy toward low-price names.

Fix: when shares=0 BUT (1-share weight ∈ [floor, ceiling]), allow buying
1 share. Defaults: floor=5%, ceiling=15% (avoid blowing the position cap
on a $1500 share for a $10k acct).

Invariant: this only fires on BUY intent (dw > 0) and never on SELL.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

TASKS_PATH = REPO_ROOT / "backtesting/renquant_104/kernel/portfolio_qp/tasks.py"
TASKS_SRC = TASKS_PATH.read_text()


class TestQpMinShareFloor:
    """2026-05-17 EQIX-class fix — small-acct + high-price stocks."""

    def test_fix_tag_present(self):
        assert "QP_MIN_SHARE_FLOOR" in TASKS_SRC
        assert "min_share_floor for high-price stocks (EQIX/META class)" in TASKS_SRC

    def test_env_carries_floor_and_ceiling(self):
        assert "min_share_floor_pct=" in TASKS_SRC
        assert "min_share_ceiling_pct=" in TASKS_SRC
        assert 'cfg.get("qp_min_share_floor_pct"' in TASKS_SRC
        assert 'cfg.get("qp_min_share_ceiling_pct"' in TASKS_SRC

    def test_only_fires_on_buy_intent(self):
        """dw > 0 guard prevents this from firing on accidental sell paths."""
        assert "shares <= 0 and dw > 0" in TASKS_SRC, \
            "min_share_floor must only fire on positive dw (BUY intent)"

    def test_ceiling_caps_overallocation(self):
        snippet_start = TASKS_SRC.index("QP_MIN_SHARE_FLOOR")
        nearby = TASKS_SRC[snippet_start - 1000: snippet_start + 600]
        # Ceiling check appears
        assert "one_share_pct <= ceiling" in nearby, \
            "ceiling must prevent buying 1 share when it'd exceed max_position cap"
        assert "floor <= one_share_pct" in nearby

    def test_defaults_are_safe(self):
        """floor=0.05 (5%) ceiling=0.15 (15%) — defensible defaults."""
        assert '0.05' in TASKS_SRC.split("qp_min_share_floor_pct")[1][:200], \
            "default floor should be 5%"
        assert '0.15' in TASKS_SRC.split("qp_min_share_ceiling_pct")[1][:200], \
            "default ceiling should be 15%"

    def test_disable_via_floor_zero(self):
        """Setting floor=0 must disable the feature (regression guard)."""
        snippet_start = TASKS_SRC.index("QP_MIN_SHARE_FLOOR")
        nearby = TASKS_SRC[snippet_start - 600: snippet_start + 200]
        assert "if floor > 0" in nearby, \
            "floor=0 must skip the min-share path (disable knob)"


class TestDeepDrawdownVetoDisabled:
    """2026-05-17: DDV disabled globally per HXZ 2020 (RFS) Replicating
    Anomalies finding that distress/loser anomaly fails to replicate
    in modern data. Config retained for regime-conditional re-enable."""

    def test_disabled_in_golden(self):
        import json
        cfg_path = REPO_ROOT / "backtesting/renquant_104/strategy_config.golden.json"
        c = json.loads(cfg_path.read_text())
        ddv = c["ranking"]["buy_quality_gates"]["deep_drawdown_veto"]
        assert ddv["enabled"] is False, "DDV should be disabled in golden"
        assert "_disable_reason_2026-05-17" in ddv, \
            "disable reason must be documented inline for future audit"
        assert "Hou-Xue-Zhang 2020" in ddv["_disable_reason_2026-05-17"]

    def test_disabled_in_live_config(self):
        import json
        cfg_path = REPO_ROOT / "backtesting/renquant_104/strategy_config.json"
        c = json.loads(cfg_path.read_text())
        ddv = c["ranking"]["buy_quality_gates"]["deep_drawdown_veto"]
        assert ddv["enabled"] is False
