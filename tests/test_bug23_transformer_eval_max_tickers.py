"""Regression test for Bug #23 (TRANSFORMER-EVAL-MAX-TICKERS, 2026-04-26 round-7).

Stage C-3 v3 hourly transformer training crashed at:
    transformer_model.py:441 _build_date_groups
    ValueError: a group has 616 rows but max_tickers=604.

Root cause: the auto-bump (line ~781-789) considered only TRAIN group sizes.
The eval panel split happened to contain a larger date-group (likely a
date with more tickers liquid at hourly bars + the auto_eval_split
landed mid-watchlist-growth). max_tickers was set to train's max=604
but eval needed 616 → hard error from the audit T-1 guard.

Fix: take max across BOTH train AND eval group_sizes before bumping.

Test strategy: source-level contract — assert the fix considers both
splits. A full reproduction would require building a synthetic panel
big enough to hit max_tickers, which is heavy for a unit test.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSFORMER_PATH = REPO_ROOT / "backtesting/renquant_104/training_panel/transformer_model.py"
TRANSFORMER_SOURCE = TRANSFORMER_PATH.read_text()


class TestBug23TransformerEvalMaxTickers:
    def test_audit_tag_present(self):
        assert "Bug #23 fix" in TRANSFORMER_SOURCE

    def test_auto_bump_considers_eval_group_sizes(self):
        """Auto-bump must read eval_group_sizes, not just group_sizes."""
        # The fix takes max across train + eval before bumping.
        anchor = "Bug #23 fix"
        idx = TRANSFORMER_SOURCE.find(anchor)
        assert idx >= 0
        block = TRANSFORMER_SOURCE[idx:idx + 1500]
        # Both train (group_sizes) and eval (eval_group_sizes) must be
        # consulted in the bump decision.
        assert "eval_group_sizes" in block, (
            "auto-bump block must read eval_group_sizes — pre-fix only "
            "checked group_sizes (train)."
        )
        assert "group_sizes" in block

    def test_log_message_mentions_train_and_eval(self):
        """Log line must clarify it considered both splits."""
        anchor = "Bug #23 fix"
        idx = TRANSFORMER_SOURCE.find(anchor)
        block = TRANSFORMER_SOURCE[idx:idx + 1500]
        # The log message changed from "train data has..." to mention
        # both splits, so future log readers know what was considered.
        assert "train+eval" in block or "across both" in block.lower()

    def test_no_bare_train_only_check_in_bump_decision(self):
        """No more `if max_gs_train > p.max_tickers:` — that's the old
        train-only path."""
        # The old code did a bare train-only check; new code uses
        # `all_sizes` or equivalent. The variable `max_gs_train` should
        # be gone or only present in non-load-bearing context.
        anchor = "Bug #23 fix"
        idx = TRANSFORMER_SOURCE.find(anchor)
        block = TRANSFORMER_SOURCE[idx:idx + 1500]
        # The fix uses `max_gs` (not `max_gs_train`) and `all_sizes`.
        assert "all_sizes" in block, (
            "fix must aggregate sizes from both train + eval into a "
            "single max calculation"
        )


class TestBug23GuardStillRaises:
    """Regression for audit T-1: silent truncation must still be a
    hard error, not a silent slice. The bump fix is for the auto-cap
    only — _build_date_groups itself should still raise on overflow."""

    def test_build_date_groups_still_raises_on_overflow(self):
        """Synthetic panel where one group exceeds max_tickers — must raise."""
        import pandas as pd
        import sys
        sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))
        from training_panel.transformer_model import _build_date_groups

        # Build 2 dates × 5 tickers, with one group having 4 rows
        # (within max_tickers=4) and the other having 5 (exceeds).
        rows = []
        date1 = pd.Timestamp("2026-01-01")
        date2 = pd.Timestamp("2026-01-02")
        for i, t in enumerate(["A", "B", "C", "D"]):
            rows.append({"date": date1, "ticker": t, "f0": float(i), "label": 0.0})
        for i, t in enumerate(["A", "B", "C", "D", "E"]):  # 5 tickers!
            rows.append({"date": date2, "ticker": t, "f0": float(i), "label": 0.0})
        panel = pd.DataFrame(rows)
        group_sizes = [4, 5]
        import numpy as np

        try:
            _build_date_groups(
                panel, np.array(group_sizes, dtype=np.int64),
                ["f0"], "label", max_tickers=4,
            )
            raised = False
        except ValueError as exc:
            raised = True
            assert "max_tickers" in str(exc)

        assert raised, (
            "audit T-1 contract: oversized group must raise ValueError, "
            "not silent-truncate. Pre-T-1 this silently lost data."
        )
