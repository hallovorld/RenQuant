"""Tests for forward_fill_per_ticker (training_panel.imputation).

2026-05-04 — added with the user-spec "数据消失时 forward fill" but
gated by an explicit whitelist + max_gap_days cap so high-frequency
intraday features can't be silently ffill'd across days.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from training_panel.imputation import forward_fill_per_ticker  # noqa: E402


def _make(rows):
    """Build a small panel from rows of (ticker, date, feat1, feat2)."""
    df = pd.DataFrame(rows, columns=["ticker", "date", "feat1", "feat2"])
    df["date"] = pd.to_datetime(df["date"])
    return df


class TestForwardFillBasic(unittest.TestCase):
    def test_fills_within_ticker_only(self):
        df = _make([
            ("A", "2024-01-01", 1.0, 10.0),
            ("A", "2024-01-02", np.nan, 11.0),
            ("A", "2024-01-03", np.nan, 12.0),
            ("B", "2024-01-01", 5.0, 50.0),
            ("B", "2024-01-02", 6.0, 51.0),
        ])
        out = forward_fill_per_ticker(df, ["feat1"], max_gap_days=5)
        # A's feat1 should be filled forward from row 0
        a_rows = out[out["ticker"] == "A"].sort_values("date")
        self.assertEqual(list(a_rows["feat1"]), [1.0, 1.0, 1.0])
        # B unaffected (no NaN in feat1)
        b_rows = out[out["ticker"] == "B"].sort_values("date")
        self.assertEqual(list(b_rows["feat1"]), [5.0, 6.0])

    def test_does_not_cross_tickers(self):
        df = _make([
            ("A", "2024-01-01", 1.0, 10.0),
            ("B", "2024-01-02", np.nan, 20.0),  # B's first row is NaN
        ])
        out = forward_fill_per_ticker(df, ["feat1"], max_gap_days=5)
        b_row = out[out["ticker"] == "B"].iloc[0]
        # Must NOT pick up A's value 1.0 — different ticker
        self.assertTrue(np.isnan(b_row["feat1"]))

    def test_respects_max_gap(self):
        # 7 consecutive NaN — max_gap=3 fills first 3, leaves last 4 NaN
        rows = [("A", f"2024-01-{i:02d}", np.nan, 0.0) for i in range(2, 9)]
        rows.insert(0, ("A", "2024-01-01", 1.0, 0.0))
        df = _make(rows)
        out = forward_fill_per_ticker(df, ["feat1"], max_gap_days=3)
        a = out.sort_values("date")
        # Index 0: 1.0 (original), 1-3: 1.0 (ffilled), 4-7: NaN (over cap)
        vals = list(a["feat1"])
        self.assertEqual(vals[0], 1.0)
        for v in vals[1:4]:
            self.assertEqual(v, 1.0)
        for v in vals[4:]:
            self.assertTrue(np.isnan(v))

    def test_only_whitelisted_cols_filled(self):
        df = _make([
            ("A", "2024-01-01", 1.0, 10.0),
            ("A", "2024-01-02", np.nan, np.nan),
        ])
        out = forward_fill_per_ticker(df, ["feat1"], max_gap_days=5)
        # feat1 ffilled, feat2 NOT ffilled
        a = out[out["ticker"] == "A"].sort_values("date")
        self.assertEqual(list(a["feat1"]), [1.0, 1.0])
        # feat2 has NaN preserved (not in whitelist)
        self.assertTrue(np.isnan(a.iloc[1]["feat2"]))

    def test_empty_cols_no_op(self):
        df = _make([
            ("A", "2024-01-01", 1.0, 10.0),
            ("A", "2024-01-02", np.nan, np.nan),
        ])
        out = forward_fill_per_ticker(df, [], max_gap_days=5)
        # Same NaN preserved
        self.assertTrue(np.isnan(out.iloc[1]["feat1"]))
        self.assertTrue(np.isnan(out.iloc[1]["feat2"]))

    def test_unknown_col_silently_skipped(self):
        df = _make([
            ("A", "2024-01-01", 1.0, 10.0),
            ("A", "2024-01-02", np.nan, np.nan),
        ])
        out = forward_fill_per_ticker(df, ["nonexistent"], max_gap_days=5)
        self.assertTrue(np.isnan(out.iloc[1]["feat1"]))
        self.assertTrue(np.isnan(out.iloc[1]["feat2"]))

    def test_does_not_mutate_input(self):
        df = _make([
            ("A", "2024-01-01", 1.0, 10.0),
            ("A", "2024-01-02", np.nan, np.nan),
        ])
        before_nan_count = df["feat1"].isna().sum()
        _ = forward_fill_per_ticker(df, ["feat1"], max_gap_days=5)
        self.assertEqual(df["feat1"].isna().sum(), before_nan_count,
                         "input panel must not be mutated")

    def test_empty_panel_no_op(self):
        df = pd.DataFrame(columns=["ticker", "date", "feat1"])
        out = forward_fill_per_ticker(df, ["feat1"], max_gap_days=5)
        self.assertTrue(out.empty)

    def test_max_gap_zero_disables_fill(self):
        # max_gap_days=0 → ffill(limit=0) which by pandas convention
        # is a no-op (fills no NaN).
        df = _make([
            ("A", "2024-01-01", 1.0, 10.0),
            ("A", "2024-01-02", np.nan, np.nan),
        ])
        # Caller should pass an explicit ffill_max_gap_days=0 to disable.
        out = forward_fill_per_ticker(df, ["feat1"], max_gap_days=0)
        # With limit=0 pandas does NOT fill — safe
        self.assertTrue(np.isnan(out.iloc[1]["feat1"]))


class TestPanelTaskWiring(unittest.TestCase):
    """Source-level: BuildPanelTask reads imputation.ffill_cols config."""

    def test_build_panel_task_imports_ffill_helper(self):
        path = (REPO / "backtesting" / "renquant_104"
                / "training_panel" / "pp_panel_training.py")
        src = path.read_text()
        idx = src.find("class BuildPanelTask")
        idx_next = src.find("class ", idx + 1)
        body = src[idx:idx_next] if idx_next > 0 else src[idx:]
        self.assertIn("forward_fill_per_ticker", body)
        self.assertIn("ffill_cols", body)
        self.assertIn("ffill_max_gap_days", body)


if __name__ == "__main__":
    unittest.main()
