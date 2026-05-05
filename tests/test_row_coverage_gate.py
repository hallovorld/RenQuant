"""Tests for row-coverage gate (kernel/row_coverage.py).

2026-05-04 P0 — proper structural fix for NaN-leaf collapse. Verifies:
  1. Helper drops rows below min_pct, keeps rows above.
  2. Disabled (min_pct=0) returns input unchanged.
  3. Empty / no-feature scenarios handled cleanly.
  4. config-knob plumbing works.
  5. BuildPanelTask source-level wiring exists.
  6. BuildFeatureMatrixTask source-level wiring exists.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.row_coverage import (  # noqa: E402
    coverage_from_config, filter_by_coverage,
)


def _make_panel(n_rows=100, n_features=10, nan_pct_per_row=None, seed=0):
    rng = np.random.default_rng(seed)
    feats = [f"f{i}" for i in range(n_features)]
    data = rng.normal(0, 1, (n_rows, n_features))
    if nan_pct_per_row is not None:
        # nan_pct_per_row[i] = fraction of features to NaN out for row i
        for i, pct in enumerate(nan_pct_per_row):
            n_nan = int(round(pct * n_features))
            cols = rng.choice(n_features, n_nan, replace=False)
            data[i, cols] = np.nan
    df = pd.DataFrame(data, columns=feats)
    df["date"] = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    df["ticker"] = [f"T{i % 10}" for i in range(n_rows)]
    return df, feats


class TestFilterByCoverage(unittest.TestCase):
    def test_disabled_returns_input(self):
        df, feats = _make_panel(50, 10)
        out, stats = filter_by_coverage(df, feats, 0.0)
        self.assertEqual(len(out), 50)
        self.assertTrue(stats.get("skipped"))
        self.assertEqual(stats["n_dropped"], 0)

    def test_drops_low_coverage_rows(self):
        # 50 rows, half have 80% NaN (= 20% coverage), half have 0% NaN
        nan_pcts = [0.8] * 25 + [0.0] * 25
        df, feats = _make_panel(50, 10, nan_pct_per_row=nan_pcts)
        out, stats = filter_by_coverage(df, feats, min_pct=0.5)
        self.assertEqual(stats["n_in"], 50)
        self.assertEqual(stats["n_out"], 25)
        self.assertEqual(stats["n_dropped"], 25)
        self.assertAlmostEqual(stats["pct_dropped"], 0.5)

    def test_keeps_all_when_full_coverage(self):
        df, feats = _make_panel(50, 10)
        out, stats = filter_by_coverage(df, feats, min_pct=1.0)
        self.assertEqual(stats["n_dropped"], 0)
        self.assertEqual(stats["n_out"], 50)

    def test_at_threshold_keeps(self):
        # Row with exactly 50% coverage: 5 NaN of 10
        nan_pcts = [0.5] * 50
        df, feats = _make_panel(50, 10, nan_pct_per_row=nan_pcts)
        # min_pct=0.5: coverage 0.5 ≥ 0.5 → KEEP all
        out, stats = filter_by_coverage(df, feats, min_pct=0.5)
        self.assertEqual(stats["n_dropped"], 0)
        # min_pct=0.51: 0.5 < 0.51 → DROP all
        out2, stats2 = filter_by_coverage(df, feats, min_pct=0.51)
        self.assertEqual(stats2["n_dropped"], 50)

    def test_empty_features_returns_input(self):
        df, feats = _make_panel(50, 10)
        out, stats = filter_by_coverage(df, [], min_pct=0.5)
        self.assertEqual(len(out), 50)
        self.assertTrue(stats.get("skipped"))

    def test_missing_feature_cols_logged_and_continued(self):
        df, feats = _make_panel(50, 10)
        out, stats = filter_by_coverage(
            df, feats + ["NEVER_EXISTED"], min_pct=0.5,
        )
        # Should still process the available cols
        self.assertEqual(stats["n_in"], 50)

    def test_min_pct_above_one_rejected(self):
        df, feats = _make_panel(50, 10)
        with self.assertRaises(ValueError):
            filter_by_coverage(df, feats, min_pct=1.5)

    def test_index_reset_in_output(self):
        nan_pcts = [0.8 if i % 2 == 0 else 0.0 for i in range(20)]
        df, feats = _make_panel(20, 10, nan_pct_per_row=nan_pcts)
        out, _ = filter_by_coverage(df, feats, min_pct=0.5)
        # Output index should be 0..n-1, not the original sparse indices.
        # Default behaviour (training: long-form panel, integer index OK).
        self.assertEqual(list(out.index), list(range(len(out))))

    def test_preserve_index_when_requested(self):
        """2026-05-05 wl183 0-trade regression: inference matrices are
        ticker-indexed. A silent reset to int64 0..n-1 broke every
        downstream `scores.get(cand.ticker)` lookup → 0/N scored on every
        bar in production. preserve_index=True keeps the original ticker
        index intact through the filter."""
        df, feats = _make_panel(5, 10, nan_pct_per_row=[0.0, 0.8, 0.0, 0.8, 0.0])
        # Re-index by tickers, like build_inference_matrix produces.
        df = df.set_index(pd.Index(["AAPL", "MSFT", "NVDA", "TSLA", "META"]))
        out, _ = filter_by_coverage(df, feats, min_pct=0.5, preserve_index=True)
        # Surviving rows: indices 0, 2, 4 → AAPL, NVDA, META
        self.assertEqual(list(out.index), ["AAPL", "NVDA", "META"])
        self.assertEqual(out.index.dtype, object)  # NOT int64

    def test_preserve_index_default_false_back_compat(self):
        """The old API contract (reset to 0..n-1) is preserved for callers
        that don't opt into preserve_index — training paths still work."""
        df, feats = _make_panel(5, 10, nan_pct_per_row=[0.0, 0.8, 0.0, 0.8, 0.0])
        df = df.set_index(pd.Index(["AAPL", "MSFT", "NVDA", "TSLA", "META"]))
        out, _ = filter_by_coverage(df, feats, min_pct=0.5)  # default
        # Default reset → integer 0..n-1
        self.assertEqual(list(out.index), [0, 1, 2])

    def test_inference_caller_passes_preserve_index(self):
        """Source-level guard: RowCoverageGateTask (inference path) MUST
        call filter_by_coverage with preserve_index=True. Without it, the
        wl183 0/N lookup miss recurs."""
        path = (REPO / "backtesting" / "renquant_104" / "kernel"
                / "panel_pipeline" / "tasks_feature_matrix.py")
        src = path.read_text()
        idx_class = src.find("class RowCoverageGateTask")
        idx_next = src.find("class ", idx_class + 1)
        body = src[idx_class:idx_next] if idx_next > 0 else src[idx_class:]
        self.assertIn("preserve_index=True", body,
            "RowCoverageGateTask must call filter_by_coverage with "
            "preserve_index=True so X.index stays as ticker symbols")


class TestCoverageFromConfig(unittest.TestCase):
    def test_default_disabled(self):
        enabled, mp = coverage_from_config({})
        self.assertFalse(enabled)
        self.assertEqual(mp, 0.5)

    def test_explicit_config(self):
        cfg = {"panel_ltr": {"row_coverage": {"enabled": True, "min_pct": 0.7}}}
        enabled, mp = coverage_from_config(cfg)
        self.assertTrue(enabled)
        self.assertEqual(mp, 0.7)


class TestPanelTaskWiring(unittest.TestCase):
    """Source-level: BuildPanelTask references row_coverage filter."""

    def test_build_panel_task_imports_filter(self):
        path = (REPO / "backtesting" / "renquant_104"
                / "training_panel" / "pp_panel_training.py")
        src = path.read_text()
        idx_class = src.find("class BuildPanelTask")
        idx_next = src.find("class ", idx_class + 1)
        body = src[idx_class:idx_next] if idx_next > 0 else src[idx_class:]
        self.assertIn("filter_by_coverage", body,
                      "BuildPanelTask must call filter_by_coverage")
        self.assertIn("coverage_from_config", body)


class TestInferenceWiring(unittest.TestCase):
    """Source-level: row_coverage filter wired into the inference path.

    2026-05-04: BuildFeatureMatrixTask was split per CLAUDE.md §1c into
    BuildFeatureMatrixJob with 4 Tasks. The row-coverage filter now
    lives in `RowCoverageGateTask` in
    `kernel/panel_pipeline/tasks_feature_matrix.py`.
    """

    def test_row_coverage_gate_task_uses_filter(self):
        path = (REPO / "backtesting" / "renquant_104"
                / "kernel" / "panel_pipeline" / "tasks_feature_matrix.py")
        src = path.read_text()
        idx_class = src.find("class RowCoverageGateTask")
        idx_next = src.find("class DriftGuardTask", idx_class)
        body = src[idx_class:idx_next] if idx_next > 0 else src[idx_class:]
        self.assertIn("filter_by_coverage", body,
                      "RowCoverageGateTask must call filter_by_coverage")
        self.assertIn("coverage_from_config", body)


if __name__ == "__main__":
    unittest.main()
