"""Regression test for Bug #21 (FEATURE-DIAG-DTYPE, 2026-04-26 round-7).

Stage C-3 v2 hourly transformer crashed at FeatureDiagnosticTask:
    File "pp_panel_training.py", line 1281, in run
        rho, _ = spearmanr(p, y)
      ...
    DTypePromotionError: The DType <class 'numpy.dtypes.DateTime64DType'>
    could not be promoted by <class 'numpy.dtypes.Float64DType'>.

Root cause: a non-numeric column (datetime, object) leaked into
ctx.feature_cols → spearmanr(p, y) called with a DateTime64 array.
The HOURLY_RES_FEATURE_COLS canonical list is clean, but defensive
filtering at the diagnostic site is the right belt-and-suspenders.

Fix: filter feature_cols to numeric dtypes upfront in
FeatureDiagnosticTask (and also coerce to float at the spearmanr call
as defense-in-depth). Non-numeric columns get a WARN log + skipped
diagnostic; the rest of the pipeline proceeds normally.

This test sets up a minimal panel with a datetime column accidentally
in feature_cols and asserts:
  1. No exception (was DTypePromotionError pre-fix).
  2. The numeric features still get diagnostics.
  3. The datetime feature is logged as skipped + dropped.
"""
from __future__ import annotations

import datetime
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from training_panel.context import PanelTrainingContext   # noqa: E402
from training_panel.pp_panel_training import FeatureDiagnosticTask  # noqa: E402


def _build_panel_with_datetime_feature() -> pd.DataFrame:
    """3 dates × 5 tickers panel where one feature_col is accidentally
    a datetime column (the bug condition)."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    tickers = ["A", "B", "C", "D", "E"]
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "date":               d,
                "ticker":             t,
                "label":              float(rng.normal()),
                "good_feat":          float(rng.normal()),
                "bad_datetime_feat":  d,            # ← datetime64 → spearmanr crashes
                "bad_object_feat":    f"obj-{t}",   # ← object dtype → also bad
            })
    return pd.DataFrame(rows)


def _ctx(panel: pd.DataFrame, feature_cols: list[str]) -> PanelTrainingContext:
    ctx = PanelTrainingContext(config={})
    ctx.panel = panel
    ctx.feature_cols = feature_cols
    return ctx


class TestBug21DtypeDefense:
    def test_no_crash_on_datetime_feature_col(self):
        """Pre-fix: DTypePromotionError. Post-fix: WARN + skip."""
        panel = _build_panel_with_datetime_feature()
        ctx = _ctx(panel, ["good_feat", "bad_datetime_feat"])
        FeatureDiagnosticTask().run(ctx)   # MUST NOT raise

    def test_numeric_feature_still_gets_diagnostic(self):
        """The good float column must still appear in feature_diagnostics."""
        panel = _build_panel_with_datetime_feature()
        ctx = _ctx(panel, ["good_feat", "bad_datetime_feat"])
        FeatureDiagnosticTask().run(ctx)
        feat_names = {row["col"] for row in ctx.feature_diagnostics}
        assert "good_feat" in feat_names

    def test_non_numeric_feature_excluded_from_diagnostics(self):
        panel = _build_panel_with_datetime_feature()
        ctx = _ctx(panel, ["good_feat", "bad_datetime_feat", "bad_object_feat"])
        FeatureDiagnosticTask().run(ctx)
        feat_names = {row["col"] for row in ctx.feature_diagnostics}
        assert "bad_datetime_feat" not in feat_names
        assert "bad_object_feat" not in feat_names

    def test_warning_logged_for_skipped_columns(self, caplog):
        panel = _build_panel_with_datetime_feature()
        ctx = _ctx(panel, ["good_feat", "bad_datetime_feat", "bad_object_feat"])
        with caplog.at_level(logging.WARNING):
            FeatureDiagnosticTask().run(ctx)
        msgs = " ".join(r.message for r in caplog.records)
        assert "skipping" in msgs.lower() or "non-numeric" in msgs.lower()
        # Both bad columns mentioned
        assert "bad_datetime_feat" in msgs
        assert "bad_object_feat" in msgs

    def test_bool_dtype_treated_as_non_numeric(self):
        """Bool columns should be skipped — spearmanr on a constant bool
        column would return NaN anyway, and treating them as numeric
        would require special-casing all-True/all-False rows."""
        rng = np.random.default_rng(0)
        dates = pd.date_range("2026-01-01", periods=3, freq="D")
        rows = []
        for d in dates:
            for t in ["A", "B", "C", "D", "E"]:
                rows.append({
                    "date":      d,
                    "ticker":    t,
                    "label":     float(rng.normal()),
                    "good_feat": float(rng.normal()),
                    "bool_feat": bool(rng.integers(0, 2)),
                })
        panel = pd.DataFrame(rows)
        ctx = _ctx(panel, ["good_feat", "bool_feat"])
        FeatureDiagnosticTask().run(ctx)
        feat_names = {row["col"] for row in ctx.feature_diagnostics}
        assert "good_feat" in feat_names
        assert "bool_feat" not in feat_names

    def test_pure_numeric_panel_unaffected(self):
        """Healthy panel — all features numeric — gets full diagnostic."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        rows = []
        for d in dates:
            for t in ["A", "B", "C", "D", "E", "F"]:
                rows.append({
                    "date":   d,
                    "ticker": t,
                    "label":  float(rng.normal()),
                    "f1":     float(rng.normal()),
                    "f2":     float(rng.normal()),
                    "f3":     float(rng.normal()),
                })
        panel = pd.DataFrame(rows)
        ctx = _ctx(panel, ["f1", "f2", "f3"])
        FeatureDiagnosticTask().run(ctx)
        feat_names = {row["col"] for row in ctx.feature_diagnostics}
        assert feat_names == {"f1", "f2", "f3"}

    def test_missing_feature_col_in_panel_skipped_silently(self):
        """If feature_cols lists a column NOT in panel.columns, skip it
        cleanly (don't KeyError). Common when feature_cols is stale."""
        panel = _build_panel_with_datetime_feature()
        ctx = _ctx(panel, ["good_feat", "ghost_feature_does_not_exist"])
        FeatureDiagnosticTask().run(ctx)   # must not KeyError
        feat_names = {row["col"] for row in ctx.feature_diagnostics}
        assert "ghost_feature_does_not_exist" not in feat_names
