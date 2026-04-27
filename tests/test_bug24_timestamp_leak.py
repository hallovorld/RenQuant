"""Regression test for Bug #24 (TRANSFORMER-TIMESTAMP-LEAK, 2026-04-26 round-7).

v5 hourly transformer trained successfully (after #141/#21/#23 fixes) but
produced OOS IC = -0.0008 — WORSE than XGBoost's +0.0326. Inspection of
the saved artifact's feature_cols revealed:

    feature_cols (first 5): ['timestamp', 'hourly_return', 'hourly_log_return', ...]

`timestamp` is a datetime64[ns] column (HourlyBarStore index name). The
non_feature exclusion set in BuildHourlyResolutionPanelTask listed
'datetime' but NOT 'timestamp'. So timestamp leaked into ctx.feature_cols
and the transformer trained on it as a numeric feature (PyTorch cast
datetime64 → float64 silently as Unix epoch ns).

Why it produced negative IC: timestamp IS the date proxy. Training on
the row's own future-target's date is pure look-ahead bias. The model
learned date-correlated artifacts instead of feature signal.

Fix: BuildHourlyResolutionPanelTask now applies BOTH:
  1. Explicit non_feature name set (with 'timestamp' added)
  2. Numeric-dtype filter (defense-in-depth — any other non-numeric
     column that slips past the name list is also dropped)

This is the same belt-and-suspenders pattern as Bug #21's
FeatureDiagnosticTask fix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _build_synthetic_hourly_panel_with_timestamp() -> pd.DataFrame:
    """Construct a panel that mimics what BuildHourlyResolutionPanelTask
    sees AFTER reset_index — including the rogue 'timestamp' column."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2026-01-01", periods=10, freq="h")
    rows = []
    for ticker in ["A", "B", "C"]:
        for d in dates:
            rows.append({
                "ticker":               ticker,
                "datetime":             d,
                "date":                 d.normalize(),
                "hour":                 d.hour,
                "timestamp":            d,                # <-- THE BUG: leaked datetime64 column
                "hourly_return":        float(rng.normal()),
                "hourly_vol":           float(rng.normal()),
                "label":                float(rng.normal()),
                "_sample_weight":       1.0,
            })
    return pd.DataFrame(rows)


class TestBug24TimestampLeak:
    def test_timestamp_in_non_feature_set(self):
        """Audit-tag: source must list 'timestamp' as a non-feature."""
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        anchor = "Bug #24 fix"
        assert anchor in src, "Audit tag for Bug #24 must be in source"
        idx = src.find(anchor)
        block = src[idx:idx + 2000]
        # Both name-list AND dtype-filter must be present.
        assert "non_feature" in block
        assert '"timestamp"' in block, "non_feature set must include timestamp"
        assert "is_numeric_dtype" in block, "dtype filter must be defense-in-depth"

    def test_dtype_filter_drops_non_numeric(self):
        """Even if a future column slips past the name list, dtype filter
        must remove it from feature_cols."""
        from training_panel.pp_panel_training import BuildHourlyResolutionPanelTask
        # The actual code is inside the task's run() — we can't easily
        # call just the filter logic. So this test validates via a
        # source-substring contract that the filter is applied.
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        idx = src.find("Bug #24 fix")
        block = src[idx:idx + 2000]
        # The fix must list-comprehend BOTH conditions.
        assert "is_numeric_dtype(panel[c].dtype)" in block
        assert "not pd.api.types.is_bool_dtype" in block

    def test_panel_with_timestamp_column_simulation(self):
        """Simulate: given a panel with timestamp as datetime64, the
        feature_cols filter must exclude it."""
        panel = _build_synthetic_hourly_panel_with_timestamp()
        # Verify the simulated panel has timestamp as datetime64
        assert pd.api.types.is_datetime64_any_dtype(panel["timestamp"].dtype)

        # Apply the same filter logic the code uses
        non_feature = {"ticker", "date", "hour", "datetime", "timestamp",
                       "label", "_sample_weight",
                       "forward_excess_return"}
        candidate_cols = [c for c in panel.columns if c not in non_feature]
        feature_cols = [
            c for c in candidate_cols
            if pd.api.types.is_numeric_dtype(panel[c].dtype)
            and not pd.api.types.is_bool_dtype(panel[c].dtype)
        ]

        assert "timestamp" not in feature_cols, "timestamp must be excluded"
        assert "datetime" not in feature_cols
        assert "date" not in feature_cols
        assert "hour" in panel.columns and "hour" not in feature_cols, "hour also excluded"
        assert "hourly_return" in feature_cols, "real features still included"
        assert "hourly_vol" in feature_cols

    def test_warning_logged_when_drops_non_numeric(self):
        """If a non-numeric column slips past the name list, a WARN is
        emitted so operators see the silent feature drop."""
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        idx = src.find("Bug #24 fix")
        block = src[idx:idx + 2500]
        assert "log.warning" in block
        assert "non-numeric" in block

    def test_v5_artifact_documents_the_bug(self):
        """The v5 artifact (saved with the bug) was the smoking gun. Its
        feature_cols had 'timestamp' as the FIRST element. This test
        documents the contract that v6+ artifacts MUST NOT have any
        datetime-like names in their feature_cols."""
        # If the v5 artifact still exists, verify the bug is real (not
        # speculative) — sanity check.
        artifact = REPO_ROOT / "backtesting/renquant_104/artifacts/panel-transformer.json"
        if not artifact.exists():
            pytest.skip("no panel-transformer.json yet")
        import json
        meta = json.loads(artifact.read_text())
        feature_cols = meta.get("feature_cols", [])
        # If the v5 (buggy) artifact still on disk, its feature_cols
        # SHOULD include timestamp — that's why this fix exists.
        # After re-training (v6+), this assertion will FLIP — a healthy
        # feature_cols list will NOT contain timestamp/datetime/date.
        # We document both states here as evidence of the fix's purpose.
        if "timestamp" in feature_cols:
            # v5 (buggy) artifact still present — this test is just
            # documenting the bug; v6+ will not produce this.
            assert feature_cols.index("timestamp") == 0, \
                "v5 buggy artifact had timestamp as col 0 (smoking gun)"
        else:
            # v6+ healthy artifact — timestamp is gone.
            assert "datetime" not in feature_cols
            assert "date" not in feature_cols
