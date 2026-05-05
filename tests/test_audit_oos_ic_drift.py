"""Tests for scripts/audit_oos_ic_drift.py — walk-forward IC drift detection."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _load_module():
    path = REPO / "scripts" / "audit_oos_ic_drift.py"
    spec = importlib.util.spec_from_file_location("audit_oos_ic_drift", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestDriftDetection(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _load_module()

    def test_stable_ic_few_drift_flags(self):
        """Stable IC ≈ +0.04 with small noise → few drift flags at 2σ.

        With Gaussian noise around the rolling mean, ~2.3% of points fall
        below mean − 2σ purely by chance. Over 30 points expect ≤ 2.
        """
        rng = np.random.default_rng(0)
        n = 30
        df = pd.DataFrame({
            "run_date": pd.date_range("2025-01-01", periods=n, freq="W"),
            "oos_mean_ic": rng.normal(0.04, 0.005, n),
            "n_tickers": [100] * n,
            "n_features": [27] * n,
        })
        out = self.mod.compute_drift_signals(df, window_size=12, drift_sigma=2.0)
        # 2σ threshold → expect very few false positives on stable data
        self.assertLessEqual(out["is_drift"].sum(), 3)

    def test_sharp_drop_flagged(self):
        """A sudden drop should trigger drift flag."""
        n = 30
        ic = [0.04] * 20 + [0.01] * 10
        df = pd.DataFrame({
            "run_date": pd.date_range("2025-01-01", periods=n, freq="W"),
            "oos_mean_ic": ic,
            "n_tickers": [100] * n,
            "n_features": [27] * n,
        })
        out = self.mod.compute_drift_signals(df, window_size=12, drift_sigma=1.0)
        # First few of the drop should be flagged
        self.assertGreater(out["is_drift"].sum(), 0)

    def test_load_training_runs_missing_db_raises_loud(self):
        """Missing DB raises (no silent empty frame masking the bug)."""
        with self.assertRaises(Exception):
            self.mod.load_training_runs(Path("/nowhere/nope.db"))


class TestCli(unittest.TestCase):
    def test_module_imports_clean(self):
        mod = _load_module()
        self.assertTrue(callable(getattr(mod, "main", None)))
        self.assertTrue(callable(getattr(mod, "compute_drift_signals", None)))
        self.assertTrue(callable(getattr(mod, "load_training_runs", None)))


if __name__ == "__main__":
    unittest.main()
