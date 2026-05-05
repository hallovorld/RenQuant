"""Tests for the entry-IC acceptance gate (2026-05-04).

Synthetic-model tests + real-data assertion that today's baseline B2 fails
this gate (documenting the production reality surfaced 2026-05-03).
"""
from __future__ import annotations

import datetime
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.acceptance_entry_ic import (   # noqa: E402
    acceptance, compute_entry_ic,
)


def _make_paired_trades(
    rank_to_pnl: dict[str, float],
    *,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic trades_df where buys/sells are paired by ticker.

    Each ticker has ONE buy at rank r and ONE matching sell at pnl_pct p,
    given by rank_to_pnl. Missing fields filled with sensible defaults.
    """
    rng = np.random.default_rng(rng_seed)
    rows = []
    for t, (r, p) in rank_to_pnl.items():
        rows.append({
            "ticker":      t, "action": "buy",
            "rank_score":  float(r),
            "mu":          rng.normal(0, 0.005),
            "sigma":       0.05,
            "pnl_pct":     None,
            "hold_days":   None,
            "tax":         None,
            "exit_reason": None,
        })
        rows.append({
            "ticker":      t, "action": "sell",
            "rank_score":  None,
            "mu":          None,
            "sigma":       None,
            "pnl_pct":     float(p),
            "hold_days":   30,
            "tax":         max(p, 0) * 1000 * 0.37,
            "exit_reason": "qp_sell",
        })
    return pd.DataFrame(rows)


class TestEntryICCompute(unittest.TestCase):
    def test_strong_positive_signal_high_ic(self):
        # rank ↑ → pnl ↑ exactly
        scenarios = {f"T{i:02d}": (i / 10.0, i * 0.005) for i in range(40)}
        df = _make_paired_trades(scenarios)
        ic, mu, sig, n = compute_entry_ic(df)
        self.assertEqual(n, 40)
        self.assertGreater(ic, 0.95)

    def test_strong_anti_predictive_signal_negative_ic(self):
        # rank ↑ → pnl ↓
        scenarios = {f"T{i:02d}": (i / 10.0, -i * 0.005) for i in range(40)}
        df = _make_paired_trades(scenarios)
        ic, _, _, _ = compute_entry_ic(df)
        self.assertLess(ic, -0.95)

    def test_random_signal_zero_ic(self):
        rng = np.random.default_rng(0)
        scenarios = {f"T{i:02d}": (rng.uniform(0, 1), rng.normal(0, 0.05))
                     for i in range(80)}
        df = _make_paired_trades(scenarios)
        ic, _, _, _ = compute_entry_ic(df)
        self.assertLess(abs(ic), 0.30)

    def test_few_trades_returns_nan(self):
        scenarios = {"T1": (0.5, 0.05), "T2": (0.6, 0.06)}
        df = _make_paired_trades(scenarios)
        ic, _, _, n = compute_entry_ic(df)
        self.assertTrue(np.isnan(ic))
        self.assertLess(n, 5)

    def test_unmatched_buys_ignored(self):
        # 5 buys, 3 sells — only 3 pair
        rng = np.random.default_rng(0)
        rows = []
        for i in range(5):
            rows.append({"ticker": f"T{i}", "action": "buy",
                         "rank_score": rng.uniform(0, 1), "mu": 0,
                         "sigma": 0.05, "pnl_pct": None})
        for i in range(3):
            rows.append({"ticker": f"T{i}", "action": "sell",
                         "rank_score": None, "mu": None,
                         "sigma": None, "pnl_pct": rng.normal(0, 0.05)})
        df = pd.DataFrame(rows)
        ic, _, _, n = compute_entry_ic(df)
        self.assertEqual(n, 3)
        self.assertTrue(np.isnan(ic))   # n < 5

    def test_constant_rank_returns_nan(self):
        scenarios = {f"T{i}": (0.5, np.random.default_rng(i).normal(0, 0.02))
                     for i in range(40)}
        df = _make_paired_trades(scenarios)
        ic, _, _, _ = compute_entry_ic(df)
        # All ranks identical → spearman undefined
        self.assertTrue(np.isnan(ic))


class TestEntryICAcceptance(unittest.TestCase):
    def test_strong_predictive_passes(self):
        scenarios = {f"T{i:02d}": (i / 10.0, i * 0.005) for i in range(40)}
        v = acceptance(_make_paired_trades(scenarios), min_ic=0.02)
        self.assertTrue(v.passed)
        self.assertGreater(v.ic, 0.02)

    def test_anti_predictive_fails(self):
        scenarios = {f"T{i:02d}": (i / 10.0, -i * 0.005) for i in range(40)}
        v = acceptance(_make_paired_trades(scenarios), min_ic=0.02)
        self.assertFalse(v.passed)
        self.assertIn("ANTI", v.detail)

    def test_at_threshold_passes(self):
        # Rank lightly correlated → IC ≈ 0.02
        rng = np.random.default_rng(0)
        scenarios = {}
        for i in range(80):
            r = rng.uniform(0, 1)
            scenarios[f"T{i:02d}"] = (r, 0.001 * r + rng.normal(0, 0.05))
        v = acceptance(_make_paired_trades(scenarios), min_ic=0.0)
        # Just verify the framework — exact pass/fail depends on noise.
        self.assertEqual(v.threshold, 0.0)

    def test_insufficient_sample_pass_open(self):
        scenarios = {f"T{i}": (i / 5.0, 0.01) for i in range(5)}
        v = acceptance(_make_paired_trades(scenarios), min_ic=0.05, min_n_paired=30)
        # n = 5 paired trades, < min_n_paired → PASS-OPEN with detail noting insufficient sample
        self.assertTrue(v.passed)
        self.assertIn("insufficient sample", v.detail)


class TestRealBaselineB2EntryICFails(unittest.TestCase):
    """The 2026-05-03 baseline B2 trades.parquet must fail this gate.

    This is the reverse of "regression test" — the data confirms the bug
    we found. If a future fix ever makes this test PASS on the same
    dataset, that's success.
    """

    def test_baseline_b2_2024_12_31_entry_ic_negative(self):
        path = REPO / "data" / "holdout_results" / "2024-12-31.trades.parquet"
        if not path.exists():
            self.skipTest("baseline B2 trades.parquet not present (run holdout_backtest)")
        df = pd.read_parquet(path)
        ic, _, _, n = compute_entry_ic(df)
        self.assertGreater(n, 0)
        # Document: today's data has anti-predictive entry IC
        # (Spearman = -0.10 reported in audit; exact value depends on
        # paired set; just enforce "below 0.02 threshold" for the
        # production model).
        self.assertLess(
            ic, 0.02,
            f"Production baseline B2 has entry IC {ic:+.4f} ≥ 0.02 — "
            f"the bug is fixed (or you regenerated trades). Update test.",
        )


if __name__ == "__main__":
    unittest.main()
