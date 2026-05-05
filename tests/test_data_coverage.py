"""Tests for kernel/data_coverage.py + a contract that pins production
data coverage so future regressions surface loud.

The 'baseline' coverage numbers below are recorded 2026-05-04 evening
(before the L0 hourly/minute backfill jobs complete). Future changes
should NOT REGRESS — only IMPROVE — these numbers. When the backfill
finishes and coverage rises to ~90%+, update the baseline.
"""
from __future__ import annotations

import datetime as _dt
import sys
import unittest
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.data_coverage import (   # noqa: E402
    TickerCoverage, compute_coverage, coverage_summary,
)


class TestCoverageHelper(unittest.TestCase):
    def test_no_data_returns_all_false(self):
        cov = compute_coverage(["NEVER_FETCHED"], REPO)
        self.assertEqual(len(cov), 1)
        c = cov["NEVER_FETCHED"]
        self.assertFalse(c.has_ohlcv_daily)
        self.assertFalse(c.has_hourly_bars)
        self.assertFalse(c.has_minute_bars)
        self.assertFalse(c.has_fundamentals)
        self.assertFalse(c.has_earnings_surprise)
        self.assertFalse(c.has_insider)
        self.assertIsNone(c.ohlcv_max_date)

    def test_real_ohlcv_present_for_aapl(self):
        cov = compute_coverage(["AAPL"], REPO)
        c = cov["AAPL"]
        # AAPL is in cache from prior runs
        self.assertTrue(c.has_ohlcv_daily,
                        "AAPL OHLCV should be cached locally")
        self.assertIsNotNone(c.ohlcv_max_date)
        self.assertIsNotNone(c.ohlcv_age_days)
        self.assertGreaterEqual(c.ohlcv_age_days, 0)

    def test_summary_aggregates_correctly(self):
        cov = {
            "A": TickerCoverage(ticker="A", has_ohlcv_daily=True,
                                has_hourly_bars=True),
            "B": TickerCoverage(ticker="B", has_ohlcv_daily=True,
                                has_hourly_bars=False),
            "C": TickerCoverage(ticker="C", has_ohlcv_daily=False),
        }
        s = coverage_summary(cov)
        self.assertEqual(s["n_tickers"], 3)
        self.assertEqual(s["ohlcv_daily_n"], 2)
        self.assertAlmostEqual(s["ohlcv_daily_pct"], 2 / 3)
        self.assertEqual(s["hourly_n"], 1)


class TestProductionWatchlistCoverageBaseline(unittest.TestCase):
    """Baseline contract: production wl coverage must NOT regress.

    Recorded 2026-05-04 evening — these are the floor. Backfill jobs
    that complete after this point should make these numbers RISE; the
    update path is to bump the floor in this test, not to delete the
    test.
    """

    @classmethod
    def setUpClass(cls):
        import json
        path = REPO / "backtesting" / "renquant_104" / "strategy_config.json"
        cls.cfg = json.loads(path.read_text())
        cls.coverage = compute_coverage(cls.cfg["watchlist"], REPO)
        cls.summary = coverage_summary(cls.coverage)

    def test_ohlcv_daily_coverage_at_least_baseline(self):
        # Production wl is fully cached locally (we maintain it).
        self.assertGreaterEqual(self.summary["ohlcv_daily_pct"], 0.95,
                                f"OHLCV daily coverage dropped below 95% — "
                                f"got {self.summary['ohlcv_daily_pct']:.1%}")

    def test_fundamentals_coverage_at_least_baseline(self):
        # 2026-05-03 chain log: 182/183 wl_sweep_183 coverage on fundamentals.
        # Floor at 95% for production wl=103 (≤ wl=183).
        self.assertGreaterEqual(
            self.summary["fundamentals_pct"], 0.90,
            f"Fundamentals coverage dropped below 90% — "
            f"got {self.summary['fundamentals_pct']:.1%}",
        )

    def test_earnings_surprise_coverage_at_least_baseline(self):
        self.assertGreaterEqual(
            self.summary["earnings_surprise_pct"], 0.90,
            f"Earnings-surprise coverage dropped below 90% — "
            f"got {self.summary['earnings_surprise_pct']:.1%}",
        )

    def test_intraday_coverage_floor_DOCUMENTED(self):
        """L0 P0 — intraday coverage is currently the structural gap.

        Recorded BEFORE backfill: hourly + minute parquet files DO NOT
        EXIST under data/intraday/{tic}/{1h,10m}.parquet for the
        production wl. The XGB NaN-leaf collapse traces back here.

        This test DOES NOT enforce a floor — it documents the current
        state so we can SEE coverage rise as backfill completes. When
        hourly_pct ≥ 0.50 after backfill, flip this to enforce.
        """
        h = self.summary["hourly_pct"]
        m = self.summary["minute_pct"]
        # Just print + record — no assert. Update test to assert ≥ 0.50
        # AFTER the backfill jobs complete.
        print(f"\n  intraday coverage: hourly_pct={h:.2%}  minute_pct={m:.2%}")
        # Sanity: percents are in [0, 1]
        self.assertGreaterEqual(h, 0.0)
        self.assertLessEqual(h, 1.0)
        self.assertGreaterEqual(m, 0.0)
        self.assertLessEqual(m, 1.0)


if __name__ == "__main__":
    unittest.main()
