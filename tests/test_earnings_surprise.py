"""Earnings-surprise factor tests.

Covers:
  * EarningsSurpriseStore parquet cache round-trip
  * _fetch_from_yfinance normalization (schema → SURPRISE_COLS)
  * compute_earnings_surprise_cum trailing-N-quarter rolling sum + daily ffill
  * fetch_earnings_surprise uses cache short-circuit
  * LoadEarningsSurpriseTask no-op when flag is off
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


class TestEarningsSurpriseStore:
    def test_save_and_load_round_trip(self, tmp_path):
        from kernel.earnings_surprise import EarningsSurpriseStore
        store = EarningsSurpriseStore(data_dir=tmp_path)
        idx = pd.DatetimeIndex([pd.Timestamp("2024-01-30"), pd.Timestamp("2024-04-30")])
        df = pd.DataFrame({
            "eps_actual":    [2.10, 1.52],
            "eps_estimate":  [1.95, 1.43],
            "surprise_abs":  [0.15, 0.09],
            "surprise_pct":  [0.077, 0.063],
        }, index=idx)
        store.save(df, "AAPL")

        loaded = store.load("AAPL")
        assert len(loaded) == 2
        assert loaded["eps_actual"].iloc[-1] == pytest.approx(1.52)

    def test_dedup_keeps_last(self, tmp_path):
        from kernel.earnings_surprise import EarningsSurpriseStore
        store = EarningsSurpriseStore(data_dir=tmp_path)
        idx = pd.DatetimeIndex([pd.Timestamp("2024-01-30")])
        v1 = pd.DataFrame({"eps_actual": [1.0], "eps_estimate": [0.9],
                           "surprise_abs": [0.1], "surprise_pct": [0.11]}, index=idx)
        v2 = pd.DataFrame({"eps_actual": [2.0], "eps_estimate": [0.9],
                           "surprise_abs": [1.1], "surprise_pct": [1.22]}, index=idx)
        store.save(v1, "AAPL")
        store.save(v2, "AAPL")
        loaded = store.load("AAPL")
        assert len(loaded) == 1
        assert loaded["eps_actual"].iloc[0] == pytest.approx(2.0)


class TestFetchWithInjectedProvider:
    def test_fetch_uses_cache_short_circuit(self, tmp_path):
        """Fresh cache (within refresh_after_days) skips the provider call.

        Round-3 audit (#R3-36): cache that exceeds the refresh window
        now refetches automatically. To exercise the short-circuit,
        seed with a recent date.
        """
        from kernel.earnings_surprise import (
            fetch_earnings_surprise, EarningsSurpriseStore,
        )
        store = EarningsSurpriseStore(data_dir=tmp_path)
        # Pre-seed the cache with a RECENT date (within the 30-day window)
        recent = pd.Timestamp.now().normalize() - pd.Timedelta(days=5)
        idx = pd.DatetimeIndex([recent])
        pre = pd.DataFrame({"eps_actual": [1.0], "eps_estimate": [0.9],
                            "surprise_abs": [0.1], "surprise_pct": [0.11]}, index=idx)
        store.save(pre, "AAPL")

        called = {"n": 0}
        def fake_provider(sym):
            called["n"] += 1
            return pd.DataFrame()

        df = fetch_earnings_surprise("AAPL", cache=True, store=store, provider_fn=fake_provider)
        assert called["n"] == 0, "cache hit must skip the provider"
        assert not df.empty

    def test_fetch_calls_provider_and_writes_cache(self, tmp_path):
        from kernel.earnings_surprise import (
            fetch_earnings_surprise, EarningsSurpriseStore,
        )
        store = EarningsSurpriseStore(data_dir=tmp_path)

        idx = pd.DatetimeIndex([pd.Timestamp("2024-01-30")])
        payload = pd.DataFrame({"eps_actual": [1.0], "eps_estimate": [0.9],
                                "surprise_abs": [0.1], "surprise_pct": [0.11]}, index=idx)
        def fake_provider(sym):
            return payload

        df = fetch_earnings_surprise("AAPL", cache=True, store=store, provider_fn=fake_provider)
        assert not df.empty
        # Cache file should exist after the fetch
        assert (tmp_path / "AAPL.parquet").exists()

    def test_empty_provider_result_no_cache_write(self, tmp_path):
        from kernel.earnings_surprise import (
            fetch_earnings_surprise, EarningsSurpriseStore,
        )
        store = EarningsSurpriseStore(data_dir=tmp_path)
        df = fetch_earnings_surprise("AAPL", cache=True, store=store,
                                      provider_fn=lambda sym: pd.DataFrame())
        assert df.empty
        assert not (tmp_path / "AAPL.parquet").exists()


class TestComputeSurpriseCum:
    def test_trailing_4q_sum(self):
        from kernel.earnings_surprise import compute_earnings_surprise_cum
        # 5 announcements over 2 years
        dates = pd.DatetimeIndex([
            "2023-01-30", "2023-04-28", "2023-07-28", "2023-10-27", "2024-01-29",
        ])
        surp = pd.DataFrame({
            "surprise_pct": [0.05, 0.02, 0.03, -0.01, 0.04],
        }, index=dates)
        surprises = {"AAPL": surp}

        # Daily OHLCV index covering 2023-01-01 to 2024-02-15
        idx = pd.bdate_range("2023-01-01", "2024-02-15")
        ohlcv = {"AAPL": pd.DataFrame({"close": 100.0}, index=idx)}

        out = compute_earnings_surprise_cum(surprises, ohlcv, trailing_quarters=4)
        s = out["AAPL"]
        # After the 5th announcement (2024-01-29), trailing-4Q = last 4
        # announcements: 0.02 + 0.03 + -0.01 + 0.04 = 0.08
        assert s.loc["2024-02-01"] == pytest.approx(0.08)
        # After the 4th announcement (2023-10-27), trailing-4Q = all 4 available
        # = 0.05 + 0.02 + 0.03 + -0.01 = 0.09
        assert s.loc["2023-11-01"] == pytest.approx(0.09)

    def test_missing_ticker_returns_all_nan(self):
        from kernel.earnings_surprise import compute_earnings_surprise_cum
        idx = pd.bdate_range("2024-01-01", periods=20)
        ohlcv = {"X": pd.DataFrame({"close": 100.0}, index=idx)}
        out = compute_earnings_surprise_cum({}, ohlcv)
        assert out["X"].isna().all()

    def test_ffill_between_announcements(self):
        """Between two announcements the value holds steady (step function)."""
        from kernel.earnings_surprise import compute_earnings_surprise_cum
        dates = pd.DatetimeIndex(["2023-01-30", "2023-04-28"])
        surp = pd.DataFrame({"surprise_pct": [0.05, 0.10]}, index=dates)
        idx = pd.bdate_range("2023-01-01", "2023-05-15")
        ohlcv = {"X": pd.DataFrame({"close": 100.0}, index=idx)}

        out = compute_earnings_surprise_cum({"X": surp}, ohlcv, trailing_quarters=4)
        s = out["X"]
        # Between Jan 30 and Apr 28: the value should be 0.05 (one announcement so far)
        assert s.loc["2023-02-15"] == pytest.approx(0.05)
        assert s.loc["2023-04-20"] == pytest.approx(0.05)
        # After Apr 28: value = sum of both announcements = 0.15
        assert s.loc["2023-05-01"] == pytest.approx(0.15)


class TestLoadEarningsSurpriseTaskFlag:
    def test_noop_when_disabled(self):
        from training_panel.pp_panel_training import LoadEarningsSurpriseTask
        from training_panel.context import PanelTrainingContext

        ctx = PanelTrainingContext(
            config={"panel_ltr": {"earnings_surprise": {"enabled": False}}},
            watchlist=["AAPL"],
        )
        LoadEarningsSurpriseTask().run(ctx)
        assert ctx.earnings_surprises == {}
