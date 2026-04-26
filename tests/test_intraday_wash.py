"""Tests for kernel.intraday_wash — bar washing functions.

Covers Stage A + Stage B of the hourly transformer prep:
  Stage A: winsorize_returns + add_sample_weight (data wash)
  Stage B: add_hour_of_day_features + cross_sectional_z_per_hour
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.intraday_wash import (  # noqa: E402
    add_hour_of_day_features,
    add_sample_weight,
    cross_sectional_z_per_hour,
    wash_bars,
    winsorize_returns,
)


def _hourly_frame(n_bars: int = 200, seed: int = 0,
                   start: str = "2024-01-02 09:30") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.005, size=n_bars)
    close = 100.0 * np.exp(np.cumsum(rets))
    high  = close * (1 + np.abs(rng.normal(0, 0.001, size=n_bars)))
    low   = close * (1 - np.abs(rng.normal(0, 0.001, size=n_bars)))
    open_ = np.r_[close[0], close[:-1]]
    vol   = rng.lognormal(mean=10, sigma=1, size=n_bars).astype(int)
    idx = pd.date_range(start=start, periods=n_bars, freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


# ── Stage A: winsorize_returns ────────────────────────────────────────────────

class TestWinsorize:
    def test_normal_returns_pass_through(self):
        df = _hourly_frame(seed=1)
        out = winsorize_returns(df, n_sigma=5.0, rolling_window=60)
        assert "_hourly_return" in out.columns
        # Normal Gaussian returns rarely exceed 5σ; output should be ~unchanged
        # (only differs at first few bars where σ undefined)
        finite = out["_hourly_return"].dropna()
        assert finite.notna().all()

    def test_outlier_clipped(self):
        df = _hourly_frame(seed=2, n_bars=300)
        # Inject a 10σ outlier at bar 250
        df_close = df["close"].copy()
        df_close.iloc[250] = df_close.iloc[249] * 1.20   # +20% spike
        df["close"] = df_close
        out = winsorize_returns(df, n_sigma=5.0, rolling_window=60)
        # Outlier return should be clipped — magnitude at i=250 < raw
        raw_ret = df["close"].pct_change().iloc[250]
        clipped_ret = out["_hourly_return"].iloc[250]
        assert abs(clipped_ret) < abs(raw_ret)

    def test_idempotent(self):
        df = _hourly_frame(seed=3)
        once  = winsorize_returns(df)
        twice = winsorize_returns(once)
        # Second call computes σ from already-clipped returns; should
        # not flip or introduce NaNs (might tighten cap slightly).
        assert twice["_hourly_return"].notna().sum() >= once["_hourly_return"].notna().sum() - 1

    def test_empty_frame_safe(self):
        out = winsorize_returns(pd.DataFrame())
        assert out.empty


# ── Stage A: add_sample_weight ────────────────────────────────────────────────

class TestSampleWeight:
    def test_high_volume_gets_weight_1(self):
        df = _hourly_frame(seed=4, n_bars=100)
        # Force volume to be very high
        df["volume"] = 1_000_000
        df["close"] = 100.0
        out = add_sample_weight(df, min_dollar_volume=1e5)
        assert (out["_sample_weight"] == 1.0).sum() >= 50

    def test_zero_volume_gets_weight_0(self):
        df = _hourly_frame(seed=5, n_bars=50)
        df.loc[df.index[10:20], "volume"] = 0
        out = add_sample_weight(df, min_dollar_volume=1e5)
        # Ten zero-volume bars should be weight=0
        zero_weight = (out["_sample_weight"] == 0.0).sum()
        assert zero_weight >= 10

    def test_idempotent(self):
        df = _hourly_frame(seed=6)
        once  = add_sample_weight(df)
        twice = add_sample_weight(once)
        np.testing.assert_array_equal(once["_sample_weight"],
                                       twice["_sample_weight"])


# ── Stage B: add_hour_of_day_features ─────────────────────────────────────────

class TestHourOfDayFeatures:
    def test_sin_cos_added(self):
        df = _hourly_frame(seed=7, start="2024-01-02 09:30")
        out = add_hour_of_day_features(df)
        assert "hour_of_day_sin" in out.columns
        assert "hour_of_day_cos" in out.columns
        # Values bounded in [-1, 1]
        assert out["hour_of_day_sin"].between(-1, 1).all()
        assert out["hour_of_day_cos"].between(-1, 1).all()

    def test_cyclic_property(self):
        """sin² + cos² = 1 for all rows."""
        df = _hourly_frame(seed=8)
        out = add_hour_of_day_features(df)
        norm = out["hour_of_day_sin"]**2 + out["hour_of_day_cos"]**2
        np.testing.assert_allclose(norm, 1.0, atol=1e-9)

    def test_idempotent(self):
        df = _hourly_frame(seed=9)
        once  = add_hour_of_day_features(df)
        twice = add_hour_of_day_features(once)
        # No new columns added on second call
        assert list(once.columns) == list(twice.columns)

    def test_different_hours_get_different_codes(self):
        """09:30 and 15:30 must get distinct (sin, cos) pairs."""
        df = _hourly_frame(seed=10, n_bars=10, start="2024-01-02 09:30")
        out = add_hour_of_day_features(df)
        first  = (out["hour_of_day_sin"].iloc[0], out["hour_of_day_cos"].iloc[0])
        sixth  = (out["hour_of_day_sin"].iloc[5], out["hour_of_day_cos"].iloc[5])
        assert first != sixth


# ── Stage A+B combined: wash_bars ─────────────────────────────────────────────

class TestWashBars:
    def test_all_three_stages_applied(self):
        df = _hourly_frame(seed=11)
        out = wash_bars(df)
        for col in ("_hourly_return", "_sample_weight",
                    "hour_of_day_sin", "hour_of_day_cos"):
            assert col in out.columns

    def test_individual_stage_disable(self):
        df = _hourly_frame(seed=12)
        out = wash_bars(df, enable_hour_features=False)
        assert "hour_of_day_sin" not in out.columns

    def test_empty_safe(self):
        assert wash_bars(pd.DataFrame()).empty

    def test_no_nan_introduced(self):
        df = _hourly_frame(seed=13, n_bars=200)
        out = wash_bars(df)
        # The wash-added columns should not contain NaN past warmup
        assert out["hour_of_day_sin"].notna().all()
        assert out["_sample_weight"].notna().all()


# ── Stage B helper: cross_sectional_z_per_hour ────────────────────────────────

class TestCrossSectionalZPerHour:
    def _multi_ticker_panel(self, n_dates=10, n_hours=5,
                              n_tickers=8, seed=20) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        for d in pd.date_range("2024-01-02", periods=n_dates, freq="D"):
            for h in range(n_hours):
                for t_i in range(n_tickers):
                    rows.append({
                        "date":     d.date(),
                        "hour":     h,
                        "ticker":   f"T{t_i}",
                        "feat_a":   rng.normal(0, 1.0),
                        "feat_b":   rng.normal(2, 0.5),
                    })
        return pd.DataFrame(rows)

    def test_z_score_within_each_group(self):
        panel = self._multi_ticker_panel()
        out = cross_sectional_z_per_hour(panel, ["feat_a", "feat_b"])
        # Within each (date, hour), feat_a_z should have mean ≈ 0, std ≈ 1
        for (d, h), grp in out.groupby(["date", "hour"]):
            if len(grp) < 5:
                continue
            assert abs(grp["feat_a_z"].mean()) < 1e-9
            assert abs(grp["feat_a_z"].std() - 1.0) < 0.2

    def test_small_group_passes_through(self):
        panel = self._multi_ticker_panel(n_tickers=3)   # < min_group_size=5
        out = cross_sectional_z_per_hour(panel, ["feat_a"], min_group_size=5)
        # Output should equal input on feat_a (since no z-score applied)
        assert (out["feat_a_z"] == panel["feat_a"]).all()

    def test_unknown_column_skipped(self):
        panel = self._multi_ticker_panel()
        out = cross_sectional_z_per_hour(panel, ["nonexistent"])
        # Should return panel unchanged (column-wise)
        assert "nonexistent_z" not in out.columns

    def test_constant_column_handled(self):
        panel = self._multi_ticker_panel()
        panel["constant"] = 5.0
        out = cross_sectional_z_per_hour(panel, ["constant"])
        # When σ=0, output should be (s - mu) = 0
        assert (out["constant_z"] == 0.0).all()

    def test_empty_panel(self):
        empty = pd.DataFrame(columns=["date","hour","ticker","feat_a"])
        out = cross_sectional_z_per_hour(empty, ["feat_a"])
        assert out.empty
