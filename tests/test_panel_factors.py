"""Tests for training_panel/factors.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _make_ohlcv(n: int = 400, seed: int = 0, drift: float = 0.0005) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    rets = rng.normal(drift, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"close": close}, index=idx)


class TestMom121:
    def test_mom_12_1_exact_formula(self):
        from training_panel.factors import compute_momentum_12_1
        df = _make_ohlcv(n=300, seed=0)
        out = compute_momentum_12_1({"A": df}, mom_window=252, skip=21)
        # Pick a bar at index 270 and compare to explicit formula
        bar = 270
        close = df["close"].values
        # Skip: close[bar-21]  vs  Full: close[bar-252]
        expected = close[bar - 21] / close[bar - 252] - 1.0
        got = out["A"].iloc[bar]
        assert abs(got - expected) < 1e-9

    def test_mom_12_1_nan_until_warmup(self):
        from training_panel.factors import compute_momentum_12_1
        df = _make_ohlcv(n=300, seed=1)
        out = compute_momentum_12_1({"A": df}, mom_window=252, skip=21)
        # First 252 bars should be NaN
        assert out["A"].iloc[:252].isna().all()


class TestRollingBeta:
    def test_beta_of_spy_against_spy_is_1(self):
        from training_panel.factors import compute_rolling_beta
        spy = _make_ohlcv(n=300, seed=2)
        out = compute_rolling_beta({"SPY": spy}, spy, window=60)
        # After warmup, β(SPY vs SPY) = 1
        vals = out["SPY"].dropna()
        assert len(vals) > 100
        assert np.allclose(vals.values, 1.0, atol=1e-9)

    def test_beta_positive_for_positively_correlated_stock(self):
        from training_panel.factors import compute_rolling_beta
        spy = _make_ohlcv(n=500, seed=3)
        # Build a stock as 1.2× SPY returns + small noise
        rng = np.random.default_rng(42)
        spy_rets = spy["close"].pct_change()
        stock_rets = 1.2 * spy_rets + pd.Series(
            rng.normal(0, 0.002, len(spy)), index=spy.index,
        )
        stock_close = (1 + stock_rets.fillna(0)).cumprod() * 100.0
        stock_df = pd.DataFrame({"close": stock_close.values}, index=spy.index)
        out = compute_rolling_beta({"A": stock_df}, spy, window=60)
        mean_beta = out["A"].dropna().mean()
        assert 1.0 < mean_beta < 1.4, f"expected β≈1.2, got {mean_beta:.3f}"


class TestResidualMomentum:
    def test_residual_momentum_orthogonal_to_spy_mom(self):
        from training_panel.factors import (
            compute_residual_momentum, compute_momentum_12_1,
        )
        rng = np.random.default_rng(5)
        spy = _make_ohlcv(n=600, seed=5)
        stocks = {f"S{i}": _make_ohlcv(n=600, seed=i + 10) for i in range(8)}
        res = compute_residual_momentum(stocks, spy, window=60, mom_window=252, skip=21)
        spy_mom = compute_momentum_12_1({"SPY": spy}, mom_window=252, skip=21)["SPY"]

        # Cross-sectional: for each date t, corr(res[:,t], spy_mom[t]) → but
        # since spy_mom is a scalar per date, we can't correlate cross-sec.
        # Instead test a time-series: mean residual momentum vs spy mom.
        # Build a panel of residuals
        frames = []
        for t, s in res.items():
            frames.append(pd.DataFrame({"t": s.index, "val": s.values}))
        long = pd.concat(frames, ignore_index=True)
        mean_res = long.groupby("t")["val"].mean()
        joint = pd.DataFrame({"res": mean_res, "spy_mom": spy_mom}).dropna()
        corr = joint["res"].corr(joint["spy_mom"])
        # After subtracting β × spy_mom, the average residual-mom should be
        # close to uncorrelated with spy_mom.
        assert abs(corr) < 0.3, f"residual mom still loaded on SPY: {corr:.3f}"


class TestSize:
    def test_size_uses_log_scale(self):
        from training_panel.factors import compute_size_feature
        df = _make_ohlcv(n=100, seed=7)
        out = compute_size_feature({"A": df})
        # log(close) should match
        assert np.allclose(out["A"].dropna().values,
                           np.log(df["close"]).values)

    def test_size_with_shares_outstanding(self):
        from training_panel.factors import compute_size_feature
        df = _make_ohlcv(n=50, seed=8)
        shares = pd.Series(1_000_000.0, index=df.index)
        out = compute_size_feature({"A": df}, shares_outstanding={"A": shares})
        expected = np.log(df["close"] * 1_000_000.0)
        assert np.allclose(out["A"].values, expected.values)


class TestCrossSectionalZscore:
    def test_zscore_mean_zero_std_one_per_date(self):
        from training_panel.factors import cross_sectional_zscore
        rng = np.random.default_rng(11)
        idx = pd.bdate_range("2024-01-01", periods=10)
        feat = {
            f"T{i:02d}": pd.Series(rng.normal(5, 2, len(idx)), index=idx)
            for i in range(30)
        }
        out = cross_sectional_zscore(feat)
        # Per-date: gather values across tickers
        per_date_vals = {d: [] for d in idx}
        for t, s in out.items():
            for d, v in s.items():
                per_date_vals[d].append(v)
        for d, xs in per_date_vals.items():
            arr = np.asarray(xs)
            assert abs(arr.mean()) < 0.1
            assert 0.9 < arr.std() < 1.1

    def test_nan_input_preserves_nan(self):
        from training_panel.factors import cross_sectional_zscore
        idx = pd.bdate_range("2024-01-01", periods=3)
        feat = {
            "A": pd.Series([1.0, np.nan, 3.0], index=idx),
            "B": pd.Series([2.0, 4.0, np.nan], index=idx),
        }
        out = cross_sectional_zscore(feat)
        assert np.isnan(out["A"].iloc[1])
        assert np.isnan(out["B"].iloc[2])


class TestBuildFactorBundle:
    def test_bundle_covers_all_tickers(self):
        from training_panel.factors import build_factor_bundle
        spy = _make_ohlcv(n=400, seed=100)
        ohlcv = {f"S{i}": _make_ohlcv(n=400, seed=i + 200) for i in range(5)}
        bundle = build_factor_bundle(ohlcv, spy)
        assert set(bundle.keys()) == set(ohlcv.keys())
        for t, df in bundle.items():
            for c in ("size_z", "mom_12_1_z", "beta_60d_z", "resid_mom_z"):
                assert c in df.columns

    def test_bundle_index_matches_ohlcv(self):
        from training_panel.factors import build_factor_bundle
        spy = _make_ohlcv(n=300, seed=300)
        ohlcv = {"A": _make_ohlcv(n=300, seed=301)}
        bundle = build_factor_bundle(ohlcv, spy)
        assert bundle["A"].index.equals(ohlcv["A"].index)
