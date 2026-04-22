"""Unit tests for Round 3 orthogonal factor functions.

Covers:
  compute_amihud_illiquidity
  compute_volume_shift
  compute_price_to_high
  compute_realized_vol
  compute_drawdown_from_peak
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _make_ohlcv(returns: np.ndarray, volume: float = 1e6) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=len(returns))
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "open": close, "high": close * 1.005, "low": close * 0.995,
        "close": close, "volume": np.full(len(returns), volume),
    }, index=idx)


class TestAmihudIlliquidity:
    def test_illiquid_stock_has_higher_score(self):
        """At same volatility, lower dollar volume → higher illiquidity."""
        from training_panel.factors import compute_amihud_illiquidity
        rng = np.random.default_rng(42)
        rets = rng.normal(0, 0.02, 500)
        out = compute_amihud_illiquidity(
            {"LIQ":   _make_ohlcv(rets, volume=1e9),
             "ILLIQ": _make_ohlcv(rets, volume=1e6)},
            window=21,
        )
        liq_last   = out["LIQ"].dropna().iloc[-1]
        illiq_last = out["ILLIQ"].dropna().iloc[-1]
        assert illiq_last > liq_last * 100, \
            "ILLIQ's score should be far higher (1000x volume ratio)"

    def test_no_volume_column_returns_all_nan(self):
        from training_panel.factors import compute_amihud_illiquidity
        df = pd.DataFrame({
            "open": [100.0] * 10, "high": [101.0] * 10, "low": [99.0] * 10,
            "close": [100.0] * 10,
        }, index=pd.bdate_range("2024-01-02", periods=10))
        out = compute_amihud_illiquidity({"X": df})
        assert out["X"].isna().all()


class TestVolumeShift:
    def test_rising_volume_yields_positive_shift(self):
        from training_panel.factors import compute_volume_shift
        idx = pd.bdate_range("2024-01-02", periods=200)
        # Flat price, but volume doubles in the last 20 days.
        vol = np.full(200, 1e6)
        vol[-20:] = 2e6
        df = pd.DataFrame({
            "open": [100.0] * 200, "high": [101.0] * 200, "low": [99.0] * 200,
            "close": [100.0] * 200, "volume": vol,
        }, index=idx)
        out = compute_volume_shift({"X": df}, short_window=20, long_window=60)
        last = out["X"].iloc[-1]
        # short-mean ≈ 2e6, long-mean ≈ mix approaching 1.33e6, ratio > 1
        assert last > 0, f"rising-volume shift should be positive, got {last}"


class TestPriceToHigh:
    def test_all_time_peak_gives_one(self):
        from training_panel.factors import compute_price_to_high
        # Monotone-increasing prices — every bar is its own peak.
        close = np.linspace(100, 200, 300)
        df = pd.DataFrame({
            "open": close, "high": close * 1.001, "low": close * 0.999,
            "close": close, "volume": np.ones(300) * 1e6,
        }, index=pd.bdate_range("2024-01-02", periods=300))
        out = compute_price_to_high({"X": df}, window=252)
        # After warmup, the ratio should be essentially 1 (price equals peak).
        assert out["X"].dropna().iloc[-1] == 1.0

    def test_deep_selloff_drops_ratio(self):
        from training_panel.factors import compute_price_to_high
        n = 300
        close = np.concatenate([
            np.linspace(100, 200, 250),  # rally to 200
            np.linspace(200, 100, 50),   # sell off back to 100
        ])
        df = pd.DataFrame({
            "open": close, "high": close * 1.001, "low": close * 0.999,
            "close": close, "volume": np.ones(n) * 1e6,
        }, index=pd.bdate_range("2024-01-02", periods=n))
        out = compute_price_to_high({"X": df}, window=252)
        # At the end, price/peak should be ≈ 100/200 = 0.5
        assert out["X"].iloc[-1] == pytest.approx(0.5, abs=0.02)


class TestRealizedVol:
    def test_low_vol_lower_value(self):
        from training_panel.factors import compute_realized_vol
        rng = np.random.default_rng(1)
        out = compute_realized_vol(
            {"LO": _make_ohlcv(rng.normal(0, 0.005, 500)),
             "HI": _make_ohlcv(rng.normal(0, 0.03, 500))},
            window=20,
        )
        lo_last = out["LO"].dropna().iloc[-1]
        hi_last = out["HI"].dropna().iloc[-1]
        assert hi_last > lo_last * 3, \
            "high-vol series should annualize to ~6x the low-vol series"

    def test_annualization(self):
        from training_panel.factors import compute_realized_vol
        n = 500
        # Known daily std=0.01 → annualized = 0.01 × sqrt(252) ≈ 0.1587
        rng = np.random.default_rng(99)
        rets = rng.normal(0, 0.01, n)
        df = _make_ohlcv(rets)
        out = compute_realized_vol({"X": df}, window=100)
        vv = out["X"].dropna()
        assert vv.mean() == pytest.approx(0.01 * np.sqrt(252), rel=0.15)


class TestDrawdownFromPeak:
    def test_at_peak_equals_zero(self):
        from training_panel.factors import compute_drawdown_from_peak
        close = np.linspace(100, 300, 300)
        df = pd.DataFrame({
            "open": close, "high": close * 1.001, "low": close * 0.999,
            "close": close, "volume": np.ones(300) * 1e6,
        }, index=pd.bdate_range("2024-01-02", periods=300))
        out = compute_drawdown_from_peak({"X": df}, window=252)
        assert out["X"].iloc[-1] == 0.0

    def test_always_non_positive(self):
        from training_panel.factors import compute_drawdown_from_peak
        rng = np.random.default_rng(7)
        df = _make_ohlcv(rng.normal(0, 0.02, 500))
        out = compute_drawdown_from_peak({"X": df}, window=252)
        vv = out["X"].dropna()
        assert (vv <= 1e-9).all(), "drawdown from rolling peak must be ≤ 0"


import pytest  # noqa: E402  (pytest.approx used above)
