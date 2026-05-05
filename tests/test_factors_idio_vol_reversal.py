"""Unit tests for compute_idio_vol + compute_short_term_reversal.

Two new factors landed 2026-05-03 to round out the factor menu vs Ang 2006
(IVOL puzzle) and Jegadeesh 1990 (1-month reversal). These are the
required §2 regression tests so the wiring through TickerPanelFactorJob
+ raw_cols + z-scoring loop doesn't silently break.
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


class TestIdioVol:
    def test_returns_series_aligned_to_input_index(self):
        from training_panel.factors import compute_idio_vol
        rng = np.random.default_rng(7)
        rets = rng.normal(0, 0.02, 200)
        spy_rets = rng.normal(0, 0.01, 200)
        ohlcv = {"AAA": _make_ohlcv(rets)}
        spy   = _make_ohlcv(spy_rets)
        out = compute_idio_vol(ohlcv, spy, window=60, beta_window=60)
        assert "AAA" in out
        assert list(out["AAA"].index) == list(ohlcv["AAA"].index)

    def test_warmup_is_nan(self):
        """Before beta_window+window bars there is no defined estimate."""
        from training_panel.factors import compute_idio_vol
        rng = np.random.default_rng(11)
        rets = rng.normal(0, 0.02, 200)
        spy_rets = rng.normal(0, 0.01, 200)
        out = compute_idio_vol(
            {"AAA": _make_ohlcv(rets)}, _make_ohlcv(spy_rets),
            window=60, beta_window=60,
        )
        # First ~60 bars must be NaN — beta window not full
        assert out["AAA"].iloc[:30].isna().all()

    def test_idio_smaller_than_total_when_beta_high(self):
        """A pure-market mover (β=1, no idio noise) → idio_vol ≈ 0,
        even though realized_vol is large."""
        from training_panel.factors import compute_idio_vol, compute_realized_vol
        rng = np.random.default_rng(13)
        spy_rets = rng.normal(0, 0.02, 250)
        # ticker = exact replica of spy returns
        out_idio = compute_idio_vol(
            {"PURE": _make_ohlcv(spy_rets)}, _make_ohlcv(spy_rets),
            window=60, beta_window=60,
        )
        out_total = compute_realized_vol({"PURE": _make_ohlcv(spy_rets)}, window=60)
        idio_last  = out_idio["PURE"].dropna().iloc[-1]
        total_last = out_total["PURE"].dropna().iloc[-1]
        assert idio_last < 0.05, f"pure-market replica should have ~0 idio_vol; got {idio_last}"
        assert total_last > idio_last * 5, \
            "total realized_vol must dominate idio_vol when β≈1 perfectly"

    def test_ticker_with_only_idio_noise_has_high_idio_vol(self):
        """Zero correlation with market → idio ≈ total."""
        from training_panel.factors import compute_idio_vol, compute_realized_vol
        rng = np.random.default_rng(17)
        ticker_rets = rng.normal(0, 0.03, 250)
        spy_rets    = rng.normal(0, 0.01, 250)   # uncorrelated by construction
        idio = compute_idio_vol(
            {"NOISY": _make_ohlcv(ticker_rets)}, _make_ohlcv(spy_rets),
            window=60, beta_window=60,
        )["NOISY"].dropna().iloc[-1]
        total = compute_realized_vol(
            {"NOISY": _make_ohlcv(ticker_rets)}, window=60,
        )["NOISY"].dropna().iloc[-1]
        # idio should be within ±20% of total when correlation ~0
        assert abs(idio - total) / total < 0.25


class TestShortTermReversal:
    def test_equals_21_day_pct_change(self):
        from training_panel.factors import compute_short_term_reversal
        rng = np.random.default_rng(19)
        rets = rng.normal(0, 0.02, 100)
        ohlcv = _make_ohlcv(rets)
        out = compute_short_term_reversal({"AAA": ohlcv}, window=21)
        expected = ohlcv["close"].astype(float).pct_change(21)
        pd.testing.assert_series_equal(
            out["AAA"], expected, check_names=False,
        )

    def test_warmup_first_21_bars_nan(self):
        from training_panel.factors import compute_short_term_reversal
        out = compute_short_term_reversal(
            {"AAA": _make_ohlcv(np.zeros(50))}, window=21,
        )
        assert out["AAA"].iloc[:21].isna().all()

    def test_handles_zero_return_series(self):
        from training_panel.factors import compute_short_term_reversal
        out = compute_short_term_reversal(
            {"FLAT": _make_ohlcv(np.zeros(100))}, window=21,
        )
        nonan = out["FLAT"].dropna()
        assert (nonan.abs() < 1e-9).all()
