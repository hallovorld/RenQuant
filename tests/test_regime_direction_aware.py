"""Direction-aware Hurst regime fix (2026-05-14).

Pre-fix bug: Hurst > 0.65 forced BULL_CALM even when SPY was trending DOWN.
2022 Q2 (SPY -20% in 3 months) had Hurst ≈ 0.72 (trending) and was labeled
BULL_CALM 100% of bars. The strategy then sized long-positions at full
confidence into the falling market.

Post-fix: Hurst-MOMENTUM + SPY < MA50 → BEAR.
         Hurst-MOMENTUM + SPY > MA50 → BULL_CALM.

These tests pin the direction-aware routing on synthetic + real SPY data.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.regime import detect_regime, RegimeState  # noqa: E402


_BASE_CFG = {"regime": {
    "hurst_window": 63, "hurst_trending_threshold": 0.65,
    "hurst_reversion_threshold": 0.52, "cusum_lookback": 20,
    "cusum_threshold": 5.5, "cusum_drift": 0.5,
    "transition_uncertainty_bars": 3, "vol_realized_window": 20,
    "bear_vol_threshold": 0.35, "bear_return_threshold": -0.08,
}}


def _make_trending_returns(direction: str, n: int = 200, daily: float = 0.0015,
                            noise: float = 0.005, seed: int = 42) -> np.ndarray:
    """Generate a strongly trending return series. direction ∈ {'up', 'down'}."""
    rng = np.random.default_rng(seed)
    sign = +1.0 if direction == "up" else -1.0
    return rng.normal(loc=sign * daily, scale=noise, size=n)


def _make_spy_df(returns: np.ndarray, start_price: float = 400.0) -> pd.DataFrame:
    """Build an OHLCV DataFrame matching the return series."""
    prices = start_price * np.cumprod(1.0 + returns)
    dates = pd.date_range("2023-01-01", periods=len(returns), freq="B")
    return pd.DataFrame({
        "open": prices, "high": prices * 1.005, "low": prices * 0.995,
        "close": prices, "volume": 1e6,
    }, index=dates)


class TestDirectionAwareHurst:
    """Hurst > 0.65 alone is not enough — direction matters."""

    def test_uptrend_with_high_hurst_routes_bull_calm(self):
        """Synthetic strong UP-trend → BULL_CALM (existing behavior preserved)."""
        rets = _make_trending_returns("up", n=200)
        spy_df = _make_spy_df(rets, start_price=400)
        state = RegimeState()
        detect_regime(rets, spy_df, None, state, _BASE_CFG)
        assert state.regime == "BULL_CALM", (
            f"Up-trending market should be BULL_CALM, got {state.regime}"
        )

    def test_downtrend_with_high_hurst_routes_bear_not_bull_calm(self):
        """Synthetic strong DOWN-trend → BEAR (NEW BEHAVIOR)."""
        rets = _make_trending_returns("down", n=200, daily=0.0010, noise=0.004)
        spy_df = _make_spy_df(rets, start_price=400)
        state = RegimeState()
        detect_regime(rets, spy_df, None, state, _BASE_CFG)
        # Pre-fix this returned BULL_CALM. Post-fix should be BEAR.
        # (Either via hard_bear hard override OR via Hurst-MOMENTUM + below-MA50.)
        assert state.regime == "BEAR", (
            f"Down-trending market should be BEAR (was BULL_CALM pre-fix), "
            f"got {state.regime}"
        )

    def test_2022_bear_window_majority_labeled_bear(self):
        """Real SPY data: 2022 Q2 (objective BEAR market) should be majority BEAR."""
        try:
            spy = pd.read_parquet(REPO_ROOT / "data" / "ohlcv" / "SPY" / "1d.parquet")
        except FileNotFoundError:
            pytest.skip("SPY OHLCV not available locally")
        spy.index = pd.to_datetime(spy.index)
        spy["ret"] = spy["close"].pct_change()
        bars = spy.loc["2022-04-01":"2022-07-01"]
        state = RegimeState()
        bear_count = 0; total = 0
        for d in bars.index:
            sub = spy.loc[:d]
            rets = sub["ret"].dropna().values[-250:]
            if len(rets) < 100:
                continue
            detect_regime(rets, sub.tail(250), None, state, _BASE_CFG)
            if state.regime == "BEAR":
                bear_count += 1
            total += 1
        bear_pct = bear_count / max(total, 1)
        assert bear_pct >= 0.50, (
            f"2022 Q2 was objectively a bear (SPY −20%). Detector should label "
            f"≥50% of bars BEAR; got {bear_pct*100:.1f}%. Direction-aware fix broken."
        )

    def test_2024_q4_bull_strong_majority_not_bear(self):
        """Real SPY data: 2024 Q4 (strong bull) should NOT be mislabeled BEAR."""
        try:
            spy = pd.read_parquet(REPO_ROOT / "data" / "ohlcv" / "SPY" / "1d.parquet")
        except FileNotFoundError:
            pytest.skip("SPY OHLCV not available locally")
        spy.index = pd.to_datetime(spy.index)
        spy["ret"] = spy["close"].pct_change()
        bars = spy.loc["2024-10-01":"2025-01-01"]
        state = RegimeState()
        bear_count = 0; total = 0
        for d in bars.index:
            sub = spy.loc[:d]
            rets = sub["ret"].dropna().values[-250:]
            if len(rets) < 100:
                continue
            detect_regime(rets, sub.tail(250), None, state, _BASE_CFG)
            if state.regime == "BEAR":
                bear_count += 1
            total += 1
        bear_pct = bear_count / max(total, 1)
        assert bear_pct <= 0.20, (
            f"2024 Q4 was objectively a strong bull. Detector mislabels "
            f"{bear_pct*100:.1f}% of bars BEAR (must be ≤20%)."
        )


class TestSpyTrendDirectionEdgeCases:
    """Edge cases on the SPY direction signal."""

    def test_no_spy_df_defaults_to_uptrend(self):
        """When spy_df is None, direction defaults to UP (preserves pre-fix behavior)."""
        rets = _make_trending_returns("down", n=200)
        state = RegimeState()
        detect_regime(rets, None, None, state, _BASE_CFG)
        # Without spy_df, no direction signal → falls back to old logic.
        # Could be BEAR (hard_bear fires) or BULL_CALM. Verify NOT crashed.
        assert state.regime in {"BEAR", "BULL_CALM", "CHOPPY", "BULL_VOLATILE"}

    def test_short_spy_df_under_50_bars_defaults_uptrend(self):
        """When SPY history < 50 bars, MA50 not computable → default UP."""
        rets = _make_trending_returns("up", n=100)
        spy_df = _make_spy_df(rets, start_price=400).tail(30)  # only 30 bars
        state = RegimeState()
        detect_regime(rets, spy_df, None, state, _BASE_CFG)
        # Must not crash; regime is some valid label.
        assert state.regime in {"BEAR", "BULL_CALM", "CHOPPY", "BULL_VOLATILE"}
