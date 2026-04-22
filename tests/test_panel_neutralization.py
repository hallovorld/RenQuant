"""Tests for training_panel/neutralization.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _make_sector_ohlcv(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    # Random walk close prices
    rets = rng.normal(0, 0.01, n)
    close = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"close": close}, index=idx)


def _make_feature_frame(sector_mom_df: pd.DataFrame, n: int = 400,
                        seed: int = 1, beta: float = 0.8,
                        include_mean_reversion: bool = True) -> pd.DataFrame:
    """Build a feature frame whose momentum features load on sector momentum."""
    rng = np.random.default_rng(seed)
    idx = sector_mom_df.index
    sec_mom_20 = sector_mom_df["mom_20d"].fillna(0).values
    sec_mom_60 = sector_mom_df["mom_60d"].fillna(0).values
    sec_trend  = sector_mom_df["trend"].fillna(1).values
    sec_tlong  = sector_mom_df["trend_long"].fillna(1).values

    noise1 = rng.normal(0, 0.01, n)
    noise2 = rng.normal(0, 0.02, n)
    noise3 = rng.normal(0, 0.03, n)
    noise4 = rng.normal(0, 0.03, n)

    cols = {
        "rel_mom_20d": beta * sec_mom_20 + noise1,
        "rel_mom_60d": beta * sec_mom_60 + noise2,
        "trend":       beta * sec_trend  + (1 - beta) + noise3,
        "trend_long":  beta * sec_tlong  + (1 - beta) + noise4,
    }
    if include_mean_reversion:
        cols["rsi"] = rng.uniform(20, 80, n)
        cols["bbp"] = rng.uniform(0, 1, n)
        cols["williams_r"] = rng.uniform(-100, 0, n)
    return pd.DataFrame(cols, index=idx)


class TestComputeSectorMomentum:
    def test_output_has_expected_columns(self):
        from training_panel.neutralization import compute_sector_momentum
        etfs = {"TECH": _make_sector_ohlcv(n=300, seed=0)}
        out = compute_sector_momentum(etfs)
        assert "TECH" in out
        for c in ("mom_20d", "mom_60d", "trend", "trend_long"):
            assert c in out["TECH"].columns

    def test_mom_matches_pct_change(self):
        from training_panel.neutralization import compute_sector_momentum
        etf = _make_sector_ohlcv(n=200, seed=1)
        out = compute_sector_momentum({"X": etf})["X"]
        expected = etf["close"].pct_change(20)
        # Compare only non-NaN
        mask = expected.notna() & out["mom_20d"].notna()
        assert np.allclose(out["mom_20d"][mask], expected[mask])


class TestNeutralizeFeatures:
    def test_neutralized_feature_uncorrelated_with_sector_momentum(self):
        from training_panel.neutralization import (
            compute_sector_momentum, neutralize_features,
        )
        etf = _make_sector_ohlcv(n=500, seed=10)
        sec_mom = compute_sector_momentum({"TECH": etf})
        ff = _make_feature_frame(sec_mom["TECH"], n=500, seed=11, beta=0.8)
        out = neutralize_features(
            {"AAA": ff}, sec_mom, {"AAA": "TECH"},
            rolling_window=120, expanding_warmup_days=120,
        )
        # Correlation between residual rel_mom_20d and sector mom_20d should be small
        r = out["AAA"]["rel_mom_20d"]
        s = sec_mom["TECH"]["mom_20d"]
        df = pd.DataFrame({"r": r, "s": s}).dropna()
        corr = df["r"].corr(df["s"])
        assert abs(corr) < 0.15, f"residual still correlated with sector: {corr:.3f}"

    def test_mean_reversion_features_unchanged(self):
        from training_panel.neutralization import (
            compute_sector_momentum, neutralize_features,
        )
        etf = _make_sector_ohlcv(n=300, seed=12)
        sec_mom = compute_sector_momentum({"TECH": etf})
        ff = _make_feature_frame(sec_mom["TECH"], n=300, seed=13, beta=0.5)
        out = neutralize_features(
            {"AAA": ff}, sec_mom, {"AAA": "TECH"},
            rolling_window=120, expanding_warmup_days=120,
        )
        for col in ("rsi", "bbp", "williams_r"):
            pd.testing.assert_series_equal(out["AAA"][col], ff[col])

    def test_missing_sector_returns_feature_unchanged(self):
        from training_panel.neutralization import neutralize_features
        etf = _make_sector_ohlcv(n=300, seed=14)
        from training_panel.neutralization import compute_sector_momentum
        sec_mom = compute_sector_momentum({"TECH": etf})
        ff = _make_feature_frame(sec_mom["TECH"], n=300, seed=15, beta=0.6)
        # Ticker AAA mapped to sector UNKNOWN which is not in sec_mom
        out = neutralize_features(
            {"AAA": ff}, sec_mom, {"AAA": "UNKNOWN"},
        )
        pd.testing.assert_frame_equal(out["AAA"], ff)

    def test_neutralization_is_purged_from_future(self):
        """Changing feature value at bar t must not affect residual at bars < t."""
        from training_panel.neutralization import (
            compute_sector_momentum, neutralize_features,
        )
        etf = _make_sector_ohlcv(n=300, seed=16)
        sec_mom = compute_sector_momentum({"TECH": etf})
        ff1 = _make_feature_frame(sec_mom["TECH"], n=300, seed=17, beta=0.5)
        ff2 = ff1.copy()
        # Perturb last bar of rel_mom_20d wildly
        ff2.iloc[-1, ff2.columns.get_loc("rel_mom_20d")] = 99.0
        out1 = neutralize_features({"A": ff1}, sec_mom, {"A": "TECH"},
                                    rolling_window=120, expanding_warmup_days=120)
        out2 = neutralize_features({"A": ff2}, sec_mom, {"A": "TECH"},
                                    rolling_window=120, expanding_warmup_days=120)
        r1 = out1["A"]["rel_mom_20d"].iloc[:-1]
        r2 = out2["A"]["rel_mom_20d"].iloc[:-1]
        mask = r1.notna() & r2.notna()
        assert np.allclose(r1[mask].values, r2[mask].values)

    def test_expanding_window_used_within_warmup(self):
        """Within warmup, changing a bar at index < 30 should NOT affect residuals
        past bar 30 (they rely on expanding from 0..t-1, so bars before t propagate).

        Restated: residual at bar t depends on feature[0..t-1]. This is the
        definition of expanding-window behavior. We verify: early residuals
        use many prior bars (via sensitivity)."""
        from training_panel.neutralization import (
            compute_sector_momentum, neutralize_features,
        )
        etf = _make_sector_ohlcv(n=300, seed=18)
        sec_mom = compute_sector_momentum({"TECH": etf})
        ff1 = _make_feature_frame(sec_mom["TECH"], n=300, seed=19, beta=0.5)
        # Perturb bar index 10 (deep inside expanding-warmup zone, warmup=120)
        ff2 = ff1.copy()
        ff2.iloc[10, ff2.columns.get_loc("rel_mom_20d")] += 5.0
        out1 = neutralize_features({"A": ff1}, sec_mom, {"A": "TECH"},
                                    rolling_window=120, expanding_warmup_days=120)
        out2 = neutralize_features({"A": ff2}, sec_mom, {"A": "TECH"},
                                    rolling_window=120, expanding_warmup_days=120)
        # Residual at bar 60 must change (bar 10 is inside its expanding window)
        r1_60 = out1["A"]["rel_mom_20d"].iloc[60]
        r2_60 = out2["A"]["rel_mom_20d"].iloc[60]
        assert r1_60 != r2_60 or (pd.isna(r1_60) and pd.isna(r2_60))
        # And residual at bar 9 (before the perturbation) should be unchanged
        r1_9 = out1["A"]["rel_mom_20d"].iloc[9]
        r2_9 = out2["A"]["rel_mom_20d"].iloc[9]
        if not (pd.isna(r1_9) and pd.isna(r2_9)):
            assert r1_9 == r2_9

    def test_rolling_window_used_after_warmup(self):
        """After warmup, a bar far in the past (outside rolling window) should
        no longer influence the current residual."""
        from training_panel.neutralization import (
            compute_sector_momentum, neutralize_features,
        )
        etf = _make_sector_ohlcv(n=500, seed=20)
        sec_mom = compute_sector_momentum({"TECH": etf})
        ff1 = _make_feature_frame(sec_mom["TECH"], n=500, seed=21, beta=0.5)
        ff2 = ff1.copy()
        # Perturb bar index 50
        ff2.iloc[50, ff2.columns.get_loc("rel_mom_20d")] += 5.0
        out1 = neutralize_features({"A": ff1}, sec_mom, {"A": "TECH"},
                                    rolling_window=120, expanding_warmup_days=120)
        out2 = neutralize_features({"A": ff2}, sec_mom, {"A": "TECH"},
                                    rolling_window=120, expanding_warmup_days=120)
        # Bar 50 is outside the 120-window ending at bar 480 → unchanged
        r1 = out1["A"]["rel_mom_20d"].iloc[480]
        r2 = out2["A"]["rel_mom_20d"].iloc[480]
        assert np.isclose(r1, r2, atol=1e-12)

    def test_other_columns_preserved(self):
        from training_panel.neutralization import (
            compute_sector_momentum, neutralize_features,
        )
        etf = _make_sector_ohlcv(n=300, seed=22)
        sec_mom = compute_sector_momentum({"TECH": etf})
        ff = _make_feature_frame(sec_mom["TECH"], n=300, seed=23, beta=0.5)
        out = neutralize_features({"A": ff}, sec_mom, {"A": "TECH"})
        # All columns still present
        assert set(out["A"].columns) == set(ff.columns)
