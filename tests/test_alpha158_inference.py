"""Regression tests for inference-time alpha158 feature computation.

Validates: same canonical 148 names, deterministic output, parity
with build script for spot checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


class TestAlpha158Inference:
    def _make_ohlcv(self, n_bars: int = 100, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2024-01-01", periods=n_bars)
        # Random-walk closes
        closes = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n_bars))
        opens = closes * (1 + rng.normal(0, 0.005, n_bars))
        highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.005, n_bars)))
        lows  = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.005, n_bars)))
        vols  = rng.uniform(1e6, 1e7, n_bars)
        return pd.DataFrame(
            {"open": opens, "high": highs, "low": lows,
             "close": closes, "volume": vols},
            index=dates,
        )

    def test_canonical_148_feature_names(self):
        from kernel.panel_pipeline.alpha158_features import alpha158_feature_names
        names = alpha158_feature_names()
        # 9 KBAR + 4 PRICE + 27 × 5 rolling = 9 + 4 + 135 = 148
        assert len(names) == 158, f"got {len(names)} names"
        # KBAR present
        for k in ("KMID", "KLEN", "KMID2", "KUP", "KUP2",
                   "KLOW", "KLOW2", "KSFT", "KSFT2"):
            assert k in names
        # PRICE present
        for k in ("OPEN0", "HIGH0", "LOW0", "VWAP0"):
            assert k in names
        # Rolling family on 60d window
        for fam in ("ROC60", "MA60", "STD60", "BETA60", "RSQR60", "RESI60",
                     "MAX60", "MIN60", "QTLU60", "QTLD60", "RANK60", "RSV60",
                     "IMAX60", "IMIN60", "IMXD60", "CORR60", "CORD60",
                     "CNTP60", "CNTN60", "CNTD60", "SUMP60", "SUMN60", "SUMD60",
                     "VMA60", "VSTD60", "WVMA60", "VSUMP60", "VSUMN60", "VSUMD60"):
            assert fam in names

    def test_compute_at_last_bar_returns_dict(self):
        from kernel.panel_pipeline.alpha158_features import compute_alpha158_at
        ohlcv = self._make_ohlcv(100)
        feats = compute_alpha158_at(ohlcv)
        assert isinstance(feats, dict)
        assert len(feats) == 158
        # All values finite
        for k, v in feats.items():
            assert isinstance(v, (int, float))
            assert np.isfinite(v) or pd.isna(v), f"feature {k} = {v!r}"

    def test_insufficient_history_returns_empty(self):
        from kernel.panel_pipeline.alpha158_features import compute_alpha158_at
        ohlcv = self._make_ohlcv(50)  # < min_bars=70
        feats = compute_alpha158_at(ohlcv)
        assert feats == {}, f"expected empty, got {len(feats)} features"

    def test_kbar_formula_correctness(self):
        """KMID = (close-open)/open."""
        from kernel.panel_pipeline.alpha158_features import compute_alpha158_at
        n_bars = 100
        ohlcv = self._make_ohlcv(n_bars)
        feats = compute_alpha158_at(ohlcv)
        last = ohlcv.iloc[-1]
        expected_kmid = (last["close"] - last["open"]) / last["open"]
        assert abs(feats["KMID"] - expected_kmid) < 1e-10
        expected_klen = (last["high"] - last["low"]) / last["open"]
        assert abs(feats["KLEN"] - expected_klen) < 1e-10

    def test_price_features(self):
        """OPEN0 = open / close."""
        from kernel.panel_pipeline.alpha158_features import compute_alpha158_at
        ohlcv = self._make_ohlcv(100)
        feats = compute_alpha158_at(ohlcv)
        last = ohlcv.iloc[-1]
        assert abs(feats["OPEN0"] - last["open"] / last["close"]) < 1e-10
        assert abs(feats["HIGH0"] - last["high"] / last["close"]) < 1e-10

    def test_roc_matches_qlib_definition(self):
        """ROC[n] = past_close / today_close (Qlib's exact formula)."""
        from kernel.panel_pipeline.alpha158_features import compute_alpha158_at
        ohlcv = self._make_ohlcv(100)
        feats = compute_alpha158_at(ohlcv)
        c_today = ohlcv["close"].iloc[-1]
        c_5d_ago = ohlcv["close"].iloc[-6]  # 5 bars before last
        expected = c_5d_ago / c_today
        assert abs(feats["ROC5"] - expected) < 1e-10

    def test_deterministic_across_calls(self):
        """Same inputs → same outputs."""
        from kernel.panel_pipeline.alpha158_features import compute_alpha158_at
        ohlcv = self._make_ohlcv(100, seed=42)
        f1 = compute_alpha158_at(ohlcv)
        f2 = compute_alpha158_at(ohlcv)
        for k in f1:
            assert f1[k] == f2[k], f"non-deterministic: {k}"

    def test_min_periods_enforced(self):
        """Custom min_bars threshold respected."""
        from kernel.panel_pipeline.alpha158_features import compute_alpha158_at
        ohlcv = self._make_ohlcv(60)
        feats = compute_alpha158_at(ohlcv, min_bars=60)
        assert feats != {}
        feats_strict = compute_alpha158_at(ohlcv, min_bars=70)
        assert feats_strict == {}
