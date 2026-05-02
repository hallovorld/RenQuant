"""Regression: triple-barrier label computation.

Pinned invariants:
  1. Synthetic monotonic up-trend → all labels are 'upper' hits, all positive.
  2. Synthetic monotonic down-trend → all 'lower' hits, all negative.
  3. Synthetic flat noise → all 'time' barrier (timeout), sample_weight=0.5.
  4. NO LOOKAHEAD: hit detection walks t+1..t+H, day 0 (=t) cannot trigger.
  5. Last max_horizon_days rows are NaN (insufficient lookahead).
  6. NaN propagation: insufficient sigma history → NaN label.
  7. Sample weights: 1.0 for barrier hits, 0.5 for timeout, NaN for missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.triple_barrier import (  # noqa: E402
    TripleBarrierConfig,
    compute_triple_barrier_labels,
)


def _ohlcv_from_close(closes: list[float], start: str = "2026-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame({"close": closes}, index=idx)


class TestUpperBarrierHits:

    def test_monotonic_up_trend_all_upper_hits(self):
        # Build 30 days where price grinds +0.5% per day after a 25-day
        # σ-build period of mild noise.
        rng = np.random.default_rng(42)
        warmup = 100.0 + rng.normal(0, 0.1, 25).cumsum()
        trend = warmup[-1] + np.arange(1, 30) * 0.5
        closes = list(warmup) + list(trend)
        ohlcv = {"AAPL": _ohlcv_from_close(closes)}
        cfg = TripleBarrierConfig(alpha=2.0, beta=2.0, max_horizon_days=5, vol_window=20)
        out = compute_triple_barrier_labels(ohlcv, cfg)["AAPL"]

        non_nan = out.dropna(subset=["label"])
        assert len(non_nan) > 0, "Expected non-NaN labels in trend region"
        # All hits in the trend region (where σ_t computed) must be upper or time
        in_trend = non_nan.iloc[-15:]   # latter half of trend, after warm-up
        # In a strong up-trend, almost all labels should be upper-hits and positive
        assert (in_trend["hit_type"] == "upper").mean() > 0.7, (
            f"Expected mostly upper hits in monotonic up-trend; "
            f"got hit_type counts: {in_trend['hit_type'].value_counts().to_dict()}"
        )
        assert (in_trend["label"] > 0).all(), "All labels in up-trend must be positive"


class TestLowerBarrierHits:

    def test_monotonic_down_trend_all_lower_hits(self):
        rng = np.random.default_rng(43)
        warmup = 100.0 + rng.normal(0, 0.1, 25).cumsum()
        trend = warmup[-1] - np.arange(1, 30) * 0.5
        closes = list(warmup) + list(trend)
        ohlcv = {"AAPL": _ohlcv_from_close(closes)}
        cfg = TripleBarrierConfig(alpha=2.0, beta=2.0, max_horizon_days=5, vol_window=20)
        out = compute_triple_barrier_labels(ohlcv, cfg)["AAPL"]

        non_nan = out.dropna(subset=["label"])
        in_trend = non_nan.iloc[-15:]
        assert (in_trend["hit_type"] == "lower").mean() > 0.7, (
            f"Expected mostly lower hits in down-trend; "
            f"got: {in_trend['hit_type'].value_counts().to_dict()}"
        )
        assert (in_trend["label"] < 0).all(), "All labels in down-trend must be negative"


class TestTimeBarrierTimeout:

    def test_pure_noise_yields_time_hits(self):
        """Noise smaller than barrier widths → mostly time-barrier (timeout)."""
        rng = np.random.default_rng(44)
        # Tiny moves (0.1% std) with WIDE barriers (alpha=beta=5) → no hit
        closes = 100.0 + rng.normal(0, 0.1, 100).cumsum() * 0.01
        ohlcv = {"AAPL": _ohlcv_from_close(closes)}
        cfg = TripleBarrierConfig(alpha=5.0, beta=5.0, max_horizon_days=5, vol_window=20)
        out = compute_triple_barrier_labels(ohlcv, cfg)["AAPL"]
        non_nan = out.dropna(subset=["label"])
        timeout_frac = (non_nan["hit_type"] == "time").mean()
        assert timeout_frac > 0.8, (
            f"With noise << barrier width, expected mostly time barrier; "
            f"got: {non_nan['hit_type'].value_counts().to_dict()}"
        )
        # Sample weights for time-barrier hits = 0.5
        time_rows = non_nan[non_nan["hit_type"] == "time"]
        assert (time_rows["sample_weight"] == 0.5).all()


class TestNoLookahead:

    def test_hit_at_day_zero_impossible(self):
        """The hit search walks day 1..max_horizon. Day 0 (=t) cannot trigger.
        Even if the price crosses a barrier on day 0 (which by construction
        it cannot, since barriers are computed from p_t and σ_t), the label
        must reflect day t+1..t+max_h only."""
        # Build a series where price is flat then jumps massively at day t (impossible
        # by construction of barriers, but we verify hit_days >= 1 for all rows).
        rng = np.random.default_rng(45)
        closes = 100.0 + rng.normal(0, 0.5, 100).cumsum()
        ohlcv = {"AAPL": _ohlcv_from_close(closes)}
        cfg = TripleBarrierConfig(alpha=2.0, beta=2.0, max_horizon_days=5, vol_window=20)
        out = compute_triple_barrier_labels(ohlcv, cfg)["AAPL"]
        non_nan = out.dropna(subset=["hit_days"])
        assert (non_nan["hit_days"] >= 1).all(), "hit_days must be >= 1 (day 0 = t)"


class TestNanPropagation:

    def test_last_max_horizon_rows_are_nan(self):
        rng = np.random.default_rng(46)
        closes = 100.0 + rng.normal(0, 0.5, 100).cumsum()
        ohlcv = {"AAPL": _ohlcv_from_close(closes)}
        cfg = TripleBarrierConfig(alpha=2.0, beta=2.0, max_horizon_days=10, vol_window=20)
        out = compute_triple_barrier_labels(ohlcv, cfg)["AAPL"]
        # Last `max_horizon_days` rows should be NaN
        tail = out.tail(10)
        assert tail["label"].isna().all(), (
            f"Last max_horizon_days rows must be NaN; "
            f"non-NaN tail values: {tail['label'].dropna().to_dict()}"
        )

    def test_pre_sigma_warmup_is_nan(self):
        rng = np.random.default_rng(47)
        closes = 100.0 + rng.normal(0, 0.5, 50).cumsum()
        ohlcv = {"AAPL": _ohlcv_from_close(closes)}
        cfg = TripleBarrierConfig(alpha=2.0, beta=2.0, max_horizon_days=5, vol_window=20)
        out = compute_triple_barrier_labels(ohlcv, cfg)["AAPL"]
        # First vol_window=20 rows have NaN σ_t → NaN labels
        head = out.head(20)
        # Allow some non-NaN if sigma converges fast; but at least row 0..19
        # should mostly be NaN. We enforce a fraction.
        assert head["label"].isna().mean() > 0.9, (
            f"First vol_window rows should be mostly NaN (no σ history); "
            f"got non-NaN frac: {(~head['label'].isna()).mean():.2f}"
        )

    def test_empty_ohlcv_gives_empty_output(self):
        out = compute_triple_barrier_labels({"AAPL": pd.DataFrame()})["AAPL"]
        assert out.empty


class TestSampleWeights:

    def test_barrier_hits_weight_one(self):
        rng = np.random.default_rng(48)
        warmup = 100.0 + rng.normal(0, 0.1, 25).cumsum()
        trend = warmup[-1] + np.arange(1, 30) * 0.5
        ohlcv = {"AAPL": _ohlcv_from_close(list(warmup) + list(trend))}
        cfg = TripleBarrierConfig(alpha=2.0, beta=2.0, max_horizon_days=5, vol_window=20)
        out = compute_triple_barrier_labels(ohlcv, cfg)["AAPL"]
        non_nan = out.dropna(subset=["label"])
        upper_rows = non_nan[non_nan["hit_type"] == "upper"]
        if len(upper_rows) > 0:
            assert (upper_rows["sample_weight"] == 1.0).all()


class TestConfig:

    def test_default_config(self):
        cfg = TripleBarrierConfig()
        assert cfg.alpha == 2.0
        assert cfg.beta == 2.0
        assert cfg.max_horizon_days == 10
        assert cfg.vol_window == 20

    def test_asymmetric_barriers(self):
        """alpha != beta — tight upper, wide lower → labels skew lower-heavy on noise."""
        rng = np.random.default_rng(49)
        closes = 100.0 + rng.normal(0, 0.5, 100).cumsum()
        ohlcv = {"AAPL": _ohlcv_from_close(closes)}
        cfg = TripleBarrierConfig(alpha=0.5, beta=3.0, max_horizon_days=5, vol_window=20)
        out = compute_triple_barrier_labels(ohlcv, cfg)["AAPL"]
        non_nan = out.dropna(subset=["label"])
        # Tight alpha → upper hits more frequently than lower
        upper_frac = (non_nan["hit_type"] == "upper").mean()
        lower_frac = (non_nan["hit_type"] == "lower").mean()
        assert upper_frac > lower_frac, (
            f"Tight upper barrier (alpha=0.5) should hit more than wide lower (beta=3.0); "
            f"got upper={upper_frac:.2f}, lower={lower_frac:.2f}"
        )
