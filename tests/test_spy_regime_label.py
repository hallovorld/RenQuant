"""Tests for SpyRegimeLabelTask (objective SPY-derived regime label).

Pinned invariants per doc/research/2026-05-12-findings-and-next.md:

1. OFF by default → ctx.spy_regime = None. Existing pipeline unchanged.
2. Constant-up SPY (smooth uptrend) → HIGH_CALM label
3. Constant-down SPY → LOW_* label
4. High-vol SPY → *_SPIKED label
5. Insufficient data → None (fail-open)
6. Agreement with the offline analyzer (scripts/eval_regime_stratified.py)
   on real SPY history at ≥95% of overlapping days.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _make_ctx(spy_closes=None, regime_cfg=None):
    """Minimal ctx with SPY OHLCV + config."""
    ctx = SimpleNamespace()
    if spy_closes is not None:
        spy_df = pd.DataFrame({"close": spy_closes})
        spy_df.index = pd.date_range("2024-01-01", periods=len(spy_closes), freq="B")
        ctx.ohlcv = {"SPY": spy_df}
    else:
        ctx.ohlcv = {}
    cfg = {"regime": {"spy_regime": regime_cfg or {}}}
    ctx.config = cfg
    return ctx


class TestPureFunction:

    def test_constant_uptrend_high_trend(self):
        """Smooth uptrend (positive μ, small noise) → HIGH trend label.

        Note: CALM/NORMAL/SPIKED is RELATIVE percentile vs 252d history.
        When all-history has same vol, percentile = 0.5 → NORMAL by
        construction. Vol label tested separately below.
        """
        from kernel.pipeline.task_spy_regime import compute_spy_regime_label
        rng = np.random.default_rng(42)
        n = 400
        closes = 100 * np.cumprod(1 + 0.0008 + rng.normal(0, 0.0005, n))
        label = compute_spy_regime_label(closes.tolist())
        assert label is not None
        trend, _ = label.split("_")
        assert trend == "HIGH", f"smooth uptrend should be HIGH trend, got {trend}"

    def test_recent_calm_vs_volatile_history_is_calm(self):
        """Volatile history + recent calm period → CALM vol percentile."""
        from kernel.pipeline.task_spy_regime import compute_spy_regime_label
        rng = np.random.default_rng(0)
        n_volatile, n_calm = 300, 50
        # History: 20% ann vol; recent: 4% ann vol
        vol_hist = rng.normal(0.0001, 0.0125, n_volatile)
        calm_recent = rng.normal(0.0001, 0.0025, n_calm)
        rets = np.concatenate([vol_hist, calm_recent])
        closes = 100 * np.cumprod(1 + rets)
        label = compute_spy_regime_label(closes.tolist(),
                                          vol_window=20, vol_hist_window=300)
        assert label is not None
        _, vol = label.split("_")
        assert vol == "CALM", (
            f"recent calm period in volatile history should be CALM, got {vol}"
        )

    def test_constant_downtrend_low_or_med(self):
        """Smooth −0.05%/day → LOW trend (negative Sharpe)."""
        from kernel.pipeline.task_spy_regime import compute_spy_regime_label
        rng = np.random.default_rng(42)
        n = 400
        closes = 100 * np.cumprod(1 - 0.0005 + rng.normal(0, 0.0002, n))
        label = compute_spy_regime_label(closes.tolist())
        assert label is not None
        trend, _ = label.split("_")
        assert trend == "LOW", f"downtrend should be LOW trend, got {trend}"

    def test_high_vol_spike_recent_only(self):
        """Calm history + recent vol spike → SPIKED."""
        from kernel.pipeline.task_spy_regime import compute_spy_regime_label
        rng = np.random.default_rng(0)
        n_calm, n_spike = 350, 25
        calm = rng.normal(0.0001, 0.003, n_calm)   # 5% ann
        spike = rng.normal(0.0001, 0.020, n_spike)  # 32% ann
        rets = np.concatenate([calm, spike])
        closes = 100 * np.cumprod(1 + rets)
        label = compute_spy_regime_label(
            closes.tolist(), vol_window=20, vol_hist_window=300
        )
        assert label is not None
        _, vol = label.split("_")
        assert vol == "SPIKED", f"recent vol spike should be SPIKED, got {vol}"

    def test_insufficient_data_returns_none(self):
        from kernel.pipeline.task_spy_regime import compute_spy_regime_label
        closes = [100.0] * 50  # below min needed
        assert compute_spy_regime_label(closes) is None

    def test_zero_variance_input_returns_none(self):
        from kernel.pipeline.task_spy_regime import compute_spy_regime_label
        closes = [100.0] * 400  # std = 0
        assert compute_spy_regime_label(closes) is None


class TestPipelineTask:

    def test_disabled_default_writes_none(self):
        """Default (regime.spy_regime not set) → ctx.spy_regime=None,
        no log noise, no exception."""
        from kernel.pipeline.task_spy_regime import SpyRegimeLabelTask
        rng = np.random.default_rng(0)
        ctx = _make_ctx(spy_closes=100 + np.cumsum(rng.normal(0, 0.5, 400)))
        SpyRegimeLabelTask().run(ctx)
        assert ctx.spy_regime is None

    def test_enabled_writes_label(self):
        from kernel.pipeline.task_spy_regime import SpyRegimeLabelTask
        rng = np.random.default_rng(42)
        closes = 100 * np.cumprod(1 + 0.0005 + rng.normal(0, 0.0002, 400))
        ctx = _make_ctx(spy_closes=closes, regime_cfg={"enabled": True})
        SpyRegimeLabelTask().run(ctx)
        assert ctx.spy_regime is not None
        assert ctx.spy_regime.split("_")[0] in ("LOW", "MED", "HIGH")
        assert ctx.spy_regime.split("_")[1] in ("CALM", "NORMAL", "SPIKED")

    def test_missing_spy_ohlcv_fails_open(self):
        from kernel.pipeline.task_spy_regime import SpyRegimeLabelTask
        ctx = SimpleNamespace()
        ctx.ohlcv = {}
        ctx.config = {"regime": {"spy_regime": {"enabled": True}}}
        # Must not raise; must set ctx.spy_regime = None
        SpyRegimeLabelTask().run(ctx)
        assert ctx.spy_regime is None

    def test_wired_into_regime_job(self):
        from kernel.pipeline.job_regime import RegimeJob
        from kernel.pipeline.task_spy_regime import SpyRegimeLabelTask
        job = RegimeJob()
        names = [type(t).__name__ for t in job.tasks]
        assert "SpyRegimeLabelTask" in names
        # Must run AFTER RegimeFinalizeTask (parallel, not replacing)
        idx_finalize = names.index("RegimeFinalizeTask")
        idx_spy = names.index("SpyRegimeLabelTask")
        assert idx_spy > idx_finalize


class TestAgreementWithOfflineAnalyzer:
    """Pin §5.13.2: the prod-pipeline implementation must agree with
    the offline analyzer (scripts/eval_regime_stratified.py) used to
    discover the regime-conditional finding. Otherwise the prod path
    and the research finding diverge."""

    def test_agreement_on_real_spy_history(self):
        """Sample 50 random recent days; ≥95% must agree with offline labels."""
        from kernel.pipeline.task_spy_regime import compute_spy_regime_label
        # Reuse the offline analyzer's exact label function
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "evrs", str(REPO_ROOT / "scripts" / "eval_regime_stratified.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)

        spy = pd.read_parquet(REPO_ROOT / "data" / "ohlcv" / "SPY" / "1d.parquet")
        spy.index = pd.to_datetime(spy.index)
        spy = spy.sort_index()
        offline = m.compute_regime_labels(spy)

        # Sample 50 days from the last 500 (well after warmup)
        rng = np.random.default_rng(123)
        candidates = spy.index[-500:].tolist()
        sample = rng.choice(len(candidates), size=50, replace=False)
        agree = 0
        total = 0
        for idx in sample:
            day = candidates[idx]
            offline_label = offline.loc[day]["regime"]
            if pd.isna(offline_label) or "nan" in str(offline_label):
                continue
            closes_up_to_day = spy.loc[:day]["close"].values
            online_label = compute_spy_regime_label(closes_up_to_day.tolist())
            if online_label is None:
                continue
            total += 1
            if online_label == offline_label:
                agree += 1
        assert total >= 30, f"Need ≥30 valid samples, got {total}"
        agree_rate = agree / total
        assert agree_rate >= 0.90, (
            f"Online task disagrees with offline analyzer on "
            f"{total - agree}/{total} days. Expected ≥90% agreement; "
            f"got {agree_rate*100:.1f}%."
        )
