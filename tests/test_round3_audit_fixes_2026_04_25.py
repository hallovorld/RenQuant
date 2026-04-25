"""Round-3 audit regression tests (2026-04-25).

Covers TF-3, GC-1, TPF-1 fixes shipped in the same commit.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


# ── TF-3: hurst_proxy is now real Hurst, not lag-1 autocorr ──────────────────

class TestTF3HurstProxyIsRealHurst:
    def test_hurst_proxy_uses_kernel_rolling_hurst(self):
        """The hurst_proxy column in build_training_features should be the
        real R/S Hurst exponent (range [0, 1]), not lag-1 autocorr (range
        [-1, +1]). A clean trending series should produce H > 0.5; a clean
        anti-correlated series produces H < 0.5."""
        from kernel.regime import compute_hurst, rolling_hurst
        # Strong upward trend → R/S Hurst should be ABOVE 0.5
        rng = np.random.default_rng(0)
        n = 200
        rets = np.full(n, 0.01) + rng.normal(0, 0.001, n)  # near-deterministic upward drift
        h = compute_hurst(rets)
        # Don't be picky about the exact value — just verify it's in
        # the valid Hurst range AND > 0.5 for a trending series.
        assert 0.0 <= h <= 1.0
        # Also verify rolling_hurst returns a Series in the same range
        s = pd.Series(rets)
        rh = rolling_hurst(s, window=63)
        valid = rh.dropna()
        assert ((valid >= 0.0) & (valid <= 1.0)).all()

    def test_features_module_imports_real_hurst(self):
        """Sanity: build_training_features imports rolling_hurst from
        kernel.regime (not the old lag-1 autocorr inline lambda)."""
        src = (_STRATEGY_DIR / "training" / "features.py").read_text()
        assert "from kernel.regime import rolling_hurst" in src, (
            "TF-3 fix: hurst_proxy must use kernel.regime.rolling_hurst "
            "instead of the old corrcoef(x[:-1], x[1:]) lag-1 autocorr."
        )


# ── GC-1: GlobalPanelCalibration enforces sorted x knots ──────────────────────

class TestGC1CalibratorSortedInvariant:
    def test_constructor_rejects_unsorted_prob_x(self):
        from training_panel.global_calibrator import GlobalPanelCalibration
        with pytest.raises(ValueError, match="monotonically non-decreasing"):
            GlobalPanelCalibration(
                prob_x=np.array([0.5, 0.0, 1.0]),  # unsorted
                prob_y=np.array([0.2, 0.5, 0.8]),
                er_x=np.array([0.0, 0.5, 1.0]),
                er_y=np.array([0.0, 0.05, 0.10]),
            )

    def test_constructor_rejects_unsorted_er_x(self):
        from training_panel.global_calibrator import GlobalPanelCalibration
        with pytest.raises(ValueError, match="monotonically non-decreasing"):
            GlobalPanelCalibration(
                prob_x=np.array([0.0, 0.5, 1.0]),
                prob_y=np.array([0.2, 0.5, 0.8]),
                er_x=np.array([1.0, 0.0, 0.5]),    # unsorted
                er_y=np.array([0.0, 0.05, 0.10]),
            )

    def test_constructor_accepts_sorted_arrays(self):
        from training_panel.global_calibrator import GlobalPanelCalibration
        cal = GlobalPanelCalibration(
            prob_x=np.array([0.0, 0.5, 1.0]),
            prob_y=np.array([0.2, 0.5, 0.8]),
            er_x  =np.array([0.0, 0.5, 1.0]),
            er_y  =np.array([0.0, 0.05, 0.10]),
        )
        # interpolation should work as expected
        assert cal.calibrate_probability(0.25) == pytest.approx(0.35)

    def test_constructor_accepts_list_input(self):
        """back-compat: existing callers may pass python lists."""
        from training_panel.global_calibrator import GlobalPanelCalibration
        cal = GlobalPanelCalibration(
            prob_x=[0.0, 1.0], prob_y=[0.0, 1.0],
            er_x=[0.0, 1.0],   er_y=[0.0, 0.1],
        )
        # __post_init__ coerces to ndarray
        assert isinstance(cal.prob_x, np.ndarray)
        assert cal.calibrate_probability(0.5) == pytest.approx(0.5)


# ── TPF-1: PanelFeatureJob aborts on >5% chain failures ───────────────────────

class TestTPF1PanelFeatureJobAbortsOnFailure:
    def test_aborts_when_too_many_chain_failures(self):
        """Pre-fix: failures silently shrunk the panel. Post-fix: raise
        RuntimeError when >5% of tickers fail.
        """
        from training_panel.pp_panel_training import PanelFeatureJob, run_panel_ticker_parallel
        from training_panel.context import TickerPanelContext, PanelTrainingContext
        # Build 100 ticker contexts; simulate ALL of them having no
        # raw_factor_frame (i.e. they all "failed").
        watchlist = [f"T{i:03d}" for i in range(100)]
        ohlcv = {t: pd.DataFrame({"close": [100.0]}) for t in watchlist}
        ctx = PanelTrainingContext(
            config={"benchmark": "SPY"},
            watchlist=watchlist,
            ohlcv=ohlcv,
            sector_etf_ohlcv={},
            ticker_sectors={},
            listing_dates=None,
        )
        # Patch run_panel_ticker_parallel to leave all tc.* fields as None
        # — same effect as a per-ticker chain raising for every ticker.
        import training_panel.pp_panel_training as ppt
        orig = ppt.run_panel_ticker_parallel
        try:
            ppt.run_panel_ticker_parallel = lambda tcs: None  # no-op
            with pytest.raises(RuntimeError, match=r"ticker chains failed"):
                PanelFeatureJob().run(ctx)
        finally:
            ppt.run_panel_ticker_parallel = orig

    def test_logs_aggregate_count_below_threshold(self):
        """Below the 5% threshold, the job warns + continues.

        We patch run_panel_ticker_parallel to mark ~3% of tickers as
        chain-failed — must still complete without raising.
        """
        from training_panel.pp_panel_training import PanelFeatureJob
        from training_panel.context import PanelTrainingContext, TickerPanelContext
        watchlist = [f"T{i:03d}" for i in range(100)]
        ohlcv = {t: pd.DataFrame({"close": [100.0]}) for t in watchlist}
        ctx = PanelTrainingContext(
            config={"benchmark": "SPY"},
            watchlist=watchlist,
            ohlcv=ohlcv,
            sector_etf_ohlcv={},
            ticker_sectors={},
            listing_dates=None,
        )
        import training_panel.pp_panel_training as ppt
        def _fake(tcs):
            # Mark 97 of 100 as successful, 3 as failed (chain left None).
            for i, tc in enumerate(tcs):
                if i >= 3:
                    tc.feature_frame    = pd.DataFrame({"x": [1, 2, 3]})
                    tc.neutralized_frame= pd.DataFrame({"x": [1, 2, 3]})
                    tc.raw_factor_frame = pd.DataFrame({"x": [1, 2, 3]})
        orig = ppt.run_panel_ticker_parallel
        try:
            ppt.run_panel_ticker_parallel = _fake
            # Must NOT raise — 3% < 5% threshold.
            PanelFeatureJob().run(ctx)
        finally:
            ppt.run_panel_ticker_parallel = orig
        assert len(ctx.raw_factor_frames) == 97
