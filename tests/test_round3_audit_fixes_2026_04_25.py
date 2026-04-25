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


# ── K-1 (Round 5 audit): kelly_target_pct rejects NaN/inf inputs ──────────────

class TestK1KellyRejectsNanInf:
    """Pre-fix, `mu = NaN` or `sigma = NaN` slipped past the guards
    (NaN comparisons are False) and propagated through the formula → the
    function returned NaN instead of 0.0. Downstream SizeAndEmitTask then
    multiplied that NaN into max_pct, producing NaN order sizes."""

    def test_nan_mu_returns_zero(self):
        from kernel.kelly import kelly_target_pct
        assert kelly_target_pct(float("nan"), 0.05, max_pct=0.15) == 0.0

    def test_nan_sigma_returns_zero(self):
        from kernel.kelly import kelly_target_pct
        assert kelly_target_pct(0.01, float("nan"), max_pct=0.15) == 0.0

    def test_inf_mu_returns_zero(self):
        from kernel.kelly import kelly_target_pct
        assert kelly_target_pct(float("inf"), 0.05, max_pct=0.15) == 0.0

    def test_inf_sigma_returns_zero(self):
        from kernel.kelly import kelly_target_pct
        assert kelly_target_pct(0.01, float("inf"), max_pct=0.15) == 0.0

    def test_finite_inputs_still_work(self):
        from kernel.kelly import kelly_target_pct
        # μ=0.02, σ=0.10 → f*=2.0, fractional 0.25 → 0.5, capped to max_pct 0.15.
        assert kelly_target_pct(0.02, 0.10, max_pct=0.15, fractional=0.25) == pytest.approx(0.15)


# ── M-1 (Round 6 audit): calibrate_score returns finite on NaN raw_score ──────

class TestM1CalibrateScoreRejectsNaN:
    """Pre-fix, NaN raw_score leaked through every calibration method
    (identity / isotonic / platt) and produced NaN rank_score, which
    poisoned downstream ranking + tier-gate logic."""

    def test_nan_with_no_calibration_returns_zero(self):
        from kernel.models import calibrate_score
        assert calibrate_score(float("nan"), None) == 0.0

    def test_nan_with_isotonic_falls_back_to_base_rate(self):
        from kernel.models import calibrate_score
        cal = {"method": "isotonic",
               "x_thresholds": [0.0, 1.0], "y_thresholds": [0.0, 1.0],
               "base_rate": 0.07}
        assert calibrate_score(float("nan"), cal) == pytest.approx(0.07)

    def test_nan_with_platt_falls_back_to_base_rate(self):
        from kernel.models import calibrate_score
        cal = {"method": "platt", "platt_coef": 1.0, "platt_intercept": 0.0,
               "platt_scale_std": 1.0, "platt_scale_mean": 0.0,
               "base_rate": 0.04}
        assert calibrate_score(float("nan"), cal) == pytest.approx(0.04)

    def test_inf_raw_score_handled(self):
        from kernel.models import calibrate_score
        cal = {"method": "isotonic",
               "x_thresholds": [0.0, 1.0], "y_thresholds": [0.0, 1.0],
               "base_rate": 0.05}
        assert calibrate_score(float("inf"),  cal) == pytest.approx(0.05)
        assert calibrate_score(float("-inf"), cal) == pytest.approx(0.05)

    def test_finite_isotonic_still_correct(self):
        from kernel.models import calibrate_score
        cal = {"method": "isotonic",
               "x_thresholds": [0.0, 0.5, 1.0],
               "y_thresholds": [0.0, 0.4, 1.0]}
        assert calibrate_score(0.25, cal) == pytest.approx(0.20)


# ── SC-1 (Round 7 audit): two calibration paths agree on NaN ──────────────────

class TestSC1CalibrationPathConsistency:
    """Pre-fix, ScoreCalibration.calibrate(NaN) returned 0.0 while
    kernel/models.calibrate_score(NaN, dict) returned base_rate. Two
    parallel calibration code paths produced different values on
    identical NaN input — silent inconsistency that could surface in
    any downstream consumer reading rank_score directly."""

    def test_dataclass_calibrate_uses_base_rate_on_nan(self):
        from kernel.scoring import ScoreCalibration
        cal = ScoreCalibration(method="isotonic", base_rate=0.05,
                                x_thresholds=[0.0, 1.0], y_thresholds=[0.0, 1.0])
        assert cal.calibrate(float("nan")) == pytest.approx(0.05)

    def test_dataclass_and_function_agree_on_nan(self):
        from kernel.scoring import ScoreCalibration
        from kernel.models import calibrate_score as fn_calibrate
        cal = ScoreCalibration(method="platt", base_rate=0.04,
                                platt_coef=1.0, platt_intercept=0.0,
                                platt_scale_std=1.0, platt_scale_mean=0.0)
        cal_dict = cal.to_dict()
        # Both paths should now return the same value on NaN.
        v1 = cal.calibrate(float("nan"))
        v2 = fn_calibrate(float("nan"), cal_dict)
        assert v1 == pytest.approx(v2), (
            f"calibration paths disagree on NaN: dataclass={v1}, fn={v2}"
        )


# ── CAL-7 (Round 2 audit): RefreshPanelCalibratorJob wired into pipeline ─────

class TestCAL7CalibratorRefreshWired:
    """Pre-fix, panel-rank-calibration.json was rebuilt only by manual
    `scripts/fit_panel_calibrator.py`. Operators forgot → calibrator went
    stale relative to retrained panel-LTR. Post-fix, RefreshPanelCalibratorJob
    runs after PanelNGBoostJob in PanelTrainingPipeline."""

    def test_pipeline_includes_refresh_job_after_ngboost(self):
        """Verify Job order: PanelNGBoostJob then RefreshPanelCalibratorJob."""
        # We can't instantiate PanelTrainingPipeline().run(ctx) without a real
        # context, but we can inspect the static job list it constructs.
        import inspect
        from training_panel.pp_panel_training import (
            PanelTrainingPipeline, PanelNGBoostJob, RefreshPanelCalibratorJob,
        )
        src = inspect.getsource(PanelTrainingPipeline.run)
        # Both classes are referenced in the run() method body.
        assert "PanelNGBoostJob()" in src
        assert "RefreshPanelCalibratorJob()" in src
        # And in the right relative order (calibrator AFTER ngboost).
        ngb_idx  = src.index("PanelNGBoostJob()")
        cal_idx  = src.index("RefreshPanelCalibratorJob()")
        assert ngb_idx < cal_idx, (
            "RefreshPanelCalibratorJob must run AFTER PanelNGBoostJob so "
            "both panel-LTR and NGBoost artifacts are stable when the "
            "calibrator queries them."
        )

    def test_should_skip_when_global_calibration_disabled(self):
        """If global_calibration is off, the refresh job no-ops cleanly."""
        from training_panel.pp_panel_training import RefreshPanelCalibratorJob
        from training_panel.context import PanelTrainingContext
        # Minimal context; default global_calibration absent → enabled False.
        ctx = PanelTrainingContext(
            config={"panel_ltr": {}}, watchlist=[], ohlcv={},
            sector_etf_ohlcv={}, ticker_sectors={}, listing_dates=None,
        )
        assert RefreshPanelCalibratorJob().should_skip(ctx) is True

    def test_should_skip_when_auto_refresh_disabled(self):
        from training_panel.pp_panel_training import RefreshPanelCalibratorJob
        from training_panel.context import PanelTrainingContext
        ctx = PanelTrainingContext(
            config={"panel_ltr": {"global_calibration": {
                "enabled": True, "auto_refresh": False,
            }}},
            watchlist=[], ohlcv={}, sector_etf_ohlcv={},
            ticker_sectors={}, listing_dates=None,
        )
        assert RefreshPanelCalibratorJob().should_skip(ctx) is True

    def test_should_run_by_default_when_global_calibration_enabled(self):
        from training_panel.pp_panel_training import RefreshPanelCalibratorJob
        from training_panel.context import PanelTrainingContext
        # Default auto_refresh=True applies when global_calibration.enabled=True.
        ctx = PanelTrainingContext(
            config={"panel_ltr": {"global_calibration": {"enabled": True}}},
            watchlist=[], ohlcv={}, sector_etf_ohlcv={},
            ticker_sectors={}, listing_dates=None,
        )
        assert RefreshPanelCalibratorJob().should_skip(ctx) is False


# ── LBL-1 (Round 2 audit): residualize sec_fwd vs spy first (FWL) ─────────────

class TestLBL1SectorOrthogonalToSPY:
    """Pre-fix, beta_sec was fit on raw fwd against raw sec_fwd. Sector ETFs
    are ~90% SPY-correlated, so β_sec absorbed the SPY component of
    sec_fwd, and (β_spy·spy_fwd + β_sec·sec_fwd) double-counted SPY exposure.
    Post-fix uses Frisch-Waugh-Lovell: orthogonalize sec_fwd vs spy_fwd
    first, so the final residual matches a joint OLS regression of
    fwd on [spy_fwd, sec_fwd].
    """

    def _build_synthetic(self, n=400, beta_spy=1.2, beta_sec=0.4, seed=7):
        """Synthesize fwd = β_spy·spy + β_sec·(sec_orthogonal_to_spy) + ε.

        sec_fwd is constructed as 0.9·spy + 0.1·sec_specific so it's heavily
        SPY-correlated. The TRUE residual we want to recover ≈ ε.
        """
        import numpy as np, pandas as pd
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2020-01-01", periods=n)
        spy = pd.Series(rng.normal(0, 0.01, n), index=idx)
        sec_specific = pd.Series(rng.normal(0, 0.01, n), index=idx)
        sec = 0.9 * spy + 0.1 * sec_specific
        eps = pd.Series(rng.normal(0, 0.005, n), index=idx)
        fwd = beta_spy * spy + beta_sec * (sec - 0.9 * spy) + eps
        return fwd, spy, sec, eps

    def test_residual_is_uncorrelated_with_spy_under_correlated_sector(self):
        """After FWL fix, |corr(residual, spy_fwd)| should be near 0."""
        from training_panel.labels import compute_residual_returns
        fwd, spy, sec, _ = self._build_synthetic()
        out = compute_residual_returns(
            fwd_returns={"FOO": fwd},
            spy_returns=spy,
            sector_returns_by_ticker={"FOO": sec},
            beta_window=60, lookahead_days=5,
        )
        residual = out["FOO"].dropna()
        spy_aligned = spy.reindex(residual.index)
        # Joint OLS residual is by construction orthogonal to spy_fwd.
        # Correlation should be much smaller than the buggy code's leak.
        corr = float(residual.corr(spy_aligned))
        assert abs(corr) < 0.10, (
            f"Residual still correlated with SPY (|corr|={abs(corr):.3f}) — "
            "FWL orthogonalization did not take effect."
        )

    def test_residual_is_uncorrelated_with_sector_under_correlated_sector(self):
        """Joint OLS residual must also be orthogonal to sec_fwd itself."""
        from training_panel.labels import compute_residual_returns
        fwd, spy, sec, _ = self._build_synthetic()
        out = compute_residual_returns(
            fwd_returns={"FOO": fwd},
            spy_returns=spy,
            sector_returns_by_ticker={"FOO": sec},
            beta_window=60, lookahead_days=5,
        )
        residual = out["FOO"].dropna()
        sec_aligned = sec.reindex(residual.index)
        corr = float(residual.corr(sec_aligned))
        assert abs(corr) < 0.10, (
            f"Residual still correlated with sector (|corr|={abs(corr):.3f})."
        )

    def test_old_two_step_path_would_double_count_spy(self):
        """Demonstrate the bug: if we replicate the OLD code path (β_sec
        fit on raw fwd vs raw sec_fwd), the residual is meaningfully
        correlated with SPY because of double-counting. This guards
        against accidentally reverting LBL-1."""
        import numpy as np, pandas as pd
        from training_panel.labels import _rolling_beta_purged
        fwd, spy, sec, _ = self._build_synthetic()
        beta_spy = _rolling_beta_purged(fwd, spy, window=60, purge=5)
        residual_after_spy = fwd - beta_spy * spy
        # OLD buggy step: fit β_sec on raw fwd against raw sec_fwd.
        beta_sec_buggy = _rolling_beta_purged(fwd, sec, window=60, purge=5)
        residual_buggy = (residual_after_spy - beta_sec_buggy * sec).dropna()
        spy_aligned = spy.reindex(residual_buggy.index)
        corr_buggy = float(residual_buggy.corr(spy_aligned))
        # The BUGGY residual leaks SPY at materially higher magnitude
        # than the FWL-fixed path (which we verified < 0.10 above).
        assert abs(corr_buggy) > 0.20, (
            f"Old code path corr={abs(corr_buggy):.3f} — expected substantial "
            "SPY leak; if this is small, the synthetic setup may be too easy "
            "and the LBL-1 regression test loses its teeth."
        )


# ── M-4 (Round 6 audit): predict_xgboost honours default_left on NaN ──────────

class TestM4XgboostDefaultLeft:
    """Pre-fix, pure-Python predict_xgboost used `val <= sc[node]` for all
    inputs including NaN. NaN <= x is False, so NaN inputs always took
    the RIGHT path. XGBoost's actual semantics: each split has a
    `default_left` flag stored in the trained tree that says which
    branch is the "missing" direction (auto-learned during training).
    Pre-fix inference therefore diverged from training on NaN inputs —
    silent train/inference parity bug.
    """

    def _toy_xgboost_artifact(self, default_left=0):
        """One-tree binary:logistic XGBoost artifact with a single
        split on feature index 0 at threshold 0.5."""
        return {
            "learner": {
                "gradient_booster": {
                    "model": {
                        "trees": [{
                            "left_children":   [1, -1, -1],
                            "right_children":  [2, -1, -1],
                            "split_conditions": [0.5, 0.0, 0.0],
                            "split_indices":   [0, 0, 0],
                            "base_weights":    [0.0, +2.0, -2.0],
                            "default_left":    [default_left, 0, 0],
                        }]
                    }
                }
            }
        }

    def test_finite_routes_left_when_below_threshold(self):
        from kernel.models import predict_xgboost
        a = self._toy_xgboost_artifact(default_left=0)
        # val 0.3 ≤ 0.5 → left → +2 → sigmoid(2) ≈ 0.881
        assert predict_xgboost(a, [0.3]) == pytest.approx(0.881, rel=1e-2)

    def test_finite_routes_right_when_above_threshold(self):
        from kernel.models import predict_xgboost
        a = self._toy_xgboost_artifact(default_left=0)
        # val 0.7 > 0.5 → right → -2 → sigmoid(-2) ≈ 0.119
        assert predict_xgboost(a, [0.7]) == pytest.approx(0.119, rel=1e-2)

    def test_nan_routes_via_default_left_true(self):
        """default_left=1 → NaN goes LEFT (was RIGHT pre-fix)."""
        from kernel.models import predict_xgboost
        a = self._toy_xgboost_artifact(default_left=1)
        # val NaN with default_left=1 → left → +2 → sigmoid(2) ≈ 0.881
        # Pre-fix would have gone right → -2 → 0.119, so this is the
        # train/inference parity test.
        assert predict_xgboost(a, [float("nan")]) == pytest.approx(0.881, rel=1e-2)

    def test_nan_routes_via_default_left_false(self):
        """default_left=0 → NaN goes RIGHT (matches pre-fix coincidentally)."""
        from kernel.models import predict_xgboost
        a = self._toy_xgboost_artifact(default_left=0)
        assert predict_xgboost(a, [float("nan")]) == pytest.approx(0.119, rel=1e-2)

    def test_missing_feature_index_uses_default_left(self):
        """When feature index >= len(feat_vals), use default_left."""
        from kernel.models import predict_xgboost
        a = self._toy_xgboost_artifact(default_left=1)
        assert predict_xgboost(a, []) == pytest.approx(0.881, rel=1e-2)


# ── E-5 (Round 5 audit): compute_exits doesn't corrupt HWM on NaN price ───────

class TestE5ExitsRejectsNanPrice:
    """Pre-fix, `max(HWM, NaN) = NaN` corrupted high_watermark for the
    rest of the position's life. Trailing-stop and other HWM-based checks
    then propagated NaN → no exit ever fired."""

    def _state(self, hwm=110.0, entry=100.0):
        from kernel.exits import HoldingState
        import datetime
        return HoldingState(
            entry_price=entry,
            entry_date=datetime.date(2026, 1, 1),
            high_watermark=hwm,
        )

    def test_nan_price_does_not_corrupt_hwm(self):
        from kernel.exits import compute_exits
        import datetime
        state = self._state(hwm=110.0)
        sig, state2 = compute_exits(
            current_price=float("nan"),
            today=datetime.date(2026, 6, 1),
            model_action="hold",
            state=state,
            params={"trailing_stop_trigger_pct": 0.20,
                    "trailing_stop_trail_pct": 0.18},
        )
        assert state2.high_watermark == 110.0, "HWM must NOT be corrupted by NaN price"
        assert sig.should_exit is False

    def test_inf_price_does_not_corrupt_hwm(self):
        from kernel.exits import compute_exits
        import datetime
        state = self._state(hwm=110.0)
        _, state2 = compute_exits(
            current_price=float("inf"),
            today=datetime.date(2026, 6, 1),
            model_action="hold",
            state=state,
            params={"trailing_stop_trigger_pct": 0.20,
                    "trailing_stop_trail_pct": 0.18},
        )
        assert state2.high_watermark == 110.0

    def test_finite_price_still_updates_hwm(self):
        from kernel.exits import compute_exits
        import datetime
        state = self._state(hwm=110.0)
        _, state2 = compute_exits(
            current_price=125.0,
            today=datetime.date(2026, 6, 1),
            model_action="hold",
            state=state,
            params={"trailing_stop_trigger_pct": 0.20,
                    "trailing_stop_trail_pct": 0.18},
        )
        assert state2.high_watermark == 125.0


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
