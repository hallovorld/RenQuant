"""Regression tests for panel-LTR bugs fixed in the April 2026 session.

Bugs covered:

1. `ApplyGlobalCalibrationTask` used to be a no-op whenever
   `ngboost.enabled=True`, but NGBoost only overrides `rank_score` in
   `score_mode="mu_minus_lambda_sigma"`. In `"additive"` mode the global
   calibrator must still run — otherwise rank_score stays as the raw
   (uncalibrated) panel score, which is typically ≪ the 0.10 tier
   threshold and produces zero trades.
   (Bug caused zero trades in panel-on sim after the first retrain.)

2. `prepare_inference_panel_frames` (the shared adapter-side
   feature-prep helper used by LEAN / live runner / SimAdapter) used to
   skip `NeutralizedFeatureZScoreTask`. Since the panel was trained on
   z-scored per-ticker indicators, inference saw raw (unscaled) values —
   panel scores became noise, calibrator pool_IC dropped to ~0.
   (Bug caused the calibrator to be re-fit on the wrong distribution.)

3. `fit_panel_calibrator.py` also skipped the cross-sectional z-score
   step for the same reason — same fix applied to the script.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


# ── Bug #1: ApplyGlobalCalibrationTask must run in additive mode ────────────

class TestApplyGlobalCalibrationTask:
    """Per 2026-04-23 task #2 reorder (commit 339944b): calibration ALWAYS
    runs after ApplyNGBoostTask, regardless of score_mode. So rank_score
    ends as the calibrated probability in both additive and μ−λσ modes.

    In μ−λσ mode ApplyNGBoostTask writes `μ − λσ` onto panel_score before
    this task fires, so the calibrator maps `μ − λσ → probability` using
    the same isotonic (same scale). The original "defers to NGBoost" test
    reflected pre-reorder semantics where calibration short-circuited in
    μ−λσ mode — that behavior was removed because it left rank_score as
    raw μ−λσ ∈ [~-0.06, +0.04], always below the 0.10 tier floor.
    """

    def _make_ctx(self, ngboost_enabled: bool, score_mode: str):
        from kernel.pipeline.context import InferenceContext
        from kernel.selection import CandidateResult
        cfg = {
            "ranking": {
                "panel_scoring": {
                    "enabled": True,
                    "global_calibration": {"enabled": True, "artifact_path": "fake.json"},
                    "ngboost": {"enabled": ngboost_enabled, "score_mode": score_mode},
                },
            },
        }
        ctx = InferenceContext(config=cfg, today=pd.Timestamp("2024-06-04").date())
        ctx.candidates = [
            CandidateResult(ticker="AAA", raw_score=0.02, rank_score=0.02,
                            rs_score=0.0, detail=""),
        ]
        ctx.candidates[0].panel_score = 0.02
        ctx.holdings = {}

        # Fake calibrator that maps ANY input to 0.75 (well above any tier floor)
        class _FakeCalib:
            def calibrate_probability(self, x):
                return 0.75
            def expected_return(self, x):
                return 0.03
        ctx._global_calibrator = _FakeCalib()
        return ctx

    def test_additive_mode_runs_calibration(self):
        """In additive mode, rank_score should be rewritten to the calibrated probability."""
        from kernel.panel_pipeline.job_panel_scoring import ApplyGlobalCalibrationTask
        ctx = self._make_ctx(ngboost_enabled=True, score_mode="additive")
        ApplyGlobalCalibrationTask().run(ctx)
        assert ctx.candidates[0].rank_score == pytest.approx(0.75)

    def test_mu_minus_lambda_sigma_also_runs_calibration(self):
        """In μ−λσ mode, calibration runs too. NGBoost writes μ−λσ onto
        panel_score first; the calibrator then maps that through the
        isotonic (same scale → directionally monotone result)."""
        from kernel.panel_pipeline.job_panel_scoring import ApplyGlobalCalibrationTask
        ctx = self._make_ctx(ngboost_enabled=True, score_mode="mu_minus_lambda_sigma")
        # rank_score starts as 0.02 (raw panel score written by ApplyScoresTask)
        assert ctx.candidates[0].rank_score == pytest.approx(0.02)
        ApplyGlobalCalibrationTask().run(ctx)
        # Post-reorder: calibration runs, rank_score becomes the fake's 0.75
        assert ctx.candidates[0].rank_score == pytest.approx(0.75)

    def test_ngboost_disabled_runs_calibration(self):
        """When NGBoost is disabled entirely, calibration must run."""
        from kernel.panel_pipeline.job_panel_scoring import ApplyGlobalCalibrationTask
        ctx = self._make_ctx(ngboost_enabled=False, score_mode="additive")
        ApplyGlobalCalibrationTask().run(ctx)
        assert ctx.candidates[0].rank_score == pytest.approx(0.75)


# ── Bug #2: prepare_inference_panel_frames must z-score per-ticker indicators ─

class TestPrepareInferenceFramesZScoring:
    """Inference-side feature prep must apply the same cross-sectional z-score
    the panel was trained with. Otherwise AAPL's RSI=70 and BRK's RSI=70 go in
    at different scales than the training distribution."""

    def test_prepare_function_calls_neutralized_zscore_task(self):
        """Source-level enforcement: the function body must reference the task."""
        from training_panel import pipeline as tp
        import inspect
        src = inspect.getsource(tp.prepare_inference_panel_frames)
        assert "NeutralizedFeatureZScoreTask" in src, \
            "prepare_inference_panel_frames must run NeutralizedFeatureZScoreTask " \
            "or panel inference will receive raw (unscaled) per-ticker indicators"
        assert "LoadFundamentalsTask" in src, \
            "prepare_inference_panel_frames must also LoadFundamentalsTask so the " \
            "panel's fundamental z-scores are populated at inference"

    def test_prepare_applies_zscore_end_to_end(self):
        """End-to-end: run prepare_inference_panel_frames and verify output
        columns have been cross-sectionally centered (mean ≈ 0 per date)."""
        from training_panel.pipeline import prepare_inference_panel_frames

        # Build 3 synthetic tickers with very different RSI distributions so we
        # can detect whether cross-sectional z-scoring happened.
        def make_ohlcv(seed, rsi_shift):
            rng = np.random.default_rng(seed)
            n = 300
            idx = pd.bdate_range("2023-01-02", periods=n)
            close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n)))
            vol   = np.ones(n) * 1e6
            return pd.DataFrame({
                "open": close, "high": close * 1.01, "low": close * 0.99,
                "close": close, "volume": vol,
            }, index=idx)

        ohlcv = {
            "AAA": make_ohlcv(1, rsi_shift=0),
            "BBB": make_ohlcv(2, rsi_shift=20),
            "CCC": make_ohlcv(3, rsi_shift=-10),
            "SPY": make_ohlcv(4, rsi_shift=0),
            "XLK": make_ohlcv(5, rsi_shift=5),
        }
        cfg = {
            "benchmark": "SPY",
            "sector_etf_map": {"tech": "XLK"},
            "sector_map": {"AAA": "tech", "BBB": "tech", "CCC": "tech"},
            "panel_ltr": {
                "fundamentals": {"enabled": False},
            },
            "indicator_spec": {},  # defaults
            "_strategy_dir": str(_STRATEGY_DIR),
        }
        ff, fac, macro = prepare_inference_panel_frames(
            watchlist=["AAA", "BBB", "CCC"], ohlcv=ohlcv,
            ticker_sectors={"AAA": "tech", "BBB": "tech", "CCC": "tech"},
            config=cfg,
        )
        # Bug #25 contract: 3rd return value is macro_frame (None when disabled)
        assert macro is None, "macro frame should be None when flag disabled"
        # For each date where all 3 tickers have RSI, cross-sectional mean of
        # the post-z-score value should be near 0 (perfect only with infinite
        # precision; allow slack).
        aaa_rsi = ff["AAA"]["rsi"]
        bbb_rsi = ff["BBB"]["rsi"]
        ccc_rsi = ff["CCC"]["rsi"]
        common_idx = aaa_rsi.dropna().index.intersection(
            bbb_rsi.dropna().index).intersection(ccc_rsi.dropna().index)
        assert len(common_idx) > 50, "fixture should produce plenty of common dates"
        for d in common_idx[-10:]:
            vals = [float(aaa_rsi.loc[d]), float(bbb_rsi.loc[d]), float(ccc_rsi.loc[d])]
            assert abs(sum(vals)) < 1e-6, \
                f"cross-sectional z-score of rsi should sum to ~0 per date (got {vals} on {d})"


# ── Bug #3: fit_panel_calibrator.py also z-scores before scoring ────────────

class TestFitPanelCalibratorZScoring:
    """Source-level enforcement for the script: must invoke the z-score task."""

    def test_fit_panel_calibrator_applies_zscore(self):
        src_path = Path(__file__).resolve().parent.parent / "scripts" / "fit_panel_calibrator.py"
        src = src_path.read_text()
        assert "NeutralizedFeatureZScoreTask" in src, \
            "fit_panel_calibrator.py must apply NeutralizedFeatureZScoreTask before " \
            "scoring — otherwise the calibrator is fit on raw (unscaled) inputs"
