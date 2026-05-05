"""Tests for training_panel/pp_panel_training.py — Job/Task pipeline.

Covers each Task and Job individually, then the full orchestrator.
Uses synthetic OHLCV so tests run fast and deterministic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


# ── Shared fixtures ──────────────────────────────────────────────────────────

def _synthetic_ohlcv(n_days=300, n_tickers=5, seed=0):
    """Produce synthetic OHLCV dict (tickers + SPY + 2 sector ETFs)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)

    def _make_close(drift=0.0, vol=0.01):
        rets = rng.normal(drift, vol, size=n_days)
        return pd.Series(100.0 * np.exp(np.cumsum(rets)), index=dates)

    ohlcv = {}
    tickers = [f"T{i}" for i in range(n_tickers)]
    for i, t in enumerate(tickers):
        close = _make_close(drift=0.0005 + 0.0001 * i, vol=0.02)
        ohlcv[t] = pd.DataFrame({
            "open":  close * 0.995,
            "high":  close * 1.01,
            "low":   close * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 10_000_000, size=n_days),
        }, index=dates)

    # Benchmark
    ohlcv["SPY"] = ohlcv[tickers[0]].copy()
    # Sector ETFs
    ohlcv["XLK"] = ohlcv[tickers[1]].copy()
    ohlcv["XLF"] = ohlcv[tickers[2]].copy()
    return ohlcv, tickers


def _make_context(tmp_path, ticker_count=5):
    from training_panel.context import PanelTrainingContext

    ohlcv, tickers = _synthetic_ohlcv(n_tickers=ticker_count)
    sector_etf_ohlcv = {"tech": ohlcv["XLK"], "finance": ohlcv["XLF"]}
    ticker_sectors = {t: ("tech" if i % 2 == 0 else "finance")
                      for i, t in enumerate(tickers)}
    config = {
        "benchmark": "SPY",
        "sector_etf_map": {"tech": "XLK", "finance": "XLF"},
        "indicator_spec": {"rsi": {"period": 14}, "macd": {}, "adx": {}},
        "model_params": {"lookahead": 5, "threshold": 0.01},
        "panel_ltr": {
            "lookahead_days": 5,
            "beta_window": 20,
            "min_history_days": 60,
            "age_warmup_days": 120,
            "cv_n_splits": 3,
            "cv_embargo_days": 5,
            "num_boost_round": 20,
            "neutralize_features": True,
            "factor_mom_window": 60,
            "factor_skip": 5,
            "neutralize_rolling_window": 60,
            "neutralize_warmup_days": 60,
            # BUG-CV-2 (2026-04-28): synthetic test data is too small for
            # best_iter ≥ 20. Disable the guard for fixture; production
            # configs leave min_best_iter at default 20.
            "min_best_iter": 0,
        },
        "_strategy_dir": str(tmp_path),
    }
    ctx = PanelTrainingContext(
        config=config,
        watchlist=tickers,
        ohlcv=ohlcv,
        sector_etf_ohlcv=sector_etf_ohlcv,
        ticker_sectors=ticker_sectors,
    )
    return ctx, tickers


# ── Phase 1 — PanelDataJob / SectorMomentumTask ──────────────────────────────

class TestSectorMomentumTask:
    def test_populates_sector_momentum(self, tmp_path):
        from training_panel.pp_panel_training import SectorMomentumTask
        ctx, _ = _make_context(tmp_path)
        SectorMomentumTask().run(ctx)
        assert set(ctx.sector_momentum.keys()) == {"tech", "finance"}
        for sec, df in ctx.sector_momentum.items():
            assert "mom_20d" in df.columns

    def test_skips_when_already_populated(self, tmp_path):
        from training_panel.pp_panel_training import SectorMomentumTask
        ctx, _ = _make_context(tmp_path)
        ctx.sector_momentum = {"sentinel": pd.DataFrame()}
        SectorMomentumTask().run(ctx)
        assert "sentinel" in ctx.sector_momentum  # untouched


class TestPanelDataJob:
    def test_should_skip_when_ohlcv_and_momentum_present(self, tmp_path):
        from training_panel.pp_panel_training import (
            PanelDataJob, SectorMomentumTask,
        )
        ctx, _ = _make_context(tmp_path)
        SectorMomentumTask().run(ctx)
        assert PanelDataJob().should_skip(ctx) is True

    def test_chain_runs_without_network(self, tmp_path):
        """OHLCV pre-populated → FetchOHLCVTask is a no-op, SectorMomentumTask runs."""
        from training_panel.pp_panel_training import PanelDataJob
        ctx, _ = _make_context(tmp_path)
        # Simulate "not yet populated" by clearing momentum only
        ctx.sector_momentum = {}
        PanelDataJob().run(ctx)
        assert ctx.sector_momentum  # now populated


# ── Phase 2 — PanelFeatureJob + per-ticker chain ─────────────────────────────

class TestPanelFeatureJob:
    def test_parallel_chain_populates_three_frame_dicts(self, tmp_path):
        from training_panel.pp_panel_training import (
            SectorMomentumTask, PanelFeatureJob,
        )
        ctx, tickers = _make_context(tmp_path)
        SectorMomentumTask().run(ctx)
        PanelFeatureJob().run(ctx)

        assert ctx.feature_frames, "Feature frames missing"
        assert ctx.neutralized_frames, "Neutralized frames missing"
        assert ctx.raw_factor_frames, "Raw factor frames missing"
        # Per-ticker outputs should align
        for t in ctx.feature_frames:
            assert t in ctx.neutralized_frames
            assert t in ctx.raw_factor_frames

    def test_skip_when_already_populated(self, tmp_path):
        from training_panel.pp_panel_training import PanelFeatureJob
        ctx, _ = _make_context(tmp_path)
        ctx.feature_frames = {"T0": pd.DataFrame({"x": [1]})}
        ctx.neutralized_frames = {"T0": pd.DataFrame({"x": [1]})}
        ctx.raw_factor_frames = {"T0": pd.DataFrame({"size": [1]})}
        assert PanelFeatureJob().should_skip(ctx) is True


class TestTickerPanelJobs:
    def test_feature_job_sets_feature_frame(self, tmp_path):
        from training_panel.pp_panel_training import (
            TickerPanelContext, TickerPanelFeatureJob,
        )
        ctx, tickers = _make_context(tmp_path)
        tc = TickerPanelContext(
            ticker=tickers[0], ohlcv=ctx.ohlcv,
            sector_momentum={}, ticker_sectors=ctx.ticker_sectors,
            config=ctx.config,
        )
        TickerPanelFeatureJob().run(tc)
        assert tc.feature_frame is not None
        assert "label" in tc.feature_frame.columns

    def test_neutralize_job_preserves_index(self, tmp_path):
        from training_panel.pp_panel_training import (
            SectorMomentumTask, TickerPanelContext,
            TickerPanelFeatureJob, TickerPanelNeutralizeJob,
        )
        ctx, tickers = _make_context(tmp_path)
        SectorMomentumTask().run(ctx)
        tc = TickerPanelContext(
            ticker=tickers[0], ohlcv=ctx.ohlcv,
            sector_momentum=ctx.sector_momentum,
            ticker_sectors=ctx.ticker_sectors,
            config=ctx.config,
        )
        TickerPanelFeatureJob().run(tc)
        TickerPanelNeutralizeJob().run(tc)
        assert tc.neutralized_frame is not None
        assert tc.neutralized_frame.index.equals(tc.feature_frame.index)

    def test_factor_job_produces_expected_columns(self, tmp_path):
        from training_panel.pp_panel_training import (
            TickerPanelContext, TickerPanelFactorJob,
        )
        ctx, tickers = _make_context(tmp_path)
        tc = TickerPanelContext(
            ticker=tickers[0], ohlcv=ctx.ohlcv,
            sector_momentum={}, ticker_sectors=ctx.ticker_sectors,
            config=ctx.config,
        )
        TickerPanelFactorJob().run(tc)
        assert tc.raw_factor_frame is not None
        expected = {
            # Core factors
            "size", "mom_12_1", "beta_60d", "resid_mom",
            # Round 3 orthogonal factors
            "amihud_illiq", "volume_shift", "price_to_high",
            "realized_vol", "drawdown_peak",
            # 2026-05-03 added factors (factors.py landed):
            # Ang 2006 IVOL puzzle + Jegadeesh 1990 1-month reversal
            "idio_vol", "mom_1m_reversal",
        }
        assert set(tc.raw_factor_frame.columns) == expected


# ── Phase 3 — PanelAssemblyJob tasks ─────────────────────────────────────────

def _run_through_phase2(ctx):
    from training_panel.pp_panel_training import (
        SectorMomentumTask, PanelFeatureJob,
    )
    SectorMomentumTask().run(ctx)
    PanelFeatureJob().run(ctx)


class TestFactorZScoreTask:
    def test_produces_z_scored_frames(self, tmp_path):
        from training_panel.pp_panel_training import FactorZScoreTask
        ctx, _ = _make_context(tmp_path)
        _run_through_phase2(ctx)
        FactorZScoreTask().run(ctx)
        assert ctx.factor_frames
        expected = {
            "size_z", "mom_12_1_z", "beta_60d_z", "resid_mom_z",
            # Round 3 orthogonal factor z-scores
            "amihud_illiq_z", "volume_shift_z", "price_to_high_z",
            "realized_vol_z", "drawdown_peak_z",
            # 2026-05-03 new factor z-scores (Ang IVOL + 1mo reversal)
            "idio_vol_z", "mom_1m_reversal_z",
        }
        for t, df in ctx.factor_frames.items():
            assert set(df.columns) == expected

    def test_skips_when_already_populated(self, tmp_path):
        from training_panel.pp_panel_training import FactorZScoreTask
        ctx, _ = _make_context(tmp_path)
        ctx.factor_frames = {"sentinel": pd.DataFrame()}
        FactorZScoreTask().run(ctx)
        assert list(ctx.factor_frames.keys()) == ["sentinel"]


class TestLabelsTask:
    def test_builds_labels(self, tmp_path):
        from training_panel.pp_panel_training import LabelsTask
        ctx, _ = _make_context(tmp_path)
        _run_through_phase2(ctx)
        LabelsTask().run(ctx)
        assert ctx.labels
        for t, series in ctx.labels.items():
            assert isinstance(series, pd.Series)


class TestBuildPanelTask:
    def test_produces_panel_with_required_columns(self, tmp_path):
        from training_panel.pp_panel_training import (
            FactorZScoreTask, LabelsTask, BuildPanelTask,
        )
        ctx, _ = _make_context(tmp_path)
        _run_through_phase2(ctx)
        FactorZScoreTask().run(ctx)
        LabelsTask().run(ctx)
        BuildPanelTask().run(ctx)
        assert ctx.panel is not None
        for col in ("date", "ticker", "label", "weight"):
            assert col in ctx.panel.columns
        assert ctx.feature_cols  # non-empty
        assert len(ctx.group_sizes) == ctx.panel["date"].nunique()


class TestPanelAssemblyJob:
    def test_runs_all_three_tasks_in_order(self, tmp_path):
        from training_panel.pp_panel_training import PanelAssemblyJob
        ctx, _ = _make_context(tmp_path)
        _run_through_phase2(ctx)
        PanelAssemblyJob().run(ctx)
        assert ctx.factor_frames and ctx.labels and ctx.panel is not None


# ── Phase 4 — PanelModelJob tasks ────────────────────────────────────────────

def _run_through_phase3(ctx):
    from training_panel.pp_panel_training import PanelAssemblyJob
    _run_through_phase2(ctx)
    PanelAssemblyJob().run(ctx)


class TestCrossValidateTask:
    def test_produces_cv_result_with_mean_ic(self, tmp_path):
        from training_panel.pp_panel_training import CrossValidateTask
        ctx, _ = _make_context(tmp_path)
        _run_through_phase3(ctx)
        CrossValidateTask().run(ctx)
        assert "mean_ic" in ctx.cv_result
        assert "per_fold_ic" in ctx.cv_result
        assert len(ctx.cv_result["per_fold_ic"]) == 3  # n_splits


class TestFinalFitTask:
    def test_produces_trained_model(self, tmp_path):
        from training_panel.pp_panel_training import (
            CrossValidateTask, FinalFitTask,
        )
        ctx, _ = _make_context(tmp_path)
        _run_through_phase3(ctx)
        CrossValidateTask().run(ctx)
        FinalFitTask().run(ctx)
        assert ctx.final_model is not None
        assert ctx.final_model.booster is not None


class TestSaveArtifactTask:
    def test_writes_json_artifact(self, tmp_path):
        from training_panel.pp_panel_training import (
            CrossValidateTask, FinalFitTask, SaveArtifactTask,
        )
        ctx, _ = _make_context(tmp_path)
        ctx.config["panel_ltr"]["artifact_path"] = str(tmp_path / "panel.json")
        _run_through_phase3(ctx)
        CrossValidateTask().run(ctx)
        FinalFitTask().run(ctx)
        SaveArtifactTask().run(ctx)
        assert ctx.artifact_path.exists()
        assert ctx.summary["mean_ic"] == ctx.cv_result["mean_ic"]


class TestPanelModelJob:
    def test_chain_runs_all_three_tasks(self, tmp_path):
        from training_panel.pp_panel_training import PanelModelJob
        ctx, _ = _make_context(tmp_path)
        ctx.config["panel_ltr"]["artifact_path"] = str(tmp_path / "panel.json")
        _run_through_phase3(ctx)
        PanelModelJob().run(ctx)
        assert ctx.cv_result
        assert ctx.final_model is not None
        assert ctx.artifact_path.exists()


# ── Orchestrator end-to-end ──────────────────────────────────────────────────

class TestPanelTrainingPipeline:
    def test_full_pipeline_end_to_end(self, tmp_path):
        from training_panel.pp_panel_training import PanelTrainingPipeline
        ctx, _ = _make_context(tmp_path)
        ctx.config["panel_ltr"]["artifact_path"] = str(tmp_path / "panel.json")
        PanelTrainingPipeline().run(ctx)
        assert ctx.panel is not None
        assert ctx.final_model is not None
        assert ctx.artifact_path.exists()
        assert "mean_ic" in ctx.summary

    def test_idempotent_reruns_skip_completed_jobs(self, tmp_path):
        from training_panel.pp_panel_training import PanelTrainingPipeline
        ctx, _ = _make_context(tmp_path)
        ctx.config["panel_ltr"]["artifact_path"] = str(tmp_path / "panel.json")
        PanelTrainingPipeline().run(ctx)
        first_artifact = ctx.artifact_path
        PanelTrainingPipeline().run(ctx)  # rerun — should skip everything via should_skip
        assert ctx.artifact_path == first_artifact
