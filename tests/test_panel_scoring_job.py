"""Tests for kernel/panel_pipeline/job_panel_scoring.py — inference Job.

Covers each Task (LoadScorerTask, BuildFeatureMatrixTask, ApplyScoresTask)
and the wrapping PanelScoringJob, using a minimal real xgboost artifact
so tests run fast without a full training loop.
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


# ── Shared fixtures ──────────────────────────────────────────────────────────

FEATURE_COLS = ["f1", "f2", "size_z"]


def _train_mini_booster():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, len(FEATURE_COLS)))
    # Label ≈ f1 so scorer has a learnable signal.
    y = (X[:, 0] + 0.1 * rng.normal(size=60) > 0).astype(int)
    d = xgb.DMatrix(X, label=y)  # no feature_names — matches training/inference code
    booster = xgb.train(
        {"objective": "binary:logistic", "eval_metric": "logloss", "verbosity": 0},
        d, num_boost_round=5,
    )
    return booster


def _write_artifact(path: Path, feature_cols=None):
    """Write a minimal panel-LTR artifact compatible with PanelScorer.load()."""
    booster = _train_mini_booster()
    raw = bytes(booster.save_raw(raw_format="json")).decode("utf-8")
    payload = {
        "version": 1,
        "trained_date": "2026-04-21",
        "feature_cols": list(feature_cols or FEATURE_COLS),
        "params": {"objective": "binary:logistic"},
        "best_iter": None,
        "booster_raw_json": raw,
        "oos_mean_ic": 0.09,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _make_feature_frames(tickers, n_days=60):
    """Per-ticker feature frames with the f1/f2 columns expected by the artifact."""
    dates = pd.bdate_range("2026-01-01", periods=n_days)
    rng = np.random.default_rng(1)
    frames = {}
    for t in tickers:
        frames[t] = pd.DataFrame({
            "f1": rng.normal(size=n_days),
            "f2": rng.normal(size=n_days),
        }, index=dates)
    return frames, dates


def _make_factor_frames(tickers, dates):
    """Per-ticker factor frames providing size_z (the other artifact feature)."""
    rng = np.random.default_rng(2)
    frames = {}
    for t in tickers:
        frames[t] = pd.DataFrame({
            "size_z": rng.normal(size=len(dates)),
        }, index=dates)
    return frames


def _candidate(ticker: str, rank_score: float = 0.5):
    from kernel.selection import CandidateResult
    return CandidateResult(
        ticker=ticker, raw_score=0.0, rank_score=rank_score,
        rs_score=0.0, detail="", expected_return=0.0,
    )


def _make_ctx(tmp_path, *, enabled=True, artifact_path="panel-ltr.json",
              tickers=("AAA", "BBB", "CCC"), with_frames=True):
    """Build a minimal InferenceContext with candidates + panel config."""
    from kernel.pipeline.context import InferenceContext

    cfg = {
        "ranking": {
            "panel_scoring": {
                "enabled": enabled,
                "artifact_path": artifact_path,
                "nan_prone_cols": [],
            },
        },
        "_strategy_dir": str(tmp_path),
    }
    today = datetime.date(2026, 3, 20)
    ctx = InferenceContext(config=cfg, today=today)
    ctx.candidates = [_candidate(t, rank_score=0.1 * (i + 1))
                      for i, t in enumerate(tickers)]
    if with_frames:
        ff, dates = _make_feature_frames(list(tickers))
        fac = _make_factor_frames(list(tickers), dates)
        ctx._panel_feature_frames = ff
        ctx._panel_factor_frames = fac
    return ctx


# ── LoadScorerTask ───────────────────────────────────────────────────────────

class TestLoadScorerTask:
    def test_returns_false_when_disabled(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import LoadScorerTask
        ctx = _make_ctx(tmp_path, enabled=False)
        out = LoadScorerTask().run(ctx)
        assert out is False
        assert getattr(ctx, "_panel_scorer", None) is None

    def test_loads_artifact_with_absolute_path(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import LoadScorerTask
        artifact = _write_artifact(tmp_path / "artifacts" / "panel-ltr.json")
        ctx = _make_ctx(tmp_path, enabled=True, artifact_path=str(artifact))
        LoadScorerTask().run(ctx)
        assert ctx._panel_scorer is not None
        assert ctx._panel_scorer.feature_cols == FEATURE_COLS

    def test_resolves_relative_path_via_strategy_dir(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import LoadScorerTask
        _write_artifact(tmp_path / "artifacts" / "panel-ltr.json")
        ctx = _make_ctx(tmp_path, enabled=True, artifact_path="artifacts/panel-ltr.json")
        LoadScorerTask().run(ctx)
        assert ctx._panel_scorer is not None

    def test_preloaded_scorer_not_reloaded(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import LoadScorerTask
        ctx = _make_ctx(tmp_path, enabled=True, artifact_path="missing.json")
        sentinel = object()
        ctx._panel_scorer = sentinel
        LoadScorerTask().run(ctx)
        assert ctx._panel_scorer is sentinel  # not overwritten

    def test_returns_false_when_no_artifact_path(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import LoadScorerTask
        ctx = _make_ctx(tmp_path, enabled=True, artifact_path="")
        out = LoadScorerTask().run(ctx)
        assert out is False

    def test_returns_false_when_load_fails(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import LoadScorerTask
        ctx = _make_ctx(tmp_path, enabled=True,
                        artifact_path=str(tmp_path / "nonexistent.json"))
        out = LoadScorerTask().run(ctx)
        assert out is False


# ── BuildFeatureMatrixTask ───────────────────────────────────────────────────

class TestBuildFeatureMatrixTask:
    def _ctx_with_scorer(self, tmp_path, tickers=("AAA", "BBB", "CCC")):
        from kernel.panel_pipeline.panel_scorer import PanelScorer
        artifact = _write_artifact(tmp_path / "artifacts" / "panel-ltr.json")
        ctx = _make_ctx(tmp_path, enabled=True,
                        artifact_path=str(artifact), tickers=tickers)
        ctx._panel_scorer = PanelScorer.load(artifact)
        return ctx

    def test_returns_false_without_candidates(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import BuildFeatureMatrixTask
        ctx = self._ctx_with_scorer(tmp_path)
        ctx.candidates = []
        out = BuildFeatureMatrixTask().run(ctx)
        assert out is False

    def test_returns_false_without_scorer(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import BuildFeatureMatrixTask
        ctx = self._ctx_with_scorer(tmp_path)
        ctx._panel_scorer = None
        out = BuildFeatureMatrixTask().run(ctx)
        assert out is False

    def test_returns_false_when_feature_frames_missing(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import BuildFeatureMatrixTask
        ctx = self._ctx_with_scorer(tmp_path)
        ctx._panel_feature_frames = None
        out = BuildFeatureMatrixTask().run(ctx)
        assert out is False

    def test_builds_matrix_keyed_by_candidate_ticker(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import BuildFeatureMatrixTask
        tickers = ("AAA", "BBB", "CCC")
        ctx = self._ctx_with_scorer(tmp_path, tickers=tickers)
        BuildFeatureMatrixTask().run(ctx)
        assert ctx._panel_matrix is not None
        assert set(ctx._panel_matrix.index) == set(tickers)
        assert list(ctx._panel_matrix.columns) == FEATURE_COLS

    def test_restricts_to_candidate_tickers(self, tmp_path):
        """Feature frames may contain more tickers than candidates — keep only candidates."""
        from kernel.panel_pipeline.job_panel_scoring import BuildFeatureMatrixTask
        ctx = self._ctx_with_scorer(tmp_path, tickers=("AAA", "BBB", "CCC"))
        # Add an extra non-candidate ticker to feature frames
        extra_dates = ctx._panel_feature_frames["AAA"].index
        ctx._panel_feature_frames["XXX"] = pd.DataFrame({
            "f1": np.zeros(len(extra_dates)),
            "f2": np.zeros(len(extra_dates)),
        }, index=extra_dates)
        BuildFeatureMatrixTask().run(ctx)
        assert "XXX" not in ctx._panel_matrix.index


# ── ApplyScoresTask ──────────────────────────────────────────────────────────

class TestApplyScoresTask:
    def _ctx_ready(self, tmp_path, tickers=("AAA", "BBB", "CCC")):
        from kernel.panel_pipeline.job_panel_scoring import (
            BuildFeatureMatrixTask, LoadScorerTask,
        )
        artifact = _write_artifact(tmp_path / "artifacts" / "panel-ltr.json")
        ctx = _make_ctx(tmp_path, enabled=True,
                        artifact_path=str(artifact), tickers=tickers)
        LoadScorerTask().run(ctx)
        BuildFeatureMatrixTask().run(ctx)
        return ctx

    def test_returns_false_without_prereqs(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import ApplyScoresTask
        ctx = _make_ctx(tmp_path, enabled=True)
        out = ApplyScoresTask().run(ctx)
        assert out is False

    def test_overwrites_candidate_rank_scores(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import ApplyScoresTask
        ctx = self._ctx_ready(tmp_path)
        before = {c.ticker: c.rank_score for c in ctx.candidates}
        ApplyScoresTask().run(ctx)
        after = {c.ticker: c.rank_score for c in ctx.candidates}
        # Every ticker got a new score; not equal to the original placeholder
        assert set(after.keys()) == set(before.keys())
        # Scores should differ across tickers (non-trivial scorer)
        assert len(set(after.values())) > 1

    def test_leaves_rank_score_untouched_when_nan(self, tmp_path):
        """Tickers absent from the matrix or with NaN score keep their prior rank_score."""
        from kernel.panel_pipeline.job_panel_scoring import ApplyScoresTask
        ctx = self._ctx_ready(tmp_path)
        # Drop one ticker from the matrix — scorer won't produce a score for it
        dropped = ctx.candidates[0].ticker
        prior = ctx.candidates[0].rank_score
        ctx._panel_matrix = ctx._panel_matrix.drop(index=dropped)
        ApplyScoresTask().run(ctx)
        survived = next(c for c in ctx.candidates if c.ticker == dropped)
        assert survived.rank_score == pytest.approx(prior)


# ── PanelScoringJob ──────────────────────────────────────────────────────────

class TestPanelScoringJob:
    def test_should_skip_when_no_candidates(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import PanelScoringJob
        ctx = _make_ctx(tmp_path, enabled=True)
        ctx.candidates = []
        assert PanelScoringJob().should_skip(ctx) is True

    def test_should_skip_when_disabled(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import PanelScoringJob
        ctx = _make_ctx(tmp_path, enabled=False)
        assert PanelScoringJob().should_skip(ctx) is True

    def test_should_not_skip_when_enabled_with_candidates(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import PanelScoringJob
        ctx = _make_ctx(tmp_path, enabled=True)
        assert PanelScoringJob().should_skip(ctx) is False

    def test_tasks_are_eight_in_order(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import (
            ApplyGlobalCalibrationTask, ApplyNGBoostTask, ApplyScoresTask,
            BuildFeatureMatrixTask, LoadGlobalCalibrationTask,
            LoadNGBoostTask, LoadScorerTask, PanelScoringJob, VetoWeakBuysTask,
        )
        tasks = PanelScoringJob().tasks
        assert len(tasks) == 8
        assert isinstance(tasks[0], LoadScorerTask)
        assert isinstance(tasks[1], BuildFeatureMatrixTask)
        assert isinstance(tasks[2], ApplyScoresTask)
        assert isinstance(tasks[3], VetoWeakBuysTask)
        assert isinstance(tasks[4], LoadGlobalCalibrationTask)
        assert isinstance(tasks[5], ApplyGlobalCalibrationTask)
        assert isinstance(tasks[6], LoadNGBoostTask)
        assert isinstance(tasks[7], ApplyNGBoostTask)

    def test_end_to_end_overrides_rank_scores(self, tmp_path):
        """Run the full Job chain — candidate rank_scores change to panel scores."""
        from kernel.panel_pipeline.job_panel_scoring import PanelScoringJob
        artifact = _write_artifact(tmp_path / "artifacts" / "panel-ltr.json")
        ctx = _make_ctx(tmp_path, enabled=True, artifact_path=str(artifact))
        before = [c.rank_score for c in ctx.candidates]

        job = PanelScoringJob()
        # Run job — the Job base class executes tasks sequentially; if any
        # task returns False the chain short-circuits.
        for task in job.tasks:
            result = task.run(ctx)
            if result is False:
                break

        after = [c.rank_score for c in ctx.candidates]
        assert before != after

    def test_end_to_end_disabled_is_noop(self, tmp_path):
        """When disabled, running the job chain leaves rank_scores intact."""
        from kernel.panel_pipeline.job_panel_scoring import PanelScoringJob
        ctx = _make_ctx(tmp_path, enabled=False)
        before = [c.rank_score for c in ctx.candidates]
        job = PanelScoringJob()
        assert job.should_skip(ctx) is True
        after = [c.rank_score for c in ctx.candidates]
        assert before == after
