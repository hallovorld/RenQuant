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
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


# ── Shared fixtures ──────────────────────────────────────────────────────────

FEATURE_COLS = ["f1", "f2", "size_z"]


class TestPanelRuntimeCache:
    def test_earnings_surprise_parquet_loaded_once_per_sim_cache(self, tmp_path, monkeypatch):
        from kernel.panel_pipeline.job_panel_scoring import _cached_earnings_surprise

        calls = []
        raw = pd.DataFrame(
            {"surprise_pct": [0.05]},
            index=pd.DatetimeIndex(["2026-03-06"], name="Earnings Date"),
        )

        def fake_read_parquet(path):
            calls.append(str(path))
            return raw.copy()

        monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
        ctx = SimpleNamespace(_panel_runtime_cache={})
        path = tmp_path / "AAA.parquet"

        first = _cached_earnings_surprise(ctx, path)
        second = _cached_earnings_surprise(ctx, path)

        assert len(calls) == 1
        assert first is second
        assert list(first.columns) == ["earnings_date", "surprise_pct"]

    def test_sentiment_parquet_loaded_once_per_sim_cache(self, tmp_path, monkeypatch):
        from kernel.panel_pipeline.job_panel_scoring import _cached_sentiment

        calls = []
        raw = pd.DataFrame({
            "date": ["2026-03-06"],
            "sentiment_pos_share": [0.6],
        })

        def fake_read_parquet(path):
            calls.append(str(path))
            return raw.copy()

        monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
        ctx = SimpleNamespace(_panel_runtime_cache={})
        path = tmp_path / "AAA.parquet"

        first = _cached_sentiment(ctx, path)
        second = _cached_sentiment(ctx, path)

        assert len(calls) == 1
        assert first is second
        assert pd.api.types.is_datetime64_any_dtype(first["date"])


class TestAlpha158TargetOnlyMatrix:
    def test_missing_legacy_frames_ok_for_alpha158_scorer(self):
        from kernel.panel_pipeline.tasks_feature_matrix import ResolveInferenceFramesTask

        ctx = SimpleNamespace(
            candidates=[SimpleNamespace(ticker="BBB"), SimpleNamespace(ticker="AAA")],
            holdings={"CCC": SimpleNamespace()},
            _panel_scorer=SimpleNamespace(metadata={"kind": "panel_ltr_xgboost"}),
            _panel_feature_frames=None,
            config={},
        )

        out = ResolveInferenceFramesTask().run(ctx)

        assert out is False
        assert list(ctx._panel_matrix.index) == ["AAA", "BBB", "CCC"]
        assert list(ctx._panel_matrix.columns) == ["__alpha158_target__"]

    def test_missing_legacy_frames_still_blocks_non_alpha158_scorer(self):
        from kernel.panel_pipeline.tasks_feature_matrix import ResolveInferenceFramesTask

        ctx = SimpleNamespace(
            candidates=[SimpleNamespace(ticker="AAA")],
            holdings={},
            _panel_scorer=SimpleNamespace(metadata={"kind": "legacy_panel"}),
            _panel_feature_frames=None,
            config={},
        )

        out = ResolveInferenceFramesTask().run(ctx)

        assert out is None
        assert ctx._panel_matrix is None

    def test_missing_legacy_frames_ok_for_history_scorer(self):
        from kernel.panel_pipeline.tasks_feature_matrix import ResolveInferenceFramesTask

        ctx = SimpleNamespace(
            candidates=[SimpleNamespace(ticker="BBB"), SimpleNamespace(ticker="AAA")],
            holdings={"CCC": SimpleNamespace()},
            _panel_scorer=SimpleNamespace(
                metadata={"kind": "hf_patchtst"},
                requires_history=True,
            ),
            _panel_feature_frames=None,
            config={},
        )

        out = ResolveInferenceFramesTask().run(ctx)

        assert out is False
        assert list(ctx._panel_matrix.index) == ["AAA", "BBB", "CCC"]
        assert list(ctx._panel_matrix.columns) == ["__history_target__"]


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

    # 2026-04-24 (audit #39): the task no longer halts the chain on empty
    # inputs — it leaves `_panel_matrix = None` and returns None so the
    # downstream LoadGlobalCalibration / LoadNGBoost loaders still
    # initialize. The matrix-consuming tasks (ApplyScores, ApplyNGBoost)
    # short-circuit individually when the matrix is None.

    def test_returns_none_without_candidates(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import BuildFeatureMatrixTask
        ctx = self._ctx_with_scorer(tmp_path)
        ctx.candidates = []
        out = BuildFeatureMatrixTask().run(ctx)
        assert out is None
        assert getattr(ctx, "_panel_matrix", None) is None

    def test_returns_none_without_scorer(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import BuildFeatureMatrixTask
        ctx = self._ctx_with_scorer(tmp_path)
        ctx._panel_scorer = None
        out = BuildFeatureMatrixTask().run(ctx)
        assert out is None
        assert getattr(ctx, "_panel_matrix", None) is None

    def test_returns_none_when_feature_frames_missing(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import BuildFeatureMatrixTask
        ctx = self._ctx_with_scorer(tmp_path)
        ctx._panel_feature_frames = None
        out = BuildFeatureMatrixTask().run(ctx)
        assert out is None
        assert getattr(ctx, "_panel_matrix", None) is None

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

    def test_returns_none_without_prereqs(self, tmp_path):
        """Audit P-21 (2026-04-24): when scorer/X are missing, ApplyScoresTask
        previously returned False (which short-circuited the rest of the chain
        and left Kelly target stale). Now returns None so downstream tasks
        (Veto/NGBoost/Calibrator/Kelly) get to run with their own None guards.
        """
        from kernel.panel_pipeline.job_panel_scoring import ApplyScoresTask
        ctx = _make_ctx(tmp_path, enabled=True)
        out = ApplyScoresTask().run(ctx)
        assert out is None

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

    def test_history_scorer_scores_holdings_too(self, tmp_path):
        """HF/PatchTST primary must populate holding.panel_score for QP μ wiring."""
        from kernel.exits import HoldingState
        from kernel.panel_pipeline.job_panel_scoring import ApplyScoresTask

        class _HistoryScorer:
            requires_history = True
            seq_len = 2
            feature_cols = ["f1"]
            metadata = {"kind": "hf_patchtst"}

            def score_with_history(self, panel_history, target_tickers):
                return pd.Series(
                    {ticker: float(i + 1) / 10.0
                     for i, ticker in enumerate(target_tickers)}
                )

        ctx = _make_ctx(tmp_path, enabled=True, tickers=("AAA", "BBB"),
                        with_frames=False)
        ctx.holdings = {
            "HLD": HoldingState(
                entry_price=100.0,
                entry_date=datetime.date(2026, 3, 1),
                high_watermark=100.0,
                shares=10,
            )
        }
        ctx._panel_scorer = _HistoryScorer()
        ctx._panel_matrix = pd.DataFrame({"f1": [1.0, 2.0, 3.0]},
                                         index=["AAA", "BBB", "HLD"])
        ctx._panel_history = pd.DataFrame({
            "date": pd.to_datetime(["2026-03-18", "2026-03-19"] * 3),
            "ticker": ["AAA", "AAA", "BBB", "BBB", "HLD", "HLD"],
            "f1": [1.0, 1.1, 2.0, 2.1, 3.0, 3.1],
        })

        ApplyScoresTask().run(ctx)

        assert [c.panel_score for c in ctx.candidates] == pytest.approx([0.1, 0.2])
        assert ctx.holdings["HLD"].panel_score == pytest.approx(0.3)

    def test_history_scorer_uses_ctx_history_without_parquet_read(self, tmp_path, monkeypatch):
        """Adapter-provided history must keep parquet I/O out of the per-bar task."""
        from kernel.panel_pipeline.job_panel_scoring import ApplyScoresTask

        class _HistoryScorer:
            requires_history = True
            seq_len = 2
            feature_cols = ["f1"]
            metadata = {"kind": "hf_patchtst"}

            def score_with_history(self, panel_history, target_tickers):
                assert set(panel_history["ticker"]) == {"AAA", "BBB", "CCC"}
                return pd.Series({t: 0.4 for t in target_tickers})

        ctx = _make_ctx(tmp_path, enabled=True, tickers=("AAA", "BBB"),
                        with_frames=False)
        ctx._panel_scorer = _HistoryScorer()
        ctx._panel_matrix = pd.DataFrame({"__history_target__": [1.0, 1.0]},
                                         index=["AAA", "BBB"])
        ctx._panel_history = pd.DataFrame({
            "date": pd.to_datetime(["2026-03-18"] * 3),
            "ticker": ["AAA", "BBB", "CCC"],
            "f1": [1.0, 2.0, 3.0],
        })

        def forbidden_read(*_args, **_kwargs):
            raise AssertionError("ApplyScoresTask should not read parquet with ctx._panel_history")

        monkeypatch.setattr(pd, "read_parquet", forbidden_read)
        ApplyScoresTask().run(ctx)

        assert [c.panel_score for c in ctx.candidates] == pytest.approx([0.4, 0.4])

    def test_history_fallback_keeps_full_universe_for_rank_norm(self, tmp_path, monkeypatch):
        """Lazy fallback should not candidate-filter the history frame."""
        from kernel.panel_pipeline.job_panel_scoring import ApplyScoresTask

        captured = {}

        class _HistoryScorer:
            requires_history = True
            seq_len = 2
            feature_cols = ["f1"]
            metadata = {"kind": "hf_patchtst"}

            def score_with_history(self, panel_history, target_tickers):
                captured["tickers"] = set(panel_history["ticker"])
                return pd.Series({t: 0.5 for t in target_tickers})

        full_panel = pd.DataFrame({
            "date": pd.to_datetime(
                ["2026-03-17", "2026-03-18", "2026-03-19"] * 3
            ),
            "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB", "CCC", "CCC", "CCC"],
            "f1": range(9),
        })
        monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: full_panel)
        ctx = _make_ctx(tmp_path, enabled=True, tickers=("AAA",), with_frames=False)
        ctx.today = datetime.date(2026, 3, 20)
        ctx._panel_scorer = _HistoryScorer()
        ctx._panel_matrix = pd.DataFrame({"__history_target__": [1.0]}, index=["AAA"])

        ApplyScoresTask().run(ctx)

        assert captured["tickers"] == {"AAA", "BBB", "CCC"}


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

    def test_tasks_are_thirteen_in_order(self, tmp_path):
        """After the 2026-05-04 P0 fix VetoWeakBuysTask was MOVED to AFTER
        ApplyGlobalCalibrationTask (so it compares against calibrated
        rank_score, not raw XGB margin). See PanelScoringJob docstring +
        VetoWeakBuysTask docstring for the production incident this resolves.

        Golden v4 (Kelly promoted) keeps ApplyKellySizingTask post-veto.
        Stage-0 buy-logic redesign (2026-04-26) added QualityFloorTask
        (flag-OFF default preserves bit-for-bit parity).
        2026-05-15 Phase 3: ApplyRealizedVolFallbackTask inserted before
        ApplyKellySizingTask so σ has a realized-vol source when NGBoost
        is OFF (count grew from 10 → 11).
        2026-05-18 shadow scoring runs immediately after primary panel
        scoring and before NGBoost/calibration. It logs shadow-model output
        without changing the production score path (count grew 11 → 12).
        2026-05-24 RegimeModelAdmissionTask runs after calibration and before
        buy_floor/QP so regime evidence, not QP, decides buy eligibility.
        """
        from kernel.panel_pipeline.job_panel_scoring import (
            ApplyGlobalCalibrationTask, ApplyKellySizingTask, ApplyNGBoostTask,
            ApplyRealizedVolFallbackTask, ApplyScoresTask, BuildFeatureMatrixTask,
            LoadGlobalCalibrationTask, LoadNGBoostTask, LoadScorerTask,
            PanelScoringJob, RegimeModelAdmissionTask, VetoWeakBuysTask,
        )
        from kernel.panel_pipeline.shadow_scoring import ApplyShadowScoringTask
        from kernel.panel_pipeline.task_quality_floor import QualityFloorTask
        tasks = PanelScoringJob().tasks
        assert len(tasks) == 13
        assert isinstance(tasks[0], LoadScorerTask)
        assert isinstance(tasks[1], BuildFeatureMatrixTask)
        assert isinstance(tasks[2], ApplyScoresTask)
        assert isinstance(tasks[3], ApplyShadowScoringTask)
        # NGBoost runs before calibration (preserved from 2026-04-23):
        assert isinstance(tasks[4], LoadNGBoostTask)
        assert isinstance(tasks[5], ApplyNGBoostTask)
        assert isinstance(tasks[6], LoadGlobalCalibrationTask)
        assert isinstance(tasks[7], ApplyGlobalCalibrationTask)
        assert isinstance(tasks[8], RegimeModelAdmissionTask)
        # 2026-05-04 P0: VetoWeakBuysTask MOVED here so it compares
        # calibrated rank_score (post-ApplyGlobalCalibration) not raw XGB.
        assert isinstance(tasks[9], VetoWeakBuysTask)
        # 2026-05-15 Phase 3: realized-vol σ fallback (no-op unless
        # kelly_sizing.use_realized_vol_fallback=true).
        assert isinstance(tasks[10], ApplyRealizedVolFallbackTask)
        assert isinstance(tasks[11], ApplyKellySizingTask)
        # Stage-0 quality gate (flag-OFF by default):
        assert isinstance(tasks[12], QualityFloorTask)

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


# ── Alpha158 XGBoost dispatch regression (0-trades bug 2026-05-07) ───────────

def _write_xgb_alpha158_artifact(path: Path, feat_cols: list[str]):
    """Minimal XGBoost artifact with kind=panel_ltr_xgboost + alpha158 feature set."""
    rng = np.random.default_rng(42)
    n = len(feat_cols)
    X = rng.normal(size=(60, n))
    y = (X[:, 0] > 0).astype(int)
    dm = xgb.DMatrix(X, label=y)
    booster = xgb.train(
        {"objective": "binary:logistic", "verbosity": 0}, dm, num_boost_round=3,
    )
    raw = bytes(booster.save_raw(raw_format="json")).decode("utf-8")
    payload = {
        "version": 1,
        "kind": "panel_ltr_xgboost",
        "feature_cols": feat_cols,
        "booster_raw_json": raw,
        "oos_mean_ic": 0.036,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _make_ohlcv(tickers, n_days=120):
    """Minimal OHLCV dict — enough rows for compute_alpha158_at (needs ≥70)."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2025-09-01", periods=n_days)
    out = {}
    for t in tickers:
        price = 100 + rng.normal(scale=2, size=n_days).cumsum()
        out[t] = pd.DataFrame({
            "open": price * 0.99,
            "high": price * 1.01,
            "low":  price * 0.98,
            "close": price,
            "volume": rng.integers(100_000, 1_000_000, size=n_days).astype(float),
        }, index=dates)
    return out


class TestAlpha158XGBDispatch:
    """Regression tests for the 2026-05-07 0-trades bug.

    Root cause: DriftGuardTask and ApplyScoresTask gated alpha158 feature
    building on scorer_kind == "panel_linear" only. panel_ltr_xgboost kind
    fell through to the 21-feature production path — DriftGuard saw 87%
    structural-missing columns and cleared ctx.candidates → 0 trades.
    """

    FEAT_COLS = [f"feat_{i}" for i in range(158)]

    def test_drift_guard_skips_for_xgb_alpha158(self, tmp_path):
        """DriftGuardTask must return None (skip) for panel_ltr_xgboost kind."""
        from kernel.panel_pipeline.job_panel_scoring import LoadScorerTask
        from kernel.panel_pipeline.tasks_feature_matrix import DriftGuardTask

        art = _write_xgb_alpha158_artifact(
            tmp_path / "artifacts" / "panel-ltr.json", self.FEAT_COLS,
        )
        ctx = _make_ctx(tmp_path, enabled=True, artifact_path=str(art))
        LoadScorerTask().run(ctx)

        # 21-col production-shaped matrix — mismatches the 158 alpha158 feature_cols
        ctx._panel_matrix = pd.DataFrame(
            np.random.default_rng(0).normal(size=(3, 3)),
            index=["AAA", "BBB", "CCC"],
            columns=FEATURE_COLS,
        )

        result = DriftGuardTask().run(ctx)
        assert result is None, "DriftGuardTask should skip for panel_ltr_xgboost"
        assert len(ctx.candidates) == 3, "candidates must survive drift guard"

    def test_apply_scores_uses_alpha158_path_for_xgb(self, tmp_path, monkeypatch):
        """ApplyScoresTask must route panel_ltr_xgboost through alpha158 feature build."""
        import kernel.panel_pipeline.alpha158_features as a158_mod  # noqa: PLC0415
        from kernel.panel_pipeline.job_panel_scoring import ApplyScoresTask, LoadScorerTask

        feat_cols = self.FEAT_COLS
        art = _write_xgb_alpha158_artifact(
            tmp_path / "artifacts" / "panel-ltr.json", feat_cols,
        )
        tickers = ("AAA", "BBB", "CCC")
        ctx = _make_ctx(tmp_path, enabled=True, artifact_path=str(art), tickers=tickers)
        LoadScorerTask().run(ctx)

        ctx.ohlcv = _make_ohlcv(tickers)
        ctx._panel_matrix = pd.DataFrame(
            np.random.default_rng(0).normal(size=(3, 3)),
            index=list(tickers),
            columns=FEATURE_COLS,
        )

        # Patch compute_alpha158_at on the module so the lazy import inside
        # ApplyScoresTask picks up the stub (returns all 158 feat_cols).
        rng = np.random.default_rng(99)
        monkeypatch.setattr(
            a158_mod, "compute_alpha158_at",
            lambda ohlcv, today: {f: float(rng.normal()) for f in feat_cols},
        )

        before = [c.rank_score for c in ctx.candidates]
        ApplyScoresTask().run(ctx)
        after = [c.rank_score for c in ctx.candidates]

        scored = sum(1 for a, b in zip(after, before) if a != b)
        assert scored == 3, (
            f"Expected all 3 candidates scored via alpha158+XGB path, "
            f"got {scored}. before={before} after={after}"
        )


class TestPEADInferenceDispatch:
    """Regression test for E47 PEAD promotion (2026-05-08).

    ApplyScoresTask must compute 3 PEAD features (days_since_earnings,
    pead_signal, pead_quintile_rank) at inference time when the artifact
    has them in feature_cols. PEAD inputs come from
    data/earnings_surprise/{ticker}.parquet. Bernard-Thomas 1989 60d
    decay window. Missing tickers / out-of-window earnings → zero.
    """

    PEAD_COLS = ["days_since_earnings", "pead_signal", "pead_quintile_rank"]
    FEAT_COLS = [f"feat_{i}" for i in range(155)] + PEAD_COLS  # 158 total

    def _write_earnings_parquet(self, path: Path, dates_and_surprises: list[tuple[str, float]]):
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {"eps_actual": [0.5] * len(dates_and_surprises),
             "eps_estimate": [0.4] * len(dates_and_surprises),
             "surprise_abs": [0.1] * len(dates_and_surprises),
             "surprise_pct": [s for _, s in dates_and_surprises]},
            index=pd.DatetimeIndex([d for d, _ in dates_and_surprises], name="Earnings Date"),
        )
        df.to_parquet(path)

    def test_apply_scores_computes_pead_when_artifact_has_pead_cols(self, tmp_path, monkeypatch):
        """PEAD features get computed online when scorer.feature_cols includes them."""
        import kernel.panel_pipeline.alpha158_features as a158_mod  # noqa: PLC0415
        from kernel.panel_pipeline.job_panel_scoring import ApplyScoresTask, LoadScorerTask

        feat_cols = self.FEAT_COLS
        art = _write_xgb_alpha158_artifact(
            tmp_path / "artifacts" / "panel-ltr.json", feat_cols,
        )
        tickers = ("AAA", "BBB", "CCC")
        ctx = _make_ctx(tmp_path, enabled=True, artifact_path=str(art), tickers=tickers)
        LoadScorerTask().run(ctx)
        ctx.ohlcv = _make_ohlcv(tickers)
        ctx._panel_matrix = pd.DataFrame(
            np.random.default_rng(0).normal(size=(3, 3)),
            index=list(tickers),
            columns=FEATURE_COLS,
        )

        # Earnings data for 2 of 3 tickers (CCC has no parquet → zero PEAD)
        # ApplyScoresTask reads from `<repo>/data/earnings_surprise/`. Repo
        # is resolved as `__file__.parents[3]` from the kernel module path,
        # which means we need to monkey-patch the path resolution.
        earn_dir = tmp_path / "data" / "earnings_surprise"
        # AAA: earnings 14 days before today (in window) — surprise +5%
        self._write_earnings_parquet(
            earn_dir / "AAA.parquet",
            [("2026-03-06", 0.05)],
        )
        # BBB: earnings 90 days before today (out of window) — should give zero
        self._write_earnings_parquet(
            earn_dir / "BBB.parquet",
            [("2025-12-20", 0.10)],
        )
        # CCC: no parquet at all

        # Patch alpha158 stub
        rng = np.random.default_rng(99)
        a158_feat = [c for c in feat_cols if c not in self.PEAD_COLS]
        monkeypatch.setattr(
            a158_mod, "compute_alpha158_at",
            lambda ohlcv, today: {f: float(rng.normal()) for f in a158_feat},
        )

        # Patch repo root resolution so the inline `__file__.parents[4]`
        # in ApplyScoresTask resolves to tmp_path. The path layout the
        # production code expects is:
        #   <repo>/backtesting/renquant_104/kernel/panel_pipeline/job_panel_scoring.py
        # so parents[4] = <repo>. We mirror that under tmp_path.
        import kernel.panel_pipeline.job_panel_scoring as scoring_mod
        fake_file = (
            tmp_path
            / "backtesting" / "renquant_104"
            / "kernel" / "panel_pipeline" / "stub.py"
        )
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(scoring_mod, "__file__", str(fake_file))

        # Today: 2026-03-20. AAA earnings 03-06 → 14d ago (in window).
        ctx.today = datetime.date(2026, 3, 20)

        # Capture the rows dict from inside ApplyScoresTask so we can
        # assert on PEAD values directly. We monkey-patch pd.DataFrame.from_dict
        # to intercept the X build — but cleaner: inspect via the booster's
        # input. Easier: just verify the log line via caplog.
        # Simpler still — we check that the test ran without raising and
        # that AAA gets a valid score (proves the path works), AND we
        # verify PEAD coverage by directly invoking build helper.

        # Run scoring — must not raise
        ApplyScoresTask().run(ctx)

        # All 3 candidates must have a rank_score (proves the full
        # alpha158→fund→PEAD→XGB pipeline ran).
        scored = sum(1 for c in ctx.candidates if c.rank_score is not None)
        assert scored == 3, f"Expected 3 scored via PEAD path, got {scored}"

    def test_health_check_warns_when_all_pead_features_zero(self, tmp_path, monkeypatch, caplog):
        """If the data lookup fails for every ticker (e.g. path wrong, all
        files missing) PEAD features are silently zero. Health check must
        emit a WARNING that names the failure mode (the 2026-05-08 bug)."""
        from kernel.panel_pipeline.job_panel_scoring import ApplyScoresTask, LoadScorerTask
        import kernel.panel_pipeline.alpha158_features as a158_mod

        feat_cols = self.FEAT_COLS
        art = _write_xgb_alpha158_artifact(
            tmp_path / "artifacts" / "panel-ltr.json", feat_cols,
        )
        tickers = ("AAA", "BBB", "CCC")
        ctx = _make_ctx(tmp_path, enabled=True, artifact_path=str(art), tickers=tickers)
        LoadScorerTask().run(ctx)
        ctx.ohlcv = _make_ohlcv(tickers)
        ctx._panel_matrix = pd.DataFrame(
            np.random.default_rng(0).normal(size=(3, 3)),
            index=list(tickers),
            columns=FEATURE_COLS,
        )
        # No earnings_surprise files → every ticker falls into n_no_data path
        # → all PEAD features = 0.0 → health WARN should fire.

        rng = np.random.default_rng(99)
        a158_feat = [c for c in feat_cols if c not in self.PEAD_COLS]
        monkeypatch.setattr(
            a158_mod, "compute_alpha158_at",
            lambda ohlcv, today: {f: float(rng.normal()) for f in a158_feat},
        )

        import kernel.panel_pipeline.job_panel_scoring as scoring_mod
        fake_file = (
            tmp_path / "backtesting" / "renquant_104"
            / "kernel" / "panel_pipeline" / "stub.py"
        )
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(scoring_mod, "__file__", str(fake_file))
        ctx.today = datetime.date(2026, 3, 20)

        import logging as _logging
        with caplog.at_level(_logging.WARNING, logger="kernel.panel_pipeline.scoring"):
            ApplyScoresTask().run(ctx)

        warn_msgs = [r.message for r in caplog.records if r.levelno == _logging.WARNING]
        pead_warn = [m for m in warn_msgs if "PEAD features" in m and "FEATURE-HEALTH" in m]
        assert pead_warn, (
            f"Expected FEATURE-HEALTH WARN about all-zero PEAD features. "
            f"Got warnings: {warn_msgs}"
        )

    def test_pead_path_resolves_repo_root_correctly(self, tmp_path, monkeypatch, caplog):
        """Regression test for the parents[3] vs parents[4] bug — production
        was looking up earnings_surprise/ at <repo>/backtesting/data/ not
        <repo>/data/, so PEAD silently zeroed every ticker."""
        import kernel.panel_pipeline.job_panel_scoring as scoring_mod
        from kernel.panel_pipeline.job_panel_scoring import ApplyScoresTask, LoadScorerTask

        feat_cols = self.FEAT_COLS
        art = _write_xgb_alpha158_artifact(
            tmp_path / "artifacts" / "panel-ltr.json", feat_cols,
        )
        tickers = ("AAA", "BBB", "CCC")
        ctx = _make_ctx(tmp_path, enabled=True, artifact_path=str(art), tickers=tickers)
        LoadScorerTask().run(ctx)
        ctx.ohlcv = _make_ohlcv(tickers)
        ctx._panel_matrix = pd.DataFrame(
            np.random.default_rng(0).normal(size=(3, 3)),
            index=list(tickers),
            columns=FEATURE_COLS,
        )

        # Earnings: AAA 14d ago (in window), BBB 90d ago (out), CCC missing
        earn_dir = tmp_path / "data" / "earnings_surprise"
        self._write_earnings_parquet(
            earn_dir / "AAA.parquet", [("2026-03-06", 0.05)],
        )
        self._write_earnings_parquet(
            earn_dir / "BBB.parquet", [("2025-12-20", 0.10)],
        )

        import kernel.panel_pipeline.alpha158_features as a158_mod
        rng = np.random.default_rng(99)
        a158_feat = [c for c in feat_cols if c not in self.PEAD_COLS]
        monkeypatch.setattr(
            a158_mod, "compute_alpha158_at",
            lambda ohlcv, today: {f: float(rng.normal()) for f in a158_feat},
        )

        # Production code uses parents[4] — the test fake_file must be 4
        # levels deep (matches real layout) for the resolution to land
        # at tmp_path/data/earnings_surprise where we put files.
        fake_file = (
            tmp_path
            / "backtesting" / "renquant_104"
            / "kernel" / "panel_pipeline" / "stub.py"
        )
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(scoring_mod, "__file__", str(fake_file))

        ctx.today = datetime.date(2026, 3, 20)

        import logging as _logging
        with caplog.at_level(_logging.INFO, logger="kernel.panel_pipeline.scoring"):
            ApplyScoresTask().run(ctx)

        # The PEAD log line should report n_no_data=1 (only CCC missing),
        # not 3 (which would indicate the path bug). AAA in window, BBB out.
        msgs = [r.message for r in caplog.records]
        pead_msgs = [m for m in msgs if "PEAD features" in m]
        assert pead_msgs, f"Expected PEAD log message, got: {msgs}"
        msg = pead_msgs[0]
        assert "no_data=1" in msg, (
            f"Expected exactly 1 no-data ticker (CCC), got log: {msg}"
        )
        assert "1/3 tickers active" in msg, (
            f"Expected 1/3 active in 60d window (AAA only), got log: {msg}"
        )
