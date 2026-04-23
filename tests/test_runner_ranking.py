"""
Tests for ranking, calibration, and blend weights.

The old integration tests for run_once_multi (TestModelScoreRanking,
TestRegressionGuards, TestTieredThresholds, TestBatchPositionFetch,
TestLiveRegimeParity, TestOhlcvFreshness, TestRunnerMaxHold,
TestRunnerOversizeFallback, TestOpenOrdersGuard) were removed when the
runner was rewritten to delegate to the kernel pipeline.  Equivalent
integration coverage now lives in tests/test_pipeline.py.

This file retains the pure-unit tests that don't depend on the old runner
helpers (_get_model_score, _get_rank_score, _ensure_fresh_ohlcv, etc.):
  - ScoreCalibration monotonicity (kernel.scoring / training.scoring)
  - Blend weight logistic estimation (scripts.recalibrate_scores)
  - predict_score_bulk for all model types (training.models)

Run with:
    cd /path/to/RenQuant
    python -m pytest tests/test_runner_ranking.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

STRATEGY_DIR = ROOT / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from kernel.scoring import ScoreCalibration
from training.scoring import fit_probability_calibration
from scripts.recalibrate_scores import _compute_blend_weights


# ── ScoreCalibration ──────────────────────────────────────────────────────────

class TestScoreCalibration:
    def test_fit_probability_calibration_is_monotonic(self):
        raw = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        future = pd.Series([-0.05, -0.03, -0.01, 0.01, 0.05, 0.08])
        calibration = fit_probability_calibration(
            pd.concat([raw] * 30, ignore_index=True),
            pd.concat([future] * 30, ignore_index=True),
            lookahead=5,
            threshold=0.02,
            score_kind="bag_learner_raw",
        )

        # 180 samples (30×6) → Platt range; both Platt and isotonic are monotone
        assert calibration.method in ("isotonic", "platt")
        assert calibration.calibrate(-1.0) <= calibration.calibrate(2.0)

    def test_calibration_calibrate_uses_thresholds(self):
        cal = ScoreCalibration(
            method="isotonic",
            score_kind="vote_count",
            x_thresholds=[0.0, 1.0, 2.0],
            y_thresholds=[0.10, 0.40, 0.85],
        )
        assert cal.calibrate(2.0) == pytest.approx(0.85)
        assert cal.calibrate(0.0) == pytest.approx(0.10)

    def test_calibration_identity_fallback(self):
        """ScoreCalibration with method='identity' passes the raw score through."""
        cal = ScoreCalibration(method="identity", score_kind="stub_raw")
        assert cal.calibrate(0.77) == pytest.approx(0.77)


# ── Blend weights ─────────────────────────────────────────────────────────────

class TestBlendWeights:
    def test_logistic_blend_weights_favor_stronger_signal(self):
        rank_scores = np.linspace(0.05, 0.95, 200)
        rs_scores = np.linspace(0.05, 0.95, 200)
        outcomes = (rank_scores > 0.55).astype(float)

        w_rank, w_rs = _compute_blend_weights([
            {
                "rank_scores": rank_scores,
                "rs_scores": rs_scores[::-1],
                "outcomes": outcomes,
            }
        ])

        assert w_rank > w_rs

    def test_logistic_blend_weights_fallback_when_labels_too_thin(self):
        rank_scores = np.linspace(0.1, 0.9, 50)
        rs_scores = np.linspace(0.2, 0.8, 50)
        outcomes = np.ones(50)

        w_rank, w_rs = _compute_blend_weights([
            {
                "rank_scores": rank_scores,
                "rs_scores": rs_scores,
                "outcomes": outcomes,
            }
        ])

        assert (w_rank, w_rs) == (0.5, 0.5)


class TestBlendScoresTaskIgnoresRs:
    """BlendScoresTask hardcodes (1.0, 0.0) — rs_score is no longer blended."""

    def _make_ctx(self, blend_weights=None):
        import datetime as dt
        from kernel.pipeline.context import InferenceContext
        from kernel.selection import CandidateResult
        ranking: dict = {}
        if blend_weights is not None:
            ranking["blend_weights"] = blend_weights
        ctx = InferenceContext(config={"ranking": ranking}, today=dt.date(2026, 4, 22))
        ctx.candidates = [
            # rank=0.1, rs=0.9 — if rs still counted it'd win. rank=0.9 must win.
            CandidateResult(ticker="RS_ONLY",   raw_score=0, rank_score=0.1,
                            rs_score=0.9, detail="", expected_return=0),
            CandidateResult(ticker="RANK_ONLY", raw_score=0, rank_score=0.9,
                            rs_score=0.1, detail="", expected_return=0),
        ]
        return ctx

    def test_blend_weights_hardcoded(self):
        from kernel.pipeline.task_ranking import BlendScoresTask
        ctx = self._make_ctx()
        BlendScoresTask().run(ctx)
        assert ctx._blend_w == (1.0, 0.0)  # noqa: SLF001

    def test_ranking_reflects_rank_only(self):
        """The rank_score ordering must match, regardless of rs_score."""
        from kernel.pipeline.task_ranking import BlendScoresTask, SortCandidatesTask
        ctx = self._make_ctx()
        BlendScoresTask().run(ctx)
        SortCandidatesTask().run(ctx)
        assert [c.ticker for c in ctx.ranked] == ["RANK_ONLY", "RS_ONLY"]

    def test_legacy_config_rs_weight_warns_and_ignored(self, caplog):
        """Stale `blend_weights` with non-zero rs contribution is logged and ignored."""
        import logging as _l
        from kernel.pipeline.task_ranking import BlendScoresTask
        ctx = self._make_ctx(blend_weights=[0.5, 0.5])  # stale
        with caplog.at_level(_l.WARNING, logger="kernel.pipeline.ranking"):
            BlendScoresTask().run(ctx)
        assert any("legacy ranking.blend_weights" in r.message for r in caplog.records)
        assert ctx._blend_w == (1.0, 0.0)  # noqa: SLF001

    def test_legacy_config_zero_rs_no_warning(self, caplog):
        """`blend_weights=[1,0]` is the silent no-op case — no warning needed."""
        import logging as _l
        from kernel.pipeline.task_ranking import BlendScoresTask
        ctx = self._make_ctx(blend_weights=[1.0, 0.0])
        with caplog.at_level(_l.WARNING, logger="kernel.pipeline.ranking"):
            BlendScoresTask().run(ctx)
        assert not any("legacy ranking.blend_weights" in r.message for r in caplog.records)


# ── predict_score_bulk on all model types ─────────────────────────────────────

class TestPredictScoreBulkAllModels:
    """Confirm predict_score_bulk works on every model type (CLAUDE.md guarantee)."""

    @pytest.fixture
    def feature_cols(self):
        return ["rsi", "macd_hist", "cci"]

    @pytest.fixture
    def train_df(self, feature_cols):
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(rng.normal(0, 1, (n, len(feature_cols))), columns=feature_cols)
        df["close"] = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
        return df

    def test_classification_predict_score_bulk(self, train_df, feature_cols):
        from training.models import create_model

        model = create_model(
            "classification", feature_columns=feature_cols,
            lookahead=5, threshold=0.02, leaf_size=10, bags=5,
            buy_threshold=0.1, sell_threshold=-0.1,
        )
        model.train(train_df)
        scores = model.predict_score_bulk(train_df)
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(train_df)

    def test_qlearning_predict_score_bulk(self, train_df, feature_cols):
        from training.models import create_model

        train_df_q = train_df.copy()
        train_df_q["position_flag"] = 0
        model = create_model("qlearning", feature_columns=feature_cols, n_bins=5, n_epochs=10)
        model.train(train_df_q)
        scores = model.predict_score_bulk(train_df_q)
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(train_df_q)

    def test_xgboost_predict_score_bulk(self, train_df, feature_cols):
        from training.models import create_model

        model = create_model(
            "xgboost", feature_columns=feature_cols,
            lookahead=5, threshold=0.02, buy_threshold=0.55, sell_threshold=0.55,
        )
        model.train(train_df)
        scores = model.predict_score_bulk(train_df)
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(train_df)
        assert scores.between(-1.0, 1.0).all()

    def test_manual_predict_score_bulk(self, train_df):
        from training.models import create_model

        rules = [
            {"col": "rsi",      "buy_below": 0.0, "sell_above": 1.0},
            {"col": "macd_hist","buy_above": 0.0, "sell_below": 0.0},
        ]
        model = create_model("manual", score_rules=rules, buy_threshold=1, sell_threshold=-1)
        scores = model.predict_score_bulk(train_df)
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(train_df)
        assert scores.dtype == float or np.issubdtype(scores.dtype, np.floating)
