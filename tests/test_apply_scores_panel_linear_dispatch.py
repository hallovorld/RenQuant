"""Regression test for ApplyScoresTask alpha158_linear dispatch (Phase 3).

When `scorer.metadata["kind"] == "panel_linear"`, ApplyScoresTask must
build alpha158 features inline from `ctx.ohlcv` and call `score_raw()`
(which applies stored ZScoreNorm + Fillna + Clip), instead of using the
21-feature matrix that BuildFeatureMatrixJob produced for XGB.

Without this, a panel_linear scorer would receive feature_cols mismatch
and crash silently in `score()`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


class TestApplyScoresPanelLinearDispatch:
    def _make_ohlcv(self, n_bars: int = 100, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2024-01-01", periods=n_bars)
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

    def test_panel_linear_dispatch_invokes_score_raw(self):
        from kernel.panel_pipeline.job_panel_scoring import ApplyScoresTask
        from training_panel.linear_ltr import PanelLinearScorer
        from kernel.panel_pipeline.alpha158_features import alpha158_feature_names

        feat_cols = alpha158_feature_names()
        # Tiny scorer: coef=0 → score=0 for everyone (smoke). Stats are
        # identity-ish so score_raw doesn't blow up on numerical edge cases.
        scorer = PanelLinearScorer(
            coef=np.zeros(len(feat_cols)),
            intercept=0.5,
            feature_cols=feat_cols,
            feature_means=np.zeros(len(feat_cols)),
            feature_stds=np.ones(len(feat_cols)),
        )
        scorer.metadata = {"kind": "panel_linear"}

        # Mock candidates + holdings + ohlcv
        ctx = SimpleNamespace(
            _panel_scorer=scorer,
            # X is the dummy 21-feature matrix from BuildFeatureMatrixJob;
            # ApplyScoresTask should IGNORE it for panel_linear and rebuild.
            _panel_matrix=pd.DataFrame(
                {"adx": [0.0, 0.0, 0.0]},
                index=["AAPL", "MSFT", "GOOG"],
            ),
            ohlcv={
                "AAPL": self._make_ohlcv(seed=1),
                "MSFT": self._make_ohlcv(seed=2),
                "GOOG": self._make_ohlcv(seed=3),
            },
            today=pd.Timestamp("2024-04-30"),
            candidates=[
                SimpleNamespace(ticker="AAPL", rank_score=None, panel_score=None),
                SimpleNamespace(ticker="MSFT", rank_score=None, panel_score=None),
            ],
            holdings={
                "GOOG": SimpleNamespace(panel_score=None),
            },
        )
        ApplyScoresTask().run(ctx)

        # Each candidate / holding should have panel_score=0.5 (= intercept)
        for cand in ctx.candidates:
            assert cand.panel_score == pytest.approx(0.5, abs=1e-6), \
                f"{cand.ticker}: panel_score={cand.panel_score}"
            assert cand.rank_score == pytest.approx(0.5, abs=1e-6)
        for ticker, hs in ctx.holdings.items():
            assert hs.panel_score == pytest.approx(0.5, abs=1e-6), \
                f"{ticker}: panel_score={hs.panel_score}"

    def test_xgb_path_unchanged_for_legacy_scorer(self):
        """Regression: panel_xgb scorers must use the legacy `score(X)`
        path with the 21-feature matrix (not invoke score_raw)."""
        from kernel.panel_pipeline.job_panel_scoring import ApplyScoresTask

        # Mock an XGB-like scorer (no `kind` metadata, has score method)
        scorer = MagicMock()
        scorer.metadata = {}   # No kind → default XGB
        scorer.score = MagicMock(return_value=pd.Series(
            [0.7, 0.3], index=["AAPL", "MSFT"], name="panel_score",
        ))
        ctx = SimpleNamespace(
            _panel_scorer=scorer,
            _panel_matrix=pd.DataFrame(
                {"adx": [1.0, 2.0], "cci": [0.5, -0.5]},
                index=["AAPL", "MSFT"],
            ),
            candidates=[
                SimpleNamespace(ticker="AAPL", rank_score=None, panel_score=None),
                SimpleNamespace(ticker="MSFT", rank_score=None, panel_score=None),
            ],
            holdings={},
        )
        ApplyScoresTask().run(ctx)
        # XGB path: scorer.score called with the 21-col matrix
        assert scorer.score.called
        scored_X = scorer.score.call_args[0][0]
        # Same shape as input — alpha158 reconstruction did NOT happen
        assert scored_X.shape == ctx._panel_matrix.shape
        assert ctx.candidates[0].panel_score == pytest.approx(0.7)
        assert ctx.candidates[1].panel_score == pytest.approx(0.3)
