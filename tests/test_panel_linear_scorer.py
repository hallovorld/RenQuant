"""Regression tests for PanelLinearScorer (Phase 1 alpha158 integration).

Closes 2026-05-06 alpha158+Linear winner integration. Per CLAUDE.md §2,
every new feature ships with tests. Tests cover:

1. Save/load roundtrip (ndarray ↔ JSON)
2. score() returns correctly indexed Series
3. PanelScorer.load dispatches `kind: panel_linear` to PanelLinearScorer
4. NaN/inf input handling (Qlib Fillna semantics)
5. Missing-column detection (KeyError)
6. from_sklearn() factory works with both LinearRegression and Ridge
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


class TestPanelLinearScorerCore:
    def test_load_preserves_kind_in_metadata(self):
        """REGRESSION (2026-05-06): early load() excluded "kind" from
        metadata, so downstream dispatch (ApplyScoresTask + DriftGuardTask)
        couldn't detect panel_linear scorers → 0-trade sim over 128 days.
        Pin: kind MUST be in metadata after load.
        """
        import tempfile
        from training_panel.linear_ltr import PanelLinearScorer
        scorer = PanelLinearScorer(
            coef=np.array([1.0]), intercept=0.0,
            feature_cols=["a"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "scorer.json"
            scorer.save(p)
            loaded = PanelLinearScorer.load(p)
            assert loaded.metadata.get("kind") == "panel_linear", (
                f"load() lost kind from metadata: {loaded.metadata}"
            )

    def test_save_load_roundtrip(self):
        from training_panel.linear_ltr import PanelLinearScorer
        coef = np.array([1.0, -0.5, 0.25])
        scorer = PanelLinearScorer(
            coef=coef, intercept=0.1,
            feature_cols=["a", "b", "c"],
            metadata={"label": "fwd_5d_excess", "training_train_ic": 0.05},
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "panel-ltr.test.json"
            scorer.save(p)
            assert p.exists()
            loaded = PanelLinearScorer.load(p)
            assert np.allclose(loaded.coef, coef)
            assert loaded.intercept == 0.1
            assert loaded.feature_cols == ["a", "b", "c"]
            assert loaded.metadata.get("label") == "fwd_5d_excess"
            assert loaded.metadata.get("training_train_ic") == 0.05

    def test_score_returns_correct_shape(self):
        from training_panel.linear_ltr import PanelLinearScorer
        scorer = PanelLinearScorer(
            coef=np.array([2.0, -1.0]), intercept=0.5,
            feature_cols=["x", "y"],
        )
        X = pd.DataFrame(
            {"x": [1.0, 2.0, 3.0], "y": [0.5, 0.0, -1.0]},
            index=["AAPL", "MSFT", "GOOG"],
        )
        scores = scorer.score(X)
        assert isinstance(scores, pd.Series)
        assert list(scores.index) == ["AAPL", "MSFT", "GOOG"]
        # y_i = 2·x - 1·y + 0.5
        expected = np.array([1*2 - 0.5*1 + 0.5,
                              2*2 - 0.0 + 0.5,
                              3*2 + 1.0 + 0.5])
        assert np.allclose(scores.values, expected)

    def test_score_handles_nan_inf_inputs(self):
        """NaN/inf in features → fillna 0.0 (matches Qlib Fillna processor)."""
        from training_panel.linear_ltr import PanelLinearScorer
        scorer = PanelLinearScorer(
            coef=np.array([1.0, 1.0]), intercept=0.0,
            feature_cols=["a", "b"],
        )
        X = pd.DataFrame({
            "a": [1.0, np.nan, np.inf],
            "b": [2.0, 3.0, -np.inf],
        })
        scores = scorer.score(X)
        # row 0: 1+2=3
        # row 1: 0+3=3 (NaN → 0)
        # row 2: 0+0=0 (both inf → 0)
        assert np.allclose(scores.values, [3.0, 3.0, 0.0])

    def test_score_raises_on_missing_column(self):
        from training_panel.linear_ltr import PanelLinearScorer
        scorer = PanelLinearScorer(
            coef=np.array([1.0, 1.0, 1.0]), intercept=0.0,
            feature_cols=["a", "b", "c"],
        )
        X = pd.DataFrame({"a": [1.0], "b": [2.0]})  # missing "c"
        with pytest.raises(KeyError, match="c"):
            scorer.score(X)

    def test_coef_shape_validation(self):
        from training_panel.linear_ltr import PanelLinearScorer
        with pytest.raises(ValueError, match="coef shape"):
            PanelLinearScorer(
                coef=np.array([1.0, 2.0]),  # 2 coefs
                intercept=0.0,
                feature_cols=["a", "b", "c"],  # 3 features
            )

    def test_load_rejects_wrong_kind(self):
        """Loading a non-panel_linear artifact via PanelLinearScorer.load fails."""
        from training_panel.linear_ltr import PanelLinearScorer
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "wrong.json"
            p.write_text(json.dumps({"kind": "panel_xgb",
                                      "coef": [1.0],
                                      "feature_cols": ["a"]}))
            with pytest.raises(ValueError, match="kind"):
                PanelLinearScorer.load(p)


class TestFromSklearn:
    def test_from_sklearn_linearregression(self):
        from sklearn.linear_model import LinearRegression
        from training_panel.linear_ltr import PanelLinearScorer
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 5))
        y = X @ np.array([1.0, -1.0, 0.5, 0.0, 2.0]) + rng.normal(scale=0.1, size=100)
        model = LinearRegression(fit_intercept=False).fit(X, y)
        scorer = PanelLinearScorer.from_sklearn(
            model, feature_cols=["f0", "f1", "f2", "f3", "f4"],
        )
        assert scorer.coef.shape == (5,)
        assert np.allclose(scorer.coef, model.coef_)

    def test_from_sklearn_ridge(self):
        from sklearn.linear_model import Ridge
        from training_panel.linear_ltr import PanelLinearScorer
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 3))
        y = rng.normal(size=50)
        model = Ridge(alpha=1.0, fit_intercept=False).fit(X, y)
        scorer = PanelLinearScorer.from_sklearn(model, feature_cols=["a", "b", "c"])
        assert np.allclose(scorer.coef, model.coef_)


class TestPanelScorerDispatch:
    """Verify `PanelScorer.load()` dispatches `kind: panel_linear`."""

    def test_panel_scorer_dispatch_loads_linear(self):
        from training_panel.linear_ltr import PanelLinearScorer
        from kernel.panel_pipeline.panel_scorer import PanelScorer
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "panel-ltr.linear_test.json"
            scorer = PanelLinearScorer(
                coef=np.array([1.0, -1.0]),
                intercept=0.0,
                feature_cols=["alpha_KMID", "alpha_ROC5"],
            )
            scorer.save(p)
            # Top-level dispatch should return a PanelLinearScorer
            loaded = PanelScorer.load(p)
            assert isinstance(loaded, PanelLinearScorer)
            assert loaded.feature_cols == ["alpha_KMID", "alpha_ROC5"]

    def test_dispatch_score_compatible_with_xgb_call_site(self):
        """Verify .score() returns same type/index pattern as XGB scorer."""
        from training_panel.linear_ltr import PanelLinearScorer
        from kernel.panel_pipeline.panel_scorer import PanelScorer
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "panel-ltr.linear_test.json"
            PanelLinearScorer(
                coef=np.array([1.0]), intercept=0.5, feature_cols=["x"],
            ).save(p)
            scorer = PanelScorer.load(p)
            X = pd.DataFrame({"x": [1.0, 2.0]}, index=["AAPL", "MSFT"])
            scores = scorer.score(X)
            assert isinstance(scores, pd.Series)
            assert scores.name == "panel_score"
            assert list(scores.index) == ["AAPL", "MSFT"]


class TestScoreRawNormalization:
    """score_raw applies stored ZScoreNorm + Fillna + Clip then predicts."""

    def test_score_raw_with_stats(self):
        from training_panel.linear_ltr import PanelLinearScorer
        # Coef 1.0 each, no intercept → score = sum of normalized features
        scorer = PanelLinearScorer(
            coef=np.array([1.0, 1.0]),
            intercept=0.0,
            feature_cols=["a", "b"],
            feature_means=np.array([10.0, 100.0]),
            feature_stds=np.array([2.0, 50.0]),
        )
        # raw input: a=12, b=200 → z-scored: (12-10)/2 = 1.0, (200-100)/50 = 2.0
        # score = 1.0 + 2.0 = 3.0
        X = pd.DataFrame({"a": [12.0], "b": [200.0]}, index=["AAPL"])
        scores = scorer.score_raw(X)
        assert abs(scores.iloc[0] - 3.0) < 1e-10

    def test_score_raw_clips_at_5_sigma(self):
        from training_panel.linear_ltr import PanelLinearScorer
        scorer = PanelLinearScorer(
            coef=np.array([1.0]),
            intercept=0.0,
            feature_cols=["a"],
            feature_means=np.array([0.0]),
            feature_stds=np.array([1.0]),
            clip_sigma=5.0,
        )
        # raw a=100 → z=100 → clip to 5.0 → score=5.0
        X = pd.DataFrame({"a": [100.0]}, index=["X"])
        assert abs(scorer.score_raw(X).iloc[0] - 5.0) < 1e-10
        # raw a=-100 → z=-100 → clip to -5.0
        X2 = pd.DataFrame({"a": [-100.0]}, index=["X"])
        assert abs(scorer.score_raw(X2).iloc[0] - (-5.0)) < 1e-10

    def test_score_raw_handles_nan(self):
        from training_panel.linear_ltr import PanelLinearScorer
        scorer = PanelLinearScorer(
            coef=np.array([1.0, 1.0]),
            intercept=0.0,
            feature_cols=["a", "b"],
            feature_means=np.array([0.0, 0.0]),
            feature_stds=np.array([1.0, 1.0]),
        )
        # NaN → fillna(0) → score for that feature = 0
        X = pd.DataFrame({"a": [np.nan, 1.0], "b": [2.0, np.nan]})
        scores = scorer.score_raw(X)
        # row 0: fillna gives a=0 (NaN→nan→0), b=2 (z=2) → 0+2=2
        # row 1: a=1 (z=1), b=0 → 1
        assert abs(scores.iloc[0] - 2.0) < 1e-10
        assert abs(scores.iloc[1] - 1.0) < 1e-10

    def test_score_raw_raises_without_stats(self):
        from training_panel.linear_ltr import PanelLinearScorer
        scorer = PanelLinearScorer(
            coef=np.array([1.0]),
            intercept=0.0,
            feature_cols=["a"],
        )
        X = pd.DataFrame({"a": [1.0]})
        with pytest.raises(ValueError, match="feature_means.*not stored"):
            scorer.score_raw(X)

    def test_save_load_with_stats_roundtrip(self):
        import tempfile
        from training_panel.linear_ltr import PanelLinearScorer
        scorer = PanelLinearScorer(
            coef=np.array([1.0, -1.0]),
            intercept=0.5,
            feature_cols=["a", "b"],
            feature_means=np.array([5.0, -3.0]),
            feature_stds=np.array([1.5, 2.0]),
            clip_sigma=4.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "scorer.json"
            scorer.save(p)
            loaded = PanelLinearScorer.load(p)
            assert np.allclose(loaded.feature_means, [5.0, -3.0])
            assert np.allclose(loaded.feature_stds, [1.5, 2.0])
            assert loaded.clip_sigma == 4.0
            # score_raw must work on loaded scorer
            X = pd.DataFrame({"a": [5.0], "b": [-3.0]})
            scores = loaded.score_raw(X)
            assert abs(scores.iloc[0] - 0.5) < 1e-10  # mean inputs → score = intercept


class TestProductionArtifactIntegrity:
    """Verify the production artifact (panel-ltr.alpha158_linear.json) loads
    via PanelScorer.load and produces sensible scores."""

    PROD_ARTIFACT = (REPO_ROOT / "backtesting" / "renquant_104" / "artifacts"
                     / "panel-ltr.alpha158_linear.json")

    @pytest.mark.skipif(not PROD_ARTIFACT.exists(),
                         reason="Production alpha158_linear artifact not built yet")
    def test_production_artifact_loads(self):
        from kernel.panel_pipeline.panel_scorer import PanelScorer
        scorer = PanelScorer.load(self.PROD_ARTIFACT)
        # Should have ~158 alpha158 features
        assert 100 <= len(scorer.feature_cols) <= 200, \
            f"Unexpected feature count: {len(scorer.feature_cols)}"
        # First few feature names should match alpha158 K-bar pattern
        feature_set = set(scorer.feature_cols)
        assert "KMID" in feature_set or "KLEN" in feature_set, \
            "Production artifact missing K-bar features"

    @pytest.mark.skipif(not PROD_ARTIFACT.exists(),
                         reason="Production alpha158_linear artifact not built yet")
    def test_production_artifact_metadata_has_ics(self):
        from kernel.panel_pipeline.panel_scorer import PanelScorer
        scorer = PanelScorer.load(self.PROD_ARTIFACT)
        # Train IC should be in the metadata (from train_panel_linear.py)
        assert scorer.metadata.get("training_train_ic") is not None, \
            "Production artifact missing training_train_ic — script may have skipped diagnostics"
        train_ic = scorer.metadata["training_train_ic"]
        assert isinstance(train_ic, (int, float))
        # Sanity: train IC shouldn't be wildly off (e.g. 0.05 ± 0.03)
        assert 0.01 < train_ic < 0.20, f"train_ic {train_ic} suspicious"
