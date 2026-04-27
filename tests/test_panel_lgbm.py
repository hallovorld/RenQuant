"""Tests for training_panel/lgbm_ltr.py — LightGBM LambdaRank backend."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


pytest.importorskip("lightgbm")
from training_panel.lgbm_ltr import (  # noqa: E402
    DEFAULT_PARAMS, PanelLGBMModel, PanelLGBMScorer, _bucketize_labels,
)


class TestDefaultParams:
    def test_audit_12_no_unused_data_random_seed(self):
        """Audit #12 fix (2026-04-27): `data_random_seed` only takes effect
        for GOSS / random-forest sampling — neither is enabled. Setting it
        gave a false sense of an extra determinism control. The remaining
        seed/bagging_seed/feature_fraction_seed cover the lambdarank+gbdt
        path. Remove it to keep DEFAULT_PARAMS honest."""
        assert "data_random_seed" not in DEFAULT_PARAMS
        # The actually-effective seeds must remain
        assert DEFAULT_PARAMS.get("seed") == 42
        assert DEFAULT_PARAMS.get("bagging_seed") == 42
        assert DEFAULT_PARAMS.get("feature_fraction_seed") == 42


def _make_easy_panel(n_dates: int = 40, n_tickers: int = 8, seed: int = 0):
    """Panel where label = x1 + small noise, all tickers share structure."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    for d in dates:
        for i in range(n_tickers):
            x1 = rng.normal()
            x2 = rng.normal()
            label = x1 + 0.1 * rng.normal()
            rows.append({
                "date": d, "ticker": f"T{i}", "x1": x1, "x2": x2,
                "label": label, "weight": 1.0,
            })
    panel = pd.DataFrame(rows)
    group_sizes = panel.groupby("date", sort=True).size().values.astype(np.int32)
    return panel, group_sizes


class TestBucketize:
    def test_output_in_range(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=200)
        out = _bucketize_labels(y, n_buckets=11)
        assert out.min() >= 0 and out.max() <= 10
        assert out.dtype == np.int32

    def test_monotone_mapping(self):
        """Sorted input → sorted output."""
        y = np.sort(np.random.default_rng(1).normal(size=500))
        out = _bucketize_labels(y, n_buckets=11)
        diffs = np.diff(out)
        assert (diffs >= 0).all(), "bucketize should be monotone"


class TestTraining:
    def test_fits_with_nonuniform_weights(self):
        """Regression: LightGBM takes PER-ROW weights (length = n_rows), not
        per-group. Earlier `PanelLGBMModel.train` passed a per-group array
        and LightGBM rejected it with
        `LightGBMError: Length of weights differs from the length of #data`.
        Would fail on the 2026-04-23 daily-104 A/B run. Fix broadcasts the
        group-mean weight back to per-row so LightGBM's length check passes.
        """
        panel, gs = _make_easy_panel(n_dates=10, n_tickers=6, seed=42)
        # Introduce NON-uniform weights so the broadcast is exercised.
        panel = panel.copy()
        panel["weight"] = 1.0 + 0.5 * (panel.index % 3)
        m = PanelLGBMModel(
            params={"learning_rate": 0.05, "num_leaves": 7, "min_data_in_leaf": 3,
                    "verbose": -1},
        )
        info = m.train(
            panel, gs, feature_cols=["x1", "x2"],
            label_col="label", weight_col="weight", num_boost_round=5,
        )
        preds = m.predict(panel)
        assert preds.shape == (len(panel),)
        assert not preds.isna().any()
        assert "train_ic" in info

    def test_fits_and_predicts_shape(self):
        panel, gs = _make_easy_panel(n_dates=30, n_tickers=6, seed=2)
        m = PanelLGBMModel(
            params={"learning_rate": 0.05, "num_leaves": 7, "min_data_in_leaf": 5},
        )
        info = m.train(
            panel, gs, feature_cols=["x1", "x2"],
            label_col="label", weight_col="weight", num_boost_round=30,
        )
        preds = m.predict(panel)
        assert preds.shape == (len(panel),)
        assert "train_ic" in info

    def test_train_ic_positive_on_easy_signal(self):
        panel, gs = _make_easy_panel(n_dates=40, n_tickers=8, seed=3)
        m = PanelLGBMModel(
            params={"learning_rate": 0.1, "num_leaves": 7, "min_data_in_leaf": 5},
        )
        info = m.train(
            panel, gs, feature_cols=["x1", "x2"],
            label_col="label", weight_col="weight", num_boost_round=50,
        )
        # Easy signal → IC should be substantially positive
        assert info["train_ic"] > 0.3


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        panel, gs = _make_easy_panel(n_dates=30, n_tickers=6, seed=5)
        m = PanelLGBMModel(
            params={"learning_rate": 0.05, "num_leaves": 7, "min_data_in_leaf": 5},
        )
        m.train(panel, gs, feature_cols=["x1", "x2"], num_boost_round=30)
        path = tmp_path / "panel-lgbm.json"
        m.save(path, metadata={"training_notes": "unit-test"})
        assert path.exists()

        m2 = PanelLGBMModel.load(path)
        np.testing.assert_allclose(m.predict(panel).values, m2.predict(panel).values,
                                    rtol=1e-9)
        assert m2.feature_cols == ["x1", "x2"]

    def test_scorer_loads_lgbm_artifact(self, tmp_path):
        panel, gs = _make_easy_panel(n_dates=30, n_tickers=6, seed=6)
        m = PanelLGBMModel(
            params={"learning_rate": 0.05, "num_leaves": 7, "min_data_in_leaf": 5},
        )
        m.train(panel, gs, feature_cols=["x1", "x2"], num_boost_round=30)
        path = tmp_path / "panel-lgbm.json"
        m.save(path)

        # PanelScorer.load dispatches on kind → should return PanelLGBMScorer
        from kernel.panel_pipeline import PanelScorer
        scorer = PanelScorer.load(path)
        assert isinstance(scorer, PanelLGBMScorer)
        # And score() works
        X = panel[["x1", "x2"]].copy()
        X.index = panel["ticker"].values
        s = scorer.score(X)
        assert len(s) == len(X)

    def test_scorer_still_loads_xgboost_artifact(self, tmp_path):
        """Scorer dispatcher must not break the existing XGBoost path."""
        from training_panel.ltr_model import PanelLTRModel
        panel, gs = _make_easy_panel(n_dates=30, n_tickers=6, seed=7)
        m = PanelLTRModel()
        m.train(panel, gs, feature_cols=["x1", "x2"],
                label_col="label", weight_col="weight", num_boost_round=20)
        path = tmp_path / "panel-xgb.json"
        m.save(path)

        from kernel.panel_pipeline import PanelScorer
        scorer = PanelScorer.load(path)
        # Default XGBoost path returns PanelScorer (not PanelLGBMScorer)
        assert isinstance(scorer, PanelScorer)

    def test_load_rejects_wrong_kind(self, tmp_path):
        import json
        path = tmp_path / "wrong.json"
        path.write_text(json.dumps({"kind": "something_else"}))
        with pytest.raises(ValueError, match="panel_lgbm"):
            PanelLGBMScorer.load(path)
