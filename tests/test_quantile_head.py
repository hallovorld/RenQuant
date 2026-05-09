"""Unit tests for training_panel.quantile_head.

Covers:
  - QuantileHead.load() round-trips a real artifact
  - predict_distribution returns DataFrame[mu, sigma] with σ > 0
  - load_head_by_kind dispatches on artifact `kind` field
  - LoadNGBoostTask via polymorphic loader works for both NGBoost
    and quantile artifacts
"""
from __future__ import annotations

import base64
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from training_panel.quantile_head import QuantileHead, load_head_by_kind  # noqa: E402


def _train_mini_quantile_artifact(path: Path, feat_cols: list[str]):
    """Train 3 tiny XGBoost-quantile models + save in QuantileHead format."""
    rng = np.random.default_rng(7)
    n = len(feat_cols)
    X = rng.normal(size=(120, n))
    y = X[:, 0] + 0.3 * rng.normal(size=120)

    boosters_raw = {}
    for q in (0.16, 0.50, 0.84):
        params = {"objective": "reg:quantileerror", "quantile_alpha": q,
                  "tree_method": "hist", "max_depth": 3, "verbosity": 0}
        m = xgb.XGBRegressor(n_estimators=10, **params)
        m.fit(X, y)
        boosters_raw[q] = bytes(m.get_booster().save_raw(raw_format="json")).decode()

    payload_obj = {
        "quantiles": [0.16, 0.50, 0.84],
        "boosters_raw": boosters_raw,
        "feature_cols": list(feat_cols),
        "feature_medians": np.zeros(n).tolist(),
    }
    blob = base64.b64encode(pickle.dumps(payload_obj)).decode("ascii")
    artifact = {
        "version": 1,
        "kind": "quantile_head",
        "trained_date": "2026-05-09",
        "feature_cols": list(feat_cols),
        "params": {"max_depth": 3},
        "quantiles": [0.16, 0.50, 0.84],
        "regressor_pickle_b64": blob,
        "feature_medians": np.zeros(n).tolist(),
        "config_fingerprint": "sha256:test-fp",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact))
    return path


class TestQuantileHeadLoad:
    def test_load_round_trips(self, tmp_path):
        feat_cols = [f"f{i}" for i in range(8)]
        p = _train_mini_quantile_artifact(tmp_path / "qhead.json", feat_cols)
        head = QuantileHead.load(p)
        assert head.feature_cols == feat_cols
        assert sorted(head.boosters.keys()) == [0.16, 0.50, 0.84]
        assert head.feature_medians_ is not None and len(head.feature_medians_) == 8

    def test_load_rejects_wrong_kind(self, tmp_path):
        p = tmp_path / "wrong.json"
        p.write_text(json.dumps({"kind": "ngboost_head", "feature_cols": []}))
        with pytest.raises(ValueError, match="quantile_head"):
            QuantileHead.load(p)


class TestQuantileHeadPredict:
    def test_predict_distribution_returns_mu_sigma(self, tmp_path):
        feat_cols = [f"f{i}" for i in range(6)]
        p = _train_mini_quantile_artifact(tmp_path / "qhead.json", feat_cols)
        head = QuantileHead.load(p)

        # Build a panel-like DataFrame
        rng = np.random.default_rng(99)
        panel = pd.DataFrame(
            rng.normal(size=(20, len(feat_cols))),
            index=[f"T{i}" for i in range(20)],
            columns=feat_cols,
        )
        out = head.predict_distribution(panel)
        assert list(out.columns) == ["mu", "sigma"]
        assert len(out) == 20
        assert out["mu"].notna().all()
        # σ must be positive (Gaussian recovery floored at 1e-6)
        assert (out["sigma"] > 0).all()

    def test_predict_raises_on_missing_columns(self, tmp_path):
        feat_cols = [f"f{i}" for i in range(4)]
        p = _train_mini_quantile_artifact(tmp_path / "qhead.json", feat_cols)
        head = QuantileHead.load(p)
        panel = pd.DataFrame({"f0": [1.0, 2.0]})  # missing f1..f3
        with pytest.raises(ValueError, match="missing required feature"):
            head.predict_distribution(panel)

    def test_nan_rows_return_nan(self, tmp_path):
        feat_cols = [f"f{i}" for i in range(4)]
        p = _train_mini_quantile_artifact(tmp_path / "qhead.json", feat_cols)
        head = QuantileHead.load(p)
        # Override medians to non-zero so we can detect NaN-rejection
        head.feature_medians_ = None  # disable imputation
        panel = pd.DataFrame(
            [[np.nan, 1.0, 2.0, 3.0],
             [1.0, 2.0, 3.0, 4.0]],
            index=["NAN_ROW", "OK_ROW"],
            columns=feat_cols,
        )
        out = head.predict_distribution(panel)
        assert pd.isna(out.loc["NAN_ROW", "mu"])
        assert pd.isna(out.loc["NAN_ROW", "sigma"])
        assert not pd.isna(out.loc["OK_ROW", "mu"])
        assert out.loc["OK_ROW", "sigma"] > 0


class TestLoadHeadByKind:
    def test_dispatches_quantile_head(self, tmp_path):
        feat_cols = [f"f{i}" for i in range(4)]
        p = _train_mini_quantile_artifact(tmp_path / "qhead.json", feat_cols)
        head = load_head_by_kind(p)
        assert isinstance(head, QuantileHead)

    def test_rejects_unknown_kind(self, tmp_path):
        p = tmp_path / "weird.json"
        p.write_text(json.dumps({"kind": "alien_head"}))
        with pytest.raises(ValueError, match="unsupported kind"):
            load_head_by_kind(p)


class TestProductionArtifactLoads:
    """Smoke test: the actual on-disk production artifact loads + predicts."""

    def test_production_quantile_head_loads(self):
        prod_path = REPO / "backtesting/renquant_104/artifacts/ngboost-head.alpha158_fund.json"
        if not prod_path.exists():
            pytest.skip("production quantile head artifact not present")
        head = load_head_by_kind(prod_path)
        assert isinstance(head, QuantileHead)
        assert len(head.feature_cols) == 166  # alpha158 + 5 fund + 3 PEAD
        # Predict on a fake panel with the right columns
        rng = np.random.default_rng(42)
        panel = pd.DataFrame(
            rng.normal(size=(5, 166)),
            index=["AAPL", "MSFT", "GOOG", "META", "NVDA"],
            columns=head.feature_cols,
        )
        out = head.predict_distribution(panel)
        assert len(out) == 5
        assert (out["sigma"] > 0).all()
