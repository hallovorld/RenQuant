"""Tests for training_panel/ltr_model.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_103"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _make_easy_panel(n_dates: int = 40, n_tickers: int = 8, seed: int = 0):
    """Panel where label = x1 + small noise (IC should be high)."""
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
    panel = panel.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    group_sizes = panel.groupby("date", sort=True).size().values.astype(np.int32)
    return panel, group_sizes


class TestTrain:
    def test_train_produces_booster(self):
        from training_panel.ltr_model import PanelLTRModel
        panel, grp = _make_easy_panel(n_dates=20, n_tickers=6, seed=1)
        m = PanelLTRModel()
        res = m.train(panel, grp, feature_cols=["x1", "x2"],
                      num_boost_round=20, early_stopping_rounds=None)
        assert m.booster is not None
        assert "train_ic" in res

    def test_ic_high_on_easy_signal(self):
        from training_panel.ltr_model import PanelLTRModel
        panel, grp = _make_easy_panel(n_dates=30, n_tickers=8, seed=2)
        m = PanelLTRModel()
        res = m.train(panel, grp, feature_cols=["x1", "x2"],
                      num_boost_round=100, early_stopping_rounds=None)
        # label = x1 + noise ⇒ train IC should be high
        assert res["train_ic"] > 0.7, f"train IC too low: {res['train_ic']:.3f}"

    def test_ic_monotonic_ish_with_rounds(self):
        from training_panel.ltr_model import PanelLTRModel
        panel, grp = _make_easy_panel(n_dates=25, n_tickers=6, seed=3)
        m1 = PanelLTRModel()
        r1 = m1.train(panel, grp, feature_cols=["x1", "x2"],
                      num_boost_round=10, early_stopping_rounds=None)
        m2 = PanelLTRModel()
        r2 = m2.train(panel, grp, feature_cols=["x1", "x2"],
                      num_boost_round=100, early_stopping_rounds=None)
        # More rounds ⇒ at least no worse (up to noise)
        assert r2["train_ic"] + 0.05 >= r1["train_ic"]

    def test_training_respects_sample_weights(self):
        from training_panel.ltr_model import PanelLTRModel
        # Build a panel where half the rows have label=x1, half have label=-x1.
        # Up-weighting the first half should flip the learned sign of x1.
        rng = np.random.default_rng(4)
        rows = []
        dates = pd.bdate_range("2024-01-01", periods=30)
        for d_i, d in enumerate(dates):
            for i in range(8):
                x1 = rng.normal()
                # Alternate by ticker index to give each date a mix
                if i < 4:
                    label = x1
                    w = 10.0
                else:
                    label = -x1
                    w = 0.01
                rows.append({"date": d, "ticker": f"T{i}", "x1": x1,
                             "label": label, "weight": w})
        panel = pd.DataFrame(rows).sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
        grp = panel.groupby("date", sort=True).size().values.astype(np.int32)

        m_heavy = PanelLTRModel()
        m_heavy.train(panel, grp, feature_cols=["x1"], weight_col="weight",
                      num_boost_round=50, early_stopping_rounds=None)
        preds_heavy = m_heavy.predict(panel).values

        # Correlation of preds with x1: should be positive (first-half dominated
        # training via high weights)
        corr_heavy = np.corrcoef(preds_heavy, panel["x1"].values)[0, 1]
        assert corr_heavy > 0.2, f"weights ignored: corr={corr_heavy:.3f}"


class TestPredict:
    def test_predict_returns_finite_scores_of_expected_length(self):
        from training_panel.ltr_model import PanelLTRModel
        panel, grp = _make_easy_panel(n_dates=20, n_tickers=6, seed=5)
        m = PanelLTRModel()
        m.train(panel, grp, feature_cols=["x1", "x2"],
                num_boost_round=20, early_stopping_rounds=None)
        preds = m.predict(panel)
        assert len(preds) == len(panel)
        assert preds.notna().all()
        assert np.isfinite(preds.values).all()

    def test_predict_before_train_raises(self):
        from training_panel.ltr_model import PanelLTRModel
        panel, _ = _make_easy_panel(n_dates=5, n_tickers=3, seed=6)
        m = PanelLTRModel()
        with pytest.raises(RuntimeError):
            m.predict(panel)


class TestSaveLoad:
    def test_save_load_roundtrip_identical_predictions(self, tmp_path):
        from training_panel.ltr_model import PanelLTRModel
        panel, grp = _make_easy_panel(n_dates=20, n_tickers=6, seed=7)
        m = PanelLTRModel()
        m.train(panel, grp, feature_cols=["x1", "x2"],
                num_boost_round=30, early_stopping_rounds=None)
        preds_orig = m.predict(panel).values

        p = tmp_path / "panel_model.json"
        m.save(p, metadata={"training_notes": "unit test"})
        assert p.exists()

        loaded = PanelLTRModel.load(p)
        preds_loaded = loaded.predict(panel).values
        assert np.allclose(preds_orig, preds_loaded, atol=1e-9)

    def test_save_before_train_raises(self, tmp_path):
        from training_panel.ltr_model import PanelLTRModel
        m = PanelLTRModel()
        with pytest.raises(RuntimeError):
            m.save(tmp_path / "x.json")


class TestGroupSizes:
    def test_group_sizes_used_in_training(self):
        """If group_sizes doesn't match panel length, XGBoost should error."""
        from training_panel.ltr_model import PanelLTRModel
        panel, _ = _make_easy_panel(n_dates=10, n_tickers=5, seed=8)
        bad_grp = np.array([1] * (len(panel) + 5), dtype=np.int32)
        m = PanelLTRModel()
        with pytest.raises(Exception):
            m.train(panel, bad_grp, feature_cols=["x1", "x2"],
                    num_boost_round=5, early_stopping_rounds=None)
