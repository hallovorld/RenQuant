"""Tests for training_panel/ngboost_head.py — NGBoost Normal(μ,σ) head."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from training_panel.ngboost_head import (  # noqa: E402
    NGBoostHead,
    combined_score,
    sigma_sizing_multiplier,
)


def _gaussian_panel(n: int = 400, sigma: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y  = 2.0 * x1 - 0.5 * x2 + rng.normal(0.0, sigma, size=n)
    df = pd.DataFrame({
        "x1": x1, "x2": x2,
        "residual_return_raw": y,
        "weight": 1.0,
    })
    return df, ["x1", "x2"]


def _heteroskedastic_panel(n: int = 600, seed: int = 0):
    """Noise scales with |x1| — σ should track |x1| at inference."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise_scale = 0.1 + 0.8 * np.abs(x1)
    y  = 1.0 * x1 + rng.normal(0.0, 1.0, size=n) * noise_scale
    df = pd.DataFrame({
        "x1": x1, "x2": x2,
        "residual_return_raw": y,
        "weight": 1.0,
    })
    return df, ["x1", "x2"]


class TestFit:
    def test_ngboost_recovers_mean_on_known_gaussian(self):
        df, feats = _gaussian_panel(n=500, sigma=0.2, seed=1)
        m = NGBoostHead({"n_estimators": 80, "learning_rate": 0.02})
        info = m.train(df, feats)
        assert info["n_rows"] == 500
        preds = m.predict_distribution(df)
        # mean predicted μ should be close to actual mean of labels
        assert abs(preds["mu"].mean() - df["residual_return_raw"].mean()) < 0.15
        assert (preds["sigma"] > 0).all()

    def test_sigma_correlates_with_label_noise(self):
        df, feats = _heteroskedastic_panel(n=600, seed=2)
        m = NGBoostHead({"n_estimators": 100, "learning_rate": 0.02})
        m.train(df, feats)
        preds = m.predict_distribution(df)
        # NGBoost σ̂ should be positively correlated with true noise scale |x1|
        rho = np.corrcoef(preds["sigma"].values, np.abs(df["x1"].values))[0, 1]
        assert rho > 0.3, f"expected σ to track |x1|, got ρ={rho:.3f}"


class TestPredictShape:
    def test_predict_distribution_shape(self):
        df, feats = _gaussian_panel(n=120, sigma=0.3, seed=3)
        m = NGBoostHead({"n_estimators": 40, "learning_rate": 0.05})
        m.train(df, feats)
        out = m.predict_distribution(df)
        assert list(out.columns) == ["mu", "sigma"]
        assert len(out) == len(df)
        assert (out["sigma"] > 0).all()

    def test_predict_mu_and_sigma_helpers(self):
        df, feats = _gaussian_panel(n=80, sigma=0.3, seed=4)
        m = NGBoostHead({"n_estimators": 30, "learning_rate": 0.05})
        m.train(df, feats)
        mu = m.predict_mu(df)
        sigma = m.predict_sigma(df)
        assert mu.name == "mu"
        assert sigma.name == "sigma"
        # equivalent to distribution columns
        full = m.predict_distribution(df)
        np.testing.assert_allclose(mu.values, full["mu"].values)
        np.testing.assert_allclose(sigma.values, full["sigma"].values)


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        df, feats = _gaussian_panel(n=150, sigma=0.3, seed=5)
        m = NGBoostHead({"n_estimators": 30, "learning_rate": 0.05})
        m.train(df, feats)
        preds_before = m.predict_distribution(df)

        path = tmp_path / "ngb_head.json"
        m.save(path, metadata={"training_notes": "unit-test"})
        assert path.exists()

        m2 = NGBoostHead.load(path)
        preds_after = m2.predict_distribution(df)
        np.testing.assert_allclose(preds_before["mu"].values,
                                   preds_after["mu"].values, rtol=1e-9)
        np.testing.assert_allclose(preds_before["sigma"].values,
                                   preds_after["sigma"].values, rtol=1e-9)
        assert m2.feature_cols == feats

    def test_load_rejects_non_ngboost_artifact(self, tmp_path):
        import json
        path = tmp_path / "wrong.json"
        path.write_text(json.dumps({"kind": "something_else"}))
        with pytest.raises(ValueError, match="ngboost_head"):
            NGBoostHead.load(path)


class TestScoringHelpers:
    def test_combined_score_prefers_high_mu_low_sigma(self):
        tickers = ["HIGH_MU_LOW_SIG", "LOW_MU_HIGH_SIG", "MID", "NEGATIVE"]
        mu    = pd.Series([0.10, 0.08, 0.02, -0.05], index=tickers)
        sigma = pd.Series([0.02, 0.20, 0.03,  0.05], index=tickers)
        score = combined_score(mu, sigma, lambda_sigma=1.0)
        # HIGH_MU_LOW_SIG = 0.08; LOW_MU_HIGH_SIG = -0.12; MID = -0.01; NEG = -0.10
        assert score.idxmax() == "HIGH_MU_LOW_SIG"
        assert score.idxmin() == "LOW_MU_HIGH_SIG"

    def test_combined_score_lambda_zero_equals_mu(self):
        mu    = pd.Series([0.1, 0.2, -0.05], index=list("abc"))
        sigma = pd.Series([0.5, 0.1,  0.3],  index=list("abc"))
        score = combined_score(mu, sigma, lambda_sigma=0.0)
        np.testing.assert_allclose(score.values, mu.values)

    def test_sigma_sizing_multiplier_bounds(self):
        # σ_median = 0.10; ratios = [1.0, 0.5, 2.0]; clipped to [floor, ceiling]
        sigma = pd.Series([0.10, 0.20, 0.05], index=list("abc"))
        mult = sigma_sizing_multiplier(sigma, floor=0.3, ceiling=1.0)
        # a: 0.10/0.10 = 1.0 → 1.0
        # b: 0.10/0.20 = 0.5 → 0.5
        # c: 0.10/0.05 = 2.0 → clipped to 1.0 (ceiling)
        np.testing.assert_allclose(mult.values, [1.0, 0.5, 1.0])
        assert mult.name == "sigma_mult"

    def test_sigma_sizing_multiplier_floor_clips_high_sigma(self):
        sigma = pd.Series([0.01, 10.0, 1.0], index=list("abc"))
        mult = sigma_sizing_multiplier(sigma, floor=0.4, ceiling=1.0)
        # median = 1.0; ratios = [100.0, 0.1, 1.0]; clipped = [1.0, 0.4, 1.0]
        np.testing.assert_allclose(mult.values, [1.0, 0.4, 1.0])

    def test_sigma_sizing_handles_nonpositive_median(self):
        # All zeros — no meaningful σ ordering, should short-circuit to 1.0
        sigma = pd.Series([0.0, 0.0, 0.0], index=list("abc"))
        mult = sigma_sizing_multiplier(sigma)
        np.testing.assert_allclose(mult.values, [1.0, 1.0, 1.0])


class TestWeighted:
    def test_training_respects_sample_weights(self):
        df, feats = _gaussian_panel(n=200, sigma=0.3, seed=7)
        # Poison the last half, but weight them to zero — fit should still
        # be close to the clean half's labels.
        df = df.copy()
        poison_start = 100
        df.loc[poison_start:, "residual_return_raw"] = 10.0
        df["weight"] = 1.0
        df.loc[poison_start:, "weight"] = 0.0

        m = NGBoostHead({"n_estimators": 60, "learning_rate": 0.02})
        m.train(df, feats, sample_weight_col="weight")
        preds = m.predict_distribution(df.iloc[:poison_start])
        # Predicted μ on clean rows shouldn't be pulled toward 10
        assert preds["mu"].mean() < 3.0


class TestSigmaOverflowClamp:
    """Audit fix NGB-OVERFLOW (2026-04-26) — predict_distribution
    must clamp pathological sigma values to a sane range and never
    propagate inf/NaN downstream.

    Triggered by `RuntimeWarning: overflow encountered in square` from
    ngboost.distns.normal:72 (`self.var = self.scale**2`) seen in the
    transformer Sunday sweep — internal NGBoost gradient excursions
    can produce huge `scale` values.
    """

    def test_sigma_clamped_to_5(self, monkeypatch):
        df, feats = _gaussian_panel(n=120, sigma=0.2, seed=1)
        m = NGBoostHead({"n_estimators": 30, "learning_rate": 0.05})
        m.train(df, feats)

        # Patch the regressor's pred_dist to simulate a blow-up: emit a
        # sigma of 1e10 for half the rows and inf for the rest.
        class _Stub:
            def __init__(self, n):
                import numpy as _np
                self.loc   = _np.zeros(n)
                self.scale = _np.full(n, 1e10)
                self.scale[: n // 2] = float("inf")

        original_pred = m.regressor.pred_dist
        monkeypatch.setattr(m.regressor, "pred_dist",
                            lambda X: _Stub(len(X)))
        out = m.predict_distribution(df.iloc[:50])
        # All sigmas in the finite, output range
        assert out["sigma"].max() <= 5.0
        assert out["sigma"].min() >= 1e-6
        assert out["sigma"].notna().all()

    def test_mu_clamped_when_extreme(self, monkeypatch):
        df, feats = _gaussian_panel(n=120, sigma=0.2, seed=1)
        m = NGBoostHead({"n_estimators": 30, "learning_rate": 0.05})
        m.train(df, feats)

        class _StubMu:
            def __init__(self, n):
                import numpy as _np
                self.loc   = _np.full(n, 1e6)        # absurd mean
                self.scale = _np.full(n, 0.1)

        monkeypatch.setattr(m.regressor, "pred_dist",
                            lambda X: _StubMu(len(X)))
        out = m.predict_distribution(df.iloc[:50])
        assert out["mu"].max() <= 1.0
        assert out["mu"].min() >= -1.0

    def test_normal_predictions_unchanged(self):
        """Sanity — normal predictions don't get accidentally clamped."""
        df, feats = _gaussian_panel(n=200, sigma=0.2, seed=7)
        m = NGBoostHead({"n_estimators": 30, "learning_rate": 0.05})
        m.train(df, feats)
        out = m.predict_distribution(df)
        # Normal sigma should be in [0.05, 1.0] for this data
        assert 0.05 < out["sigma"].mean() < 1.0
        # And in any case not at the floor or ceiling
        assert (out["sigma"] >= 1e-6).all()
        assert (out["sigma"] <= 5.0).all()

    def test_warning_when_many_clipped(self, monkeypatch, caplog):
        """When >1% of rows hit the ceiling, emit a warning log."""
        import logging
        df, feats = _gaussian_panel(n=120, sigma=0.2, seed=1)
        m = NGBoostHead({"n_estimators": 30, "learning_rate": 0.05})
        m.train(df, feats)

        class _StubAllHigh:
            def __init__(self, n):
                import numpy as _np
                self.loc   = _np.zeros(n)
                self.scale = _np.full(n, 100.0)   # all > ceil

        monkeypatch.setattr(m.regressor, "pred_dist",
                            lambda X: _StubAllHigh(len(X)))
        with caplog.at_level(logging.WARNING, logger="ngboost"):
            m.predict_distribution(df.iloc[:50])
        msgs = [rec.message for rec in caplog.records
                if rec.name == "ngboost"]
        assert any("clipped to ceil" in m for m in msgs), \
            f"expected clipping warning, got {msgs}"
