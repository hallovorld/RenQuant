"""Unit tests for training_panel/transformer_model.py.

Scope for commit 2a: standalone model class only — no pipeline hookup yet.
Covers shape, pad-mask correctness, deterministic seed reproducibility,
save/load round-trip, and a "model learns ranking on synthetic signal" sanity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

torch = pytest.importorskip("torch")

from training_panel.transformer_model import (  # noqa: E402
    PanelTransformerModel,
    TransformerParams,
    _listnet_loss,
    _build_date_groups,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_synthetic_panel(
    n_dates: int = 40, n_tickers: int = 8, n_features: int = 6, seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Build a panel where label = w · features + small noise, so a competent
    ranker can learn to predict ranking within each date-group.
    """
    rng = np.random.default_rng(seed)
    feature_cols = [f"f{i}" for i in range(n_features)]
    true_w = rng.normal(size=n_features).astype(np.float32)

    rows = []
    group_sizes = []
    for d in range(n_dates):
        # Occasionally drop a ticker to test variable group sizes.
        active = n_tickers if d % 5 else n_tickers - 2
        X = rng.normal(size=(active, n_features)).astype(np.float32)
        noise = rng.normal(size=active).astype(np.float32) * 0.3
        y = X @ true_w + noise
        for t in range(active):
            rows.append({
                "date": d,
                "ticker": f"T{t}",
                **{c: float(X[t, i]) for i, c in enumerate(feature_cols)},
                "label":  float(y[t]),
                "weight": 1.0,
            })
        group_sizes.append(active)
    return pd.DataFrame(rows), np.array(group_sizes, dtype=int), feature_cols


# ── Shape / pad-mask tests ───────────────────────────────────────────────────

class TestShapeAndPadMask:
    def test_build_date_groups_shapes(self):
        panel, gs, fc = _make_synthetic_panel()
        # Audit T-1 (2026-04-25): _build_date_groups now returns 4-tuple
        # (x, y, pad_mask, nan_label_mask).
        x, y, pad, nan_y = _build_date_groups(
            panel, gs, fc, "label", max_tickers=10,
        )
        assert x.shape == (len(gs), 10, len(fc))
        assert y.shape == (len(gs), 10)
        assert pad.shape == (len(gs), 10)
        assert nan_y.shape == (len(gs), 10)
        # Padded positions are zero in x/y and True in pad.
        for gi, g in enumerate(gs):
            assert pad[gi, :g].sum() == 0, "non-pad positions must not be masked"
            assert pad[gi, g:].sum() == 10 - g, "pad positions must all be masked"
            assert np.allclose(x[gi, g:], 0.0)
            assert np.allclose(y[gi, g:], 0.0)
            # Synthetic panel has no NaN labels → mask all-False.
            assert nan_y[gi].sum() == 0

    def test_pad_mask_excludes_padding_from_softmax(self):
        """ListNet softmax must treat padded positions as zero-weight.

        We build a 2-element group with one valid and one padded ticker.
        Placing a huge score on the padded slot must NOT affect the loss.
        """
        scores = torch.tensor([[100.0, 0.5], [1.0, -1.0]])        # (2, 2)
        labels = torch.tensor([[0.0, 0.3], [0.7, -0.2]])
        pad    = torch.tensor([[True, False], [False, False]])
        loss_a = _listnet_loss(scores, labels, pad)

        # Now change the padded slot to a completely different value.
        scores2 = scores.clone()
        scores2[0, 0] = -99.0
        loss_b = _listnet_loss(scores2, labels, pad)
        assert torch.allclose(loss_a, loss_b), (
            "padded-slot score must not influence ListNet loss"
        )


# ── Determinism ──────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_two_runs_same_seed_match(self):
        panel, gs, fc = _make_synthetic_panel(n_dates=20, n_tickers=6)
        p = {"max_epochs": 4, "d_model": 32, "n_heads": 2, "n_layers": 2,
             "batch_size": 8, "device": "cpu", "seed": 7}
        m1 = PanelTransformerModel(params=p)
        m1.train(panel, gs, fc, num_boost_round=4)
        out1 = m1.predict(panel).values

        m2 = PanelTransformerModel(params=p)
        m2.train(panel, gs, fc, num_boost_round=4)
        out2 = m2.predict(panel).values

        assert np.allclose(out1, out2, atol=1e-5), (
            "same seed + deterministic settings must reproduce identical predictions"
        )

    def test_different_seed_diverges(self):
        panel, gs, fc = _make_synthetic_panel(n_dates=20, n_tickers=6)
        base = {"max_epochs": 4, "d_model": 32, "n_heads": 2, "n_layers": 2,
                "batch_size": 8, "device": "cpu"}
        m1 = PanelTransformerModel(params={**base, "seed": 1})
        m1.train(panel, gs, fc, num_boost_round=4)
        m2 = PanelTransformerModel(params={**base, "seed": 2})
        m2.train(panel, gs, fc, num_boost_round=4)
        assert not np.allclose(m1.predict(panel).values, m2.predict(panel).values, atol=1e-3), (
            "different seeds should produce different trajectories"
        )


# ── Save / load round-trip ────────────────────────────────────────────────────

class TestSaveLoadRoundtrip:
    def test_load_reproduces_predictions(self, tmp_path: Path):
        panel, gs, fc = _make_synthetic_panel(n_dates=20, n_tickers=6)
        m = PanelTransformerModel(params={
            "max_epochs": 3, "d_model": 32, "n_heads": 2, "n_layers": 2,
            "batch_size": 8, "device": "cpu", "seed": 42,
        })
        m.train(panel, gs, fc, num_boost_round=3)
        before = m.predict(panel).values

        art = tmp_path / "panel-transformer.pt"
        m.save(art, metadata={"note": "unit test"})
        assert art.exists()
        assert art.with_suffix(".json").exists()

        m2 = PanelTransformerModel.load(art)
        after = m2.predict(panel).values
        assert np.allclose(before, after, atol=1e-5), (
            "load round-trip must preserve predictions bit-for-bit (within fp tol)"
        )
        # Sidecar schema sanity
        meta = json.loads(art.with_suffix(".json").read_text())
        assert meta["kind"] == "panel_transformer"
        assert meta["feature_cols"] == fc
        assert "history" in meta

    def test_save_rejects_untrained(self, tmp_path: Path):
        m = PanelTransformerModel()
        with pytest.raises(RuntimeError):
            m.save(tmp_path / "nope.pt")


# ── Model actually learns on synthetic signal ─────────────────────────────────

class TestLearnsSignal:
    def test_train_ic_rises_above_random(self):
        panel, gs, fc = _make_synthetic_panel(n_dates=80, n_tickers=10, n_features=5)
        m = PanelTransformerModel(params={
            "max_epochs": 20, "d_model": 64, "n_heads": 4, "n_layers": 2,
            "batch_size": 16, "device": "cpu", "seed": 0,
            "label_smoothing": 0.0, "ticker_dropout": 0.0, "feature_dropout": 0.0,
        })
        res = m.train(panel, gs, fc, num_boost_round=20)
        assert res["train_ic"] > 0.30, (
            f"model should learn ranking on a linear synthetic signal — "
            f"got train_ic={res['train_ic']:.3f}"
        )


# ── Early stopping ────────────────────────────────────────────────────────────

class TestPredictRobustness:
    """Regression: predict() used to silently truncate groups > max_tickers
    to uninitialized memory (causing NaN IC in CV). Now it either splits
    the group into chunks or raises when inputs are ambiguous.
    """

    def _trained_model(self):
        panel, gs, fc = _make_synthetic_panel(n_dates=20, n_tickers=6, n_features=4)
        m = PanelTransformerModel(params={
            "max_epochs": 3, "d_model": 16, "n_heads": 2, "n_layers": 1,
            "batch_size": 4, "device": "cpu", "seed": 0, "max_tickers": 6,
        })
        m.train(panel, gs, fc, num_boost_round=3)
        return m, fc

    def test_predict_raises_without_date_or_group_sizes(self):
        m, fc = self._trained_model()
        rng = np.random.default_rng(0)
        bad = pd.DataFrame(rng.normal(size=(12, len(fc))).astype(np.float32),
                           columns=fc)
        # No `date` column → must raise rather than silently misgroup.
        with pytest.raises(ValueError, match="date|group_sizes"):
            m.predict(bad)

    def test_predict_accepts_explicit_group_sizes(self):
        m, fc = self._trained_model()
        rng = np.random.default_rng(1)
        frame = pd.DataFrame(rng.normal(size=(12, len(fc))).astype(np.float32),
                             columns=fc)
        # 12 rows arranged as 2 date-groups of 6 tickers each.
        out = m.predict(frame, group_sizes=np.array([6, 6], dtype=int))
        assert len(out) == 12
        assert not np.isnan(out.values).any()

    def test_predict_splits_oversized_groups(self):
        """A single flat panel of 20 rows with max_tickers=6 must emit 20
        finite scores (split into chunks) — not 6 valid + 14 NaN."""
        m, fc = self._trained_model()
        rng = np.random.default_rng(2)
        frame = pd.DataFrame(rng.normal(size=(20, len(fc))).astype(np.float32),
                             columns=fc)
        out = m.predict(frame, group_sizes=np.array([20], dtype=int))
        assert len(out) == 20
        assert not np.isnan(out.values).any(), (
            "oversized group must be chunk-split, not silently truncated to NaN"
        )

    def test_predict_detects_group_sizes_sum_mismatch(self):
        m, fc = self._trained_model()
        rng = np.random.default_rng(3)
        frame = pd.DataFrame(rng.normal(size=(10, len(fc))).astype(np.float32),
                             columns=fc)
        # Sum 4+3 = 7, but panel has 10 rows. Must raise.
        with pytest.raises(ValueError, match="group_sizes"):
            m.predict(frame, group_sizes=np.array([4, 3], dtype=int))


class TestEarlyStopping:
    def test_patience_breaks_when_eval_ic_stalls(self):
        panel, gs, fc = _make_synthetic_panel(n_dates=40, n_tickers=6)
        # Random labels in eval → no improvement possible → should stop at patience+1.
        rng = np.random.default_rng(0)
        eval_panel = panel.copy()
        eval_panel["label"] = rng.normal(size=len(eval_panel))

        m = PanelTransformerModel(params={
            "max_epochs": 50, "d_model": 16, "n_heads": 2, "n_layers": 1,
            "batch_size": 8, "device": "cpu", "patience": 3, "seed": 0,
        })
        m.train(
            panel, gs, fc, num_boost_round=50, early_stopping_rounds=3,
            eval_panel=eval_panel, eval_group_sizes=gs,
        )
        assert len(m.history) < 50, (
            f"early-stopping should halt before max_epochs — ran {len(m.history)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
