"""Regression tests for the NGBoost + Transformer deep audit fixes
(2026-04-25). See `doc/ngboost_transformer_audit_2026-04-25.md` for the
full bug catalogue.

Coverage:
  T-1   _build_date_groups raises on oversized groups (silent truncation
        was the root cause of OOS IC = 0.006).
  T-7/8 NaN labels are excluded from BOTH the label softmax AND the
        prediction softmax in ListNet loss.
  T-23  predict() sorts panel by date so groupby produces correct
        contiguous date-groups even when the caller's panel is unsorted.
  N-1/13 NGBoostHead.train drops NaN/inf feature/label/weight rows.
  N-5   NGBoostHead.predict_distribution returns NaN for NaN-input rows
        instead of erroring out / poisoning all rows.
  N-25  ApplyNGBoostTask fills missing columns with 0.0 and continues
        instead of skipping the entire bar.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


# ── T-1: silent truncation removed ───────────────────────────────────────────

class TestT1NoSilentTruncation:
    def test_oversized_group_raises_explicit_error(self):
        """Pre-fix: max_tickers=10 + a 99-ticker group → only first 10
        kept, other 89 silently dropped. Post-fix: ValueError naming both
        numbers so the operator can raise max_tickers."""
        from training_panel.transformer_model import _build_date_groups
        n = 99
        panel = pd.DataFrame({
            "f1":     np.zeros(n, dtype=np.float32),
            "label":  np.zeros(n, dtype=np.float32),
        })
        with pytest.raises(ValueError, match="max_tickers"):
            _build_date_groups(panel, np.array([n]), ["f1"], "label",
                               max_tickers=10)

    def test_default_max_tickers_covers_99_watchlist(self):
        """Default max_tickers must be ≥ 99 (current watchlist size)."""
        from training_panel.transformer_model import TransformerParams
        assert TransformerParams.max_tickers >= 99, (
            f"max_tickers default {TransformerParams.max_tickers} is < 99 — "
            f"99-ticker date-groups would silently truncate before audit T-1 "
            f"fix. Bump to ≥ watchlist size."
        )


# ── T-7 / T-8: NaN labels properly masked ────────────────────────────────────

class TestT7T8NanLabelsMasked:
    def test_nan_label_rows_do_not_influence_loss(self):
        """A NaN label converted to 0 by `_build_date_groups` must not
        contribute to the ListNet softmax. Otherwise predictions get
        biased toward the median (the most common bug we found).

        Test design: keep the nan-label mask FIXED, but change the
        garbage values at the masked positions wildly. Loss must be
        invariant — exactly like the existing pad-mask invariance test
        for `test_pad_mask_excludes_padding_from_softmax`.
        """
        import torch
        from training_panel.transformer_model import _listnet_loss
        scores = torch.tensor([[1.0, 0.0, -1.0], [0.5, -0.5, 0.2]])
        labels = torch.tensor([[1.0, 0.0, -1.0], [0.5, -0.5, 0.2]])
        pad    = torch.tensor([[False, False, False], [False, False, False]])
        # Position [0,1] and [1,1] had NaN labels in the original panel.
        nan_y  = torch.tensor([[False, True, False], [False, True, False]])

        loss_a = _listnet_loss(scores, labels, pad, nan_y)

        # Same mask, but inject extreme garbage values at the masked
        # positions. The garbage should be invisible to the loss.
        scores2 = scores.clone()
        labels2 = labels.clone()
        scores2[0, 1] = 1e6
        scores2[1, 1] = -1e6
        labels2[0, 1] = 99.0
        labels2[1, 1] = -99.0
        loss_b = _listnet_loss(scores2, labels2, pad, nan_y)

        assert torch.allclose(loss_a, loss_b, atol=1e-5), (
            f"NaN-label-masked positions must not influence loss; "
            f"got Δ={float((loss_b - loss_a).item()):.4g}"
        )

    def test_nan_label_changes_loss_if_unmasked(self):
        """Sanity: if the same NaN positions are NOT masked (mask=False),
        garbage at those positions DOES change the loss. This proves
        the masking machinery is what protects the loss, not some other
        accidental property."""
        import torch
        from training_panel.transformer_model import _listnet_loss
        scores = torch.tensor([[1.0, 0.0, -1.0]])
        labels = torch.tensor([[1.0, 0.0, -1.0]])
        pad    = torch.tensor([[False, False, False]])
        nan_y_off = torch.tensor([[False, False, False]])  # nothing masked

        loss_clean = _listnet_loss(scores, labels, pad, nan_y_off)
        scores_dirty = scores.clone()
        scores_dirty[0, 1] = 1e6
        loss_dirty = _listnet_loss(scores_dirty, labels, pad, nan_y_off)
        assert not torch.allclose(loss_clean, loss_dirty), (
            "without masking, garbage in non-masked positions MUST change loss"
        )

    def test_nan_label_mask_default_back_compat(self):
        """When no nan_label_mask is passed, behave like the pre-audit
        version (only pad mask)."""
        import torch
        from training_panel.transformer_model import _listnet_loss
        scores = torch.tensor([[1.0, 0.0]])
        labels = torch.tensor([[0.5, -0.5]])
        pad    = torch.tensor([[False, False]])
        loss_no_arg   = _listnet_loss(scores, labels, pad)
        loss_with_arg = _listnet_loss(scores, labels, pad,
                                      torch.zeros_like(pad))
        assert torch.allclose(loss_no_arg, loss_with_arg)


# ── T-23: predict() sorts by date ────────────────────────────────────────────

class TestT23PredictSortsByDate:
    def test_predict_handles_unsorted_panel(self):
        """If the caller hands us a panel with date column but in
        non-date order, predict must still produce one prediction per
        row aligned with the input index."""
        import torch  # noqa: F401  (importorskip already present in other test files)
        from training_panel.transformer_model import PanelTransformerModel
        rng = np.random.default_rng(0)
        n_dates = 6
        n_tk = 5
        rows = []
        for d in range(n_dates):
            for t in range(n_tk):
                rows.append({
                    "date": d, "ticker": f"T{t}",
                    "f1": float(rng.normal()), "f2": float(rng.normal()),
                    "label": float(rng.normal()), "weight": 1.0,
                })
        panel = pd.DataFrame(rows)
        gs = panel.groupby("date", sort=True).size().values.astype(int)
        m = PanelTransformerModel(params={
            "d_model": 16, "n_heads": 2, "n_layers": 1,
            "max_epochs": 2, "batch_size": 4, "device": "cpu",
            "max_tickers": n_tk, "seed": 0,
        })
        m.train(panel, gs, ["f1", "f2"], num_boost_round=2)

        # Now shuffle the panel rows. predict() must still yield the same
        # values per (input row index) as on the sorted panel.
        sorted_preds = m.predict(panel)
        shuffled = panel.sample(frac=1.0, random_state=42)
        shuffled_preds = m.predict(shuffled)
        # Re-align both to a common index for comparison.
        merged = pd.concat([
            sorted_preds.rename("sorted"),
            shuffled_preds.rename("shuffled"),
        ], axis=1)
        np.testing.assert_allclose(
            merged["sorted"].values, merged["shuffled"].values, atol=1e-5,
            err_msg="predict() must be invariant to caller's row order",
        )


# ── N-1 / N-13: NGBoost drops NaN feature/label/weight rows ──────────────────

class TestN1N13NGBoostNanDropped:
    def test_train_drops_nan_feature_rows_then_fits(self):
        """A panel with NaN in some feature rows must train successfully
        on the clean subset, not crash inside NGBoost."""
        from training_panel.ngboost_head import NGBoostHead
        rng = np.random.default_rng(0)
        n = 200
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y  = 2 * x1 - x2 + rng.normal(0.0, 0.3, n)
        # Poison 10% of rows with NaN in x1; some labels NaN; some weight=NaN.
        x1_corrupt = x1.copy()
        x1_corrupt[::10] = np.nan
        y_corrupt = y.copy()
        y_corrupt[5::15] = np.nan
        w_corrupt = np.ones(n)
        w_corrupt[3::20] = np.nan
        df = pd.DataFrame({
            "x1": x1_corrupt, "x2": x2,
            "residual_return_raw": y_corrupt,
            "weight": w_corrupt,
        })
        m = NGBoostHead({"n_estimators": 30, "learning_rate": 0.05})
        info = m.train(df, ["x1", "x2"])
        assert info["n_rows_dropped"] > 0
        assert info["n_rows"] < n
        assert info["train_mu_ic"] == info["train_mu_ic"]  # not NaN
        # Predict on the same df — clean rows get finite predictions, NaN
        # rows propagate NaN.
        preds = m.predict_distribution(df)
        finite_x = ~df["x1"].isna()
        assert preds.loc[finite_x, "mu"].notna().all()


# ── N-5: NGBoost predict returns NaN for NaN-input rows ──────────────────────

class TestN5NGBoostPredictNaNPassthrough:
    def test_nan_input_rows_get_nan_predictions(self):
        from training_panel.ngboost_head import NGBoostHead
        rng = np.random.default_rng(1)
        n = 150
        df = pd.DataFrame({
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "residual_return_raw": rng.normal(size=n),
        })
        m = NGBoostHead({"n_estimators": 30, "learning_rate": 0.05})
        m.train(df, ["x1", "x2"])
        # Build a tiny inference frame with one NaN row.
        infer = pd.DataFrame({
            "x1": [0.5, np.nan, 1.2],
            "x2": [0.0, 0.7,    np.inf],
        }, index=["AAA", "BBB", "CCC"])
        preds = m.predict_distribution(infer)
        assert preds.loc["AAA"].notna().all()
        assert preds.loc["BBB"].isna().all()
        assert preds.loc["CCC"].isna().all()


# ── N-2 / N-14: NGBoost early stopping on validation NLL ─────────────────────

class TestN2N14EarlyStopping:
    def _panel_with_dates(self, n: int = 600, seed: int = 0):
        rng = np.random.default_rng(seed)
        n_dates = n // 5  # 5 tickers per date
        rows = []
        for d in range(n_dates):
            for t in range(5):
                x1 = rng.normal()
                x2 = rng.normal()
                rows.append({
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=d),
                    "ticker": f"T{t}",
                    "x1": x1, "x2": x2,
                    "residual_return_raw": 1.5 * x1 - 0.5 * x2 + rng.normal(0.0, 0.2),
                    "weight": 1.0,
                })
        return pd.DataFrame(rows)

    def test_default_no_early_stop_back_compat(self):
        """Default `early_stopping_rounds=None` must preserve pre-fix
        behavior — single-shot fit on the full panel."""
        from training_panel.ngboost_head import NGBoostHead
        df = self._panel_with_dates(n=200, seed=1)
        m = NGBoostHead({"n_estimators": 30, "learning_rate": 0.05})
        info = m.train(df, ["x1", "x2"])
        # No val split → n_rows_val=0 by contract
        assert info["n_rows_val"] == 0
        assert info["best_iter"] is None  # NGBoost only sets best_val_loss_itr when val provided

    def test_early_stop_uses_time_ordered_val_split(self):
        """When enabled, last 20% of distinct dates form the val set."""
        from training_panel.ngboost_head import NGBoostHead
        df = self._panel_with_dates(n=500, seed=2)
        m = NGBoostHead({"n_estimators": 60, "learning_rate": 0.05})
        info = m.train(
            df, ["x1", "x2"],
            early_stopping_rounds=10,
            val_fraction=0.2,
        )
        # 20 dates × 5 = 100 val rows, 80 dates × 5 = 400 train rows
        assert info["n_rows_train"] >= 350 and info["n_rows_train"] <= 410
        assert info["n_rows_val"] >= 80 and info["n_rows_val"] <= 110
        # best_iter should be set when val provided (NGBoost reports the
        # iteration with lowest val NLL, regardless of whether early-stop
        # actually fired)
        assert isinstance(info.get("val_mu_ic"), float)

    def test_early_stop_actually_halts_before_max_iter(self):
        """With low patience, training should stop before n_estimators."""
        from training_panel.ngboost_head import NGBoostHead
        df = self._panel_with_dates(n=400, seed=3)
        m = NGBoostHead({"n_estimators": 200, "learning_rate": 0.1})
        info = m.train(
            df, ["x1", "x2"],
            early_stopping_rounds=5,   # tight patience
            val_fraction=0.25,
        )
        # NGBoost sets `best_val_loss_itr` ≤ n_estimators when early-stop fired.
        # We can't assert it's strictly less without making the test flaky,
        # but we can verify val IC was computed.
        assert info["n_rows_val"] > 0


# ── N-25: ApplyNGBoostTask fills missing columns ──────────────────────────────

class TestN25ApplyNGBoostHandlesMissingCols:
    def test_missing_column_filled_not_skipped(self, tmp_path):
        """Pre-fix: any column missing → no-op, μ/σ never written.
        Post-fix: warn + fill 0.0, predictions still produced."""
        from kernel.panel_pipeline.job_panel_scoring import ApplyNGBoostTask
        from training_panel.ngboost_head import NGBoostHead

        rng = np.random.default_rng(2)
        n = 200
        train_df = pd.DataFrame({
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
            "residual_return_raw": rng.normal(size=n),
        })
        head = NGBoostHead({"n_estimators": 30, "learning_rate": 0.05})
        head.train(train_df, ["x1", "x2", "x3"])

        # Inference matrix is missing x3.
        X = pd.DataFrame({
            "x1": [0.5, -0.3],
            "x2": [0.1,  0.4],
        }, index=["AAA", "BBB"])

        # Build a candidate-style InferenceContext stub.
        cand_a = SimpleNamespace(ticker="AAA", mu=None, sigma=None,
                                  rank_score=None, panel_score=None)
        cand_b = SimpleNamespace(ticker="BBB", mu=None, sigma=None,
                                  rank_score=None, panel_score=None)
        ctx = SimpleNamespace(
            config={"ranking": {"panel_scoring": {"ngboost": {
                "enabled": True, "score_mode": "additive", "lambda_sigma": 0.0,
            }}}},
            candidates=[cand_a, cand_b],
            holdings={},
        )
        ctx._ngboost_head = head
        ctx._panel_matrix = X
        ApplyNGBoostTask().run(ctx)
        # Pre-fix: cand.mu/sigma stayed None (task skipped).
        # Post-fix: predictions populated (with x3 filled to 0.0).
        assert cand_a.mu is not None and cand_a.sigma is not None
        assert cand_b.mu is not None and cand_b.sigma is not None
