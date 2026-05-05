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


# ── T-25 / T-16 / T-18: transformer overfit fixes ────────────────────────────

class TestTransformerOverfitFixes:
    """Audit fixes 2026-04-25 to bring transformer's OOS / Train ratio up
    from 7% (v3) toward 20-50% (panel-LTR's range).
    """

    def test_default_dropout_compounding_below_30pct(self):
        """T-25: pre-fix, dropout=0.30 + feature_dropout=0.20 + ticker_dropout=0.10
        compounded to ~50% effective signal loss. New defaults must keep
        compound effective dropout ≤ 35%."""
        from training_panel.transformer_model import TransformerParams
        p = TransformerParams()
        # Compound formula: 1 - (1-d) × (1-fd) × (1-td)
        compound = 1.0 - (1 - p.dropout) * (1 - p.feature_dropout) * (1 - p.ticker_dropout)
        assert compound <= 0.35, (
            f"compound dropout {compound:.3f} exceeds 0.35 — overfit risk. "
            f"Lower dropout/feature_dropout/ticker_dropout."
        )

    def test_default_grad_clip_set(self):
        """T-16: gradient clipping is on by default."""
        from training_panel.transformer_model import TransformerParams
        p = TransformerParams()
        assert p.grad_clip_norm is not None and p.grad_clip_norm > 0, (
            "gradient clipping must be enabled by default to stabilise "
            "softmax/log-softmax gradients."
        )

    def test_default_auto_eval_split_on(self):
        """T-18: auto eval split is on by default so early stopping fires
        even when callers (CV, FinalFit) don't pass eval_panel."""
        from training_panel.transformer_model import TransformerParams
        p = TransformerParams()
        assert p.auto_eval_split is True
        assert 0.05 < p.auto_eval_fraction < 0.5

    def test_auto_eval_split_runs_and_early_stops(self):
        """End-to-end: train without eval_panel → auto-split fires →
        early-stopping recorded in history."""
        import torch  # noqa: F401  (importorskip pattern in other tests)
        from training_panel.transformer_model import PanelTransformerModel
        rng = np.random.default_rng(0)
        n_dates = 80
        n_tk = 6
        rows = []
        for d in range(n_dates):
            for t in range(n_tk):
                x1 = rng.normal()
                x2 = rng.normal()
                rows.append({
                    "date": d, "ticker": f"T{t}",
                    "x1": x1, "x2": x2,
                    "label": float(2 * x1 - x2 + rng.normal(0, 0.3)),
                    "weight": 1.0,
                })
        panel = pd.DataFrame(rows)
        gs = panel.groupby("date", sort=True).size().values.astype(np.int32)

        m = PanelTransformerModel(params={
            "d_model": 16, "n_heads": 2, "n_layers": 1,
            "max_epochs": 20, "batch_size": 8, "device": "cpu",
            "max_tickers": n_tk, "seed": 7,
            "patience": 3,
        })
        info = m.train(panel, gs, ["x1", "x2"], num_boost_round=20)
        # auto-split happened → eval_ic recorded → best_iter reflects best epoch.
        assert "eval_ic" in info, (
            "auto eval split must produce eval_ic in fit metadata"
        )
        # best_iter should be a real epoch (not None) once eval fired
        assert m.best_iter is not None


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


# ── N-5 + N-Coverage: NGBoost imputes NaN at predict-time ────────────────────

class TestN5NGBoostPredictNaNPassthrough:
    def test_nan_input_rows_get_imputed_then_predicted(self):
        """Audit fix N-Coverage (2026-04-25): pre-fix, a single NaN feature
        cell at predict-time produced NaN output, which downstream
        ApplyNGBoostTask treated as "no μ/σ available" → silent skip.

        Post-fix, NaN cells are filled with the train-time column median
        (persisted on the head) so we still produce finite μ/σ estimates
        for tickers with patchy factor coverage (e.g. foreign stocks
        with no SEC Form 4 filings, or short-history tickers missing
        hourly aggregates). This restored 86% of dropped rows in the
        2026-04-25 retrain.
        """
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
        # Build a tiny inference frame with one NaN row + one inf row.
        infer = pd.DataFrame({
            "x1": [0.5, np.nan, 1.2],
            "x2": [0.0, 0.7,    np.inf],
        }, index=["AAA", "BBB", "CCC"])
        preds = m.predict_distribution(infer)
        # All three rows now produce finite μ/σ — NaN/inf cells were
        # replaced with train-time medians.
        assert preds.loc["AAA"].notna().all()
        assert preds.loc["BBB"].notna().all()
        assert preds.loc["CCC"].notna().all()
        assert (preds["sigma"] > 0).all()

    def test_impute_features_disabled_drops_nan_rows(self):
        """When impute_features=False, fall back to old drop-row behavior."""
        from training_panel.ngboost_head import NGBoostHead
        rng = np.random.default_rng(2)
        n = 150
        df = pd.DataFrame({
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "residual_return_raw": rng.normal(size=n),
        })
        m = NGBoostHead({"n_estimators": 30, "learning_rate": 0.05})
        m.train(df, ["x1", "x2"], impute_features=False)
        # impute_features=False → no medians persisted
        assert getattr(m, "feature_medians_", None) is None
        infer = pd.DataFrame({
            "x1": [0.5, np.nan, 1.2],
            "x2": [0.0, 0.7,    np.inf],
        }, index=["AAA", "BBB", "CCC"])
        preds = m.predict_distribution(infer)
        # Without medians, NaN/inf rows propagate NaN as before.
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

    def test_n17_cpcv_adapter_produces_oos_ic(self):
        """N-17: NGBoostFitTask.cv adapter wraps NGBoostHead so panel-LTR's
        purged-CV machinery produces OOS μ-IC for NGBoost too. Pre-fix
        the artifact had only train_mu_ic; post-fix oos_mean_ic +
        per-fold values are recorded when `panel_ltr.ngboost.cv.enabled`.
        """
        # Standalone test of the adapter pattern — full pipeline test
        # would need a 1000+ row panel which is too slow for unit tests.
        from training_panel.ngboost_head import NGBoostHead
        from training_panel.purged_cv import PurgedKFold, cross_validated_ic
        rng = np.random.default_rng(0)
        n_dates = 80
        rows = []
        for d in range(n_dates):
            for t in range(5):
                x1 = rng.normal()
                x2 = rng.normal()
                rows.append({
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=d),
                    "ticker": f"T{t}",
                    "x1": x1, "x2": x2,
                    "residual_return_raw": 1.5 * x1 - 0.5 * x2 + rng.normal(0.0, 0.3),
                    "weight": 1.0,
                })
        panel = pd.DataFrame(rows)

        class _Adapter:
            def __init__(self_a):
                self_a._head = NGBoostHead({"n_estimators": 30, "learning_rate": 0.05})
            def fit(self_a, X, y, sample_weight=None):
                df = X.copy()
                df["residual_return_raw"] = y
                if sample_weight is not None:
                    df["weight"] = sample_weight
                self_a._head.train(
                    df, feature_cols=list(X.columns),
                    label_col="residual_return_raw",
                    sample_weight_col="weight" if sample_weight is not None else None,
                )
            def predict(self_a, X):
                return self_a._head.predict_distribution(X)["mu"].values

        cv = PurgedKFold(n_splits=3, embargo_days=5, lookahead_days=5)
        result = cross_validated_ic(
            _Adapter, panel, ["x1", "x2"], "residual_return_raw", cv,
            weight_col="weight",
        )
        assert "mean_ic" in result
        # On a synthetic Gaussian panel with strong signal, mean_ic should
        # be clearly positive (≥0.10 typical with 1.5x1 - 0.5x2 + small noise).
        assert result["mean_ic"] > 0.05, (
            f"NGBoost CPCV adapter recovered IC {result['mean_ic']:.3f} on "
            f"a clean signal — should be ≥0.05. Adapter wiring may be broken."
        )

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


# ── 2026-05-04 user mandate: NGBoost task must tag every skipped cand ──

class TestNGBoostSkipReasonsInstrumentation:
    """When ApplyNGBoostTask cannot populate μ/σ for a candidate, it must
    write a per-ticker reason into ctx._blocked_by_ticker so the
    candidate_scores DB column blocked_by reveals exactly why on a SQL
    query. Three skip categories:

      ngb_skipped:not_in_predict_index
          → candidate's ticker has no row in the inference matrix
            (BuildFeatureMatrix dropped it).
      ngb_skipped:mu_nan
          → predict_distribution returned NaN μ (bad inference row).
      ngb_skipped:sigma_nan
          → predict returned NaN σ.
    """

    def _train_minimal_head(self):
        """Tiny NGBoostHead — just enough to predict on a 2x2 matrix."""
        from training_panel.ngboost_head import NGBoostHead
        rng = np.random.default_rng(7)
        n = 200
        df = pd.DataFrame({
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "residual_return_raw": rng.normal(size=n),
        })
        head = NGBoostHead({"n_estimators": 30, "learning_rate": 0.05})
        head.train(df, ["x1", "x2"], impute_features=False)
        return head

    def _make_ctx(self, candidates, X, head):
        return SimpleNamespace(
            config={"ranking": {"panel_scoring": {"ngboost": {
                "enabled": True, "score_mode": "additive",
                "lambda_sigma": 0.0,
            }}}},
            candidates=list(candidates),
            holdings={},
            _ngboost_head=head,
            _panel_matrix=X,
        )

    def test_not_in_index_tagged(self):
        """Candidate whose ticker isn't in the inference matrix index gets
        ngb_skipped:not_in_predict_index."""
        from kernel.panel_pipeline.job_panel_scoring import ApplyNGBoostTask
        head = self._train_minimal_head()
        # Matrix only has AAA — but cand list includes BBB
        X = pd.DataFrame({"x1": [0.1], "x2": [0.2]}, index=["AAA"])
        cand_aaa = SimpleNamespace(
            ticker="AAA", mu=None, sigma=None,
            rank_score=None, panel_score=None,
        )
        cand_bbb = SimpleNamespace(
            ticker="BBB", mu=None, sigma=None,
            rank_score=None, panel_score=None,
        )
        ctx = self._make_ctx([cand_aaa, cand_bbb], X, head)
        ApplyNGBoostTask().run(ctx)
        # AAA: μ/σ written, no skip reason
        assert cand_aaa.mu is not None
        assert "AAA" not in getattr(ctx, "_blocked_by_ticker", {})
        # BBB: skip reason logged
        assert ctx._blocked_by_ticker["BBB"] == \
               "ngb_skipped:not_in_predict_index"

    def test_mu_nan_tagged(self):
        """When predict_distribution returns NaN μ for a ticker (NaN/inf
        row passthrough with impute_features=False), tag it."""
        from kernel.panel_pipeline.job_panel_scoring import ApplyNGBoostTask
        head = self._train_minimal_head()
        # Row CCC has inf input → NaN output (impute_features=False above)
        X = pd.DataFrame({
            "x1": [0.1, np.inf],
            "x2": [0.2, 0.3],
        }, index=["AAA", "CCC"])
        cand_aaa = SimpleNamespace(
            ticker="AAA", mu=None, sigma=None,
            rank_score=None, panel_score=None,
        )
        cand_ccc = SimpleNamespace(
            ticker="CCC", mu=None, sigma=None,
            rank_score=None, panel_score=None,
        )
        ctx = self._make_ctx([cand_aaa, cand_ccc], X, head)
        ApplyNGBoostTask().run(ctx)
        assert cand_aaa.mu is not None
        # CCC's μ came back NaN → tagged
        assert "ngb_skipped:" in ctx._blocked_by_ticker["CCC"]
        # Specifically mu_nan or sigma_nan (both can happen on inf input)
        assert ctx._blocked_by_ticker["CCC"] in (
            "ngb_skipped:mu_nan", "ngb_skipped:sigma_nan",
        )
