"""Regression tests for scripts/patchtst_hf.py — HF Trainer + multi-task head.

Pins the canonical-lib mandate per CLAUDE.md §5.12: HF Trainer + canonical
torch losses (margin_ranking_loss, StudentT.log_prob) over hand-rolled
train loop. Distributional (μ, σ) head replaces deprecated NGBoost σ wire.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts/patchtst_hf.py"
sys.path.insert(0, str(REPO))


def _load_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location("patchtst_hf", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── Source contract tests (compile-time guarantees) ────────────────────────


class TestSourceContracts:
    """Pin design intent in source so future refactors can't silently drift."""

    def test_script_exists(self):
        assert SCRIPT.exists()

    def test_uses_hf_patchtst_backbone(self):
        src = SCRIPT.read_text()
        assert "PatchTSTConfig" in src and "PatchTSTModel" in src
        assert "from transformers import" in src

    def test_uses_hf_trainer(self):
        """§5.12 — canonical lib over hand-rolled train loop."""
        src = SCRIPT.read_text()
        assert "Trainer" in src and "TrainingArguments" in src
        assert "TrainerCallback" in src

    def test_uses_load_best_model_at_end(self):
        """Solves prior best-epoch save bug — Trainer loads best at end."""
        src = SCRIPT.read_text()
        assert "load_best_model_at_end=True" in src
        assert "metric_for_best_model" in src

    def test_uses_min_regime_ic_for_selection(self):
        """PRIME DIRECTIVE — min-across-regime IC, not pooled mean."""
        src = SCRIPT.read_text()
        assert "eval_min_regime_ic" in src
        assert "min(per_regime" in src.replace(" ", "")

    def test_uses_cosine_warmup_schedule(self):
        """No more fixed-LR (DOE warmup main-effect noise + best practice)."""
        src = SCRIPT.read_text()
        assert "lr_scheduler" in src
        assert "warmup_ratio" in src
        assert "cosine" in src

    def test_uses_canonical_margin_ranking_loss(self):
        """CIKM 2025 (arXiv 2510.14156) — Margin Ranking beats pairwise BCE
        on portfolio Sharpe. Canonical torch.nn.functional, not custom."""
        src = SCRIPT.read_text()
        assert "margin_ranking_loss" in src

    def test_uses_canonical_student_t_distribution(self):
        """torch.distributions.StudentT — replaces NGBoost σ wire."""
        src = SCRIPT.read_text()
        assert "torch.distributions.StudentT" in src

    def test_no_handwritten_attention_or_patch_embed(self):
        src = SCRIPT.read_text()
        for f in ("class TransformerEncoder", "MultiHeadAttention",
                  "PatchEmbed(", "patch_embed = nn", "SinusoidalPos",
                  "ScaledDotProduct"):
            assert f not in src, f"forbidden custom-arch token: {f}"

    def test_no_handwritten_train_loop(self):
        """No `for ep in range(...)` train loop — Trainer owns it now."""
        src = SCRIPT.read_text()
        # The only `for` over epochs left should be inside HF Trainer's
        # train() method (not in this wrapper).
        forbidden_patterns = [
            "for ep in range",
            "optimizer.step()",
            "model.train()  # in train",
        ]
        for f in forbidden_patterns:
            assert f not in src, f"forbidden hand-rolled-loop token: {f}"

    def test_uses_walk_forward_split(self):
        src = SCRIPT.read_text()
        assert "from kernel.walk_forward_splits import" in src
        assert "build_default_cuts" in src

    def test_dumps_val_preds_for_regime_eval(self):
        src = SCRIPT.read_text()
        assert "val_preds.parquet" in src

    def test_loc_budget(self):
        """HF Trainer wrapper + dual head + FiLM + artifact contract: 660 LOC.

        Trajectory: 350 (pre-refactor) → 450 (HF Trainer + dual head + per-
        regime callback) → 550 (+ FiLM Pillar B) → 660 (train-fit
        preprocessing + model contract stamp). All additions are config flags /
        canonical-lib glue / audit metadata, not custom training infrastructure."""
        src = SCRIPT.read_text()
        loc = sum(1 for line in src.splitlines()
                  if line.strip() and not line.strip().startswith("#"))
        assert loc <= 660, f"wrapper grew to {loc} LOC — too thick"

    def test_film_layer_class_exported(self):
        """FiLM (Perez 2017) regime conditioning module — Pillar B foundation."""
        src = SCRIPT.read_text()
        assert "class FiLMLayer" in src
        assert "Perez 2017" in src or "1709.07871" in src


# ─── Forward pass + heads ───────────────────────────────────────────────────


class TestModelArchitecture:

    def test_forward_returns_dict_with_score(self):
        from transformers import PatchTSTConfig
        import torch
        mod = _load_mod()
        cfg = PatchTSTConfig(num_input_channels=8, context_length=16,
                              patch_length=4, patch_stride=4, d_model=32,
                              num_attention_heads=2, num_hidden_layers=1,
                              ffn_dim=64)
        model = mod.HFPatchTSTRanker(cfg, use_distributional_head=False)
        x = torch.randn(5, 16, 8)
        out = model(x)
        assert isinstance(out, dict)
        assert "score" in out
        assert out["score"].shape == (5,)
        assert "loc" not in out, "no distributional head → no μ/σ"

    def test_forward_distributional_head_emits_df_loc_scale(self):
        from transformers import PatchTSTConfig
        import torch
        mod = _load_mod()
        cfg = PatchTSTConfig(num_input_channels=8, context_length=16,
                              patch_length=4, patch_stride=4, d_model=32,
                              num_attention_heads=2, num_hidden_layers=1,
                              ffn_dim=64)
        model = mod.HFPatchTSTRanker(cfg, use_distributional_head=True)
        x = torch.randn(5, 16, 8)
        out = model(x)
        for key in ("score", "df", "loc", "scale"):
            assert key in out, f"missing distributional output: {key}"
        # Student-t constraints: df > 2, scale > 0
        assert (out["df"] > 2.0 - 1e-6).all()
        assert (out["scale"] > 0).all()


class TestLosses:

    def test_margin_ranking_loss_zero_for_perfect_margin(self):
        import torch
        mod = _load_mod()
        scores = torch.tensor([50.0, 40.0, 30.0, 20.0, 10.0])
        labels = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0])
        loss = mod.margin_ranking_loss(scores, labels, margin=0.1)
        # Wide margin — all pairs satisfied — loss should be 0
        assert loss < 0.01

    def test_margin_ranking_loss_positive_for_random(self):
        import torch
        mod = _load_mod()
        torch.manual_seed(0)
        scores = torch.randn(20)
        labels = torch.randn(20)
        loss = mod.margin_ranking_loss(scores, labels, margin=0.1)
        # Should be positive (most pairs violate margin)
        assert loss > 0.0

    def test_student_t_nll_finite_on_normal_inputs(self):
        import torch
        mod = _load_mod()
        df = torch.full((10,), 5.0)
        loc = torch.zeros(10)
        scale = torch.full((10,), 0.1)
        target = torch.randn(10) * 0.1
        nll = mod.student_t_nll(df, loc, scale, target)
        assert torch.isfinite(nll).item()

    def test_student_t_nll_lower_when_calibrated(self):
        """NLL should be lower when σ matches actual noise scale."""
        import torch
        mod = _load_mod()
        df = torch.full((100,), 10.0)
        loc = torch.zeros(100)
        target = torch.randn(100) * 0.1  # true σ ≈ 0.1
        nll_well_calibrated = mod.student_t_nll(
            df, loc, torch.full((100,), 0.1), target)
        nll_too_wide = mod.student_t_nll(
            df, loc, torch.full((100,), 1.0), target)
        nll_too_narrow = mod.student_t_nll(
            df, loc, torch.full((100,), 0.01), target)
        assert nll_well_calibrated < nll_too_wide
        assert nll_well_calibrated < nll_too_narrow


# ─── Per-day batching ───────────────────────────────────────────────────────


class TestPerDayDataset:

    def test_dataset_yields_per_day_batches(self):
        import pandas as pd
        import numpy as np
        mod = _load_mod()
        rng = np.random.default_rng(0)
        dates = pd.date_range("2024-01-01", periods=20)
        rows = []
        for d in dates:
            for tkr in "ABCDEFGH":
                rows.append({"date": d, "ticker": tkr,
                              "f1": rng.normal(), "f2": rng.normal(),
                              "fwd_60d_excess": rng.normal(),
                              "split_label": "train"})
        panel = pd.DataFrame(rows)
        ds = mod.PerDayDataset(panel, ["f1", "f2"], "fwd_60d_excess",
                                seq_len=5, split="train")
        assert len(ds) > 0
        sample = ds[0]
        assert "past_values" in sample and "labels" in sample
        assert sample["past_values"].dim() == 3  # (N_tickers, seq_len, n_feat)
        assert sample["past_values"].shape[1] == 5  # seq_len
        assert sample["past_values"].shape[2] == 2  # n_feat

    def test_identity_collator_unwraps_single_batch(self):
        mod = _load_mod()
        sample = {"past_values": "X", "labels": "Y"}
        out = mod.identity_collator([sample])
        assert out is sample


# ─── Preprocessing ──────────────────────────────────────────────────────────


class TestPreprocessing:

    def test_csrank_norm_range_minus_half_to_plus_half(self):
        import pandas as pd
        import numpy as np
        mod = _load_mod()
        rng = np.random.default_rng(0)
        dates = pd.date_range("2024-01-01", periods=10)
        panel = pd.DataFrame({
            "date": np.tile(dates, 5),
            "ticker": np.repeat(list("ABCDE"), 10),
            "f1": rng.normal(0, 100, 50),
            "f2": rng.normal(0, 0.01, 50),
        })
        out = mod.csrank_norm_per_day(panel, ["f1", "f2"])
        assert out["f1"].min() >= -0.5 - 1e-9
        assert out["f1"].max() <= +0.5 + 1e-9

    def test_csrank_norm_per_day_independence(self):
        import pandas as pd
        mod = _load_mod()
        panel = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01")] * 5 + [pd.Timestamp("2024-01-02")] * 5,
            "ticker": list("ABCDE") * 2,
            "f1": [1, 2, 3, 4, 5, 100, 200, 300, 400, 500],
        })
        out = mod.csrank_norm_per_day(panel, ["f1"])
        d1 = out[out["date"] == "2024-01-01"]["f1"].sort_values().tolist()
        d2 = out[out["date"] == "2024-01-02"]["f1"].sort_values().tolist()
        assert d1 == pytest.approx(d2)

    def test_winsorize_caps_extremes(self):
        import pandas as pd
        import numpy as np
        mod = _load_mod()
        rng = np.random.default_rng(0)
        labels = rng.normal(0, 1, 1000).tolist() + [1000.0, -1000.0]
        panel = pd.DataFrame({"y": labels})
        out = mod.winsorize_label(panel, "y", pct=0.005)
        assert out["y"].max() < 10.0
        assert out["y"].min() > -10.0

    def test_load_panel_winsorizes_from_train_split_only(self, tmp_path):
        import pandas as pd
        import numpy as np
        mod = _load_mod()

        dates = pd.date_range("2024-01-01", periods=10)
        rows = []
        for d_i, d in enumerate(dates):
            for t_i, ticker in enumerate(list("ABCDE")):
                is_val_tail = d_i >= 8
                rows.append({
                    "date": d,
                    "ticker": ticker,
                    "f1": float(d_i + t_i),
                    "fwd_60d_excess": 1000.0 if is_val_tail else float(d_i * 5 + t_i),
                })
        path = tmp_path / "panel.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)

        panel, _ = mod.load_panel_with_split(
            path, "all", "fwd_60d_excess", preprocess=True,
            val_tail_pct=0.2, embargo_days=1)

        meta = panel.attrs["label_winsor"]
        assert meta["fit_split"] == "train"
        assert meta["upper"] < 1000.0
        assert (panel["split_label"] == "embargo").any()
        val = panel[panel["split_label"] == "val"]["fwd_60d_excess"]
        assert np.isclose(val.max(), meta["upper"])

    def test_training_contract_stamps_hyperparameters(self):
        import argparse
        import pandas as pd
        mod = _load_mod()
        args = argparse.Namespace(
            dataset="data/example.parquet", cut="all", label="fwd_60d_excess",
            seed=44, seq_len=24, patch_length=4, d_model=64, n_heads=4,
            n_layers=2, epochs=8, lr=1e-4, weight_decay=0.3,
            lr_scheduler="cosine", warmup_ratio=0.1,
            early_stopping_patience=2, nll_loss_weight=0.5,
            ranking_margin=0.1, distributional_head=True,
            film_regime_cond=False, cross_stock_attn=False, device="cpu",
            embargo_days=60,
        )
        panel = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=4),
            "split_label": ["train", "train", "val", "test"],
        })
        panel.attrs["label_winsor"] = {
            "enabled": True, "fit_split": "train",
            "lower": -1.0, "upper": 1.0,
        }

        contract = mod.build_training_contract(
            args, ["f1", "f2"], panel, n_params=1234,
            total_steps=80, warmup_steps=8,
            metric_for_best="eval_min_regime_ic",
            final_metrics={"eval_min_regime_ic": 0.12},
        )

        assert contract["hyperparameters"]["seq_len"] == 24
        assert contract["hyperparameters"]["lr"] == 1e-4
        assert contract["hyperparameters"]["weight_decay"] == 0.3
        assert contract["preprocessing"]["label_winsor"]["fit_split"] == "train"
        assert contract["selection"]["metric_for_best_model"] == "eval_min_regime_ic"


# ─── FiLM regime conditioning ───────────────────────────────────────────────


class TestFiLMRegimeConditioning:
    """Perez 2017 (arXiv 1709.07871) Feature-wise Linear Modulation:
    γ, β = MLP(regime_context); h' = γ ⊙ h + β. Init must be identity."""

    def test_regimes_tuple_matches_hmm_emitter(self):
        """The 4-tuple must match kernel/regime.py production emitter."""
        mod = _load_mod()
        assert mod.REGIMES == ("BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR")

    def test_regime_to_onehot_known_labels(self):
        mod = _load_mod()
        import numpy as np
        for i, r in enumerate(mod.REGIMES):
            oh = mod.regime_to_onehot(r)
            assert oh.shape == (4,)
            expected = np.zeros(4, dtype=np.float32)
            expected[i] = 1.0
            assert np.array_equal(oh, expected)

    def test_regime_to_onehot_unknown_returns_zero(self):
        """Unknown label → all-zero vector (model gets no regime signal,
        safer than guessing)."""
        import numpy as np
        mod = _load_mod()
        oh = mod.regime_to_onehot("UNKNOWN_REGIME")
        assert np.array_equal(oh, np.zeros(4, dtype=np.float32))

    def test_film_is_identity_at_init(self):
        """Zero-init last layer → (γ, β) = (1, 0) → FiLM(h, ctx) == h
        regardless of ctx. Strict superset property of no-FiLM baseline."""
        import torch
        mod = _load_mod()
        film = mod.FiLMLayer(d_model=32, n_regimes=4)
        h = torch.randn(7, 32)
        ctx = torch.randn(7, 4)
        out = film(h, ctx)
        assert torch.allclose(out, h, atol=1e-7), \
            f"FiLM not identity at init: max diff {(out - h).abs().max().item():.2e}"

    def test_film_gradients_flow(self):
        """After backward, FiLM params have non-zero grad — modulation
        is learnable."""
        import torch
        mod = _load_mod()
        film = mod.FiLMLayer(d_model=32, n_regimes=4)
        h = torch.randn(7, 32, requires_grad=True)
        ctx = torch.randn(7, 4)
        loss = film(h, ctx).sum()
        loss.backward()
        # FiLM's last layer was zero-init; after one backward must have grad
        assert film.net[-1].weight.grad is not None
        assert film.net[-1].weight.grad.abs().sum() > 0

    def test_model_with_film_identity_at_init(self):
        """Model with FiLM ON + ctx given == model with FiLM ON + no ctx
        at init (because FiLM is identity at init when ctx provided, AND
        is skipped when ctx is None)."""
        import torch
        from transformers import PatchTSTConfig
        mod = _load_mod()
        cfg = PatchTSTConfig(num_input_channels=8, context_length=16,
                              patch_length=4, patch_stride=4, d_model=32,
                              num_attention_heads=2, num_hidden_layers=1,
                              ffn_dim=64)
        model = mod.HFPatchTSTRanker(cfg, use_distributional_head=False,
                                       use_film_regime=True)
        model.eval()
        x = torch.randn(5, 16, 8)
        ctx = torch.tensor([[1.0, 0, 0, 0]] * 5)
        with torch.no_grad():
            out_ctx = model(past_values=x, regime_context=ctx)["score"]
            out_no = model(past_values=x)["score"]
        assert torch.allclose(out_ctx, out_no, atol=1e-6)

    def test_dataset_injects_regime_context_when_hmm_provided(self):
        import pandas as pd
        import numpy as np
        mod = _load_mod()
        rng = np.random.default_rng(0)
        dates = pd.date_range("2024-01-01", periods=10)
        rows = []
        for d in dates:
            for tkr in "ABCDEFGH":
                rows.append({"date": d, "ticker": tkr,
                              "f1": rng.normal(), "f2": rng.normal(),
                              "fwd_60d_excess": rng.normal(),
                              "split_label": "train"})
        panel = pd.DataFrame(rows)
        hmm = pd.DataFrame({"date": dates,
                             "regime": ["BULL_CALM"] * 5 + ["BEAR"] * 5})
        ds = mod.PerDayDataset(panel, ["f1", "f2"], "fwd_60d_excess",
                                seq_len=3, split="train", hmm_labels=hmm)
        assert len(ds) > 0
        sample = ds[0]
        assert "regime_context" in sample
        # All rows for the same day share the same regime one-hot
        ctx = sample["regime_context"]
        assert ctx.shape[1] == 4
        assert (ctx[0] == ctx).all()

    def test_dataset_omits_regime_context_when_no_hmm(self):
        import pandas as pd
        import numpy as np
        mod = _load_mod()
        rng = np.random.default_rng(0)
        dates = pd.date_range("2024-01-01", periods=10)
        rows = []
        for d in dates:
            for tkr in "ABCDEFGH":
                rows.append({"date": d, "ticker": tkr,
                              "f1": rng.normal(), "f2": rng.normal(),
                              "fwd_60d_excess": rng.normal(),
                              "split_label": "train"})
        panel = pd.DataFrame(rows)
        ds = mod.PerDayDataset(panel, ["f1", "f2"], "fwd_60d_excess",
                                seq_len=3, split="train")  # no hmm_labels
        assert len(ds) > 0
        assert "regime_context" not in ds[0]


# ─── Cross-stock attention (Tier 2 T2.1, iTransformer-style) ───────────────


class TestCrossStockAttention:
    """iTransformer-style variate-as-token attention (Liu 2024, arXiv 2310.06625).
    Each ticker attends to all other tickers on the same day. Addresses
    PatchTST's documented #1 failure for cross-sectional finance."""

    def test_module_exists(self):
        mod = _load_mod()
        assert hasattr(mod, "CrossStockAttentionLayer")

    def test_identity_at_init(self):
        """Learnable alpha gate, init=0 → output exactly equals input.
        Strict superset of cross-stock-OFF baseline (worst case = same)."""
        import torch
        mod = _load_mod()
        csa = mod.CrossStockAttentionLayer(d_model=32, n_heads=4)
        csa.eval()
        h = torch.randn(7, 32)
        with torch.no_grad():
            out = csa(h)
        assert torch.allclose(out, h, atol=1e-6), \
            f"CSA not identity at init: max|diff| = {(out-h).abs().max():.2e}"

    def test_gradient_flows_through_alpha(self):
        import torch
        mod = _load_mod()
        csa = mod.CrossStockAttentionLayer(d_model=32, n_heads=4)
        h = torch.randn(7, 32, requires_grad=True)
        loss = csa(h).sum()
        loss.backward()
        assert csa.alpha.grad is not None
        assert csa.alpha.grad.abs().item() > 0, \
            "alpha must be learnable (non-zero grad after backward)"

    def test_output_shape_preserved(self):
        import torch
        mod = _load_mod()
        csa = mod.CrossStockAttentionLayer(d_model=64, n_heads=8)
        h = torch.randn(20, 64)
        out = csa(h)
        assert out.shape == h.shape

    def test_each_ticker_can_attend_to_others_after_training(self):
        """After training (alpha grows non-zero AND projections trained),
        each ticker's output should be a function of OTHER tickers' inputs.
        Simulate by manually fill ing alpha + projections."""
        import torch
        mod = _load_mod()
        torch.manual_seed(0)
        csa = mod.CrossStockAttentionLayer(d_model=16, n_heads=2)
        # Simulate trained state: alpha non-zero AND projections re-initialized
        with torch.no_grad():
            csa.alpha.fill_(1.0)
            nn = torch.nn
            nn.init.normal_(csa.attn.out_proj.weight, std=0.1)
            nn.init.normal_(csa.ffn[-1].weight, std=0.1)
        h1 = torch.zeros(5, 16)
        h1[0] = 1.0  # only first ticker has signal
        h2 = h1.clone()
        h2[2] = 1.0  # add signal to third ticker
        with torch.no_grad():
            out1 = csa(h1)
            out2 = csa(h2)
        # Ticker 0's output should differ between scenarios (info-routing)
        diff_t0 = (out1[0] - out2[0]).abs().max().item()
        assert diff_t0 > 1e-4, \
            "ticker 0's output unchanged when ticker 2's input changes — " \
            "cross-stock attention is NOT routing information after training"

    def test_model_with_cross_stock_identity_at_init(self):
        """Model with cross-stock ON should give same output as OFF at init."""
        import torch
        from transformers import PatchTSTConfig
        mod = _load_mod()
        cfg = PatchTSTConfig(num_input_channels=8, context_length=16,
                              patch_length=4, patch_stride=4, d_model=32,
                              num_attention_heads=4, num_hidden_layers=1,
                              ffn_dim=64)
        m_csa_on = mod.HFPatchTSTRanker(cfg, use_distributional_head=False,
                                          use_cross_stock_attn=True)
        m_csa_off = mod.HFPatchTSTRanker(cfg, use_distributional_head=False,
                                           use_cross_stock_attn=False)
        # Sync backbone + rank_head weights (only CSA differs)
        m_csa_off.load_state_dict(
            {k: v for k, v in m_csa_on.state_dict().items()
             if 'cross_stock' not in k}, strict=False)
        m_csa_on.eval(); m_csa_off.eval()
        x = torch.randn(5, 16, 8)
        with torch.no_grad():
            s_on = m_csa_on(past_values=x)["score"]
            s_off = m_csa_off(past_values=x)["score"]
        assert torch.allclose(s_on, s_off, atol=1e-6), \
            f"cross_stock ON ≠ OFF at init: max|diff|={(s_on-s_off).abs().max():.2e}"


# ─── Per-regime IC callback ─────────────────────────────────────────────────


class TestPerRegimeICCallback:

    def test_callback_class_exists(self):
        mod = _load_mod()
        assert hasattr(mod, "PerRegimeICCallback")

    def test_callback_inherits_from_hf_trainer_callback(self):
        from transformers import TrainerCallback
        mod = _load_mod()
        assert issubclass(mod.PerRegimeICCallback, TrainerCallback)

    def test_callback_on_evaluate_populates_metrics(self):
        """When the callback runs, it should add eval_min_regime_ic key
        and per-regime ic_BULL_CALM-style keys to the metrics dict."""
        import pandas as pd
        import numpy as np
        import torch
        from transformers import PatchTSTConfig
        mod = _load_mod()
        # Build a tiny eval dataset
        rng = np.random.default_rng(0)
        dates = pd.date_range("2020-03-01", periods=20)
        rows = []
        for d in dates:
            for tkr in "ABCDEFGH":
                rows.append({"date": d, "ticker": tkr,
                              "f1": rng.normal(), "f2": rng.normal(),
                              "fwd_60d_excess": rng.normal(),
                              "split_label": "val"})
        panel = pd.DataFrame(rows)
        ds = mod.PerDayDataset(panel, ["f1", "f2"], "fwd_60d_excess",
                                seq_len=5, split="val")
        # Fake HMM labels — all BULL_CALM (will require ≥5 days, we have 15+)
        hmm = pd.DataFrame({
            "date": dates,
            "regime": ["BULL_CALM"] * 20,
        })
        cb = mod.PerRegimeICCallback(ds, hmm)
        cfg = PatchTSTConfig(num_input_channels=2, context_length=5,
                              patch_length=1, patch_stride=1, d_model=16,
                              num_attention_heads=2, num_hidden_layers=1,
                              ffn_dim=32)
        model = mod.HFPatchTSTRanker(cfg, use_distributional_head=False)
        metrics = {}
        cb.on_evaluate(None, None, None, model=model, metrics=metrics)
        assert "eval_min_regime_ic" in metrics
        assert "eval_ic_BULL_CALM" in metrics


# ─── End-to-end smoke ───────────────────────────────────────────────────────


class TestSmokeEndToEnd:

    @pytest.mark.skipif(
        not (REPO / "data/transformer_v4_wl200_clean.parquet").exists(),
        reason="dataset not present")
    def test_2_epoch_smoke_run(self, tmp_path):
        """2-epoch run on cut1_covid must:
        - complete without error
        - write val_preds.parquet
        - write *_model.pt with --save-model
        - log per-regime IC + cosine schedule
        """
        import subprocess, sys
        out_dir = tmp_path / "hf_smoke"
        result = subprocess.run([
            sys.executable, str(SCRIPT),
            "--cut", "cut1_covid",
            "--epochs", "2",
            "--device", "cpu",
            "--seq-len", "8",
            "--d-model", "16", "--n-heads", "2", "--n-layers", "1",
            "--patch-length", "4",
            "--seed", "42",
            "--save-model",
            "--output-dir", str(out_dir),
        ], capture_output=True, text=True, timeout=1800)
        combined = result.stdout + result.stderr
        assert result.returncode == 0, (
            f"rc={result.returncode}\nstderr_tail: {result.stderr[-1500:]}"
        )
        import glob
        preds = glob.glob(str(out_dir / "*val_preds.parquet"))
        models = glob.glob(str(out_dir / "*_model.pt"))
        assert len(preds) == 1, f"expected 1 val_preds, found {preds}"
        assert len(models) == 1, f"expected 1 model.pt, found {models}"
        # Per-regime callback ran
        assert "per-regime IC" in combined or "PerRegimeICCallback wired" in combined


class TestSmokeArtifacts:

    @pytest.mark.skipif(
        not (REPO / "artifacts/hf_smoke/hf_patchtst_cut1_covid_seed42_val_preds.parquet").exists(),
        reason="manual smoke artifact not present")
    def test_smoke_val_preds_have_correct_columns(self):
        import pandas as pd
        vp = pd.read_parquet(
            REPO / "artifacts/hf_smoke/hf_patchtst_cut1_covid_seed42_val_preds.parquet")
        # date + pred + label always; mu/sigma when distributional head ON
        assert {"date", "pred", "label"}.issubset(set(vp.columns))
        assert len(vp) > 0
