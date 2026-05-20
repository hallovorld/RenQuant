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
        """Multi-task head + HF Trainer wrapper budget: 450 LOC.

        Raised from 350 to absorb distributional head + per-regime callback
        + cosine+warmup + argparse expansion. All additions are config flags
        or canonical-lib glue, not custom training code (which is gone)."""
        src = SCRIPT.read_text()
        loc = sum(1 for line in src.splitlines()
                  if line.strip() and not line.strip().startswith("#"))
        assert loc <= 450, f"wrapper grew to {loc} LOC — too thick"


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
