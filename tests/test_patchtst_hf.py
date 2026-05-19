"""Regression tests for scripts/patchtst_hf.py — HF-based PatchTST wrapper.

Pins the 3rd-party-lib mandate per 2026-05-18 user directive:
"尽量用第三方lib". Verifies the wrapper uses HF transformers
PatchTSTModel as backbone and contains minimal custom-code surface area.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts/patchtst_hf.py"
sys.path.insert(0, str(REPO))


class TestSourceContracts:
    """Pin design intent in source so future refactors can't silently drift."""

    def test_script_exists(self):
        assert SCRIPT.exists()

    def test_uses_hf_patchtst_backbone(self):
        src = SCRIPT.read_text()
        assert "from transformers import PatchTSTConfig, PatchTSTModel" in src

    def test_no_handwritten_attention_or_patch_embed(self):
        """If we're rolling out HF, we must NOT carry over custom transformer
        impl from transformer_v4.py."""
        src = SCRIPT.read_text()
        forbidden = [
            "class TransformerEncoder", "MultiHeadAttention",
            "PatchEmbed(", "patch_embed = nn", "SinusoidalPos",
            "ScaledDotProduct",
        ]
        for f in forbidden:
            assert f not in src, f"forbidden custom-arch token: {f}"

    def test_uses_walk_forward_split_not_2023_only(self):
        """PRIME DIRECTIVE: must use kernel.walk_forward_splits, not
        the buggy 2023-only val that the deleted transformer_v4 used."""
        src = SCRIPT.read_text()
        assert "from kernel.walk_forward_splits import" in src
        assert "build_default_cuts" in src

    def test_pairwise_ranknet_loss(self):
        """Burges 2005 RankNet is the standard pairwise ranking loss."""
        src = SCRIPT.read_text()
        assert "binary_cross_entropy_with_logits" in src
        assert "RankNet" in src or "Burges" in src

    def test_dumps_val_preds_for_regime_eval(self):
        src = SCRIPT.read_text()
        assert "val_preds.parquet" in src

    def test_loc_budget(self):
        """Wrapper must stay thin. Custom-code budget: 250 LOC max."""
        src = SCRIPT.read_text()
        loc = sum(1 for line in src.splitlines()
                  if line.strip() and not line.strip().startswith("#"))
        assert loc <= 250, f"wrapper grew to {loc} LOC — too thick for 'thin wrapper' mandate"


class TestModelArchitecture:
    """Probe that the HF-backboned model trains end-to-end on a tiny input."""

    def test_forward_pass_shape(self):
        from transformers import PatchTSTConfig
        import torch
        # Direct import without running main()
        import importlib.util
        spec = importlib.util.spec_from_file_location("patchtst_hf", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        cfg = PatchTSTConfig(
            num_input_channels=8, context_length=16, patch_length=4,
            patch_stride=4, d_model=32, num_attention_heads=2,
            num_hidden_layers=1, ffn_dim=64,
        )
        model = mod.HFPatchTSTRanker(cfg)
        x = torch.randn(5, 16, 8)  # (B=5 tickers, seq=16, ch=8)
        out = model(x)
        assert out.shape == (5,), f"expected (5,) got {out.shape}"

    def test_pairwise_loss_zero_for_perfect_ranking(self):
        import importlib.util
        import torch
        spec = importlib.util.spec_from_file_location("patchtst_hf", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Sharp ranking margin → sigmoid saturates → loss near 0
        scores = torch.tensor([50.0, 40.0, 30.0, 20.0, 10.0])
        labels = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0])
        loss = mod.pairwise_rank_loss(scores, labels)
        assert loss < 0.01, f"sharp-margin perfect ranking should give ~0 loss, got {loss}"

    def test_pairwise_loss_high_for_random(self):
        import importlib.util
        import torch
        spec = importlib.util.spec_from_file_location("patchtst_hf", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Random scores should give loss near ln(2) ≈ 0.693
        torch.manual_seed(0)
        scores = torch.randn(20)
        labels = torch.randn(20)
        loss = mod.pairwise_rank_loss(scores, labels)
        assert 0.4 < loss < 1.0, f"random loss should be near ln 2, got {loss}"


class TestPreprocessing:
    """Variance-reduction preprocessing per roadmap PatchTST §variance protocol:
    CSRankNorm features per-day + Winsorize label ±0.5%.
    """

    def _load_mod(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("patchtst_hf", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_csrank_norm_range_minus_half_to_plus_half(self):
        import pandas as pd
        import numpy as np
        mod = self._load_mod()
        rng = np.random.default_rng(0)
        dates = pd.date_range("2024-01-01", periods=10)
        panel = pd.DataFrame({
            "date": np.tile(dates, 5),
            "ticker": np.repeat(list("ABCDE"), 10),
            "f1": rng.normal(0, 100, 50),  # large scale
            "f2": rng.normal(0, 0.01, 50),  # small scale
        })
        out = mod.csrank_norm_per_day(panel, ["f1", "f2"])
        # All normalized to [-0.5, +0.5]
        assert out["f1"].min() >= -0.5 - 1e-9
        assert out["f1"].max() <= +0.5 + 1e-9
        assert out["f2"].min() >= -0.5 - 1e-9
        assert out["f2"].max() <= +0.5 + 1e-9

    def test_csrank_norm_per_day_independence(self):
        """Each day's normalization must be independent of other days."""
        import pandas as pd
        import numpy as np
        mod = self._load_mod()
        # Day 1: values [1,2,3,4,5]; Day 2: values [100,200,300,400,500]
        # After CSRankNorm both should be uniformly ranked to [-0.5, +0.5]
        panel = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01")] * 5 + [pd.Timestamp("2024-01-02")] * 5,
            "ticker": list("ABCDE") * 2,
            "f1": [1, 2, 3, 4, 5, 100, 200, 300, 400, 500],
        })
        out = mod.csrank_norm_per_day(panel, ["f1"])
        # Both days should give the same rank-norm values
        day1_vals = out[out["date"] == "2024-01-01"]["f1"].sort_values().tolist()
        day2_vals = out[out["date"] == "2024-01-02"]["f1"].sort_values().tolist()
        assert day1_vals == pytest.approx(day2_vals)

    def test_winsorize_caps_extremes(self):
        import pandas as pd
        import numpy as np
        mod = self._load_mod()
        # Most values in [-1, +1], with a few extreme outliers
        rng = np.random.default_rng(0)
        labels = rng.normal(0, 1, 1000).tolist() + [1000.0, -1000.0, 999.0, -999.0]
        panel = pd.DataFrame({"y": labels})
        out = mod.winsorize_label(panel, "y", pct=0.005)
        # All extreme outliers should be clipped to within sane range
        assert out["y"].max() < 10.0, f"max not clipped: {out['y'].max()}"
        assert out["y"].min() > -10.0, f"min not clipped: {out['y'].min()}"
        # Non-extreme values preserved
        assert (out["y"].abs() <= 1.0).sum() >= 600  # most still ≤1

    def test_winsorize_preserves_median(self):
        import pandas as pd
        import numpy as np
        mod = self._load_mod()
        rng = np.random.default_rng(0)
        labels = rng.normal(0, 1, 1000)
        panel = pd.DataFrame({"y": labels})
        out = mod.winsorize_label(panel, "y", pct=0.005)
        # Median should be roughly preserved (symmetric winsorize)
        assert abs(out["y"].median() - np.median(labels)) < 0.01


class TestSmokeArtifacts:
    """If the smoke run artifact exists (from manual smoke), check shape."""

    @pytest.mark.skipif(
        not (REPO / "artifacts/hf_smoke/hf_patchtst_cut1_covid_seed42_val_preds.parquet").exists(),
        reason="manual smoke artifact not present")
    def test_smoke_val_preds_have_correct_columns(self):
        import pandas as pd
        vp = pd.read_parquet(
            REPO / "artifacts/hf_smoke/hf_patchtst_cut1_covid_seed42_val_preds.parquet")
        assert set(vp.columns) == {"date", "pred", "label"}
        assert len(vp) > 0
        assert vp["pred"].notna().all()
