"""Regression tests for HFPatchTSTPanelScorer.

Pins the interface (mirror of PatchTSTPanelScorer) + integration with
model_registry so shadow + future primary swap can rely on it.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting/renquant_104"))


@pytest.fixture(scope="module")
def scorer_mod():
    from kernel.panel_pipeline import hf_patchtst_scorer
    return hf_patchtst_scorer


class TestSourceContracts:
    """Pin design intent so future refactors can't drift."""

    def test_module_exists(self):
        path = (REPO / "backtesting/renquant_104/kernel/panel_pipeline"
                / "hf_patchtst_scorer.py")
        assert path.exists()

    def test_class_name_matches_legacy(self, scorer_mod):
        """Same naming convention as PatchTSTPanelScorer (just HF prefix)."""
        assert hasattr(scorer_mod, "HFPatchTSTPanelScorer")

    def test_required_attrs(self, scorer_mod):
        src = (REPO / "backtesting/renquant_104/kernel/panel_pipeline"
               / "hf_patchtst_scorer.py").read_text()
        # Same API as legacy PatchTSTPanelScorer
        for required in ("feature_cols", "seq_len", "requires_history",
                          "score_with_history", "load", "metadata"):
            assert required in src, f"missing required attr/method: {required}"

    def test_no_handwritten_attention_or_model(self, scorer_mod):
        """Must use HF transformers, not roll its own arch."""
        src = (REPO / "backtesting/renquant_104/kernel/panel_pipeline"
               / "hf_patchtst_scorer.py").read_text()
        # Loads via importlib from scripts/patchtst_hf.py which uses HF
        assert "HFPatchTSTRanker" in src
        for f in ("class TransformerEncoder", "MultiHeadAttention",
                  "class PatchEmbed"):
            assert f not in src

    def test_csranknorm_at_inference(self, scorer_mod):
        """PRIME risk: training uses CSRankNorm; inference MUST too or model
        sees out-of-distribution scales → garbage scores."""
        src = (REPO / "backtesting/renquant_104/kernel/panel_pipeline"
               / "hf_patchtst_scorer.py").read_text()
        assert "_csrank_norm_per_day" in src
        # Score path actually CALLS it
        assert "ph = _csrank_norm_per_day" in src

    def test_omp_fix_at_top(self, scorer_mod):
        """OMP=1 prevents xgboost ↔ HF torch SIGSEGV at coexistence."""
        src = (REPO / "backtesting/renquant_104/kernel/panel_pipeline"
               / "hf_patchtst_scorer.py").read_text()
        assert 'os.environ.setdefault("OMP_NUM_THREADS"' in src


class TestModelRegistryIntegration:
    def test_hf_patchtst_kind_registered(self):
        from kernel.panel_pipeline.model_registry import registry
        h = registry.get("hf_patchtst")
        assert h.kind == "hf_patchtst"
        assert h.requires_history is True
        assert callable(h.scorer_loader)
        assert callable(h.train_cmd)

    def test_train_cmd_uses_save_model(self):
        from kernel.panel_pipeline.model_registry import registry
        h = registry.get("hf_patchtst")
        class Args:
            dataset = "/tmp/x.parquet"
            label = "fwd_60d_excess"
            output_dir = "/tmp/out"
        cmd = h.train_cmd(Args)
        assert "--save-model" in cmd
        assert "scripts/patchtst_hf.py" in " ".join(cmd)


class TestLoadScore:
    """End-to-end load + score test using the prod-style artifact (if it
    exists from earlier training). Skipped if no artifact present yet."""

    @pytest.mark.skipif(
        not list((REPO / "artifacts/hf_patchtst_prod").rglob("*_model.pt")),
        reason="no HF PatchTST model artifact yet (train pending)")
    def test_load_returns_scorer(self, scorer_mod):
        artifact = next((REPO / "artifacts/hf_patchtst_prod").rglob("*_model.pt"))
        scorer = scorer_mod.HFPatchTSTPanelScorer.load(artifact)
        assert scorer.requires_history is True
        assert scorer.seq_len > 0
        assert len(scorer.feature_cols) > 0

    @pytest.mark.skipif(
        not list((REPO / "artifacts/hf_patchtst_prod").rglob("*_model.pt")),
        reason="no HF PatchTST model artifact yet")
    def test_score_with_history_runs(self, scorer_mod):
        artifact = next((REPO / "artifacts/hf_patchtst_prod").rglob("*_model.pt"))
        scorer = scorer_mod.HFPatchTSTPanelScorer.load(artifact)
        # Synthetic panel: 3 tickers × seq_len+5 dates × feature_cols
        rng = np.random.default_rng(0)
        n_dates = scorer.seq_len + 5
        dates = pd.date_range("2024-01-01", periods=n_dates)
        rows = []
        for ticker in ("AAA", "BBB", "CCC"):
            for d in dates:
                row = {"ticker": ticker, "date": d}
                for c in scorer.feature_cols:
                    row[c] = float(rng.normal(0, 1))
                rows.append(row)
        panel = pd.DataFrame(rows)
        scores = scorer.score_with_history(panel, ["AAA", "BBB", "CCC"])
        assert len(scores) == 3
        assert scores.notna().all()
        assert scores.name == "panel_score"
