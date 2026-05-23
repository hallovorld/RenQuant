"""Regression tests for HFPatchTSTPanelScorer.

Pins the interface (mirror of PatchTSTPanelScorer) + integration with
model_registry so shadow + future primary swap can rely on it.
"""
from __future__ import annotations
import sys
import importlib.util
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

    def test_calibrator_script_replays_sequences_and_stamps_fingerprint(self):
        """PatchTST calibrator must be model-specific, not copied from XGB."""
        script = REPO / "scripts" / "fit_hf_patchtst_calibrator.py"
        src = script.read_text()
        assert "_csrank_norm_per_day" in src
        assert "score_sequences" in src
        assert "scorer_artifact_fingerprint" in src
        assert "raw_return_units_required" in src

    def test_calibrator_script_infers_raw_er_label(self):
        script = REPO / "scripts" / "fit_hf_patchtst_calibrator.py"
        spec = importlib.util.spec_from_file_location("fit_hf_patchtst_calibrator", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod._infer_raw_er_label("fwd_60d_excess") == "fwd_60d_excess_raw"
        assert mod._infer_raw_er_label("fwd_60d_excess_raw") == "fwd_60d_excess_raw"

    def test_training_checkpoint_stamps_provenance_for_leakage_guard(self):
        src = (REPO / "scripts" / "patchtst_hf.py").read_text()
        for required in (
            "trained_date",
            "effective_train_cutoff_date",
            "lookahead_days",
            "split_date_ranges",
            "config_fingerprint",
            "config_fingerprint_fields",
            "trained_watchlist_n",
        ):
            assert required in src

    def test_loader_exposes_training_contract_metadata(self, scorer_mod):
        src = (REPO / "backtesting/renquant_104/kernel/panel_pipeline"
               / "hf_patchtst_scorer.py").read_text()
        for required in (
            "training_contract",
            "effective_train_cutoff_date",
            "effective_selection_cutoff_date",
            "split_date_ranges",
            "lookahead_days",
            "_load_contract_sidecar",
            "contract_sidecar_path",
        ):
            assert required in src


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
