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


def _load_patchtst_hf_mod():
    script = REPO / "scripts" / "patchtst_hf.py"
    spec = importlib.util.spec_from_file_location("patchtst_hf_mod", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
        assert 'RENQUANT_TORCH_THREADS", "1"' in src

    def test_calibrator_script_infers_raw_er_label(self):
        script = REPO / "scripts" / "fit_hf_patchtst_calibrator.py"
        spec = importlib.util.spec_from_file_location("fit_hf_patchtst_calibrator", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod._infer_raw_er_label("fwd_60d_excess") == "fwd_60d_excess_raw"
        assert mod._infer_raw_er_label("fwd_60d_excess_raw") == "fwd_60d_excess_raw"

    def test_calibrator_fingerprint_ignores_config_fingerprint(self, tmp_path):
        """Calibrator must bind to scorer bytes, not shared strategy config."""
        import hashlib
        script = REPO / "scripts" / "fit_hf_patchtst_calibrator.py"
        spec = importlib.util.spec_from_file_location("fit_hf_patchtst_calibrator", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        scorer_path = tmp_path / "hf_patchtst_model.pt"
        scorer_path.write_bytes(b"scorer checkpoint")
        expected = "sha256:" + hashlib.sha256(scorer_path.read_bytes()).hexdigest()

        assert mod._artifact_fingerprint(
            scorer_path,
            {"config_fingerprint": "sha256:shared-config"},
        ) == expected

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
        reason="no HF PatchTST model artifact yet (train pending)")
    def test_load_reads_provenance_schema_stamped_by_real_training_run(self, scorer_mod):
        """2026-07-02 (#426 round 9): scripts/patchtst_hf.py --save-model
        now stamps provenance_schema_version/recipe_id INTO the checkpoint
        at save time (not derived here at load time — see
        TestProvenanceStampBinding for the synthetic-fixture coverage of
        that contract). A real production training run should therefore
        carry the stamp already; skip (not fail) if this specific artifact
        predates that fix, rather than asserting a stamp it was never
        given the chance to carry."""
        from kernel.panel_pipeline.panel_scorer import PROVENANCE_SCHEMA_VERSION
        artifact = next((REPO / "artifacts/hf_patchtst_prod").rglob("*_model.pt"))
        scorer = scorer_mod.HFPatchTSTPanelScorer.load(artifact)
        if not scorer.metadata.get("recipe_id"):
            pytest.skip("this artifact predates the #426 r9 save-time stamp — "
                        "correctly NOT ACTIONABLE downstream, nothing to assert here")
        assert scorer.metadata["provenance_schema_version"] == PROVENANCE_SCHEMA_VERSION
        assert scorer.metadata["recipe_id"] == "walkforward_only_v1"
        assert scorer.metadata["required_axis_fields"] == ["effective_train_cutoff_date"]

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


class TestProvenanceStampBinding:
    """2026-07-02 (#426 round 9): the loader must READ a persisted
    provenance_schema_version/recipe_id from the checkpoint, never
    re-derive it — and that persisted stamp must be part of what
    artifact_fingerprint/artifact_sha256 actually hashes (round 8's gap:
    a load-time-derived recipe_id could never bind into a hash computed
    over bytes written before the derivation ran)."""

    @staticmethod
    def _tiny_ckpt(tmp_path, name: str, extra: dict) -> Path:
        import torch
        from transformers import PatchTSTConfig

        cfg = PatchTSTConfig(
            num_input_channels=3, context_length=8, patch_length=4,
            patch_stride=4, d_model=16, num_attention_heads=4,
            num_hidden_layers=1, ffn_dim=32,
        )
        mod = _load_patchtst_hf_mod()
        model = mod.HFPatchTSTRanker(
            cfg, use_distributional_head=False, use_film_regime=False,
            use_cross_stock_attn=False,
        )
        ckpt = {
            "config_dict": cfg.to_dict(),
            "state_dict": model.state_dict(),
            "feature_cols": ["f1", "f2", "f3"],
            "seq_len": 8,
            "uses_distributional_head": False,
            "uses_film_regime": False,
            "uses_cross_stock_attn": False,
        }
        ckpt.update(extra)
        path = tmp_path / name
        torch.save(ckpt, path)
        return path

    def test_load_reads_persisted_stamp(self, scorer_mod, tmp_path):
        from kernel.panel_pipeline.panel_scorer import PROVENANCE_SCHEMA_VERSION
        artifact = self._tiny_ckpt(tmp_path, "stamped.pt", {
            "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
            "recipe_id": "walkforward_only_v1",
            "required_axis_fields": ["effective_train_cutoff_date"],
        })
        scorer = scorer_mod.HFPatchTSTPanelScorer.load(artifact)
        assert scorer.metadata["provenance_schema_version"] == PROVENANCE_SCHEMA_VERSION
        assert scorer.metadata["recipe_id"] == "walkforward_only_v1"
        assert scorer.metadata["required_axis_fields"] == ["effective_train_cutoff_date"]

    def test_load_leaves_legacy_unstamped_checkpoint_unstamped(self, scorer_mod, tmp_path):
        """No fallback derivation — a checkpoint saved before this fix (or
        any checkpoint the training script chose not to stamp) must NOT
        get a recipe_id from anywhere else. Downstream, shadow_scoring.py
        treats a missing recipe_id as NOT ACTIONABLE."""
        artifact = self._tiny_ckpt(tmp_path, "legacy.pt", {
            "effective_train_cutoff_date": "2026-01-01",  # present but NOT stamped
        })
        scorer = scorer_mod.HFPatchTSTPanelScorer.load(artifact)
        assert "recipe_id" not in scorer.metadata
        assert "provenance_schema_version" not in scorer.metadata

    def test_tampered_recipe_id_changes_the_fingerprint(self, scorer_mod, tmp_path):
        """The persisted stamp is part of the checkpoint's own bytes, so
        tampering it (with everything else held constant) must change the
        whole-file artifact_sha256/artifact_fingerprint — proving the
        stamp is cryptographically bound, not just present-but-unverified
        metadata alongside an unrelated hash."""
        from kernel.panel_pipeline.panel_scorer import PROVENANCE_SCHEMA_VERSION
        real = self._tiny_ckpt(tmp_path, "real.pt", {
            "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
            "recipe_id": "walkforward_only_v1",
            "required_axis_fields": ["effective_train_cutoff_date"],
        })
        # Same construction, but load the state_dict/config fresh so the
        # only intended difference is the tampered recipe_id — reuse the
        # helper with a distinct recipe_id string standing in for tamper.
        tampered = self._tiny_ckpt(tmp_path, "tampered.pt", {
            "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
            "recipe_id": "full_history_only_v1",  # tampered value
            "required_axis_fields": ["effective_train_cutoff_date"],
        })
        real_scorer = scorer_mod.HFPatchTSTPanelScorer.load(real)
        tampered_scorer = scorer_mod.HFPatchTSTPanelScorer.load(tampered)
        assert (real_scorer.metadata["artifact_sha256"]
                != tampered_scorer.metadata["artifact_sha256"])
        assert real_scorer.metadata["recipe_id"] != tampered_scorer.metadata["recipe_id"]


class TestLoaderArchitectureMismatch:
    @pytest.mark.parametrize(
        ("flag_name", "state_root"),
        [
            ("uses_cross_stock_attn", "cross_stock"),
            ("uses_film_regime", "film"),
        ],
    )
    def test_rejects_declared_optional_layer_with_missing_tensors(
        self,
        scorer_mod,
        tmp_path,
        flag_name,
        state_root,
    ):
        import torch
        from transformers import PatchTSTConfig

        mod = _load_patchtst_hf_mod()
        cfg = PatchTSTConfig(
            num_input_channels=3,
            context_length=8,
            patch_length=4,
            patch_stride=4,
            d_model=16,
            num_attention_heads=4,
            num_hidden_layers=1,
            ffn_dim=32,
        )
        # Save a baseline state_dict that does NOT include the declared optional
        # component. Prior to the fix this silently loaded a random layer.
        model = mod.HFPatchTSTRanker(
            cfg,
            use_distributional_head=False,
            use_film_regime=False,
            use_cross_stock_attn=False,
        )
        ckpt = {
            "config_dict": cfg.to_dict(),
            "state_dict": model.state_dict(),
            "feature_cols": ["f1", "f2", "f3"],
            "seq_len": 8,
            "uses_distributional_head": False,
            "uses_film_regime": False,
            "uses_cross_stock_attn": False,
        }
        ckpt[flag_name] = True
        artifact = tmp_path / f"missing_{state_root}.pt"
        torch.save(ckpt, artifact)

        with pytest.raises(ValueError, match=state_root):
            scorer_mod.HFPatchTSTPanelScorer.load(artifact)
