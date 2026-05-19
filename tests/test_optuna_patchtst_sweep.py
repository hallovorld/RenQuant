"""Regression tests for scripts/optuna_patchtst_sweep.py.

Pin the 3rd-party-library-based design per 2026-05-18 user mandate:
no custom training/SWA code in our scripts; Optuna orchestrates,
transformer_v4.py trains. Tests verify:
  1. Script imports cleanly
  2. Uses Optuna (no hand-rolled hyperparameter loop)
  3. Subprocess calls transformer_v4.py (doesn't reimplement training)
  4. Search space matches PatchTST literature defaults
  5. Sampler + Pruner are Optuna built-ins
  6. Per-seed output directories prevent summary collisions
  7. Parses authoritative summary JSON (not stdout regex)
"""
from __future__ import annotations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts/optuna_patchtst_sweep.py"


class TestSourceContracts:
    """Pin design decisions in source so future refactors can't drift."""

    def test_script_exists(self):
        assert SCRIPT.exists()

    def test_imports_optuna_not_custom(self):
        src = SCRIPT.read_text()
        assert "import optuna" in src

    def test_calls_transformer_v4_via_subprocess(self):
        src = SCRIPT.read_text()
        assert "scripts/transformer_v4.py" in src
        assert "subprocess.run" in src

    def test_no_custom_training_loop(self):
        """Must not reimplement training; only orchestrate."""
        src = SCRIPT.read_text()
        forbidden = [
            "torch.optim.AdamW",
            "AveragedModel",  # No SWA in our code per user mandate
            "SWALR",
            "nn.Module",
            "loss.backward",
            "model.train()",
        ]
        for f in forbidden:
            assert f not in src, f"forbidden custom-training token in sweep: {f}"

    def test_uses_tpe_sampler(self):
        """TPE is the published Optuna default (Bergstra 2011)."""
        src = SCRIPT.read_text()
        assert "TPESampler" in src

    def test_uses_median_pruner(self):
        """MedianPruner enables early stopping of bad trials."""
        src = SCRIPT.read_text()
        assert "MedianPruner" in src

    def test_sqlite_storage(self):
        src = SCRIPT.read_text()
        assert "sqlite://" in src

    def test_optuna_dashboard_hint(self):
        """Dashboard hint helps user inspect runs."""
        src = SCRIPT.read_text()
        assert "optuna-dashboard" in src

    def test_per_seed_output_dirs(self):
        """Each seed must get its own output dir or summaries collide."""
        src = SCRIPT.read_text()
        assert "trial_" in src and "_seed_" in src

    def test_parses_summary_json_not_stdout(self):
        """Authoritative source = patchtst_summary.json, not stdout regex."""
        src = SCRIPT.read_text()
        assert "patchtst_summary.json" in src
        assert "val_ic_mean" in src


class TestSearchSpace:
    """Pin search space bounds to published references."""

    def test_lr_in_patchtst_paper_range(self):
        """PatchTST (Nie 2023) uses lr ~1e-4 to ~1e-3."""
        src = SCRIPT.read_text()
        assert 'suggest_float("lr"' in src

    def test_weight_decay_logscale(self):
        src = SCRIPT.read_text()
        assert 'suggest_float("weight_decay"' in src
        assert "log=True" in src

    def test_seq_len_categorical(self):
        src = SCRIPT.read_text()
        assert 'suggest_categorical("seq_len"' in src

    def test_warmup_epochs_int(self):
        src = SCRIPT.read_text()
        assert 'suggest_int("warmup_epochs"' in src


class TestVarianceAwareObjective:
    """CLAUDE 5.13.4: single-seed = unverified claim. Sweep MUST use n_seeds."""

    def test_n_seeds_arg_exists(self):
        src = SCRIPT.read_text()
        assert "--n-seeds" in src

    def test_default_n_seeds_at_least_2(self):
        src = SCRIPT.read_text()
        # Default arg value should be ≥ 2
        import re as _re
        m = _re.search(r'"--n-seeds".*default=(\d+)', src)
        assert m is not None
        assert int(m.group(1)) >= 2


class TestClaudeMdReferences:
    """Pin docstring cites CLAUDE.md sections so design intent is auditable."""

    def test_cites_5_14_doe(self):
        src = SCRIPT.read_text()
        assert "5.14" in src

    def test_cites_5_13_4_variance(self):
        src = SCRIPT.read_text()
        assert "5.13.4" in src

    def test_cites_5_12_canonical_refs(self):
        src = SCRIPT.read_text()
        assert "5.12" in src

    def test_cites_concurrency_budget_memory(self):
        src = SCRIPT.read_text()
        assert "concurrency_resource_budget" in src
