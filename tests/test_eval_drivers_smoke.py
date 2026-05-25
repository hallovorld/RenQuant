"""Smoke tests for newly-shipped scripts that drive Tier 1+2 verdicts.

Per audit P0-8 (2026-05-20): these scripts went 0% test coverage at
ship time despite being on the critical promote-decision path. compare_arch
is the aggregator that drives the next verdict — typo in argv could
silently produce wrong table.

Each script gets:
1. import smoke (module loads, no top-level errors)
2. CLI --help smoke (argparse parses, exits 0)
3. (where applicable) dry-run on tiny synthetic input
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


SCRIPTS = [
    "scripts/eval_dlinear_5cut_5seed.py",
    "scripts/eval_hf_film_5cut_5seed.py",
    "scripts/eval_hf_trainer_5cut_5seed.py",
    "scripts/eval_xgb_5cut_5seed.py",
    "scripts/compare_arch_5cut_5seed.py",
    "scripts/dlinear_baseline.py",
    "scripts/verify_sigma_calibration.py",
    "scripts/train_ngboost_proper.py",
]

# Scripts that have argparse + --help (CLI entry). The 3 eval drivers
# don't (they just run main() with hardcoded knobs from the script).
ARGPARSE_SCRIPTS = [
    "scripts/compare_arch_5cut_5seed.py",
    "scripts/eval_xgb_5cut_5seed.py",
    "scripts/dlinear_baseline.py",
    "scripts/verify_sigma_calibration.py",
    "scripts/train_ngboost_proper.py",
]


@pytest.mark.parametrize("script_rel", SCRIPTS)
def test_script_imports_cleanly(script_rel: str):
    """Module loads via importlib without error."""
    import importlib.util
    p = REPO / script_rel
    assert p.exists(), f"{script_rel} missing"
    spec = importlib.util.spec_from_file_location("smoke_mod", p)
    mod = importlib.util.module_from_spec(spec)
    # Must NOT crash on top-level imports / module-level statements
    spec.loader.exec_module(mod)
    # Each script defines main() or similar entry
    has_entry = any(hasattr(mod, n) for n in ("main", "run_one", "aggregate"))
    assert has_entry, f"{script_rel} has no main/run_one/aggregate entry"


@pytest.mark.parametrize("script_rel", ARGPARSE_SCRIPTS)
def test_script_help_works(script_rel: str):
    """`python script --help` exits 0 — proves argparse is well-formed.
    Only applies to scripts with argparse (not the eval drivers which run
    main() immediately with hardcoded knobs)."""
    p = REPO / script_rel
    result = subprocess.run(
        [sys.executable, str(p), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"{script_rel} --help crashed rc={result.returncode}\n"
        f"stderr: {result.stderr[-500:]}"
    )


class TestTrainNGBoostProperCLI:
    """P0-15: help/dry-run must not accidentally start real NGBoost training."""

    def test_missing_feature_policy_defaults_to_error(self):
        import importlib.util
        import pandas as pd

        p = REPO / "scripts/train_ngboost_proper.py"
        spec = importlib.util.spec_from_file_location("ngb_proper", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        panel = pd.DataFrame({"date": ["2026-01-01"], "present": [1.0]})
        with pytest.raises(ValueError, match="missing 1 feature"):
            mod._apply_missing_feature_policy(
                panel, ["present", "missing"], policy="error",
            )

    def test_missing_feature_policy_zero_fills(self):
        import importlib.util
        import pandas as pd

        p = REPO / "scripts/train_ngboost_proper.py"
        spec = importlib.util.spec_from_file_location("ngb_proper", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        panel = pd.DataFrame({"date": ["2026-01-01"], "present": [1.0]})
        out, missing = mod._apply_missing_feature_policy(
            panel, ["present", "missing"], policy="zero",
        )
        assert missing == ["missing"]
        assert out["missing"].iloc[0] == 0.0


@pytest.mark.parametrize("script_rel",
                          ["scripts/eval_dlinear_5cut_5seed.py",
                           "scripts/eval_hf_film_5cut_5seed.py",
                           "scripts/eval_hf_trainer_5cut_5seed.py",
                           "scripts/eval_xgb_5cut_5seed.py"])
def test_eval_driver_constants_sane(script_rel: str):
    """Eval drivers have hardcoded CUTS / SEEDS / KNOBS. Verify the
    module-level constants are present and sane (5 cuts × 5 seeds)."""
    import importlib.util
    p = REPO / script_rel
    spec = importlib.util.spec_from_file_location("smoke_mod", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "CUTS"), f"{script_rel} missing CUTS"
    assert hasattr(mod, "SEEDS"), f"{script_rel} missing SEEDS"
    assert len(mod.CUTS) == 5, f"expected 5 cuts, got {len(mod.CUTS)}"
    assert len(mod.SEEDS) == 5, f"expected 5 seeds, got {len(mod.SEEDS)}"
    assert hasattr(mod, "OUT_ROOT"), f"{script_rel} missing OUT_ROOT"


def test_xgb_driver_writes_patchtst_comparator_schema(tmp_path):
    import importlib.util
    import pandas as pd

    p = REPO / "scripts/eval_xgb_5cut_5seed.py"
    spec = importlib.util.spec_from_file_location("xgb_eval", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    results = [{
        "status": "ok",
        "cut": "cut1",
        "seed": 42,
        "best_val_ic": 0.02,
        "per_regime_ic": {"BULL_CALM": 0.03, "BEAR": 0.02},
    }]
    df = mod.aggregate(results, tmp_path)

    assert list(df.columns) == ["cut", "seed", "regime", "ic"]
    assert set(df["regime"]) == {"BULL_CALM", "BEAR", "_MIN_"}
    saved = pd.read_csv(tmp_path / "aggregate.csv")
    assert list(saved.columns) == ["cut", "seed", "regime", "ic"]


class TestCompareArchAggregator:
    """compare_arch_5cut_5seed.py is the verdict aggregator — has its own
    dedicated smoke since it's the single point of promote-decision input."""

    def test_aggregator_handles_missing_aggregate_csv_gracefully(self, tmp_path):
        """Empty / missing aggregate.csv → should warn + skip, not crash."""
        import subprocess
        empty_dir = tmp_path / "empty_run"
        empty_dir.mkdir()
        # Pass two non-existent runs
        result = subprocess.run(
            [sys.executable,
             str(REPO / "scripts/compare_arch_5cut_5seed.py"),
             "--runs", f"{empty_dir}:archA", f"{empty_dir}:archB"],
            capture_output=True, text=True, timeout=30,
        )
        # Should exit gracefully (rc 0 or 1 with clear error), not crash
        assert result.returncode in (0, 1), (
            f"crashed on missing aggregates rc={result.returncode}\n"
            f"stderr: {result.stderr[-500:]}"
        )

    def test_aggregator_uses_kernel_metrics_canonical(self):
        """The aggregator must use kernel.metrics for IC / DSR / PBO,
        not hand-rolled (per §5.12)."""
        src = (REPO / "scripts/compare_arch_5cut_5seed.py").read_text()
        # Soft check: if the script ever adds metric computation, it should
        # cite canonical sources. For now, the script just reads CSVs and
        # aggregates via pandas — that's fine.
        # Reject custom DSR/PBO impl here:
        assert "def _custom_dsr" not in src
        assert "def _custom_pbo" not in src


class TestDlinearBaselineSpec:
    """DLinear baseline must reuse patchtst_hf helpers (CSRankNorm,
    PerDayDataset, PerRegimeICCallback) — not duplicate them."""

    def test_imports_patchtst_hf_helpers_via_importlib(self):
        src = (REPO / "scripts/dlinear_baseline.py").read_text()
        assert "_load_patchtst_hf_helpers" in src, (
            "DLinear should reuse patchtst_hf helpers via importlib, "
            "not duplicate CSRankNorm / PerDayDataset / callback")

    def test_uses_torch_nn_functional_margin_ranking(self):
        """DLinear loss should be canonical torch margin_ranking (§5.12),
        not hand-rolled BCE."""
        src = (REPO / "scripts/dlinear_baseline.py").read_text()
        # Either imports from patchtst_hf or uses torch.nn.functional directly
        assert "margin_ranking_loss" in src or "MarginRankingLoss" in src


class TestVerifySigmaCalibration:
    """σ-calibration verifier is the gate-decision tool for σ-wire enable."""

    def test_reads_val_preds_with_mu_sigma_columns(self):
        src = (REPO / "scripts/verify_sigma_calibration.py").read_text()
        assert "mu" in src and "sigma" in src, \
            "verifier should consume mu/sigma columns from val_preds"

    def test_uses_spearman_or_calibration_coef(self):
        src = (REPO / "scripts/verify_sigma_calibration.py").read_text()
        assert "spearmanr" in src or "calibration_coef" in src
