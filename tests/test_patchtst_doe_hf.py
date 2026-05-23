"""Regression tests for scripts/patchtst_doe_hf.py."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts/patchtst_doe_hf.py"
POSTPROCESS = REPO / "scripts/postprocess_doe_hf.py"
sys.path.insert(0, str(REPO))


class TestSourceContracts:
    def test_script_exists(self):
        assert SCRIPT.exists()

    def test_uses_pydoe2(self):
        src = SCRIPT.read_text()
        assert "import pyDOE2" in src
        assert 'fracfact("a b c abc")' in src

    def test_uses_hf_wrapper(self):
        src = SCRIPT.read_text()
        assert "scripts/patchtst_hf.py" in src
        assert "subprocess.run" in src

    def test_uses_walk_forward_cuts(self):
        src = SCRIPT.read_text()
        assert "from kernel.walk_forward_splits import build_default_cuts" in src

    def test_uses_hmm_regime_objective(self):
        src = SCRIPT.read_text()
        assert "from kernel.hmm_regime_labels import" in src
        assert "bull_regime_ic" in src

    def test_predict_averaging_ensemble(self):
        """Lakshminarayanan 2017 ensemble — predict-average across seeds."""
        src = SCRIPT.read_text()
        assert "np.mean" in src and "ensembled" in src

    def test_knob_ranges_tightened(self):
        """Per prior DOE main effects: lr top 1e-4, seq top 24, wd ≥ 1e-2."""
        src = SCRIPT.read_text()
        # Tightened lr range — must NOT include 1e-3 (prior catastrophic)
        assert '"lr",            1e-5, 1e-4' in src
        assert '"seq_len",       8,    24' in src
        # wd: high regularization region
        assert '"weight_decay",  1e-2, 3e-1' in src
        # warmup: long
        assert '"warmup_epochs", 4,    10' in src


class TestDesignMatrix:
    def test_9_design_points(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("patchtst_doe_hf", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        d = mod.build_design_matrix()
        assert len(d) == 9
        assert d["is_center"].sum() == 1

    def test_coded_to_real_log_lr(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("patchtst_doe_hf", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # lr: log range 1e-5 → 1e-4
        assert abs(mod.coded_to_real("lr", -1) - 1e-5) < 1e-9
        assert abs(mod.coded_to_real("lr", +1) - 1e-4) < 1e-9
        # center geometric mean: 10^-4.5 = 3.16e-5
        assert abs(mod.coded_to_real("lr", 0) - 10 ** -4.5) < 1e-9

    def test_coded_to_real_linear_seq(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("patchtst_doe_hf", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.coded_to_real("seq_len", -1) == 8
        assert mod.coded_to_real("seq_len", +1) == 24
        assert mod.coded_to_real("seq_len", 0) == 16  # arithmetic mean


class TestPostprocessContracts:
    def test_counts_evaluated_points_not_planned_points(self, tmp_path):
        design = pd.DataFrame({
            "point_id": [0, 1, 2, 3, 4],
            "lr": [1e-5, 1e-4, 1e-5, 1e-4, 3.2e-5],
            "weight_decay": [0.01, 0.01, 0.3, 0.3, 0.055],
            "warmup_epochs": [4, 4, 10, 10, 7],
            "seq_len": [8, 24, 8, 24, 16],
            "lr_coded": [-1, 1, -1, 1, 0],
            "weight_decay_coded": [-1, -1, 1, 1, 0],
            "warmup_epochs_coded": [-1, -1, 1, 1, 0],
            "seq_len_coded": [-1, 1, -1, 1, 0],
            "is_center": [False, False, False, False, True],
        })
        design.to_csv(tmp_path / "design.csv", index=False)
        runs = pd.DataFrame([
            {"point_id": pid, "cut": cut, "bull_regime_ic": ic}
            for pid, base in [(0, 0.01), (1, 0.03), (2, 0.02), (3, 0.04)]
            for cut, ic in [("c1", base), ("c2", base + 0.01)]
        ])
        runs.to_csv(tmp_path / "runs.csv", index=False)

        subprocess.run(
            [sys.executable, str(POSTPROCESS), "--doe-dir", str(tmp_path)],
            cwd=REPO,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        pbo = pd.read_csv(tmp_path / "pbo_summary.csv").iloc[0]
        assert int(pbo["n_design_points"]) == 4
        assert int(pbo["n_planned_design_points"]) == 5
        assert int(pbo["n_evaluated_design_points"]) == 4
        assert int(pbo["n_missing_design_points"]) == 1
        assert str(int(float(pbo["missing_point_ids"]))) == "4"
        summary = (tmp_path / "summary_full.md").read_text()
        assert "Evaluated design points**: 4 / 5" in summary
        assert "Missing point ids**: 4" in summary
