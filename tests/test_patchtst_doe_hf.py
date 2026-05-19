"""Regression tests for scripts/patchtst_doe_hf.py."""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts/patchtst_doe_hf.py"
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
