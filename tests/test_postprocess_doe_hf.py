"""Regression tests for scripts/postprocess_doe_hf.py — §5.14.4 + §5.14.6."""
from __future__ import annotations
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts/postprocess_doe_hf.py"
sys.path.insert(0, str(REPO))


def _load_mod():
    spec = importlib.util.spec_from_file_location("pp", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestSourceContracts:
    def test_script_exists(self):
        assert SCRIPT.exists()

    def test_cites_bailey_papers(self):
        src = SCRIPT.read_text()
        assert "Bailey" in src and "2014" in src  # DSR
        assert "2015" in src  # PBO/CSCV

    def test_outputs_required_files(self):
        src = SCRIPT.read_text()
        for f in ("dsr.csv", "pbo_summary.csv", "main_effects.csv",
                  "interactions.csv", "summary_full.md"):
            assert f in src


class TestDSR:
    """Bailey-LdP 2014: DSR should be > 0 for genuinely significant points,
    < 0 for noise."""

    def test_dsr_positive_for_significant_point(self):
        np.random.seed(42)
        mod = _load_mod()
        # Consistently strong IC: mean 0.06, low variance
        ic = np.array([0.05, 0.06, 0.07])
        dsr = mod.compute_dsr(ic, n_trials=9)
        assert dsr > 0, f"strong point should pass DSR > 0, got {dsr}"

    def test_dsr_negative_for_noise(self):
        np.random.seed(42)
        mod = _load_mod()
        # Random near-zero IC
        ic = np.array([0.001, -0.001, 0.002])
        dsr = mod.compute_dsr(ic, n_trials=9)
        assert dsr < 0, f"noise should fail DSR > 0, got {dsr}"

    def test_dsr_nan_for_single_sample(self):
        mod = _load_mod()
        ic = np.array([0.05])
        dsr = mod.compute_dsr(ic, n_trials=9)
        assert np.isnan(dsr)


class TestPBO:
    """Bailey-Borwein-LdP-Zhu 2015 CSCV: PBO low for consistent rankings,
    high for chaotic."""

    def test_pbo_low_for_consistent_winner(self):
        mod = _load_mod()
        # Same point wins in all cuts
        df = pd.DataFrame({
            "point_id": [0, 1, 2, 3] * 3,
            "cut": ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"],
            "bull_regime_ic": [
                0.10, 0.05, 0.02, -0.01,
                0.08, 0.06, 0.04, 0.00,
                0.09, 0.07, 0.05, 0.01,
            ],
        })
        pbo = mod.compute_pbo(df)
        assert pbo < 0.5, f"consistent winner should give PBO < 0.5, got {pbo}"

    def test_pbo_high_for_rank_chaos(self):
        mod = _load_mod()
        df = pd.DataFrame({
            "point_id": [0, 1, 2, 3] * 3,
            "cut": ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"],
            "bull_regime_ic": [
                0.10, 0.05, 0.02, -0.01,  # cut a: 0 best
                -0.05, -0.02, 0.05, 0.10,  # cut b: 3 best (inverted)
                0.01, 0.10, -0.05, 0.05,   # cut c: 1 best
            ],
        })
        pbo = mod.compute_pbo(df)
        assert pbo > 0.5, f"rank chaos should give PBO > 0.5, got {pbo}"

    def test_pbo_nan_for_single_cut(self):
        mod = _load_mod()
        df = pd.DataFrame({
            "point_id": [0, 1, 2, 3],
            "cut": ["a"] * 4,
            "bull_regime_ic": [0.10, 0.05, 0.02, -0.01],
        })
        pbo = mod.compute_pbo(df)
        assert np.isnan(pbo)


class TestEffectsFit:
    """§5.14.6 main effects + 2-way interactions tables."""

    def test_fit_recovers_known_main_effect(self):
        mod = _load_mod()
        # Synthetic: y depends linearly on lr_coded only
        coded = np.array([[-1, -1, -1, -1], [+1, -1, -1, -1],
                          [-1, +1, -1, -1], [+1, +1, -1, -1],
                          [-1, -1, +1, -1], [+1, -1, +1, -1],
                          [-1, +1, +1, -1], [+1, +1, +1, -1],
                          [0, 0, 0, 0]])
        points = pd.DataFrame({
            "point_id": range(9),
            "lr_coded": coded[:, 0],
            "weight_decay_coded": coded[:, 1],
            "warmup_epochs_coded": coded[:, 2],
            "seq_len_coded": coded[:, 3],
            # y = +0.1 * lr_coded + small noise
            "bull_ic_mean": 0.1 * coded[:, 0] + np.array(
                [0.001, 0.002, -0.001, 0.001, -0.002, 0.001, -0.001, 0.002, 0.0]),
        })
        main_df, inter_df = mod.fit_effects(points)
        assert not main_df.empty
        # lr should be the dominant main effect with β ≈ +0.1
        top = main_df.iloc[0]
        assert top["knob"] == "lr"
        assert top["beta"] > 0.05

    def test_empty_input_returns_empty(self):
        mod = _load_mod()
        empty = pd.DataFrame()
        main_df, inter_df = mod.fit_effects(empty)
        assert main_df.empty and inter_df.empty
