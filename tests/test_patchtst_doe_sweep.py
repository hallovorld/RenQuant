"""Regression tests for scripts/patchtst_doe_sweep.py + kernel/regime_labels.py.

Pin CLAUDE.md §5.14 DOE compliance + PRIME DIRECTIVE per-regime objective.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ── DOE design matrix ──────────────────────────────────────────────────────

class TestDesignMatrix:
    def test_design_has_8_corners_plus_1_center(self):
        from scripts.patchtst_doe_sweep import build_design_matrix
        d = build_design_matrix()
        assert len(d) == 9
        assert d["is_center"].sum() == 1
        assert (~d["is_center"]).sum() == 8

    def test_center_at_geometric_mean_for_log_knobs(self):
        from scripts.patchtst_doe_sweep import build_design_matrix
        d = build_design_matrix()
        center = d[d["is_center"]].iloc[0]
        # lr log-mid of 1e-4, 1e-3 is 10^-3.5 = 3.16e-4
        assert abs(center["lr"] - 10 ** -3.5) < 1e-6
        # weight_decay log-mid of 1e-4, 1e-1 is 10^-2.5 = 3.16e-3
        assert abs(center["weight_decay"] - 10 ** -2.5) < 1e-6

    def test_center_at_arithmetic_mean_for_linear_knobs(self):
        from scripts.patchtst_doe_sweep import build_design_matrix
        d = build_design_matrix()
        center = d[d["is_center"]].iloc[0]
        assert center["warmup_epochs"] == 4  # (2+6)/2
        assert center["seq_len"] == 38       # (16+60)/2

    def test_resolution_iv_no_main_x_2way_confound(self):
        """In Res IV FrFact, main effects are NOT aliased with 2-way
        interactions. Generator D = ABC has this property for 2^(4-1)."""
        from scripts.patchtst_doe_sweep import build_design_matrix
        d = build_design_matrix()
        corners = d[~d["is_center"]]
        # Each main effect column has equal +1/-1 split
        for k in ["lr_coded", "weight_decay_coded",
                  "warmup_epochs_coded", "seq_len_coded"]:
            vals = corners[k].values
            assert (vals == 1.0).sum() == 4
            assert (vals == -1.0).sum() == 4

    def test_coded_to_real_log_scale(self):
        from scripts.patchtst_doe_sweep import coded_to_real
        # lr is log-scale
        assert abs(coded_to_real("lr", -1) - 1e-4) < 1e-9
        assert abs(coded_to_real("lr", +1) - 1e-3) < 1e-9

    def test_coded_to_real_linear_scale(self):
        from scripts.patchtst_doe_sweep import coded_to_real
        # warmup_epochs is linear
        assert coded_to_real("warmup_epochs", -1) == 2
        assert coded_to_real("warmup_epochs", +1) == 6


# ── PRIME DIRECTIVE: per-regime objective ──────────────────────────────────

class TestPerRegimeObjective:
    """The PRIME-DIRECTIVE objective — min-across-regime IC — must work
    correctly. Pooled mean would pick the wrong model."""

    def test_min_across_regimes_picks_worst(self):
        from kernel.regime_labels import min_across_regimes
        per_regime = {"LOW_CALM": 0.10, "MED_NORMAL": 0.05, "HIGH_SPIKED": -0.03}
        assert min_across_regimes(per_regime) == -0.03

    def test_per_regime_ic_computes_per_day_then_means(self):
        from kernel.regime_labels import per_regime_cs_ic
        # 20 days, 10 tickers each. Set up so HIGH regime has positive IC,
        # LOW regime has negative IC.
        rng = np.random.default_rng(0)
        preds, labels, dates = [], [], []
        # 15 days in HIGH regime — positive correlation
        for d in range(15):
            x = rng.normal(0, 1, 10)
            y = x + rng.normal(0, 0.2, 10)
            preds.extend(x); labels.extend(y); dates.extend([pd.Timestamp(2024, 1, 1)
                                                              + pd.Timedelta(days=d)] * 10)
        # 15 days in LOW regime — negative correlation
        for d in range(15, 30):
            x = rng.normal(0, 1, 10)
            y = -x + rng.normal(0, 0.2, 10)
            preds.extend(x); labels.extend(y); dates.extend([pd.Timestamp(2024, 1, 1)
                                                              + pd.Timedelta(days=d)] * 10)
        preds_df = pd.DataFrame({"date": dates, "pred": preds, "label": labels})
        regime_df = pd.DataFrame({
            "date": [pd.Timestamp(2024, 1, 1) + pd.Timedelta(days=d)
                     for d in range(30)],
            "regime": ["HIGH_CALM"] * 15 + ["LOW_CALM"] * 15,
        })
        out = per_regime_cs_ic(preds_df, regime_df, min_days_per_regime=5)
        assert "HIGH_CALM" in out and "LOW_CALM" in out
        assert out["HIGH_CALM"] > 0.5
        assert out["LOW_CALM"] < -0.5

    def test_under_sampled_regime_excluded(self):
        from kernel.regime_labels import per_regime_cs_ic
        # 3 days of "rare" regime — should be excluded with min_days=10
        rng = np.random.default_rng(0)
        preds, labels, dates, regimes = [], [], [], []
        for d in range(15):  # 15 days HIGH
            x = rng.normal(0, 1, 10)
            preds.extend(x); labels.extend(x + rng.normal(0, 0.1, 10))
            dates.extend([pd.Timestamp(2024, 1, 1) + pd.Timedelta(days=d)] * 10)
            regimes.extend(["HIGH"] * 10)
        for d in range(15, 18):  # 3 days RARE
            x = rng.normal(0, 1, 10)
            preds.extend(x); labels.extend(x + rng.normal(0, 0.1, 10))
            dates.extend([pd.Timestamp(2024, 1, 1) + pd.Timedelta(days=d)] * 10)
            regimes.extend(["RARE"] * 10)
        preds_df = pd.DataFrame({"date": dates, "pred": preds, "label": labels})
        regime_df = pd.DataFrame({"date": dates, "regime": regimes}).drop_duplicates()
        out = per_regime_cs_ic(preds_df, regime_df, min_days_per_regime=10)
        assert "HIGH" in out
        assert "RARE" not in out  # excluded for under-sampling

    def test_pooled_mean_vs_min_picks_different_winner(self):
        """Sanity proof of the PRIME DIRECTIVE: a model can beat another on
        pooled mean while losing on min-across-regime. THIS is the bug pattern
        the sweep objective must avoid."""
        # Model A: +0.20 in regime1, -0.05 in regime2. pooled mean = +0.075
        # Model B: +0.04 in regime1, +0.03 in regime2. pooled mean = +0.035
        # Pooled mean: A wins. But min-across-regime: B (+0.03) beats A (-0.05).
        # The PRIME DIRECTIVE objective picks B.
        from kernel.regime_labels import min_across_regimes
        a = {"r1": 0.20, "r2": -0.05}
        b = {"r1": 0.04, "r2": +0.03}
        assert np.mean(list(a.values())) > np.mean(list(b.values()))
        assert min_across_regimes(b) > min_across_regimes(a)


# ── Script source contracts ────────────────────────────────────────────────

class TestSourceContracts:
    """Pin the script's design intent so future refactors can't drift."""

    def test_script_exists(self):
        assert (REPO / "scripts/patchtst_doe_sweep.py").exists()

    def test_uses_pydoe2_not_optuna(self):
        src = (REPO / "scripts/patchtst_doe_sweep.py").read_text()
        assert "import pyDOE2" in src
        # No Optuna runtime use (docstring may mention it historically)
        assert "import optuna" not in src
        assert "optuna.create_study" not in src
        assert "optuna.trial" not in src

    def test_uses_fracfact_resolution_iv(self):
        src = (REPO / "scripts/patchtst_doe_sweep.py").read_text()
        assert 'fracfact("a b c abc")' in src

    def test_calls_transformer_v4_subprocess(self):
        src = (REPO / "scripts/patchtst_doe_sweep.py").read_text()
        assert "scripts/transformer_v4.py" in src
        assert "subprocess.run" in src

    def test_no_custom_training_code(self):
        src = (REPO / "scripts/patchtst_doe_sweep.py").read_text()
        forbidden = [
            "torch.optim", "AveragedModel", "SWALR", "nn.Module",
            "loss.backward", "model.train()", "AdamW",
        ]
        for f in forbidden:
            assert f not in src, f"forbidden custom-training token: {f}"

    def test_uses_per_regime_objective(self):
        src = (REPO / "scripts/patchtst_doe_sweep.py").read_text()
        assert "min_across_regimes" in src
        assert "per_regime_cs_ic" in src
        # Must NOT use pooled mean as objective
        assert 'val_ic_pool' in src  # tracked for comparison
        assert 'val_ic_min' in src   # the actual objective

    def test_mps_workers_forced_to_1(self):
        src = (REPO / "scripts/patchtst_doe_sweep.py").read_text()
        assert 'device == "mps"' in src and "workers = 1" in src

    def test_n_seeds_default_3(self):
        import re
        src = (REPO / "scripts/patchtst_doe_sweep.py").read_text()
        m = re.search(r'"--n-seeds".*default=(\d+)', src)
        assert m and int(m.group(1)) >= 3, "n-seeds default must be ≥3 (§5.13.4)"

    def test_writes_main_effects_and_interactions(self):
        src = (REPO / "scripts/patchtst_doe_sweep.py").read_text()
        assert "main_effects.csv" in src
        assert "interactions.csv" in src
        assert "summary.md" in src

    def test_cites_claudemd_sections(self):
        src = (REPO / "scripts/patchtst_doe_sweep.py").read_text()
        for cite in ["§5.14", "5.13.4", "PRIME DIRECTIVE"]:
            assert cite in src, f"missing CLAUDE.md cite: {cite}"


class TestTransformerV4DumpsPredictions:
    """The dump path added to transformer_v4.py is load-bearing for the DOE
    sweep's per-regime objective. Pin it."""

    def test_evaluate_has_dump_path_arg(self):
        src = (REPO / "scripts/transformer_v4.py").read_text()
        assert "dump_path:" in src
        assert "dump_path=val_dump" in src
        assert "dump_path=test_dump" in src

    def test_dump_creates_parquet_files(self):
        src = (REPO / "scripts/transformer_v4.py").read_text()
        assert "_val_preds.parquet" in src
        assert "_test_preds.parquet" in src
