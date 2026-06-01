"""Regression tests for scripts/train_walkforward_patchtst.py."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "train_walkforward_patchtst.py"
sys.path.insert(0, str(REPO))


def _load_mod():
    spec = importlib.util.spec_from_file_location("train_walkforward_patchtst", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(**overrides):
    base = dict(
        artifact_root="walkforward_patchtst_test",
        dataset="data/transformer_v4_wl200_clean.parquet",
        raw_label_panel="data/alpha158_291_fundamental_dataset_rawlabel.parquet",
        label="fwd_60d_excess",
        seed=44,
        epochs=5,
        seq_len=32,
        patch_length=4,
        d_model=64,
        n_heads=4,
        n_layers=2,
        lr=3e-4,
        weight_decay=1e-3,
        device="cpu",
        strategy_config=None,
        film_regime_cond=False,
        cross_stock_attn=False,
        jobs=1,
        skip_calibrators=False,
        calibrator_batch_size=512,
        calibrator_method="platt",
        calibrator_min_rows=1000,
        allow_partial_manifest=False,
        reuse_existing=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_data_end_for_cutoff_uses_label_lookahead():
    mod = _load_mod()
    assert mod.data_end_for_cutoff(
        pd.Timestamp("2024-04-01"), "fwd_5d_excess"
    ) == "2024-03-25"


def test_train_cmd_uses_patchtst_point_in_time_contract(tmp_path):
    mod = _load_mod()
    args = _args(epochs=2, seq_len=8, d_model=16)
    cmd = mod.train_cmd(args, pd.Timestamp("2024-04-01"), tmp_path)
    joined = " ".join(cmd)
    assert "-m renquant_model_patchtst.hf_trainer" in joined
    assert "scripts/patchtst_hf.py" not in joined
    assert "--cut all" in joined
    assert "--train-cutoff 2024-04-01" in joined
    assert "--save-model" in cmd
    assert "--output-dir" in cmd


def test_calibrator_cmd_is_causal_to_cutoff(tmp_path):
    mod = _load_mod()
    args = _args(label="fwd_20d_excess", calibrator_method="isotonic")
    model_path = tmp_path / "hf_patchtst_all_seed44_model.pt"
    cal_path = tmp_path / "cal.json"
    cmd = mod.calibrator_cmd(
        args, pd.Timestamp("2024-04-01"), model_path, cal_path,
    )
    joined = " ".join(cmd)
    assert "-m renquant_model_patchtst.fit_calibrator" in joined
    assert "scripts/fit_hf_patchtst_calibrator.py" not in joined
    assert "--scorer-artifact" in cmd
    assert "--panel data/transformer_v4_wl200_clean.parquet" in joined
    assert "--raw-label-panel data/alpha158_291_fundamental_dataset_rawlabel.parquet" in joined
    assert "--data-end 2024-03-04" in joined
    assert "--batch-size 512" in joined
    assert "--method isotonic" in joined
    assert "--min-rows 1000" in joined


def test_build_entry_reads_sidecar_contract(tmp_path):
    mod = _load_mod()
    model_path = tmp_path / "hf_patchtst_all_seed44_model.pt"
    model_path.write_bytes(b"fake")
    sidecar = model_path.with_name(model_path.name + ".metadata.json")
    sidecar.write_text(json.dumps({
        "training_contract": {
            "trained_date": "2026-05-24",
            "effective_train_cutoff_date": "2024-01-15",
        },
    }))
    cal_path = tmp_path / "hf_patchtst-calibration.json"

    entry = mod.build_entry(
        pd.Timestamp("2024-04-01"), model_path, cal_path, "fwd_60d_excess",
    )

    assert entry.artifact_uri == str(model_path)
    assert entry.calibrator_uri == str(cal_path)
    assert entry.lookahead_days == 60
    assert entry.effective_train_cutoff_date == pd.Timestamp("2024-01-15")


def test_reuse_existing_skips_completed_subprocesses(tmp_path, monkeypatch):
    mod = _load_mod()
    monkeypatch.setattr(mod, "STRATEGY_DIR", tmp_path)
    args = _args(artifact_root="wf_pt", reuse_existing=True)
    cutoff = pd.Timestamp("2025-01-02")
    out_dir = mod.artifact_dir(args, cutoff)
    out_dir.mkdir(parents=True)
    model_path = mod.model_path_for(out_dir, args.seed)
    model_path.write_bytes(b"fake")
    model_path.with_name(model_path.name + ".metadata.json").write_text(json.dumps({
        "training_contract": {
            "trained_date": "2026-05-24",
            "effective_train_cutoff_date": "2024-10-09",
        },
    }))
    mod.calibrator_path_for(model_path).write_text("{}")

    def fail_run_subprocess(cmd, label):
        raise AssertionError(f"should reuse existing artifact, got {label}: {cmd}")

    monkeypatch.setattr(mod, "run_subprocess", fail_run_subprocess)

    _, entry, err = mod.train_one_cutoff(args, cutoff)

    assert err == ""
    assert entry.artifact_uri == str(model_path)
