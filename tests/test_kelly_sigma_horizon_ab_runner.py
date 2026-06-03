from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _load_module():
    path = REPO / "scripts" / "run_kelly_sigma_horizon_ab.py"
    spec = importlib.util.spec_from_file_location("run_kelly_sigma_horizon_ab", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_seeds_accepts_count_and_list() -> None:
    mod = _load_module()

    assert mod.parse_seeds("5") == (0, 1, 2, 3, 4)
    assert mod.parse_seeds("3,7,11") == (3, 7, 11)


def test_build_variants_pairs_real_ab_and_aa_resplit(tmp_path: Path) -> None:
    mod = _load_module()
    base = tmp_path / "base.json"
    treatment = tmp_path / "treatment.json"

    variants = mod.build_variants(
        base_config_path=base,
        treatment_config_path=treatment,
        seeds=(0, 1, 2, 3, 4),
        aa_seed_offset=100,
    )

    assert [variant.name for variant in variants] == [
        "A_golden",
        "B_sigma_horizon_60",
        "AA_golden_resplit",
    ]
    assert variants[0].seeds == variants[1].seeds == (0, 1, 2, 3, 4)
    assert variants[2].config_path == base
    assert variants[2].seeds == (100, 101, 102, 103, 104)


def test_build_plan_keeps_treatment_config_in_output_dir(tmp_path: Path) -> None:
    mod = _load_module()
    args = Namespace(
        base_config=str(tmp_path / "base.json"),
        treatment_config="",
        sigma_horizon_days=60,
        start="2024-01-02",
        end="2026-03-28",
        initial_cash=100_000.0,
        seeds="0,1,2,3,4",
        aa_seed_offset=1000,
        output_dir=str(tmp_path / "out"),
        placebo_json=[],
        execute=False,
    )

    plan = mod.build_plan(args)

    assert plan["mode"] == "dry_run"
    assert plan["treatment_config_path"].startswith(str(tmp_path / "out"))
    assert plan["mandatory_checks"]["multi_seed_floor"] == 5
    assert plan["mandatory_checks"]["real_ab"] == ["A_golden", "B_sigma_horizon_60"]


def test_promotion_verdict_blocks_without_placebo() -> None:
    mod = _load_module()
    metrics = {
        "A_golden": {"apy_mean": 0.10, "sharpe_mean": 1.0},
        "B_sigma_horizon_60": {
            "apy_mean": 0.12,
            "sharpe_mean": 1.2,
            "dsr": 0.7,
            "pbo": 0.4,
        },
        "AA_golden_resplit": {"apy_mean": 0.101, "sharpe_mean": 1.01},
    }

    verdict = mod.promotion_verdict(metrics, {"provided": False, "passed": False})

    assert verdict["tier3_ready"] is False
    assert "shuffle/time-shift placebo evidence missing" in verdict["blocked_reasons"]


def test_promotion_verdict_passes_synthetic_tier3() -> None:
    mod = _load_module()
    metrics = {
        "A_golden": {"apy_mean": 0.10, "sharpe_mean": 1.0},
        "B_sigma_horizon_60": {
            "apy_mean": 0.12,
            "sharpe_mean": 1.2,
            "dsr": 0.7,
            "pbo": 0.4,
        },
        "AA_golden_resplit": {"apy_mean": 0.101, "sharpe_mean": 1.01},
    }
    placebo = {"provided": True, "passed": True, "items": []}

    verdict = mod.promotion_verdict(metrics, placebo)

    assert verdict["tier3_ready"] is True
    assert verdict["blocked_reasons"] == []
    assert verdict["deltas"]["sharpe_lift"] == 0.19999999999999996


def test_load_placebo_evidence_reads_manifest_diagnostic(tmp_path: Path) -> None:
    mod = _load_module()
    path = tmp_path / "placebo.json"
    path.write_text(json.dumps({
        "interpretation": {
            "promotion_evidence": True,
            "aligned_real_60_ic": 0.05,
            "placebo_60_ic": 0.01,
            "label_autocorr_60_ic": 0.02,
        }
    }))

    evidence = mod.load_placebo_evidence([str(path)])

    assert evidence["provided"] is True
    assert evidence["passed"] is True
    assert evidence["items"][0]["placebo_60_ic"] == 0.01
