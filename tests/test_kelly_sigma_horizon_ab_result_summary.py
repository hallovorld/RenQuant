from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def _load_module():
    path = REPO / "scripts" / "summarize_kelly_sigma_horizon_ab_results.py"
    spec = importlib.util.spec_from_file_location(
        "summarize_kelly_sigma_horizon_ab_results",
        path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summarize_rows_reports_per_variant_and_regime_mean_std() -> None:
    mod = _load_module()
    rows = [
        {
            "variant": "A_golden",
            "control_type": "real_control",
            "seed": 0,
            "regime": "BULL_CALM",
            "apy": 0.10,
            "sharpe": 0.50,
            "maxdd": -0.08,
            "cash_pct": 0.52,
            "kelly_target_pct": 0.03,
            "dsr": 0.40,
            "pbo": 0.60,
        },
        {
            "variant": "A_golden",
            "control_type": "real_control",
            "seed": 1,
            "regime": "BULL_CALM",
            "apy": 0.14,
            "sharpe": 0.70,
            "maxdd": -0.10,
            "cash_pct": 0.48,
            "kelly_target_pct": 0.05,
            "dsr": 0.42,
            "pbo": 0.58,
        },
        {
            "variant": "B_sigma_horizon_60",
            "control_type": "real_treatment",
            "seed": 0,
            "regime": "BULL_CALM",
            "apy": 0.20,
            "sharpe": 0.90,
            "maxdd": -0.12,
            "cash_pct": 0.19,
            "kelly_target_pct": 0.12,
            "dsr": 0.61,
            "pbo": 0.41,
        },
        {
            "variant": "B_sigma_horizon_60",
            "control_type": "real_treatment",
            "seed": 1,
            "regime": "CHOPPY",
            "apy": 0.06,
            "sharpe": 0.20,
            "maxdd": -0.06,
            "cash_pct": 0.33,
            "kelly_target_pct": 0.07,
            "dsr": 0.63,
            "pbo": 0.39,
        },
    ]

    summary = mod.summarize_rows(rows)

    a = summary["by_variant"]["A_golden"]
    assert a["n_rows"] == 2
    assert a["n_seeds"] == 2
    assert a["control_types"] == ["real_control"]
    assert a["metrics"]["apy"]["mean"] == pytest.approx(0.12)
    assert a["metrics"]["apy"]["std"] == pytest.approx(math.sqrt(0.0008))
    assert a["metrics"]["dsr"]["mean"] == pytest.approx(0.41)
    assert a["metrics"]["dsr"]["status"] == "available"

    bull = summary["by_variant_regime"]["A_golden"]["BULL_CALM"]
    assert bull["metrics"]["cash_pct"]["mean_pm_std"].startswith("0.5 +/- ")
    assert summary["by_regime"]["BULL_CALM"]["n_rows"] == 3

    treatment_choppy = summary["by_variant_regime"]["B_sigma_horizon_60"]["CHOPPY"]
    assert treatment_choppy["n_rows"] == 1
    assert treatment_choppy["metrics"]["sharpe"]["mean"] == 0.20
    assert treatment_choppy["metrics"]["sharpe"]["std"] is None

    comparison = summary["comparisons"]
    assert comparison["control_variant"] == "A_golden"
    assert comparison["treatment_variant"] == "B_sigma_horizon_60"
    assert comparison["by_variant"]["status"] == "available"
    assert comparison["by_variant"]["metrics"]["apy"]["delta"] == pytest.approx(0.01)
    bull_delta = comparison["by_regime"]["BULL_CALM"]["metrics"]
    assert bull_delta["cash_pct"]["delta"] == pytest.approx(-0.31)
    assert bull_delta["kelly_target_pct"]["delta"] == pytest.approx(0.08)
    assert comparison["by_regime"]["CHOPPY"]["status"] == "missing_control"


def test_missing_optional_kelly_and_dsr_pbo_columns_emit_placeholders() -> None:
    mod = _load_module()
    summary = mod.summarize_rows([
        {
            "variant": "A_golden",
            "seed": 0,
            "regime": "CHOPPY",
            "apy": 0.01,
            "sharpe": 0.10,
            "max_dd": -0.04,
            "cash_pct": 0.50,
        }
    ])

    metrics = summary["by_variant"]["A_golden"]["metrics"]
    assert metrics["kelly_target_pct"] == {
        "n": 0,
        "mean": None,
        "std": None,
        "mean_pm_std": "null +/- null",
        "status": "not_provided",
    }
    assert metrics["dsr"]["status"] == "not_provided"
    assert metrics["pbo"]["status"] == "not_provided"
    assert metrics["maxdd"]["mean"] == -0.04
    assert summary["comparisons"]["by_variant"]["status"] == "missing_treatment"


def test_load_rows_accepts_csv_json_and_jsonl(tmp_path: Path) -> None:
    mod = _load_module()
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text(
        "variant,seed,regime,apy,sharpe,maxdd,cash_pct,dsr,pbo\n"
        "A,0,BULL_CALM,10%,0.4,-5%,50%,0.4,0.6\n"
    )
    json_path = tmp_path / "rows.json"
    json_path.write_text(json.dumps({
        "results": [
            {
                "variant": "B",
                "seed": 0,
                "regime": "BULL_CALM",
                "apy": 0.2,
                "sharpe": 0.8,
                "maxdd": -0.07,
                "cash_pct": 0.2,
            }
        ]
    }))
    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text(json.dumps({
        "variant": "B",
        "seed": 1,
        "regime": "CHOPPY",
        "apy": 0.03,
        "sharpe": 0.2,
        "maxdd": -0.03,
        "cash_pct": 0.35,
    }) + "\n")

    rows = mod.load_rows([str(csv_path), str(json_path), str(jsonl_path)])
    summary = mod.summarize_rows(rows)

    assert summary["n_rows"] == 3
    assert summary["by_variant"]["A"]["metrics"]["apy"]["mean"] == 0.10
    assert summary["by_variant"]["A"]["metrics"]["maxdd"]["mean"] == -0.05
    assert sorted(summary["by_variant"]) == ["A", "B"]


def test_cli_prints_json_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(
        json.dumps({
            "variant": "B_sigma_horizon_60",
            "control_type": "real_treatment",
            "seed": 0,
            "regime": "BULL_VOLATILE",
            "apy": 0.11,
            "sharpe": 0.44,
            "maxdd": -0.09,
            "cash_pct": 0.21,
            "kelly_target_pct": 0.08,
            "dsr": 0.55,
            "pbo": 0.45,
        }) + "\n"
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "summarize_kelly_sigma_horizon_ab_results.py"),
            str(input_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    group = payload["by_variant"]["B_sigma_horizon_60"]
    assert group["metrics"]["kelly_target_pct"]["mean"] == 0.08
    assert group["metrics"]["dsr"]["status"] == "available"
    assert payload["schema"]["dsr_pbo"] == "passed-through only; not computed by this script"
    assert payload["schema"]["comparisons"] == "treatment mean minus control mean"
    assert payload["comparisons"]["by_variant"]["status"] == "missing_control"
