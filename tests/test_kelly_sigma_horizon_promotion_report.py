from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _load_module():
    path = REPO / "scripts" / "render_kelly_sigma_horizon_promotion_report.py"
    spec = importlib.util.spec_from_file_location(
        "render_kelly_sigma_horizon_promotion_report",
        path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _plan_payload() -> dict:
    return {
        "plan": {
            "variants": [
                {"name": "A_golden", "role": "real_control", "seeds": [0, 1]},
                {"name": "B_sigma_horizon_60", "role": "real_treatment", "seeds": [0, 1]},
                {"name": "AA_golden_resplit", "role": "aa_resplit", "seeds": [100, 101]},
            ],
            "mandatory_checks": {
                "per_regime": ["BULL_CALM", "BULL_VOLATILE", "CHOPPY"],
            },
            "placebo_requirements": {"required": True},
        },
        "promotion_verdict": {
            "tier3_ready": False,
            "blocked_reasons": ["placebo evidence did not pass"],
            "thresholds": {
                "required_regimes": ["BULL_CALM", "BULL_VOLATILE", "CHOPPY"],
            },
        },
    }


def _summary_payload() -> dict:
    metric = lambda mean, status="available": {  # noqa: E731
        "mean": mean,
        "std": None,
        "mean_pm_std": f"{mean} +/- null",
        "status": status,
    }
    delta = lambda control, treatment: {  # noqa: E731
        "control_mean": control,
        "treatment_mean": treatment,
        "delta": treatment - control,
        "control_status": "available",
        "treatment_status": "available",
    }
    return {
        "by_variant": {
            "B_sigma_horizon_60": {
                "metrics": {
                    "dsr": metric(0.61),
                    "pbo": metric(0.41),
                }
            }
        },
        "by_variant_regime": {
            "A_golden": {
                "BULL_CALM": {
                    "n_rows": 2,
                    "n_seeds": 2,
                    "metrics": {
                        "apy": metric(0.1),
                        "sharpe": metric(0.5),
                        "maxdd": metric(-0.08),
                        "cash_pct": metric(0.5),
                        "kelly_target_pct": metric(0.03),
                    },
                },
                "CHOPPY": {
                    "n_rows": 1,
                    "n_seeds": 1,
                    "metrics": {"sharpe": metric(0.1)},
                },
            },
            "B_sigma_horizon_60": {
                "BULL_CALM": {
                    "n_rows": 2,
                    "n_seeds": 2,
                    "metrics": {
                        "apy": metric(0.2),
                        "sharpe": metric(0.9),
                        "maxdd": metric(-0.06),
                        "cash_pct": metric(0.3),
                        "kelly_target_pct": metric(0.08),
                    },
                },
                "BULL_VOLATILE": {
                    "n_rows": 1,
                    "n_seeds": 1,
                    "metrics": {"sharpe": metric(0.4)},
                },
            },
        },
        "comparisons": {
            "control_variant": "A_golden",
            "treatment_variant": "B_sigma_horizon_60",
            "by_variant": {
                "status": "available",
                "metrics": {
                    "apy": delta(0.1, 0.2),
                    "sharpe": delta(0.5, 0.9),
                    "maxdd": delta(-0.08, -0.06),
                    "cash_pct": delta(0.5, 0.3),
                    "kelly_target_pct": delta(0.03, 0.08),
                },
            },
            "by_regime": {
                "BULL_CALM": {
                    "status": "available",
                    "metrics": {
                        "apy": delta(0.1, 0.2),
                        "sharpe": delta(0.5, 0.9),
                        "maxdd": delta(-0.08, -0.06),
                        "cash_pct": delta(0.5, 0.3),
                        "kelly_target_pct": delta(0.03, 0.08),
                    },
                },
                "BULL_VOLATILE": {
                    "status": "missing_control",
                    "metrics": {
                        "sharpe": {
                            "control_mean": None,
                            "treatment_mean": 0.4,
                            "delta": None,
                            "control_status": "missing",
                            "treatment_status": "available",
                        },
                    },
                },
            },
        },
    }


def test_render_report_includes_promotion_checklist_phrases(tmp_path: Path) -> None:
    mod = _load_module()
    placebo = tmp_path / "placebo.json"
    placebo.write_text(json.dumps({
        "interpretation": {
            "promotion_evidence": True,
            "aligned_real_60_ic": 0.08,
            "placebo_60_ic": 0.01,
            "label_autocorr_60_ic": 0.02,
        }
    }))

    report = mod.render_report(
        plan_payload=_plan_payload(),
        summary=_summary_payload(),
        placebo_paths=[str(placebo)],
    )

    assert "A/B variants" in report
    assert "A_golden" in report
    assert "B_sigma_horizon_60" in report
    assert "Seed count: 2" in report
    assert "Required regimes: BULL_CALM, BULL_VOLATILE, CHOPPY" in report
    assert "Tier 3 ready: no" in report
    assert "placebo evidence did not pass" in report
    assert "DSR/PBO status" in report
    assert "Placebo evidence status: passed" in report
    assert "A/B comparison deltas" in report
    assert "Comparison: A_golden -> B_sigma_horizon_60" in report
    assert "BULL_CALM: status=available" in report
    assert "cash_pct_delta=-0.2 (0.5 -> 0.3)" in report
    assert "BULL_CALM metrics" in report
    assert "BULL_VOLATILE metrics" in report
    assert "CHOPPY metrics" in report


def test_cli_writes_output_markdown(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    summary = tmp_path / "summary.json"
    output = tmp_path / "report.md"
    plan.write_text(json.dumps(_plan_payload()))
    summary.write_text(json.dumps(_summary_payload()))

    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "render_kelly_sigma_horizon_promotion_report.py"),
            "--plan-json",
            str(plan),
            "--summary-json",
            str(summary),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = output.read_text()
    assert report.startswith("# Kelly Sigma-Horizon Promotion Evidence")
    assert "Placebo evidence status: required, not provided" in report
