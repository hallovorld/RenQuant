from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from build_kelly_sigma_horizon_ab_config import (  # noqa: E402
    SIGMA_HORIZON_PATH,
    build_kelly_sigma_horizon_ab_config,
    changed_dotted_paths,
)
from validate_sim_config_active import static_validate  # noqa: E402


def _baseline_config() -> dict:
    return {
        "ranking": {
            "kelly_sizing": {
                "enabled": True,
                "use_realized_vol_fallback": True,
                "realized_vol_window_days": 60,
            },
            "panel_scoring": {"enabled": True},
        },
        "rotation": {"joint_actions": {"enabled": True}},
    }


def test_builder_sets_only_sigma_horizon_days() -> None:
    baseline = _baseline_config()

    treatment = build_kelly_sigma_horizon_ab_config(baseline)

    assert treatment["ranking"]["kelly_sizing"]["sigma_horizon_days"] == 60
    assert SIGMA_HORIZON_PATH not in changed_dotted_paths(baseline, baseline)
    assert changed_dotted_paths(baseline, treatment) == [SIGMA_HORIZON_PATH]
    assert "sigma_horizon_days" not in baseline["ranking"]["kelly_sizing"]


def test_builder_keeps_existing_fields_and_changes_only_value() -> None:
    baseline = _baseline_config()
    baseline["ranking"]["kelly_sizing"]["sigma_horizon_days"] = 252

    treatment = build_kelly_sigma_horizon_ab_config(
        baseline,
        sigma_horizon_days=60,
    )

    assert changed_dotted_paths(baseline, treatment) == [SIGMA_HORIZON_PATH]
    assert treatment["ranking"]["kelly_sizing"]["sigma_horizon_days"] == 60


def test_sigma_horizon_path_is_static_validator_active() -> None:
    baseline = _baseline_config()
    treatment = build_kelly_sigma_horizon_ab_config(baseline)

    ok, report = static_validate(baseline, treatment)

    assert ok is True
    assert any(f"{SIGMA_HORIZON_PATH}:" in row and "ACTIVE" in row for row in report)


def test_cli_writes_derived_config_and_reports_single_diff(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    out_path = tmp_path / "treatment.json"
    baseline_path.write_text(json.dumps(_baseline_config(), indent=2))

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "build_kelly_sigma_horizon_ab_config.py"),
            "--base-config",
            str(baseline_path),
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=REPO,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    treatment = json.loads(out_path.read_text())
    assert payload["changed_paths"] == [SIGMA_HORIZON_PATH]
    assert treatment["ranking"]["kelly_sizing"]["sigma_horizon_days"] == 60
