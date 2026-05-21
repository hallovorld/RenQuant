"""Regression tests for QP/Kelly static contract gates."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from qp_contracts import validate_qp_contract_config  # noqa: E402


def _base_qp_config() -> dict:
    return {
        "rotation": {
            "joint_actions": {
                "enabled": True,
                "solver": "qp",
            },
        },
        "ranking": {
            "panel_scoring": {
                "global_calibration": {"enabled": True},
                "ngboost": {"enabled": False},
            },
            "kelly_sizing": {"enabled": True},
        },
    }


def test_stale_qp_config_without_mu_sigma_contract_fails() -> None:
    cfg = _base_qp_config()
    report = validate_qp_contract_config(cfg)
    assert report.qp_enabled is True
    assert report.passed is False
    assert "qp_mu_contract must be strict" in report.summary()
    assert "no legal QP" in report.summary()
    assert "Kelly enabled without NGBoost" in report.summary()


def test_strict_calibrator_mu_with_realized_vol_fallback_passes() -> None:
    cfg = _base_qp_config()
    cfg["rotation"]["joint_actions"]["qp_mu_contract"] = "strict"
    cfg["ranking"]["kelly_sizing"].update(
        use_calibrator_mu=True,
        use_realized_vol_fallback=True,
    )
    report = validate_qp_contract_config(cfg)
    assert report.passed is True
    assert report.evidence["calibrator_mu_enabled"] is True
    assert report.evidence["realized_vol_fallback_enabled"] is True


def test_renquant104_qp_configs_have_strict_mu_sigma_contract() -> None:
    config_names = [
        "strategy_config.json",
        "strategy_config.golden.json",
        "strategy_config.sim_wl200.json",
    ]
    for name in config_names:
        cfg = json.loads((REPO / "backtesting/renquant_104" / name).read_text())
        report = validate_qp_contract_config(cfg)
        assert report.passed is True, f"{name}: {report.summary()}"
        assert report.evidence["qp_mu_contract"] == "strict"
