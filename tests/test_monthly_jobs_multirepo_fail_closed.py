"""Regression guards for monthly maintenance multirepo fail-closed wrappers."""
from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def _non_comment_source(script: str) -> str:
    path = REPO / "scripts" / script
    return "\n".join(
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


def test_monthly_calibrator_fails_closed_without_model_repo() -> None:
    src = _non_comment_source("monthly_calibrator_refresh.sh")

    assert "renquant_model_gbdt.fit_calibrator_alpha158_fund" in src
    assert "monthly calibrator fails closed" in src
    assert "renquant_strict_enabled RQ_MONTHLY_CALIBRATOR_STRICT" in src
    assert "scripts/fit_panel_calibrator.py" not in src
    assert "falling back to umbrella fit_panel_calibrator.py" not in src


def test_monthly_meta_label_fails_closed_without_model_or_backtesting_repo() -> None:
    src = _non_comment_source("monthly_meta_label_retrain.sh")

    assert "renquant_backtesting.wf_gate.sim_driver" in src
    assert "renquant_model_common.meta_label_exit" in src
    assert "monthly job fails closed" in src
    assert "renquant_strict_enabled RQ_META_LABEL_STRICT" in src
    assert "renquant_strict_enabled RQ_META_LABEL_SIM_STRICT" in src
    assert "snapshot sim failed" in src
    assert "scripts/run_sim_104.py" not in src
    assert "scripts/_meta_label_generate.py" not in src
    assert "scripts/_meta_label_train.py" not in src
    assert "falling back to umbrella" not in src
