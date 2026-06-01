"""Regression guards for the alpha158 retrain multirepo wrapper."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_umbrella_retrain_pipeline_remains_present() -> None:
    src = (
        REPO
        / "backtesting"
        / "renquant_104"
        / "training_panel"
        / "daily_retrain_alpha158_fund.py"
    ).read_text()
    assert "class DailyRetrainAlpha158FundPipeline" in src
    assert "class RefitCalibratorTask" in src


def test_retrain_wrapper_defaults_to_orchestrator_with_rollback() -> None:
    src = (REPO / "scripts" / "daily_retrain_alpha158_fund.sh").read_text()
    assert 'RQ_RETRAIN_RUNNER:-multirepo' in src
    assert "RQ_RETRAIN_RUNNER=umbrella" in src
    assert "renquant_orchestrator.retrain_alpha158_fund" in src
    assert "renquant-orchestrator/src" in src
    assert "renquant-model/src" in src
    assert "renquant-pipeline/src" in src
    assert "renquant-execution/src" in src
    assert "training_panel.daily_retrain_alpha158_fund" in src
    assert "RQ_RETRAIN_STRICT" in src
    assert "renquant_orchestrator.retrain_alpha158_fund={m.__file__}" in src
    assert "RETRAIN_FALLBACK" in src
    assert "Priority: low" in src
    assert ".subrepo_fallback_alert_stamp" in src
    assert 'grep -q -- "--staged"' in src
    assert "TODO(renquant-orchestrator#2)" in src


def test_weekly_still_calls_wrapper_with_explicit_staging_paths() -> None:
    weekly = (REPO / "scripts" / "weekly_wf_promote.sh").read_text()
    assert "bash scripts/daily_retrain_alpha158_fund.sh" in weekly
    assert '--xgb-artifact-out "$STAGING_ART"' in weekly
    assert '--calibrator-out "$STAGING_CAL"' in weekly
