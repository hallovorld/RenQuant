"""Regression guards for the alpha158 retrain multirepo wrapper."""
from __future__ import annotations

import json
import re
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
    assert "scripts/subrepo_env.sh" in src
    assert 'renquant_load_subrepo_env "$REPO_DIR"' in src
    assert 'export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"' in src
    assert 'renquant_subrepo_pythonpath "$SUBREPO_ROOT"' in src
    assert "renquant-orchestrator" in src
    assert "renquant-model" in src
    assert "renquant-pipeline" in src
    assert "renquant-execution" in src
    assert "training_panel.daily_retrain_alpha158_fund" in src
    assert "RQ_RETRAIN_STRICT" in src
    assert "renquant_orchestrator.retrain_alpha158_fund={m.__file__}" in src
    assert "RETRAIN_FALLBACK" in src
    assert "Priority: low" in src
    assert ".subrepo_fallback_alert_stamp" in src
    assert 'grep -q -- "--staged"' not in src
    assert "--staged is umbrella-only" not in src


def test_weekly_still_calls_wrapper_with_explicit_staging_paths() -> None:
    weekly = (REPO / "scripts" / "weekly_wf_promote.sh").read_text()
    assert "scripts/subrepo_env.sh" in weekly
    assert 'renquant_load_subrepo_env "$REPO_DIR"' in weekly
    assert 'export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"' in weekly
    assert "bash scripts/daily_retrain_alpha158_fund.sh" in weekly
    assert "renquant_backtesting.wf_gate" in weekly
    assert 'RQ_WF_GATE_RUNNER:-multirepo' in weekly
    assert "RQ_WF_GATE_STRICT" in weekly
    assert "scripts/run_wf_gate.py" in weekly
    assert "renquant_backtesting.forensics.model_acceptance" in weekly
    assert '--xgb-artifact-out "$STAGING_ART"' in weekly
    assert '--calibrator-out "$STAGING_CAL"' in weekly


def test_weekly_wf_manifest_and_base_config_exist_and_match() -> None:
    weekly = (REPO / "scripts" / "weekly_wf_promote.sh").read_text()
    manifest_match = re.search(r'^WF_MANIFEST="([^"]+)"', weekly, flags=re.MULTILINE)
    strategy_match = re.search(r"--strategy-config\s+([^\s\\]+)", weekly)
    assert manifest_match, "weekly_wf_promote.sh must define WF_MANIFEST"
    assert strategy_match, "weekly_wf_promote.sh must pass --strategy-config"

    strategy_dir = REPO / "backtesting" / "renquant_104"
    manifest_rel = manifest_match.group(1)
    config_name = strategy_match.group(1)
    manifest_path = strategy_dir / manifest_rel
    config_path = strategy_dir / config_name

    assert manifest_path.exists(), f"weekly WF manifest is missing: {manifest_rel}"
    assert config_path.exists(), f"weekly WF base config is missing: {config_name}"

    config = json.loads(config_path.read_text())
    assert config["walkforward"]["manifest_path"] == manifest_rel

    manifest = json.loads(manifest_path.read_text())
    assert manifest.get("retrains"), f"weekly WF manifest has no retrains: {manifest_rel}"
