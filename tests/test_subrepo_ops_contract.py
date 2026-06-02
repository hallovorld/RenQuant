"""Regression guards for the production multirepo ops contract."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "subrepo_ops_contract.py"


def _load_contract_module():
    spec = importlib.util.spec_from_file_location("subrepo_ops_contract_for_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_subrepo_ops_contract_passes_current_entrypoints() -> None:
    mod = _load_contract_module()
    result = mod.run_contract()

    assert result["ok"], result["failures"]
    assert not result["failures"]
    assert "daily_live_defaults_to_multirepo" in result["passed"]
    assert "alpha158_linear_retrain_defaults_to_orchestrator" in result["passed"]
    assert "conditional_trigger_uses_orchestrator" in result["passed"]
    assert "weekly_fundamental_refresh_uses_base_data_earnings" in result["passed"]
    assert "weekly_fundamental_refresh_uses_base_data_sec" in result["passed"]
    assert "event_sec_schema_change_uses_base_data" in result["passed"]
    assert "daily_iv_snapshot_uses_base_data" in result["passed"]
    assert "daily_news_fetch_uses_base_data" in result["passed"]
    assert "daily_news_sentiment_uses_model_repo" in result["passed"]
    assert "screen_watchlist_uses_base_data" in result["passed"]
    assert "model_smoke_test_uses_backtesting_repo" in result["passed"]
    assert "lean_watchlist_export_uses_backtesting_repo" in result["passed"]
    assert "portfolio_metrics_uses_backtesting_repo" in result["passed"]
    assert "forward_returns_backfill_uses_backtesting_repo" in result["passed"]
    assert "dashboard_builder_uses_backtesting_repo" in result["passed"]
    assert "wf_fingerprint_stamping_uses_backtesting_repo" in result["passed"]
    assert "strategy_config_drift_uses_strategy_repo" in result["passed"]
    assert "monthly_meta_label_uses_model_repo" in result["passed"]
    assert "monthly_meta_label_snapshot_uses_backtesting_repo" in result["passed"]
    assert "monthly_calibrator_refresh_uses_model_repo" in result["passed"]
    assert "legacy_retrain_panel_delegates_to_weekly_wf" in result["passed"]
    assert "launchd_plists_parseable" in result["passed"]
    assert "launchd_program_arguments_multirepo" in result["passed"]
    assert "launchd_uses_project_venv" in result["passed"]
    assert "state_backup_uses_orchestrator" in result["passed"]
    assert "weekly_apy_uses_orchestrator" in result["passed"]
    assert "subrepo_pin_ci_green_gate_available" in result["passed"]
    assert "subrepo_lock_refresh_prewrite_ci_gate_available" in result["passed"]
    assert "subrepo_pin_ci_green_workflow_wired" in result["passed"]
    assert result["known_gaps"] == []


def test_subrepo_ops_contract_cli_json() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    assert '"ok": true' in proc.stdout
    assert '"known_gaps": []' in proc.stdout
