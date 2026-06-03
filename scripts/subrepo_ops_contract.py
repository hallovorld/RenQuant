#!/usr/bin/env python3
"""Validate that scheduled RenQuant ops default to pinned multirepo paths.

This is a fast structural contract for launchd/shell entrypoints. It does not
run broker code or retrain models; it fails when an active production wrapper
drifts back to direct umbrella execution or an old Python environment.
"""
from __future__ import annotations

import argparse
import json
import plistlib
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONDA_PREFIX = "/Users/renhao/miniconda3"
# Personal-workstation contract: active launchd plists are installed for this
# operator account and must resolve the project venv at this absolute path.
CANONICAL_ROOT = "/Users/renhao/git/github/RenQuant"
VENV_BIN = "/Users/renhao/git/github/RenQuant/.venv/bin"


@dataclass(frozen=True)
class Check:
    name: str
    path: str
    required: tuple[str, ...]
    forbidden: tuple[str, ...] = ()


@dataclass(frozen=True)
class LaunchdProgram:
    path: str
    expected_args: tuple[str, ...]


def _script(name: str) -> str:
    return f"{CANONICAL_ROOT}/scripts/{name}"


CHECKS: tuple[Check, ...] = (
    Check(
        name="subrepo_env_supports_runtime_root",
        path="scripts/subrepo_env.sh",
        required=(
            "RENQUANT_SUBREPO_ROOT",
            "renquant_load_subrepo_env()",
            "renquant_subrepo_root()",
            "renquant_subrepo_src()",
            "renquant_strategy_config()",
            "renquant_subrepo_pythonpath()",
            "renquant_strict_enabled()",
            "RENQUANT_OPS_FAIL_CLOSED",
        ),
    ),
    Check(
        name="subrepo_paths_support_runtime_root",
        path="scripts/subrepo_paths.py",
        required=(
            "RENQUANT_SUBREPO_ROOT",
            "RENQUANT_ASSEMBLY_DIR",
            ".subrepo_assembly",
            "current.env",
            "current.json",
        ),
    ),
    Check(
        name="python_delegate_exports_strategy_config",
        path="scripts/subrepo_module_delegate.py",
        required=(
            "_pinned_strategy_config",
            "renquant-strategy-104",
            "strategy_config.json",
            "RENQUANT_SUBREPO_ROOT",
            "RENQUANT_STRATEGY_CONFIG",
            "RENQUANT_STRICT_SUBREPO_PATHS",
            "RENQUANT_OPS_FAIL_CLOSED",
            "GLOBAL_STRICT_ENV",
            "_strict_enabled",
            "os.environ.setdefault",
        ),
    ),
    Check(
        name="subrepo_daily_contract_uses_runtime_root",
        path="scripts/subrepo_daily_contract.py",
        required=(
            "from subrepo_paths import resolve_subrepo_root",
            "SUBREPO_ROOT = resolve_subrepo_root(ROOT)",
            'SUBREPO_ROOT / entry["name"] / "src"',
            '_subrepo_path("renquant-strategy-104")',
        ),
        forbidden=("Path(_entry(\"renquant-strategy-104\")[\"local_path\"])",),
    ),
    Check(
        name="daily_live_defaults_to_multirepo",
        path="scripts/daily_104.sh",
        required=(
            'RQ_DAILY_RUNNER:-multirepo',
            'source "$REPO_DIR/scripts/subrepo_env.sh"',
            'renquant_load_subrepo_env "$REPO_DIR"',
            'export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"',
            'renquant_strategy_config "$SUBREPO_ROOT" strategy_config.json',
            "PROD_STRATEGY_CONFIG",
            'RUNNER_ARGS=("$REPO_DIR/scripts/daily_multirepo.py")',
            'RUNNER_ARGS=(-m live.runner)',
            'runner = [sys.executable, "$REPO_DIR/scripts/live_multirepo.py"]',
        ),
        forbidden=(),
    ),
    Check(
        name="live_multirepo_uses_strategy_config_subrepo",
        path="scripts/live_multirepo.py",
        required=(
            "--strategy-config-path",
            "renquant-strategy-104",
            "configs",
            "_with_pinned_strategy_config",
        ),
    ),
    Check(
        name="daily_multirepo_uses_strategy_config_subrepo",
        path="scripts/daily_multirepo.py",
        required=(
            "--strategy-config-path",
            "renquant-strategy-104",
            "configs",
            "_with_pinned_strategy_config",
        ),
    ),
    Check(
        name="strategy_config_subrepo_sync_test_present",
        path="tests/test_live_multirepo_entrypoints.py",
        required=(
            "test_strategy_subrepo_configs_match_umbrella_rollback_copies",
            'git", "-C", str(repo), "show"',
            "strategy_config.shadow.json",
        ),
    ),
    Check(
        name="intraday_sell_defaults_to_multirepo",
        path="scripts/intraday_sell_104.sh",
        required=(
            'RQ_DAILY_RUNNER:-multirepo',
            'source "$REPO_DIR/scripts/subrepo_env.sh"',
            'renquant_load_subrepo_env "$REPO_DIR"',
            'export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"',
            'RUNNER_ARGS=("$REPO_DIR/scripts/live_multirepo.py")',
            "--sell-only --intraday",
        ),
    ),
    Check(
        name="preopen_gate_defaults_to_execution_subrepo",
        path="scripts/preopen_cancel_gate.sh",
        required=(
            'RQ_PREOPEN_GATE_RUNNER:-multirepo',
            'source "$PWD/scripts/subrepo_env.sh"',
            'renquant_load_subrepo_env "$PWD"',
            'export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"',
            'renquant_subrepo_root "$PWD" "$GITHUB_DIR"',
            'renquant_subrepo_src "$SUBREPO_ROOT" renquant-execution',
            'renquant_subrepo_src "$SUBREPO_ROOT" renquant-common',
            "python -m renquant_execution.preopen_cancel_gate",
            "RQ_PREOPEN_GATE_STRICT",
        ),
    ),
    Check(
        name="weekly_retrain_delegates_to_orchestrator_wrapper",
        path="scripts/weekly_wf_promote.sh",
        required=(
            'source "$REPO_DIR/scripts/subrepo_env.sh"',
            'renquant_load_subrepo_env "$REPO_DIR"',
            'export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"',
            'renquant_strategy_config "$SUBREPO_ROOT" strategy_config.json',
            'export RENQUANT_STRATEGY_CONFIG="$PROD_STRATEGY_CONFIG"',
            "bash scripts/daily_retrain_alpha158_fund.sh",
            "renquant_backtesting.wf_gate",
            'RQ_WF_GATE_RUNNER:-multirepo',
            "scripts/run_wf_gate.py",
            "renquant_backtesting.forensics.model_acceptance",
            '--fingerprint-config "$PROD_STRATEGY_CONFIG"',
            "--strict",
            "set RQ_WF_GATE_RUNNER=umbrella for explicit rollback",
        ),
        forbidden=("RQ_ALLOW_NO_WF=1", "falling back to umbrella run_wf_gate.py"),
    ),
    Check(
        name="legacy_retrain_panel_delegates_to_weekly_wf",
        path="scripts/retrain_panel.sh",
        required=(
            "Compatibility wrapper for the old Sunday retrain agent",
            "weekly_wf_promote already ran today",
            "delegating to the strict trust boundary",
            "bash scripts/weekly_wf_promote.sh",
        ),
        forbidden=("scripts/train_104.py", "RQ_ALLOW_NO_WF=1"),
    ),
    Check(
        name="conditional_trigger_uses_orchestrator",
        path="scripts/conditional_retrain_104.sh",
        required=(
            'RQ_CONDITIONAL_TRIGGER_RUNNER:-multirepo',
            'source "$REPO_DIR/scripts/subrepo_env.sh"',
            'renquant_load_subrepo_env "$REPO_DIR"',
            'export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"',
            'renquant_subrepo_root "$REPO_DIR" "$GITHUB_DIR"',
            'renquant_subrepo_src "$SUBREPO_ROOT" renquant-orchestrator',
            "renquant_orchestrator.anomaly_triggers",
            "set RQ_CONDITIONAL_TRIGGER_RUNNER=legacy for explicit rollback",
        ),
        forbidden=("falling back to umbrella trigger check",),
    ),
    Check(
        name="alpha158_fund_retrain_defaults_to_orchestrator",
        path="scripts/daily_retrain_alpha158_fund.sh",
        required=(
            'RQ_RETRAIN_RUNNER:-multirepo',
            'source "$REPO_DIR/scripts/subrepo_env.sh"',
            'renquant_load_subrepo_env "$REPO_DIR"',
            'export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"',
            'renquant_subrepo_pythonpath "$SUBREPO_ROOT"',
            "renquant_orchestrator.retrain_alpha158_fund",
            "renquant-orchestrator",
            "renquant-model",
            "renquant-pipeline",
            "renquant-execution",
            "set RQ_RETRAIN_RUNNER=umbrella for explicit rollback",
        ),
        forbidden=("falling back to umbrella retrain", "RETRAIN_FALLBACK"),
    ),
    Check(
        name="alpha158_linear_retrain_defaults_to_orchestrator",
        path="scripts/retrain_alpha158_linear.sh",
        required=(
            'RQ_ALPHA158_LINEAR_RUNNER:-multirepo',
            'source "$REPO_DIR/scripts/subrepo_env.sh"',
            'renquant_load_subrepo_env "$REPO_DIR"',
            'export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"',
            'renquant_subrepo_pythonpath "$SUBREPO_ROOT"',
            "renquant_orchestrator.retrain_alpha158_linear",
            "renquant-orchestrator",
            "renquant-model",
            "renquant-pipeline",
            "renquant-execution",
            "set RQ_ALPHA158_LINEAR_RUNNER=umbrella for explicit rollback",
        ),
        forbidden=("falling back to umbrella retrain",),
    ),
    Check(
        name="weekly_fundamental_refresh_uses_base_data_earnings",
        path="scripts/weekly_fundamental_refresh.sh",
        required=(
            "renquant_base_data.earnings_surprise_refresh",
            'source "$REPO_DIR/scripts/subrepo_env.sh"',
            'renquant_load_subrepo_env "$REPO_DIR"',
            'export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"',
            'renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-base-data renquant-common',
            'renquant_strategy_config "$SUBREPO_ROOT" strategy_config.json',
            '--strategy-config "$STRATEGY_CONFIG"',
            "RQ_DATA_REFRESH_STRICT",
        ),
        forbidden=('--strategy-config "$REPO_DIR/backtesting/renquant_104/strategy_config.json"',),
    ),
    Check(
        name="weekly_fundamental_refresh_uses_base_data_sec",
        path="scripts/weekly_fundamental_refresh.sh",
        required=(
            "renquant_base_data.sec_fundamentals",
            "--mode both",
            'source "$REPO_DIR/scripts/subrepo_env.sh"',
            'renquant_load_subrepo_env "$REPO_DIR"',
            'export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"',
            'renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-base-data renquant-common',
            'renquant_strategy_config "$SUBREPO_ROOT" strategy_config.json',
            "RQ_DATA_REFRESH_STRICT",
        ),
    ),
    Check(
        name="event_sec_schema_change_uses_base_data",
        path="scripts/event_sec_schema_change.sh",
        required=(
            "renquant_base_data.sec_fundamentals",
            "--mode both",
            'source "$REPO_DIR/scripts/subrepo_env.sh"',
            'renquant_load_subrepo_env "$REPO_DIR"',
            'export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"',
            'renquant_subrepo_src "$SUBREPO_ROOT" renquant-base-data',
            "RQ_EVENT_SEC_REFRESH_STRICT",
        ),
        forbidden=("/Users/renhao/miniconda3",),
    ),
    Check(
        name="daily_iv_snapshot_uses_base_data",
        path="scripts/daily_iv_snapshot.sh",
        required=(
            "renquant_base_data.options_iv_refresh",
            'source "$REPO_DIR/scripts/subrepo_env.sh"',
            'renquant_load_subrepo_env "$REPO_DIR"',
            'export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"',
            'renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-base-data renquant-common',
            'renquant_strategy_config "$SUBREPO_ROOT" strategy_config.json',
            '--strategy-config "$STRATEGY_CONFIG"',
            "RQ_DAILY_IV_STRICT",
        ),
        forbidden=('--strategy-config "$REPO_DIR/backtesting/renquant_104/strategy_config.json"',),
    ),
    Check(
        name="daily_news_fetch_uses_base_data",
        path="scripts/daily_news_sentiment_refresh.sh",
        required=(
            "renquant_base_data.alpaca_news_refresh",
            'source "$REPO_DIR/scripts/subrepo_env.sh"',
            'renquant_load_subrepo_env "$REPO_DIR"',
            'export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"',
            'renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-model renquant-base-data renquant-common',
            'renquant_strategy_config "$SUBREPO_ROOT" strategy_config.json',
            '--strategy-config "$STRATEGY_CONFIG"',
            "RQ_DAILY_NEWS_STRICT",
        ),
        forbidden=('--strategy-config "$REPO_DIR/backtesting/renquant_104/strategy_config.json"',),
    ),
    Check(
        name="daily_news_sentiment_uses_model_repo",
        path="scripts/daily_news_sentiment_refresh.sh",
        required=(
            "renquant_model_common.news_sentiment_finbert",
            'renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-model renquant-base-data renquant-common',
            "RQ_DAILY_NEWS_SENTIMENT_STRICT",
        ),
    ),
    Check(
        name="screen_watchlist_uses_base_data",
        path="scripts/screen_watchlist.py",
        required=(
            "renquant_base_data.watchlist_screen",
            "from subrepo_paths import resolve_subrepo_root",
            "resolve_subrepo_root(REPO_ROOT)",
            "renquant-base-data/src",
            "renquant-strategy-104",
            "strategy_config.json",
            "RENQUANT_STRICT_SUBREPO_PATHS",
            "RENQUANT_OPS_FAIL_CLOSED",
            "_strict_multirepo_enabled",
            "RQ_SCREEN_WATCHLIST_STRICT",
        ),
    ),
    Check(
        name="model_smoke_test_uses_backtesting_repo",
        path="scripts/smoke_test_model.py",
        required=(
            "from subrepo_module_delegate import delegate_to_subrepo_module",
            "delegate_to_subrepo_module(",
            "renquant_backtesting.analysis.smoke_test_model",
            "renquant-backtesting",
            "RQ_BACKTESTING_OPS_RUNNER",
            "RQ_BACKTESTING_OPS_STRICT",
        ),
    ),
    Check(
        name="lean_watchlist_export_uses_backtesting_repo",
        path="scripts/export_lean_watchlist.py",
        required=(
            "from subrepo_module_delegate import delegate_to_subrepo_module",
            "delegate_to_subrepo_module(",
            "renquant_backtesting.lean_export.export_lean_watchlist",
            "renquant-backtesting",
            "RQ_BACKTESTING_OPS_RUNNER",
            "RQ_BACKTESTING_OPS_STRICT",
        ),
    ),
    Check(
        name="portfolio_metrics_uses_backtesting_repo",
        path="scripts/compute_portfolio_metrics.py",
        required=(
            "from subrepo_module_delegate import delegate_to_subrepo_module",
            "delegate_to_subrepo_module(",
            "renquant_backtesting.analysis.compute_portfolio_metrics",
            "renquant-backtesting",
            "RQ_BACKTESTING_OPS_RUNNER",
            "RQ_BACKTESTING_OPS_STRICT",
        ),
    ),
    Check(
        name="forward_returns_backfill_uses_backtesting_repo",
        path="scripts/backfill_forward_returns.py",
        required=(
            "from subrepo_module_delegate import delegate_to_subrepo_module",
            "delegate_to_subrepo_module(",
            "renquant_backtesting.analysis.backfill_forward_returns",
            "renquant-backtesting",
            "RQ_BACKTESTING_OPS_RUNNER",
            "RQ_BACKTESTING_OPS_STRICT",
        ),
    ),
    Check(
        name="dashboard_builder_uses_backtesting_repo",
        path="scripts/build_dashboard.py",
        required=(
            "from subrepo_module_delegate import delegate_to_subrepo_module",
            "delegate_to_subrepo_module(",
            "renquant_backtesting.reporting.build_dashboard",
            "renquant-backtesting",
            "RQ_BACKTESTING_OPS_RUNNER",
            "RQ_BACKTESTING_OPS_STRICT",
        ),
    ),
    Check(
        name="wf_fingerprint_stamping_uses_backtesting_repo",
        path="scripts/stamp_walkforward_fingerprints.py",
        required=(
            "from subrepo_module_delegate import delegate_to_subrepo_module",
            "delegate_to_subrepo_module(",
            "renquant_backtesting.wf_gate.stamp_walkforward_fingerprints",
            "renquant-backtesting",
            "RQ_BACKTESTING_OPS_RUNNER",
            "RQ_BACKTESTING_OPS_STRICT",
        ),
    ),
    Check(
        name="strategy_config_drift_uses_strategy_repo",
        path="scripts/check_config_drift.py",
        required=(
            "from subrepo_module_delegate import delegate_to_subrepo_module",
            "delegate_to_subrepo_module(",
            "renquant_strategy_104.config_drift",
            "renquant-strategy-104",
            "RQ_STRATEGY_OPS_RUNNER",
            "RQ_STRATEGY_OPS_STRICT",
        ),
    ),
    Check(
        name="monthly_meta_label_uses_model_repo",
        path="scripts/monthly_meta_label_retrain.sh",
        required=(
            "renquant_model_common.meta_label_exit",
            'source "$REPO_DIR/scripts/subrepo_env.sh"',
            'renquant_load_subrepo_env "$REPO_DIR"',
            'export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"',
            'renquant_subrepo_pythonpath "$SUBREPO_ROOT"',
            'renquant_strategy_config "$SUBREPO_ROOT" strategy_config.json',
            'json.load(open("$PROD_STRATEGY_CONFIG"))',
            "renquant-model",
            "RQ_META_LABEL_STRICT",
        ),
        forbidden=('json.load(open("backtesting/renquant_104/strategy_config.json"))',),
    ),
    Check(
        name="monthly_meta_label_snapshot_uses_backtesting_repo",
        path="scripts/monthly_meta_label_retrain.sh",
        required=(
            "renquant_backtesting.wf_gate.sim_driver",
            'renquant_subrepo_pythonpath "$SUBREPO_ROOT"',
            "renquant-backtesting",
            "RQ_META_LABEL_SIM_STRICT",
        ),
    ),
    Check(
        name="monthly_calibrator_refresh_uses_model_repo",
        path="scripts/monthly_calibrator_refresh.sh",
        required=(
            "renquant_model_gbdt.fit_calibrator_alpha158_fund",
            'source "$REPO_DIR/scripts/subrepo_env.sh"',
            'renquant_load_subrepo_env "$REPO_DIR"',
            'export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"',
            'renquant_subrepo_pythonpath "$SUBREPO_ROOT"',
            'renquant_strategy_config "$SUBREPO_ROOT" strategy_config.json',
            "PROD_SCORER",
            '--scorer-artifact "$PROD_SCORER"',
            "Path('$PROD_STRATEGY_CONFIG').read_text()",
            "renquant-model",
            "RQ_MONTHLY_CALIBRATOR_STRICT",
        ),
        forbidden=(
            '--scorer-artifact "$REPO_DIR/backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json"',
            "cfg = json.loads((sd / 'strategy_config.json').read_text())",
        ),
    ),
    Check(
        name="patchtst_wf_uses_model_repo",
        path="scripts/train_walkforward_patchtst.py",
        required=(
            'TRAIN_MODULE = "renquant_model_patchtst.hf_trainer"',
            'CALIBRATOR_MODULE = "renquant_model_patchtst.fit_calibrator"',
        ),
    ),
    Check(
        name="wf_calibrators_use_model_repos",
        path="scripts/fit_walkforward_calibrators.py",
        required=(
            'GBDT_FITTER_MODULE = "renquant_model_gbdt.fit_calibrator_alpha158_fund"',
            'PATCHTST_FITTER_MODULE = "renquant_model_patchtst.fit_calibrator"',
        ),
    ),
    Check(
        name="state_backup_uses_orchestrator",
        path="scripts/backup_to_github.sh",
        required=(
            'RQ_STATE_BACKUP_RUNNER:-multirepo',
            "renquant_orchestrator.state_backup",
            'source "$REPO_ROOT/scripts/subrepo_env.sh"',
            'renquant_load_subrepo_env "$REPO_ROOT"',
            'export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"',
            'renquant_subrepo_src "$SUBREPO_ROOT" renquant-orchestrator',
            "RQ_STATE_BACKUP_STRICT",
        ),
    ),
    Check(
        name="weekly_apy_uses_orchestrator",
        path="scripts/weekly_apy_check.py",
        required=(
            'os.environ.get("RQ_WEEKLY_APY_RUNNER", "multirepo")',
            "renquant_orchestrator.weekly_apy_monitor",
            "from subrepo_paths import resolve_subrepo_root",
            "resolve_subrepo_root(REPO_ROOT)",
            'subrepo_root / "renquant-orchestrator" / "src"',
            "RQ_WEEKLY_APY_STRICT",
            "_strict_multirepo_enabled",
            "RENQUANT_OPS_FAIL_CLOSED",
        ),
    ),
    Check(
        name="subrepo_pin_ci_green_gate_available",
        path="scripts/check_lock_pins_ci_green.py",
        required=(
            "actions/runs?",
            "commits/{full_sha}/check-runs",
            "commits/{full_sha}/status",
            "compare/{full_sha}...{encoded_branch}",
            "subrepos.lock.json",
            "CI is red, pending, or missing",
        ),
    ),
    Check(
        name="subrepo_lock_refresh_prewrite_ci_gate_available",
        path="scripts/refresh_subrepo_lock.py",
        required=(
            "check_lock_func(candidate_path, only_subrepos=changed_names)",
            "candidate subrepo pin failed CI-green gate",
            "subrepo_lock_ci_green_force_override",
            "os.replace(tmp_path, path)",
            "--force",
        ),
    ),
    Check(
        name="subrepo_pin_ci_green_workflow_wired",
        path=".github/workflows/subrepo-pin-ci-green.yml",
        required=(
            "subrepos.lock.json",
            "scripts/check_lock_pins_ci_green.py",
            "tests/test_check_lock_pins_ci_green.py",
            "Detect subrepos.lock.json changes",
            "Verify pinned subrepo commits have green CI",
        ),
    ),
)


LAUNCHD_PLISTS: tuple[str, ...] = (
    "scripts/launchd/com.renquant.conditional-retrain104.plist",
    "scripts/launchd/com.renquant.daily104.plist",
    "scripts/launchd/com.renquant.daily-iv-snapshot.plist",
    "scripts/launchd/com.renquant.daily-news-sentiment.plist",
    "scripts/launchd/com.renquant.intraday104.plist",
    "scripts/launchd/com.renquant.monthly-calibrator-refresh.plist",
    "scripts/launchd/com.renquant.monthly-meta-label-retrain.plist",
    "scripts/launchd/com.renquant.preopen-cancel-gate.plist",
    "scripts/launchd/com.renquant.retrain-alpha158-linear.plist",
    "scripts/launchd/com.renquant.retrain-panel104.plist",
    "scripts/launchd/com.renquant.screen-watchlist.plist",
    "scripts/launchd/com.renquant.weekly-fundamental-refresh.plist",
    "scripts/launchd/com.renquant.weekly-apy104.plist",
    "scripts/launchd/com.renquant.weekly-wf-promote.plist",
    "scripts/com.renquant.backup.plist",
)


LAUNCHD_PROGRAMS: tuple[LaunchdProgram, ...] = (
    LaunchdProgram(
        "scripts/com.renquant.backup.plist",
        ("/bin/bash", _script("backup_to_github.sh")),
    ),
    LaunchdProgram(
        "scripts/launchd/com.renquant.conditional-retrain104.plist",
        (_script("conditional_retrain_104.sh"),),
    ),
    LaunchdProgram(
        "scripts/launchd/com.renquant.daily104.plist",
        (_script("daily_104.sh"),),
    ),
    LaunchdProgram(
        "scripts/launchd/com.renquant.daily-iv-snapshot.plist",
        (_script("daily_iv_snapshot.sh"),),
    ),
    LaunchdProgram(
        "scripts/launchd/com.renquant.daily-news-sentiment.plist",
        (_script("daily_news_sentiment_refresh.sh"),),
    ),
    LaunchdProgram(
        "scripts/launchd/com.renquant.intraday104.plist",
        (_script("intraday_sell_104.sh"),),
    ),
    LaunchdProgram(
        "scripts/launchd/com.renquant.monthly-calibrator-refresh.plist",
        (_script("monthly_calibrator_refresh.sh"),),
    ),
    LaunchdProgram(
        "scripts/launchd/com.renquant.monthly-meta-label-retrain.plist",
        (_script("monthly_meta_label_retrain.sh"),),
    ),
    LaunchdProgram(
        "scripts/launchd/com.renquant.preopen-cancel-gate.plist",
        (_script("preopen_cancel_gate.sh"),),
    ),
    LaunchdProgram(
        "scripts/launchd/com.renquant.retrain-alpha158-linear.plist",
        (_script("retrain_alpha158_linear.sh"),),
    ),
    LaunchdProgram(
        "scripts/launchd/com.renquant.retrain-panel104.plist",
        (_script("retrain_panel.sh"),),
    ),
    LaunchdProgram(
        "scripts/launchd/com.renquant.screen-watchlist.plist",
        (f"{VENV_BIN}/python", _script("screen_watchlist.py")),
    ),
    LaunchdProgram(
        "scripts/launchd/com.renquant.weekly-apy104.plist",
        (f"{VENV_BIN}/python", _script("weekly_apy_check.py")),
    ),
    LaunchdProgram(
        "scripts/launchd/com.renquant.weekly-fundamental-refresh.plist",
        (_script("weekly_fundamental_refresh.sh"),),
    ),
    LaunchdProgram(
        "scripts/launchd/com.renquant.weekly-wf-promote.plist",
        (_script("weekly_wf_promote.sh"),),
    ),
)


KNOWN_GAPS: tuple[dict[str, str], ...] = ()


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _non_comment_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


def run_contract() -> dict[str, object]:
    failures: list[dict[str, str]] = []
    passed: list[str] = []
    launchd_programs = {program.path: program for program in LAUNCHD_PROGRAMS}

    for check in CHECKS:
        text = _read(check.path)
        executable_text = _non_comment_text(text)
        for needle in check.required:
            if needle not in text:
                failures.append({
                    "check": check.name,
                    "path": check.path,
                    "reason": f"missing required text: {needle}",
                })
        for needle in check.forbidden:
            if needle in executable_text:
                failures.append({
                    "check": check.name,
                    "path": check.path,
                    "reason": f"forbidden text present: {needle}",
                })
        if not any(f["check"] == check.name for f in failures):
            passed.append(check.name)

    for rel in LAUNCHD_PLISTS:
        path = ROOT / rel
        try:
            payload = plistlib.loads(path.read_bytes())
        except Exception as exc:
            failures.append({
                "check": "launchd_plist_parseable",
                "path": rel,
                "reason": f"plist XML is not parseable: {exc}",
            })
            payload = {}

        expected_program = launchd_programs.get(rel)
        if expected_program is None:
            failures.append({
                "check": "launchd_program_arguments_multirepo",
                "path": rel,
                "reason": "active launchd plist has no ProgramArguments contract",
            })
        else:
            actual_args = payload.get("ProgramArguments")
            if not isinstance(actual_args, list):
                failures.append({
                    "check": "launchd_program_arguments_multirepo",
                    "path": rel,
                    "reason": "ProgramArguments must be a list",
                })
            elif tuple(actual_args) != expected_program.expected_args:
                failures.append({
                    "check": "launchd_program_arguments_multirepo",
                    "path": rel,
                    "reason": (
                        "ProgramArguments drifted from pinned multirepo wrapper: "
                        f"expected={list(expected_program.expected_args)!r} "
                        f"actual={actual_args!r}"
                    ),
                })

        text = _read(rel)
        if CONDA_PREFIX in text:
            failures.append({
                "check": "launchd_uses_project_venv",
                "path": rel,
                "reason": f"forbidden old conda path present: {CONDA_PREFIX}",
            })
        if "EnvironmentVariables" not in text or "PATH" not in text:
            failures.append({
                "check": "launchd_uses_project_venv",
                "path": rel,
                "reason": "active launchd plist must declare EnvironmentVariables.PATH",
            })
        elif VENV_BIN not in text:
            failures.append({
                "check": "launchd_uses_project_venv",
                "path": rel,
                "reason": f"launchd PATH should include project venv: {VENV_BIN}",
            })

    if not any(f["check"] == "launchd_plist_parseable" for f in failures):
        passed.append("launchd_plists_parseable")
    if not any(f["check"] == "launchd_program_arguments_multirepo" for f in failures):
        passed.append("launchd_program_arguments_multirepo")
    if not any(f["check"] == "launchd_uses_project_venv" for f in failures):
        passed.append("launchd_uses_project_venv")

    return {
        "ok": not failures,
        "passed": passed,
        "failures": failures,
        "known_gaps": list(KNOWN_GAPS),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args(argv)

    result = run_contract()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["known_gaps"]:
            print("Known gaps are informational; failures block the contract.", file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
