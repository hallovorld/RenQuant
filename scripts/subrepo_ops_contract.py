#!/usr/bin/env python3
"""Validate that scheduled RenQuant ops default to pinned multirepo paths.

This is a fast structural contract for launchd/shell entrypoints. It does not
run broker code or retrain models; it fails when an active production wrapper
drifts back to direct umbrella execution or an old Python environment.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONDA_PREFIX = "/Users/renhao/miniconda3"
VENV_BIN = "/Users/renhao/git/github/RenQuant/.venv/bin"


@dataclass(frozen=True)
class Check:
    name: str
    path: str
    required: tuple[str, ...]
    forbidden: tuple[str, ...] = ()


CHECKS: tuple[Check, ...] = (
    Check(
        name="daily_live_defaults_to_multirepo",
        path="scripts/daily_104.sh",
        required=(
            'RQ_DAILY_RUNNER:-multirepo',
            'RUNNER_ARGS=("$REPO_DIR/scripts/daily_multirepo.py")',
            'RUNNER_ARGS=(-m live.runner)',
            'runner = [sys.executable, "$REPO_DIR/scripts/live_multirepo.py"]',
        ),
        forbidden=(),
    ),
    Check(
        name="intraday_sell_defaults_to_multirepo",
        path="scripts/intraday_sell_104.sh",
        required=(
            'RQ_DAILY_RUNNER:-multirepo',
            'RUNNER_ARGS=("$REPO_DIR/scripts/live_multirepo.py")',
            "--sell-only --intraday",
        ),
    ),
    Check(
        name="preopen_gate_defaults_to_execution_subrepo",
        path="scripts/preopen_cancel_gate.sh",
        required=(
            'RQ_PREOPEN_GATE_RUNNER:-multirepo',
            "../renquant-execution/src",
            "python -m renquant_execution.preopen_cancel_gate",
            "RQ_PREOPEN_GATE_STRICT",
        ),
    ),
    Check(
        name="weekly_retrain_delegates_to_orchestrator_wrapper",
        path="scripts/weekly_wf_promote.sh",
        required=(
            "bash scripts/daily_retrain_alpha158_fund.sh",
            "scripts/run_wf_gate.py",
            "--strict",
        ),
        forbidden=("RQ_ALLOW_NO_WF=1",),
    ),
    Check(
        name="alpha158_fund_retrain_defaults_to_orchestrator",
        path="scripts/daily_retrain_alpha158_fund.sh",
        required=(
            'RQ_RETRAIN_RUNNER:-multirepo',
            "renquant_orchestrator.retrain_alpha158_fund",
            "$GITHUB_DIR/renquant-orchestrator/src",
            "RQ_RETRAIN_STRICT",
        ),
    ),
    Check(
        name="alpha158_linear_retrain_defaults_to_orchestrator",
        path="scripts/retrain_alpha158_linear.sh",
        required=(
            'RQ_ALPHA158_LINEAR_RUNNER:-multirepo',
            "renquant_orchestrator.retrain_alpha158_linear",
            "$GITHUB_DIR/renquant-orchestrator/src",
            "RQ_ALPHA158_LINEAR_STRICT",
        ),
    ),
    Check(
        name="weekly_fundamental_refresh_uses_base_data_earnings",
        path="scripts/weekly_fundamental_refresh.sh",
        required=(
            "renquant_base_data.earnings_surprise_refresh",
            "$GITHUB_DIR/renquant-base-data/src",
            "RQ_DATA_REFRESH_STRICT",
        ),
    ),
    Check(
        name="monthly_calibrator_refresh_uses_model_repo",
        path="scripts/monthly_calibrator_refresh.sh",
        required=(
            "renquant_model_gbdt.fit_calibrator_alpha158_fund",
            "$GITHUB_DIR/renquant-model/src",
            "RQ_MONTHLY_CALIBRATOR_STRICT",
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
)


LAUNCHD_PLISTS: tuple[str, ...] = (
    "scripts/launchd/com.renquant.conditional-retrain104.plist",
    "scripts/launchd/com.renquant.daily-iv-snapshot.plist",
    "scripts/launchd/com.renquant.daily-news-sentiment.plist",
    "scripts/launchd/com.renquant.monthly-calibrator-refresh.plist",
    "scripts/launchd/com.renquant.monthly-meta-label-retrain.plist",
    "scripts/launchd/com.renquant.preopen-cancel-gate.plist",
    "scripts/launchd/com.renquant.retrain-alpha158-linear.plist",
    "scripts/launchd/com.renquant.screen-watchlist.plist",
    "scripts/launchd/com.renquant.weekly-fundamental-refresh.plist",
    "scripts/launchd/com.renquant.weekly-wf-promote.plist",
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
        text = _read(rel)
        if CONDA_PREFIX in text:
            failures.append({
                "check": "launchd_uses_project_venv",
                "path": rel,
                "reason": f"forbidden old conda path present: {CONDA_PREFIX}",
            })
        if "EnvironmentVariables" in text and "PATH" in text and VENV_BIN not in text:
            failures.append({
                "check": "launchd_uses_project_venv",
                "path": rel,
                "reason": f"launchd PATH should include project venv: {VENV_BIN}",
            })

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
