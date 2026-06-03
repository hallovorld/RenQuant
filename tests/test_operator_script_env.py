"""Operator scripts should use the project venv, not stale conda paths."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def _non_comment(path: Path) -> str:
    return "\n".join(
        line for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


def test_challenger_window_uses_project_venv() -> None:
    src = (REPO / "scripts" / "check_challenger_window.sh").read_text(encoding="utf-8")
    non_comment = _non_comment(REPO / "scripts" / "check_challenger_window.sh")

    assert "miniconda" not in non_comment
    assert "CONDA_PREFIX" not in non_comment
    assert 'PYTHON="${REPO_ROOT}/.venv/bin/python"' in src
    assert '"$PYTHON" "${REPO_ROOT}/scripts/finalize_challenger.py"' in src


def test_manual_promote_uses_project_venv() -> None:
    src = (REPO / "scripts" / "manual_promote.sh").read_text(encoding="utf-8")
    non_comment = _non_comment(REPO / "scripts" / "manual_promote.sh")

    assert "miniconda" not in non_comment
    assert "CONDA_PREFIX" not in non_comment
    assert 'PYTHON="$REPO_DIR/.venv/bin/python"' in src


def test_multirepo_shell_wrappers_use_shared_strict_helper() -> None:
    env_src = (REPO / "scripts" / "subrepo_env.sh").read_text(encoding="utf-8")
    assert "renquant_strict_enabled()" in env_src
    assert "RENQUANT_OPS_FAIL_CLOSED" in env_src

    wrappers = (
        "backup_to_github.sh",
        "conditional_retrain_104.sh",
        "daily_iv_snapshot.sh",
        "daily_news_sentiment_refresh.sh",
        "daily_retrain_alpha158_fund.sh",
        "event_sec_schema_change.sh",
        "monthly_calibrator_refresh.sh",
        "monthly_meta_label_retrain.sh",
        "preopen_cancel_gate.sh",
        "retrain_alpha158_linear.sh",
        "weekly_fundamental_refresh.sh",
        "weekly_wf_promote.sh",
    )
    for script in wrappers:
        src = (REPO / "scripts" / script).read_text(encoding="utf-8")
        assert "renquant_strict_enabled" in src, script

    weekly_apy = (REPO / "scripts" / "weekly_apy_check.py").read_text(encoding="utf-8")
    assert "_strict_multirepo_enabled" in weekly_apy
    assert "RENQUANT_OPS_FAIL_CLOSED" in weekly_apy


def test_weekly_apy_default_fails_closed_without_orchestrator(tmp_path) -> None:
    missing_runtime = tmp_path / "repos"
    missing_runtime.mkdir()

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "weekly_apy_check.py"),
            "--quiet",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "RENQUANT_NO_NOTIFY": "1",
            "RENQUANT_OPS_FAIL_CLOSED": "0",
            "RENQUANT_STRICT_SUBREPO_PATHS": "0",
            "RENQUANT_SUBREPO_ROOT": str(missing_runtime),
            "RQ_WEEKLY_APY_STRICT": "0",
            "RQ_WEEKLY_APY_RUNNER": "multirepo",
        },
    )

    assert proc.returncode == 2
    assert "weekly APY defaults to fail-closed multirepo mode" in proc.stderr
    assert "falling back" not in proc.stderr
