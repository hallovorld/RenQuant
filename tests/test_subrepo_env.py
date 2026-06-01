"""Tests for shared shell subrepo path helpers."""
from __future__ import annotations

import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def _bash(script: str) -> str:
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=REPO,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def test_subrepo_root_defaults_to_sibling_github_dir() -> None:
    out = _bash(
        'source scripts/subrepo_env.sh; '
        'renquant_subrepo_root "$PWD" "$(cd "$PWD/.." && pwd)"'
    )
    assert out == str(REPO.parent)


def test_subrepo_root_honors_runtime_override() -> None:
    out = _bash(
        'source scripts/subrepo_env.sh; '
        'RENQUANT_SUBREPO_ROOT=/tmp/renquant-runtime; '
        'renquant_subrepo_src "$(renquant_subrepo_root "$PWD")" renquant-model'
    )
    assert out == "/tmp/renquant-runtime/renquant-model/src"


def test_subrepo_pythonpath_preserves_repo_order() -> None:
    out = _bash(
        'source scripts/subrepo_env.sh; '
        'RENQUANT_SUBREPO_ROOT=/tmp/renquant-runtime; '
        'renquant_subrepo_pythonpath "$(renquant_subrepo_root "$PWD")" '
        'renquant-orchestrator renquant-common'
    )
    assert out == (
        "/tmp/renquant-runtime/renquant-orchestrator/src:"
        "/tmp/renquant-runtime/renquant-common/src"
    )
