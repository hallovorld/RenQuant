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


def test_subrepo_root_loads_current_env_runtime_root(tmp_path: Path) -> None:
    repo = tmp_path / "RenQuant"
    runtime = tmp_path / "runtime" / "repos"
    env_dir = repo / ".subrepo_assembly"
    env_dir.mkdir(parents=True)
    runtime.mkdir(parents=True)
    (env_dir / "current.env").write_text(
        f"export RENQUANT_SUBREPO_ROOT={runtime}\n",
        encoding="utf-8",
    )

    out = _bash(
        'source scripts/subrepo_env.sh; '
        f'renquant_load_subrepo_env "{repo}"; '
        f'renquant_subrepo_root "{repo}" "/fallback"'
    )
    assert out == str(runtime)


def test_subrepo_root_resolves_relative_runtime_root_against_repo(tmp_path: Path) -> None:
    repo = tmp_path / "RenQuant"
    env_dir = repo / ".subrepo_assembly"
    env_dir.mkdir(parents=True)
    (env_dir / "current.env").write_text(
        "export RENQUANT_SUBREPO_ROOT=.subrepo_runtime/repos\n",
        encoding="utf-8",
    )

    out = _bash(
        'source scripts/subrepo_env.sh; '
        f'renquant_load_subrepo_env "{repo}"; '
        f'renquant_subrepo_root "{repo}" "/fallback"'
    )
    assert out == str(repo / ".subrepo_runtime" / "repos")


def test_subrepo_root_uses_loaded_assembly_repos(tmp_path: Path) -> None:
    repo = tmp_path / "RenQuant"
    assembly = tmp_path / "assembly"
    repos = assembly / "repos"
    env_dir = repo / ".subrepo_assembly"
    env_dir.mkdir(parents=True)
    repos.mkdir(parents=True)
    (env_dir / "current.env").write_text(
        f"export RENQUANT_ASSEMBLY_DIR={assembly}\n",
        encoding="utf-8",
    )

    out = _bash(
        'source scripts/subrepo_env.sh; '
        f'renquant_load_subrepo_env "{repo}"; '
        f'renquant_subrepo_root "{repo}" "/fallback"'
    )
    assert out == str(repos)
