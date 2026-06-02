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


def test_strategy_config_resolves_from_strategy_subrepo(tmp_path: Path) -> None:
    root = tmp_path / "runtime" / "repos"
    config = root / "renquant-strategy-104" / "configs" / "strategy_config.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")

    out = _bash(
        'source scripts/subrepo_env.sh; '
        f'renquant_strategy_config "{root}" strategy_config.json'
    )
    assert out == str(config)


def test_strategy_config_fails_when_missing(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source scripts/subrepo_env.sh; '
            f'renquant_strategy_config "{tmp_path}" strategy_config.json',
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
    )

    assert result.returncode == 1
    assert result.stdout == ""


def test_strict_helper_honors_wrapper_env() -> None:
    out = _bash(
        'source scripts/subrepo_env.sh; '
        'RQ_FAKE_STRICT=1; '
        'if renquant_strict_enabled RQ_FAKE_STRICT; then echo strict; else echo loose; fi'
    )
    assert out == "strict"


def test_strict_helper_honors_global_ops_fail_closed() -> None:
    out = _bash(
        'source scripts/subrepo_env.sh; '
        'RENQUANT_OPS_FAIL_CLOSED=1; '
        'if renquant_strict_enabled RQ_FAKE_STRICT; then echo strict; else echo loose; fi'
    )
    assert out == "strict"


def test_strict_helper_defaults_loose() -> None:
    out = _bash(
        'source scripts/subrepo_env.sh; '
        'if renquant_strict_enabled RQ_FAKE_STRICT; then echo strict; else echo loose; fi'
    )
    assert out == "loose"


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
