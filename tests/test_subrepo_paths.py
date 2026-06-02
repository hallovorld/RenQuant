from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "subrepo_paths.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("subrepo_paths_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_lock(repo: Path, root: Path, *, missing: str | None = None) -> None:
    payload = {
        "subrepos": [
            {
                "name": "renquant-orchestrator",
                "local_path": str(root / "renquant-orchestrator"),
            },
            {
                "name": "renquant-strategy-104",
                "local_path": str(root / "renquant-strategy-104"),
            },
        ]
    }
    repo.mkdir(parents=True)
    for entry in payload["subrepos"]:
        if entry["name"] == missing:
            continue
        (Path(entry["local_path"]) / "src").mkdir(parents=True)
    (repo / "subrepos.lock.json").write_text(json.dumps(payload), encoding="utf-8")


def test_resolve_subrepo_root_prefers_env_override(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module()
    runtime = tmp_path / "runtime" / "repos"
    monkeypatch.setenv("RENQUANT_SUBREPO_ROOT", str(runtime))

    assert mod.resolve_subrepo_root(tmp_path / "RenQuant") == runtime


def test_resolve_subrepo_root_reads_current_env_runtime_root(
    monkeypatch, tmp_path: Path,
) -> None:
    mod = _load_module()
    monkeypatch.delenv("RENQUANT_SUBREPO_ROOT", raising=False)
    monkeypatch.delenv("RENQUANT_ASSEMBLY_DIR", raising=False)
    repo = tmp_path / "RenQuant"
    runtime = tmp_path / "runtime" / "repos"
    (repo / ".subrepo_assembly").mkdir(parents=True)
    (repo / ".subrepo_assembly" / "current.env").write_text(
        f"export RENQUANT_SUBREPO_ROOT={runtime}\n",
        encoding="utf-8",
    )

    assert mod.resolve_subrepo_root(repo) == runtime


def test_resolve_subrepo_root_makes_current_env_runtime_root_absolute(
    monkeypatch, tmp_path: Path,
) -> None:
    mod = _load_module()
    monkeypatch.delenv("RENQUANT_SUBREPO_ROOT", raising=False)
    monkeypatch.delenv("RENQUANT_ASSEMBLY_DIR", raising=False)
    repo = tmp_path / "RenQuant"
    (repo / ".subrepo_assembly").mkdir(parents=True)
    (repo / ".subrepo_assembly" / "current.env").write_text(
        "export RENQUANT_SUBREPO_ROOT=.subrepo_runtime/repos\n",
        encoding="utf-8",
    )

    assert mod.resolve_subrepo_root(repo) == repo / ".subrepo_runtime" / "repos"


def test_resolve_subrepo_root_reads_current_env_assembly(
    monkeypatch, tmp_path: Path,
) -> None:
    mod = _load_module()
    monkeypatch.delenv("RENQUANT_SUBREPO_ROOT", raising=False)
    monkeypatch.delenv("RENQUANT_ASSEMBLY_DIR", raising=False)
    repo = tmp_path / "RenQuant"
    assembly = tmp_path / "assembly"
    (assembly / "repos").mkdir(parents=True)
    (repo / ".subrepo_assembly").mkdir(parents=True)
    (repo / ".subrepo_assembly" / "current.env").write_text(
        f"export RENQUANT_ASSEMBLY_DIR={assembly}\n",
        encoding="utf-8",
    )

    assert mod.resolve_subrepo_root(repo) == assembly / "repos"


def test_resolve_subrepo_root_reads_current_json(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module()
    monkeypatch.delenv("RENQUANT_SUBREPO_ROOT", raising=False)
    monkeypatch.delenv("RENQUANT_ASSEMBLY_DIR", raising=False)
    repo = tmp_path / "RenQuant"
    assembly = tmp_path / "assembly"
    (assembly / "repos").mkdir(parents=True)
    (repo / ".subrepo_assembly").mkdir(parents=True)
    (repo / ".subrepo_assembly" / "current.json").write_text(
        json.dumps({"current": str(assembly)}),
        encoding="utf-8",
    )

    assert mod.resolve_subrepo_root(repo) == assembly / "repos"


def test_resolve_subrepo_root_falls_back_to_sibling_root(
    monkeypatch, tmp_path: Path,
) -> None:
    mod = _load_module()
    monkeypatch.delenv("RENQUANT_SUBREPO_ROOT", raising=False)
    monkeypatch.delenv("RENQUANT_ASSEMBLY_DIR", raising=False)
    monkeypatch.setenv("RENQUANT_SUBREPO_ENV", str(tmp_path / "missing.env"))
    repo = tmp_path / "RenQuant"

    assert mod.resolve_subrepo_root(repo) == tmp_path


def test_resolve_subrepo_root_infers_common_parent_from_lock_local_paths(
    tmp_path: Path, monkeypatch,
) -> None:
    mod = _load_module()
    repo = tmp_path / "worktrees" / "RenQuant"
    root = tmp_path / "github"
    _write_lock(repo, root)
    monkeypatch.delenv("RENQUANT_SUBREPO_ROOT", raising=False)
    monkeypatch.delenv("RENQUANT_ASSEMBLY_DIR", raising=False)
    monkeypatch.setenv("RENQUANT_SUBREPO_ENV", str(tmp_path / "missing.env"))

    assert mod.resolve_subrepo_root(repo) == root


def test_resolve_subrepo_root_falls_back_when_lock_local_paths_incomplete(
    tmp_path: Path, monkeypatch,
) -> None:
    mod = _load_module()
    repo = tmp_path / "worktrees" / "RenQuant"
    root = tmp_path / "github"
    _write_lock(repo, root, missing="renquant-strategy-104")
    monkeypatch.delenv("RENQUANT_SUBREPO_ROOT", raising=False)
    monkeypatch.delenv("RENQUANT_ASSEMBLY_DIR", raising=False)
    monkeypatch.setenv("RENQUANT_SUBREPO_ENV", str(tmp_path / "missing.env"))

    assert mod.resolve_subrepo_root(repo) == repo.parent


def test_resolve_subrepo_root_env_override_wins_over_lock_local_paths(
    tmp_path: Path, monkeypatch,
) -> None:
    mod = _load_module()
    repo = tmp_path / "worktrees" / "RenQuant"
    root = tmp_path / "github"
    override = tmp_path / "runtime" / "repos"
    _write_lock(repo, root)
    monkeypatch.setenv("RENQUANT_SUBREPO_ROOT", str(override))

    assert mod.resolve_subrepo_root(repo) == override
