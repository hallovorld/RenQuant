"""Tests for Python subrepo runtime path resolution."""
from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from subrepo_paths import resolve_subrepo_root  # noqa: E402


def test_resolve_subrepo_root_prefers_env_override(monkeypatch, tmp_path: Path) -> None:
    runtime = tmp_path / "runtime" / "repos"
    monkeypatch.setenv("RENQUANT_SUBREPO_ROOT", str(runtime))
    assert resolve_subrepo_root(tmp_path / "RenQuant") == runtime


def test_resolve_subrepo_root_reads_current_env_runtime_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RENQUANT_SUBREPO_ROOT", raising=False)
    monkeypatch.delenv("RENQUANT_ASSEMBLY_DIR", raising=False)
    repo = tmp_path / "RenQuant"
    runtime = tmp_path / "runtime" / "repos"
    (repo / ".subrepo_assembly").mkdir(parents=True)
    (repo / ".subrepo_assembly" / "current.env").write_text(
        f"export RENQUANT_SUBREPO_ROOT={runtime}\n",
        encoding="utf-8",
    )
    assert resolve_subrepo_root(repo) == runtime


def test_resolve_subrepo_root_makes_current_env_runtime_root_absolute(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("RENQUANT_SUBREPO_ROOT", raising=False)
    monkeypatch.delenv("RENQUANT_ASSEMBLY_DIR", raising=False)
    repo = tmp_path / "RenQuant"
    (repo / ".subrepo_assembly").mkdir(parents=True)
    (repo / ".subrepo_assembly" / "current.env").write_text(
        "export RENQUANT_SUBREPO_ROOT=.subrepo_runtime/repos\n",
        encoding="utf-8",
    )
    assert resolve_subrepo_root(repo) == repo / ".subrepo_runtime" / "repos"


def test_resolve_subrepo_root_reads_current_env_assembly(monkeypatch, tmp_path: Path) -> None:
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
    assert resolve_subrepo_root(repo) == assembly / "repos"


def test_resolve_subrepo_root_reads_current_json(monkeypatch, tmp_path: Path) -> None:
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
    assert resolve_subrepo_root(repo) == assembly / "repos"


def test_resolve_subrepo_root_falls_back_to_sibling_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RENQUANT_SUBREPO_ROOT", raising=False)
    monkeypatch.delenv("RENQUANT_ASSEMBLY_DIR", raising=False)
    repo = tmp_path / "RenQuant"
    assert resolve_subrepo_root(repo) == tmp_path
