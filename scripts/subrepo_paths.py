"""Shared helpers for resolving RenQuant subrepo runtime roots."""
from __future__ import annotations

import json
import os
import shlex
from pathlib import Path


def _read_export(path: Path, name: str) -> str | None:
    if not path.exists():
        return None
    prefix = f"export {name}="
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith(prefix):
            return shlex.split(line[len("export ") :], posix=True)[0].split("=", 1)[1]
    return None


def _abs_path(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else repo_root / path


def _lock_local_path_root(repo_root: Path) -> Path | None:
    lock_path = repo_root / "subrepos.lock.json"
    if not lock_path.exists():
        return None
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    parents: set[Path] = set()
    n_local_paths = 0
    n_existing = 0
    for entry in lock.get("subrepos", []):
        raw = entry.get("local_path")
        if not raw:
            continue
        n_local_paths += 1
        path = _abs_path(repo_root, str(raw))
        if not (path / "src").is_dir():
            continue
        n_existing += 1
        parents.add(path.parent)
    if n_local_paths > 0 and n_existing == n_local_paths and len(parents) == 1:
        return next(iter(parents))
    return None


def resolve_subrepo_root(repo_root: Path) -> Path:
    """Return runtime root, current assembly repos dir, or sibling checkout root."""
    if root := os.environ.get("RENQUANT_SUBREPO_ROOT"):
        return _abs_path(repo_root, root)

    assembly_dir = os.environ.get("RENQUANT_ASSEMBLY_DIR")
    if assembly_dir:
        repos = _abs_path(repo_root, assembly_dir) / "repos"
        if repos.exists():
            return repos

    env_path = Path(
        os.environ.get(
            "RENQUANT_SUBREPO_ENV",
            str(repo_root / ".subrepo_assembly" / "current.env"),
        )
    )
    if root := _read_export(env_path, "RENQUANT_SUBREPO_ROOT"):
        return _abs_path(repo_root, root)
    if assembly_dir := _read_export(env_path, "RENQUANT_ASSEMBLY_DIR"):
        repos = _abs_path(repo_root, assembly_dir) / "repos"
        if repos.exists():
            return repos

    current_json = repo_root / ".subrepo_assembly" / "current.json"
    if current_json.exists():
        try:
            current = Path(json.loads(current_json.read_text(encoding="utf-8"))["current"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            current = None
        if current is not None and (current / "repos").exists():
            return current / "repos"

    if root := _lock_local_path_root(repo_root):
        return root

    return repo_root.parent
