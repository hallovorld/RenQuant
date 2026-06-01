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


def resolve_subrepo_root(repo_root: Path) -> Path:
    """Return runtime root, current assembly repos dir, or sibling checkout root."""
    if root := os.environ.get("RENQUANT_SUBREPO_ROOT"):
        return Path(root)

    assembly_dir = os.environ.get("RENQUANT_ASSEMBLY_DIR")
    if assembly_dir and (Path(assembly_dir) / "repos").exists():
        return Path(assembly_dir) / "repos"

    env_path = Path(
        os.environ.get(
            "RENQUANT_SUBREPO_ENV",
            str(repo_root / ".subrepo_assembly" / "current.env"),
        )
    )
    if root := _read_export(env_path, "RENQUANT_SUBREPO_ROOT"):
        return Path(root)
    if assembly_dir := _read_export(env_path, "RENQUANT_ASSEMBLY_DIR"):
        repos = Path(assembly_dir) / "repos"
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

    return repo_root.parent
