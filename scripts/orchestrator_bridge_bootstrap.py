#!/usr/bin/env python
"""Bootstrap helpers for compatibility wrappers that delegate to orchestrator."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

STRICT_PIN_ENVS = (
    "RENQUANT_STRICT_SUBREPO_PATHS",
    "RENQUANT_STRICT_SUBREPO_PINS",
    "RENQUANT_OPS_FAIL_CLOSED",
)


def resolve_orchestrator_src(repo_root: Path, siblings: Path) -> Path:
    """Return the orchestrator src root, preferring the pinned runtime root."""
    runtime_root = _runtime_root(repo_root)
    if runtime_root:
        return runtime_root / "renquant-orchestrator" / "src"
    if locked := _lock_local_src(repo_root):
        return locked
    if _strict_pin_enabled():
        raise SystemExit(
            "RENQUANT strict subrepo mode requires a pinned renquant-orchestrator "
            "runtime root or lock local_path matching subrepos.lock.json"
        )
    return siblings / "renquant-orchestrator" / "src"


def _runtime_root(repo_root: Path) -> Path | None:
    if raw := os.environ.get("RENQUANT_SUBREPO_ROOT"):
        return _abs_path(repo_root, raw)

    if raw := os.environ.get("RENQUANT_ASSEMBLY_DIR"):
        return _abs_path(repo_root, raw) / "repos"

    env_path = Path(
        os.environ.get(
            "RENQUANT_SUBREPO_ENV",
            str(repo_root / ".subrepo_assembly" / "current.env"),
        )
    )
    if raw := _read_export(env_path, "RENQUANT_SUBREPO_ROOT"):
        return _abs_path(repo_root, raw)
    if raw := _read_export(env_path, "RENQUANT_ASSEMBLY_DIR"):
        return _abs_path(repo_root, raw) / "repos"

    current_json = repo_root / ".subrepo_assembly" / "current.json"
    if current_json.exists():
        try:
            current = Path(json.loads(current_json.read_text(encoding="utf-8"))["current"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            current = None
        if current is not None and (current / "repos").exists():
            return current / "repos"
    return None


def _read_export(path: Path, name: str) -> str | None:
    if not path.exists():
        return None
    prefix = f"export {name}="
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith(prefix):
            return shlex.split(line[len("export ") :], posix=True)[0].split("=", 1)[1]
    return None


def _lock_local_src(repo_root: Path) -> Path | None:
    lock_path = repo_root / "subrepos.lock.json"
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for entry in lock.get("subrepos", []):
        if entry.get("name") != "renquant-orchestrator" or not entry.get("local_path"):
            continue
        repo_path = _abs_path(repo_root, str(entry["local_path"]))
        src_path = repo_path / "src"
        if not src_path.is_dir():
            continue
        if _strict_pin_enabled():
            _validate_lock_entry(repo_path, entry)
        return src_path
    return None


def _validate_lock_entry(repo_path: Path, entry: dict) -> None:
    expected = str(entry.get("commit", ""))
    expected_remote = str(entry.get("remote", ""))
    try:
        head = _git(repo_path, "log", "-1", "--format=%H")
        remote = _git(repo_path, "remote", "get-url", "origin")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"renquant-orchestrator git metadata failed: {exc}") from exc
    if expected and not head.startswith(expected):
        raise SystemExit(
            "renquant-orchestrator local_path HEAD "
            f"{head[:12]} does not match lock commit {expected}"
        )
    if expected_remote and _norm_remote(remote) != _norm_remote(expected_remote):
        raise SystemExit(
            "renquant-orchestrator local_path remote "
            f"{remote} does not match lock remote {expected_remote}"
        )


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(repo), *args), text=True).strip()


def _strict_pin_enabled() -> bool:
    return any(os.environ.get(name) == "1" for name in STRICT_PIN_ENVS)


def _norm_remote(remote: str) -> str:
    return remote.removesuffix(".git").rstrip("/")


def _abs_path(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else repo_root / path
