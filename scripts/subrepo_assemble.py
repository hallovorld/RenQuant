#!/usr/bin/env python3
"""Create a deterministic local assembly from pinned RenQuant subrepos.

The assembly is an output directory. It never deletes or rewrites the source
umbrella repo, and it does not mutate subrepos unless --sync is explicitly
passed. By default it verifies that local subrepo checkouts already match
subrepos.lock.json, then writes a timestamped bundle containing repo symlinks,
an env file, and a machine-readable manifest.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "subrepos.lock.json"
ASSEMBLY_ROOT = ROOT / ".subrepo_assembly"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(repo), *args), text=True).strip()


def _run(args: tuple[str, ...], *, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def _norm_remote(remote: str) -> str:
    return remote.removesuffix(".git").rstrip("/")


def _is_dirty(repo: Path) -> bool:
    return bool(_git(repo, "status", "--porcelain"))


def _ensure_repo(entry: dict[str, Any], *, sync: bool) -> None:
    path = Path(entry["local_path"])
    if not path.exists():
        if not sync:
            raise RuntimeError(f"{entry['name']} missing at {path}; rerun with --sync to clone")
        path.parent.mkdir(parents=True, exist_ok=True)
        _run(("git", "clone", entry["remote"], str(path)))

    remote = _git(path, "remote", "get-url", "origin")
    if _norm_remote(remote) != _norm_remote(entry["remote"]):
        raise RuntimeError(f"{entry['name']} remote mismatch: local={remote} lock={entry['remote']}")

    commit = _git(path, "rev-parse", "HEAD")
    branch = _git(path, "rev-parse", "--abbrev-ref", "HEAD")
    if commit == entry["commit"] and branch == entry["branch"]:
        return

    if not sync:
        raise RuntimeError(
            f"{entry['name']} not pinned: branch={branch} commit={commit}; "
            f"expected branch={entry['branch']} commit={entry['commit']}"
        )
    if _is_dirty(path):
        raise RuntimeError(f"{entry['name']} is dirty; refusing --sync checkout")
    _run(("git", "fetch", "origin"), cwd=path)
    _run(("git", "checkout", entry["branch"]), cwd=path)
    _run(("git", "checkout", entry["commit"]), cwd=path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        raise RuntimeError(f"assembly path already exists: {dst}")
    dst.symlink_to(src, target_is_directory=True)


def build_assembly(lock: dict[str, Any], *, sync: bool, dry_run: bool) -> Path | None:
    for entry in lock["subrepos"]:
        _ensure_repo(entry, sync=sync)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    assembly = ASSEMBLY_ROOT / ts
    if dry_run:
        return None

    repos_dir = assembly / "repos"
    repos_dir.mkdir(parents=True, exist_ok=False)
    python_paths: list[str] = []

    for entry in lock["subrepos"]:
        repo_path = Path(entry["local_path"])
        _symlink(repo_path, repos_dir / entry["name"])
        src = repo_path / "src"
        if src.exists():
            python_paths.append(str(src))

    manifest = {
        "created_at": ts,
        "source_repo": lock["source_repo"],
        "subrepos": lock["subrepos"],
        "pythonpath": python_paths,
    }
    _write_text(assembly / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    _write_text(assembly / "pythonpath.txt", "\n".join(python_paths) + "\n")
    env = [
        "# source this file to use the pinned RenQuant subrepo assembly",
        f"export RENQUANT_ASSEMBLY_DIR={assembly}",
        f"export PYTHONPATH={':'.join(python_paths)}:${{PYTHONPATH:-}}",
        "",
    ]
    _write_text(assembly / "env.sh", "\n".join(env))
    _write_text(ASSEMBLY_ROOT / "current.json", json.dumps({"current": str(assembly)}, indent=2) + "\n")
    return assembly


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync", action="store_true", help="Clone/fetch/checkout missing or unpinned clean repos")
    parser.add_argument("--dry-run", action="store_true", help="Verify only; do not write assembly output")
    args = parser.parse_args()

    lock = json.loads(LOCK_PATH.read_text())
    if lock["source_repo"].get("never_delete") is not True:
        raise SystemExit("source_repo.never_delete must be true")

    try:
        assembly = build_assembly(lock, sync=args.sync, dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001
        print(f"subrepo assembly failed: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(json.dumps({"ok": True, "dry_run": True}, indent=2))
    else:
        print(json.dumps({"ok": True, "assembly": str(assembly)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
