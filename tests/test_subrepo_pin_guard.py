from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "subrepo_pin_guard.py"
    spec = importlib.util.spec_from_file_location("subrepo_pin_guard", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(repo), *args), text=True).strip()


def _make_repo(path: Path) -> str:
    path.mkdir()
    (path / "src").mkdir()
    (path / "src" / "pkg.py").write_text("VALUE = 1\n")
    subprocess.run(("git", "init"), cwd=path, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(("git", "config", "user.email", "test@example.com"), cwd=path, check=True)
    subprocess.run(("git", "config", "user.name", "Test"), cwd=path, check=True)
    subprocess.run(("git", "add", "."), cwd=path, check=True)
    subprocess.run(("git", "commit", "-m", "init"), cwd=path, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(("git", "remote", "add", "origin", "https://github.com/hallovorld/test-repo"), cwd=path, check=True)
    return _git(path, "rev-parse", "--short=7", "HEAD")


def _write_lock(path: Path, repo_path: Path, commit: str) -> Path:
    lock = {
        "subrepos": [
            {
                "name": "test-repo",
                "local_path": str(repo_path),
                "remote": "https://github.com/hallovorld/test-repo",
                "branch": "main",
                "commit": commit,
            }
        ]
    }
    lock_path = path / "subrepos.lock.json"
    lock_path.write_text(json.dumps(lock))
    return lock_path


def test_pin_guard_accepts_matching_clean_repo(tmp_path):
    module = _load_module()
    repo = tmp_path / "test-repo"
    commit = _make_repo(repo)
    lock_path = _write_lock(tmp_path, repo, commit)

    roots, issues = module.resolve_subrepo_src_roots(
        lock_file=lock_path,
        names=["test-repo"],
        siblings=tmp_path,
    )

    assert roots == [repo / "src"]
    assert issues == []


def test_pin_guard_reports_dirty_repo(tmp_path):
    module = _load_module()
    repo = tmp_path / "test-repo"
    commit = _make_repo(repo)
    lock_path = _write_lock(tmp_path, repo, commit)
    (repo / "src" / "pkg.py").write_text("VALUE = 2\n")

    _, issues = module.resolve_subrepo_src_roots(
        lock_file=lock_path,
        names=["test-repo"],
        siblings=tmp_path,
    )

    assert [issue.reason for issue in issues] == ["working tree is dirty"]
