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


def test_pin_metadata_cache_hits_without_second_git_call(tmp_path, monkeypatch):
    module = _load_module()
    repo = tmp_path / "test-repo"
    _make_repo(repo)
    module.CACHE_PATH = tmp_path / "cache.json"

    expected = module._pin_metadata(repo)

    def fail_git(*_args, **_kwargs):
        raise AssertionError("cache miss")

    monkeypatch.setattr(module, "_git", fail_git)

    assert module._pin_metadata(repo) == expected


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
        check_dirty=True,
    )

    assert [issue.reason for issue in issues] == ["working tree is dirty"]


def test_pin_guard_skips_dirty_check_by_default(tmp_path):
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

    assert issues == []


def test_old_strict_subrepo_paths_env_still_fails_closed(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("RENQUANT_STRICT_SUBREPO_PATHS", "1")

    issue = module.PinIssue(
        repo="test-repo",
        path="/tmp/test-repo",
        reason="HEAD abc does not match lock commit def",
    )

    try:
        module.enforce_or_warn([issue])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit")


def test_global_ops_fail_closed_env_fails_pin_drift(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("RENQUANT_OPS_FAIL_CLOSED", "1")

    issue = module.PinIssue(
        repo="test-repo",
        path="/tmp/test-repo",
        reason="HEAD abc does not match lock commit def",
    )

    try:
        module.enforce_or_warn([issue])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit")


def test_dirty_requires_clean_strict_env(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("RENQUANT_STRICT_SUBREPO_PATHS", "1")
    issue = module.PinIssue(
        repo="test-repo",
        path="/tmp/test-repo",
        reason="working tree is dirty",
        kind="dirty",
    )

    module.enforce_or_warn([issue])

    monkeypatch.setenv("RENQUANT_STRICT_SUBREPO_CLEAN", "1")
    try:
        module.enforce_or_warn([issue])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit")
