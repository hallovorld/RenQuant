from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "subrepo_assemble.py"
    spec = importlib.util.spec_from_file_location("subrepo_assemble", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(repo), *args), text=True).strip()


def test_runtime_root_output_is_gitignored() -> None:
    repo = Path(__file__).resolve().parents[1]
    subprocess.run(
        ("git", "check-ignore", ".subrepo_runtime/repos/renquant-common/.git"),
        cwd=repo,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _make_remote(tmp_path: Path, name: str) -> tuple[Path, str]:
    src = tmp_path / f"{name}-src"
    (src / "src").mkdir(parents=True)
    (src / "src" / "pkg.py").write_text("VALUE = 1\n")
    subprocess.run(("git", "init", "-b", "main"), cwd=src, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(("git", "config", "user.email", "test@example.com"), cwd=src, check=True)
    subprocess.run(("git", "config", "user.name", "Test"), cwd=src, check=True)
    subprocess.run(("git", "add", "."), cwd=src, check=True)
    subprocess.run(("git", "commit", "-m", "init"), cwd=src, check=True, stdout=subprocess.DEVNULL)
    commit = _git(src, "rev-parse", "HEAD")
    remote = tmp_path / f"{name}.git"
    subprocess.run(("git", "clone", "--bare", str(src), str(remote)), check=True, stdout=subprocess.DEVNULL)
    return remote, commit


def _lock(name: str, *, local_path: Path, remote: Path, commit: str) -> dict:
    return {
        "source_repo": {"name": "RenQuant", "never_delete": True},
        "subrepos": [
            {
                "name": name,
                "role": "test repo",
                "local_path": str(local_path),
                "remote": str(remote),
                "branch": "main",
                "commit": commit[:7],
                "test_command": "true",
                "status": "test",
            }
        ],
    }


def test_runtime_root_clones_pins_without_touching_dev_worktree(tmp_path):
    module = _load_module()
    name = "renquant-test"
    remote, commit = _make_remote(tmp_path, name)
    dev_path = tmp_path / "dev" / name
    dev_path.mkdir(parents=True)
    (dev_path / "UNTOUCHED").write_text("developer worktree marker\n")

    runtime_root = tmp_path / "runtime"
    assembly_root = tmp_path / "assembly"
    assembly = module.build_assembly(
        _lock(name, local_path=dev_path, remote=remote, commit=commit),
        sync=True,
        dry_run=False,
        runtime_root=runtime_root,
        assembly_root=assembly_root,
    )

    runtime_repo = runtime_root / name
    assert assembly is not None
    assert (runtime_repo / "src" / "pkg.py").read_text() == "VALUE = 1\n"
    assert _git(runtime_repo, "log", "-1", "--format=%H") == commit
    assert (dev_path / "UNTOUCHED").read_text() == "developer worktree marker\n"

    manifest = json.loads((assembly / "manifest.json").read_text())
    assert manifest["runtime_repo_root"] == str(runtime_root)
    assert manifest["repo_paths"] == {name: str(runtime_repo)}
    assert manifest["pythonpath"] == [str(runtime_repo / "src")]

    env = (assembly / "env.sh").read_text()
    assert f"export RENQUANT_SUBREPO_ROOT={runtime_root}" in env
    assert "export RENQUANT_STRICT_SUBREPO_PATHS=1" in env
    assert (assembly_root / "current.env").read_text() == env
    assert json.loads((assembly_root / "current.json").read_text()) == {
        "current": str(assembly),
        "runtime_repo_root": str(runtime_root),
    }


def test_runtime_root_missing_clone_requires_sync(tmp_path):
    module = _load_module()
    name = "renquant-test"
    remote, commit = _make_remote(tmp_path, name)

    try:
        module.build_assembly(
            _lock(name, local_path=tmp_path / "dev" / name, remote=remote, commit=commit),
            sync=False,
            dry_run=False,
            runtime_root=tmp_path / "runtime",
            assembly_root=tmp_path / "assembly",
        )
    except RuntimeError as exc:
        assert "missing" in str(exc)
        assert "rerun with --sync" in str(exc)
    else:
        raise AssertionError("expected missing runtime clone failure")
