from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _load_module():
    path = REPO / "scripts" / "check_ops_deployment_ready.py"
    spec = importlib.util.spec_from_file_location("check_ops_deployment_ready", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _patch_green_dependencies(monkeypatch, module, *, branch: str = "main") -> None:
    def fake_git(repo_root: Path, *args: str) -> str:
        joined = " ".join(args)
        if "--abbrev-ref" in joined:
            return branch
        if "--verify origin/main" in joined:
            return "a" * 40
        if args == ("rev-parse", "HEAD"):
            return "a" * 40
        if args == ("status", "--porcelain"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(module, "_git", fake_git)
    monkeypatch.setattr(module, "run_contract", lambda: {"ok": True, "failures": []})
    monkeypatch.setattr(
        module,
        "inspect_launchagents",
        lambda **kwargs: {"ok": True, "issues": [], "entries": []},
    )


def test_read_exports_parses_current_env(tmp_path: Path) -> None:
    module = _load_module()
    env = tmp_path / "current.env"
    env.write_text(
        "# comment\n"
        "export RENQUANT_ASSEMBLY_DIR=/tmp/assembly\n"
        "export RENQUANT_SUBREPO_ROOT='/tmp/runtime/repos'\n",
        encoding="utf-8",
    )
    assert module._read_exports(env) == {
        "RENQUANT_ASSEMBLY_DIR": "/tmp/assembly",
        "RENQUANT_SUBREPO_ROOT": "/tmp/runtime/repos",
    }


def test_readiness_requires_runtime_root(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    _patch_green_dependencies(monkeypatch, module)

    result = module.run_readiness(
        repo_root=tmp_path,
        canonical_repo=tmp_path,
        allow_non_canonical=True,
    )

    assert result["ok"] is False
    assert any(issue["check"] == "runtime_root" for issue in result["issues"])


def test_readiness_passes_with_runtime_root_and_green_checks(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    _patch_green_dependencies(monkeypatch, module)
    runtime = tmp_path / ".subrepo_runtime" / "repos"
    runtime.mkdir(parents=True)
    env_dir = tmp_path / ".subrepo_assembly"
    env_dir.mkdir()
    (env_dir / "current.env").write_text(
        f"export RENQUANT_SUBREPO_ROOT={runtime}\n",
        encoding="utf-8",
    )

    result = module.run_readiness(
        repo_root=tmp_path,
        canonical_repo=tmp_path,
        allow_non_canonical=True,
    )

    assert result["ok"] is True
    assert result["details"]["subrepo_root"] == str(runtime)


def test_readiness_blocks_non_main_branch(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    _patch_green_dependencies(monkeypatch, module, branch="feature")
    runtime = tmp_path / ".subrepo_runtime" / "repos"
    runtime.mkdir(parents=True)
    env_dir = tmp_path / ".subrepo_assembly"
    env_dir.mkdir()
    (env_dir / "current.env").write_text(
        f"export RENQUANT_SUBREPO_ROOT={runtime}\n",
        encoding="utf-8",
    )

    result = module.run_readiness(repo_root=tmp_path, canonical_repo=tmp_path)

    assert result["ok"] is False
    assert any(issue["check"] == "git_branch" for issue in result["issues"])
