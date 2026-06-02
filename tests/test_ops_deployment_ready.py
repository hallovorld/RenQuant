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


def _patch_green_dependencies(
    monkeypatch,
    module,
    *,
    branch: str = "main",
    runtime_head: str | None = None,
    dirty: bool = False,
    dirty_status: str | None = None,
) -> None:
    def fake_git(repo_root: Path, *args: str) -> str:
        joined = " ".join(args)
        if "--abbrev-ref" in joined:
            return branch
        if args == ("log", "-1", "--format=%H"):
            return runtime_head or ("a" * 40)
        if "--verify origin/main" in joined:
            return "a" * 40
        if args == ("rev-parse", "HEAD"):
            return "a" * 40
        if args == ("status", "--porcelain"):
            if repo_root.name == "renquant-common":
                return ""
            if dirty_status is not None:
                return dirty_status
            return " M scripts/daily_104.sh" if dirty else ""
        raise AssertionError(args)

    monkeypatch.setattr(module, "_git", fake_git)
    monkeypatch.setattr(module, "run_contract", lambda: {"ok": True, "failures": []})
    monkeypatch.setattr(
        module,
        "inspect_launchagents",
        lambda **kwargs: {"ok": True, "issues": [], "entries": []},
    )


def _write_pinned_runtime(tmp_path: Path, *, commit: str = "a" * 40) -> Path:
    runtime = tmp_path / ".subrepo_runtime" / "repos"
    (runtime / "renquant-common").mkdir(parents=True)
    (tmp_path / "subrepos.lock.json").write_text(
        '{"subrepos":[{"name":"renquant-common","commit":"' + commit + '"}]}',
        encoding="utf-8",
    )
    env_dir = tmp_path / ".subrepo_assembly"
    env_dir.mkdir()
    (env_dir / "current.env").write_text(
        f"export RENQUANT_SUBREPO_ROOT={runtime}\n"
        "export RENQUANT_STRICT_SUBREPO_PATHS=1\n"
        "export RENQUANT_OPS_FAIL_CLOSED=1\n",
        encoding="utf-8",
    )
    return runtime


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


def test_git_preserves_porcelain_status_prefix(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()

    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda *args, **kwargs: " M backtesting/renquant_104/artifacts/shadow/result.json\n",
    )

    assert module._git(tmp_path, "status", "--porcelain").startswith(" M backtesting/")


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
    runtime = _write_pinned_runtime(tmp_path)

    result = module.run_readiness(
        repo_root=tmp_path,
        canonical_repo=tmp_path,
        allow_non_canonical=True,
    )

    assert result["ok"] is True
    assert result["details"]["subrepo_root"] == str(runtime)
    assert result["details"]["strict_subrepo_paths"] == "1"
    assert result["details"]["ops_fail_closed"] == "1"
    assert result["details"]["runtime_pins_ok"] is True
    assert result["runtime_pins"]["entries"][0]["name"] == "renquant-common"


def test_readiness_blocks_runtime_root_without_strict_env(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    _patch_green_dependencies(monkeypatch, module)
    runtime = _write_pinned_runtime(tmp_path)
    env_path = tmp_path / ".subrepo_assembly" / "current.env"
    env_path.write_text(
        f"export RENQUANT_SUBREPO_ROOT={runtime}\n",
        encoding="utf-8",
    )

    result = module.run_readiness(
        repo_root=tmp_path,
        canonical_repo=tmp_path,
        allow_non_canonical=True,
    )

    assert result["ok"] is False
    assert any(issue["check"] == "runtime_strict_env" for issue in result["issues"])


def test_readiness_warns_runtime_root_without_global_fail_closed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    _patch_green_dependencies(monkeypatch, module)
    runtime = _write_pinned_runtime(tmp_path)
    env_path = tmp_path / ".subrepo_assembly" / "current.env"
    env_path.write_text(
        f"export RENQUANT_SUBREPO_ROOT={runtime}\n"
        "export RENQUANT_STRICT_SUBREPO_PATHS=1\n",
        encoding="utf-8",
    )

    result = module.run_readiness(
        repo_root=tmp_path,
        canonical_repo=tmp_path,
        allow_non_canonical=True,
    )

    assert result["ok"] is True
    assert any(
        issue["severity"] == "warning" and issue["check"] == "runtime_fail_closed_env"
        for issue in result["issues"]
    )


def test_readiness_can_skip_launchagents_for_preinstall(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    _patch_green_dependencies(monkeypatch, module)
    _write_pinned_runtime(tmp_path)

    def fail_if_called(**kwargs):
        raise AssertionError("launchagent inspection must be skipped in pre-install mode")

    monkeypatch.setattr(module, "inspect_launchagents", fail_if_called)

    result = module.run_readiness(
        repo_root=tmp_path,
        canonical_repo=tmp_path,
        allow_non_canonical=True,
        skip_launchagents=True,
    )

    assert result["ok"] is True
    assert result["details"]["launchagents_ok"] is None
    assert result["launchagents"]["skipped"] is True


def test_readiness_still_blocks_launchagent_drift_by_default(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    _patch_green_dependencies(monkeypatch, module)
    _write_pinned_runtime(tmp_path)
    monkeypatch.setattr(
        module,
        "inspect_launchagents",
        lambda **kwargs: {"ok": False, "issues": [{"reason": "drift"}], "entries": []},
    )

    result = module.run_readiness(
        repo_root=tmp_path,
        canonical_repo=tmp_path,
        allow_non_canonical=True,
    )

    assert result["ok"] is False
    assert any(issue["check"] == "launchagents" for issue in result["issues"])


def test_readiness_blocks_stale_runtime_pin(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    _patch_green_dependencies(monkeypatch, module, runtime_head="b" * 40)
    _write_pinned_runtime(tmp_path, commit="a" * 40)

    result = module.run_readiness(
        repo_root=tmp_path,
        canonical_repo=tmp_path,
        allow_non_canonical=True,
    )

    assert result["ok"] is False
    assert result["details"]["runtime_pins_ok"] is False
    assert any(issue["check"] == "runtime_pins" for issue in result["issues"])
    assert result["runtime_pins"]["failures"][0]["name"] == "renquant-common"


def test_readiness_blocks_dirty_worktree(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    _patch_green_dependencies(monkeypatch, module, dirty=True)
    _write_pinned_runtime(tmp_path)

    result = module.run_readiness(
        repo_root=tmp_path,
        canonical_repo=tmp_path,
        allow_non_canonical=True,
    )

    assert result["ok"] is False
    assert result["details"]["dirty"] is True
    assert any(issue["check"] == "git_dirty" for issue in result["issues"])


def test_readiness_allows_runtime_dirty_outputs(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    _patch_green_dependencies(
        monkeypatch,
        module,
        dirty_status=(
            " M backtesting/renquant_104/live_state.alpaca.json\n"
            " M backtesting/renquant_104/artifacts/shadow/panel-rank-calibration.json\n"
            "?? backtesting/renquant_104/artifacts/cache/tmp.json\n"
            " M doc/dashboard.md"
        ),
    )
    _write_pinned_runtime(tmp_path)

    result = module.run_readiness(
        repo_root=tmp_path,
        canonical_repo=tmp_path,
        allow_non_canonical=True,
    )

    assert result["ok"] is True
    assert result["details"]["dirty"] is True
    assert result["details"]["blocking_dirty_paths"] == []
    assert "backtesting/renquant_104/live_state.alpaca.json" in result["details"]["runtime_dirty_paths"]
    assert not any(issue["check"] == "git_dirty" for issue in result["issues"])


def test_readiness_blocks_non_main_branch(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    _patch_green_dependencies(monkeypatch, module, branch="feature")
    _write_pinned_runtime(tmp_path)

    result = module.run_readiness(repo_root=tmp_path, canonical_repo=tmp_path)

    assert result["ok"] is False
    assert any(issue["check"] == "git_branch" for issue in result["issues"])
