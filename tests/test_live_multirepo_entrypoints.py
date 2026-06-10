"""Regression guards for scheduled live.runner multirepo entry points."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _orchestrator_bridge_path() -> Path:
    lock = json.loads((REPO / "subrepos.lock.json").read_text())
    entry = next(e for e in lock["subrepos"] if e["name"] == "renquant-orchestrator")
    return Path(entry["local_path"]) / "src" / "renquant_orchestrator" / "live_bridge.py"


ORCH_BRIDGE = _orchestrator_bridge_path()


def _load_bridge_module():
    path = REPO / "scripts" / "live_multirepo.py"
    spec = importlib.util.spec_from_file_location("live_multirepo_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_daily_module():
    path = REPO / "scripts" / "daily_multirepo.py"
    spec = importlib.util.spec_from_file_location("daily_multirepo_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_bootstrap_module():
    path = REPO / "scripts" / "orchestrator_bridge_bootstrap.py"
    spec = importlib.util.spec_from_file_location("orchestrator_bridge_bootstrap_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clear_orchestrator_modules() -> None:
    for name in list(sys.modules):
        if name == "renquant_orchestrator" or name.startswith("renquant_orchestrator."):
            sys.modules.pop(name, None)


def _write_fake_live_bridge(root: Path) -> Path:
    package = root / "renquant-orchestrator" / "src" / "renquant_orchestrator"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "live_bridge.py").write_text(
        textwrap.dedent(
            """
            DEFAULT_PIN_SRCS = []

            def _arg_value(argv, flag, default=None):
                return default

            def _without_arg(argv, flag):
                return argv

            def _strategy_config_name(argv):
                return "strategy_config.json"

            def _with_pinned_strategy_config(argv, *, repo_root):
                return argv

            def _subrepo_src_roots(**kwargs):
                return [], []

            def _force_alias(alias, target, aliased):
                aliased.append(alias)

            def bootstrap_multirepo(**kwargs):
                return []

            def main(*, mode, repo_root):
                return 0
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return package.parent


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(repo), *args), text=True).strip()


def _init_git_repo(path: Path, remote: str = "https://github.com/hallovorld/renquant-orchestrator") -> str:
    (path / "src").mkdir(parents=True)
    (path / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.check_call(("git", "init", str(path)), stdout=subprocess.DEVNULL)
    subprocess.check_call(("git", "-C", str(path), "remote", "add", "origin", remote))
    subprocess.check_call(("git", "-C", str(path), "add", "."))
    subprocess.check_call((
        "git", "-C", str(path),
        "-c", "user.name=Test",
        "-c", "user.email=test@example.com",
        "commit", "-m", "fixture",
    ), stdout=subprocess.DEVNULL)
    return _git(path, "rev-parse", "HEAD")


@pytest.mark.parametrize(
    ("script", "mode"),
    [
        ("live_multirepo.py", "live"),
        ("daily_multirepo.py", "daily"),
    ],
)
def test_multirepo_wrappers_delegate_to_orchestrator_bridge(script: str, mode: str) -> None:
    src = (REPO / "scripts" / script).read_text()
    assert "orchestrator-owned" in src
    assert "from orchestrator_bridge_bootstrap import resolve_orchestrator_src" in src
    assert "ORCH_SRC = resolve_orchestrator_src(REPO, SIBLINGS)" in src
    assert "from renquant_orchestrator import live_bridge as _bridge" in src
    assert "_PIN_SRCS = list(_bridge.DEFAULT_PIN_SRCS)" in src
    assert "return _bridge._with_pinned_strategy_config(argv, repo_root=REPO)" in src
    assert "return _bridge.bootstrap_multirepo(" in src
    assert f'return _bridge.main(mode="{mode}", repo_root=REPO)' in src

    assert 'importlib.import_module("live.runner")' not in src
    assert "from subrepo_paths import resolve_subrepo_root" not in src
    assert "renquant_pipeline.kernel.panel_pipeline.job_panel_scoring" not in src


def test_orchestrator_bridge_bootstrap_prefers_runtime_root(monkeypatch, tmp_path) -> None:
    mod = _load_bootstrap_module()
    repo = tmp_path / "RenQuant"
    siblings = tmp_path / "siblings"
    runtime = tmp_path / "runtime" / "repos"
    monkeypatch.setenv("RENQUANT_SUBREPO_ROOT", str(runtime))

    assert mod.resolve_orchestrator_src(repo, siblings) == (
        runtime / "renquant-orchestrator" / "src"
    )


def test_orchestrator_bridge_bootstrap_reads_assembly_dir_env(monkeypatch, tmp_path) -> None:
    mod = _load_bootstrap_module()
    repo = tmp_path / "RenQuant"
    siblings = tmp_path / "siblings"
    assembly = tmp_path / "assembly"
    monkeypatch.delenv("RENQUANT_SUBREPO_ROOT", raising=False)
    monkeypatch.setenv("RENQUANT_ASSEMBLY_DIR", str(assembly))

    assert mod.resolve_orchestrator_src(repo, siblings) == (
        assembly / "repos" / "renquant-orchestrator" / "src"
    )


def test_orchestrator_bridge_bootstrap_reads_current_env(monkeypatch, tmp_path) -> None:
    mod = _load_bootstrap_module()
    repo = tmp_path / "RenQuant"
    siblings = tmp_path / "siblings"
    env_dir = repo / ".subrepo_assembly"
    env_dir.mkdir(parents=True)
    (env_dir / "current.env").write_text(
        "export RENQUANT_SUBREPO_ROOT=.subrepo_runtime/repos\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("RENQUANT_SUBREPO_ROOT", raising=False)
    monkeypatch.delenv("RENQUANT_SUBREPO_ENV", raising=False)

    assert mod.resolve_orchestrator_src(repo, siblings) == (
        repo / ".subrepo_runtime" / "repos" / "renquant-orchestrator" / "src"
    )


def test_orchestrator_bridge_bootstrap_reads_current_json(monkeypatch, tmp_path) -> None:
    mod = _load_bootstrap_module()
    repo = tmp_path / "RenQuant"
    siblings = tmp_path / "siblings"
    assembly = tmp_path / "assembly"
    (assembly / "repos").mkdir(parents=True)
    (repo / ".subrepo_assembly").mkdir(parents=True)
    (repo / ".subrepo_assembly" / "current.json").write_text(
        json.dumps({"current": str(assembly)}),
        encoding="utf-8",
    )
    monkeypatch.delenv("RENQUANT_SUBREPO_ROOT", raising=False)
    monkeypatch.delenv("RENQUANT_ASSEMBLY_DIR", raising=False)
    monkeypatch.delenv("RENQUANT_SUBREPO_ENV", raising=False)

    assert mod.resolve_orchestrator_src(repo, siblings) == (
        assembly / "repos" / "renquant-orchestrator" / "src"
    )


def test_orchestrator_bridge_bootstrap_falls_back_to_sibling(monkeypatch, tmp_path) -> None:
    mod = _load_bootstrap_module()
    repo = tmp_path / "RenQuant"
    siblings = tmp_path / "siblings"
    monkeypatch.delenv("RENQUANT_SUBREPO_ROOT", raising=False)
    monkeypatch.delenv("RENQUANT_SUBREPO_ENV", raising=False)

    assert mod.resolve_orchestrator_src(repo, siblings) == (
        siblings / "renquant-orchestrator" / "src"
    )


def test_orchestrator_bridge_bootstrap_uses_lock_local_path(monkeypatch, tmp_path) -> None:
    mod = _load_bootstrap_module()
    repo = tmp_path / "RenQuant"
    siblings = tmp_path / "siblings"
    orch = tmp_path / "locked-orchestrator"
    (orch / "src").mkdir(parents=True)
    repo.mkdir()
    (repo / "subrepos.lock.json").write_text(
        json.dumps({
            "subrepos": [
                {"name": "renquant-orchestrator", "local_path": str(orch)},
            ],
        }),
        encoding="utf-8",
    )
    monkeypatch.delenv("RENQUANT_SUBREPO_ROOT", raising=False)
    monkeypatch.delenv("RENQUANT_SUBREPO_ENV", raising=False)

    assert mod.resolve_orchestrator_src(repo, siblings) == orch / "src"


def test_orchestrator_bridge_bootstrap_validates_lock_local_path_in_strict_mode(
    monkeypatch, tmp_path,
) -> None:
    mod = _load_bootstrap_module()
    repo = tmp_path / "RenQuant"
    siblings = tmp_path / "siblings"
    orch = tmp_path / "locked-orchestrator"
    head = _init_git_repo(orch)
    repo.mkdir()
    (repo / "subrepos.lock.json").write_text(
        json.dumps({
            "subrepos": [{
                "name": "renquant-orchestrator",
                "local_path": str(orch),
                "commit": head,
                "remote": "https://github.com/hallovorld/renquant-orchestrator",
            }],
        }),
        encoding="utf-8",
    )
    monkeypatch.delenv("RENQUANT_SUBREPO_ROOT", raising=False)
    monkeypatch.delenv("RENQUANT_ASSEMBLY_DIR", raising=False)
    monkeypatch.delenv("RENQUANT_SUBREPO_ENV", raising=False)
    monkeypatch.setenv("RENQUANT_STRICT_SUBREPO_PATHS", "1")

    assert mod.resolve_orchestrator_src(repo, siblings) == orch / "src"


def test_orchestrator_bridge_bootstrap_blocks_stale_lock_local_path_in_strict_mode(
    monkeypatch, tmp_path,
) -> None:
    mod = _load_bootstrap_module()
    repo = tmp_path / "RenQuant"
    siblings = tmp_path / "siblings"
    orch = tmp_path / "locked-orchestrator"
    _init_git_repo(orch)
    repo.mkdir()
    (repo / "subrepos.lock.json").write_text(
        json.dumps({
            "subrepos": [{
                "name": "renquant-orchestrator",
                "local_path": str(orch),
                "commit": "deadbeef",
                "remote": "https://github.com/hallovorld/renquant-orchestrator",
            }],
        }),
        encoding="utf-8",
    )
    monkeypatch.delenv("RENQUANT_SUBREPO_ROOT", raising=False)
    monkeypatch.delenv("RENQUANT_ASSEMBLY_DIR", raising=False)
    monkeypatch.delenv("RENQUANT_SUBREPO_ENV", raising=False)
    monkeypatch.setenv("RENQUANT_STRICT_SUBREPO_PATHS", "1")

    with pytest.raises(SystemExit, match="does not match lock commit"):
        mod.resolve_orchestrator_src(repo, siblings)


def test_orchestrator_bridge_bootstrap_blocks_sibling_fallback_in_strict_mode(
    monkeypatch, tmp_path,
) -> None:
    mod = _load_bootstrap_module()
    repo = tmp_path / "RenQuant"
    siblings = tmp_path / "siblings"
    repo.mkdir()
    (repo / "subrepos.lock.json").write_text(json.dumps({"subrepos": []}), encoding="utf-8")
    monkeypatch.delenv("RENQUANT_SUBREPO_ROOT", raising=False)
    monkeypatch.delenv("RENQUANT_ASSEMBLY_DIR", raising=False)
    monkeypatch.delenv("RENQUANT_SUBREPO_ENV", raising=False)
    monkeypatch.setenv("RENQUANT_OPS_FAIL_CLOSED", "1")

    with pytest.raises(SystemExit, match="requires a pinned renquant-orchestrator"):
        mod.resolve_orchestrator_src(repo, siblings)


def test_live_multirepo_imports_orchestrator_bridge_from_runtime_root(
    monkeypatch, tmp_path,
) -> None:
    runtime = tmp_path / "runtime" / "repos"
    fake_src = _write_fake_live_bridge(runtime)
    monkeypatch.setenv("RENQUANT_SUBREPO_ROOT", str(runtime))
    _clear_orchestrator_modules()
    try:
        mod = _load_bridge_module()
        assert Path(mod._bridge.__file__).resolve() == (
            fake_src / "renquant_orchestrator" / "live_bridge.py"
        ).resolve()
    finally:
        _clear_orchestrator_modules()
        while str(fake_src) in sys.path:
            sys.path.remove(str(fake_src))


@pytest.mark.parametrize(
    ("loader", "mode"),
    [
        (_load_bridge_module, "live"),
        (_load_daily_module, "daily"),
    ],
)
def test_multirepo_wrapper_main_forwards_mode_and_repo(monkeypatch, loader, mode: str) -> None:
    mod = loader()
    calls = []

    def fake_main(*, mode: str, repo_root: Path) -> int:
        calls.append((mode, repo_root))
        return 17

    monkeypatch.setattr(mod._bridge, "main", fake_main)

    assert mod.main() == 17
    assert calls == [(mode, REPO)]


def test_orchestrator_live_bridge_owns_bootstrap_and_strategy_config_logic() -> None:
    src = ORCH_BRIDGE.read_text()
    assert "def _with_pinned_strategy_config(" in src
    assert "--strategy-config-path" in src
    assert "renquant-strategy-104" in src
    assert "configs" in src
    assert "def bootstrap_multirepo(" in src
    assert 'importlib.import_module("live.runner")' in src
    assert "critical multirepo module unavailable" in src
    assert "Critical production modules must not silently fall back" in src
    assert '"renquant_pipeline.kernel.preflight"' in src
    assert '"renquant_pipeline.kernel.panel_pipeline"' in src
    assert '"renquant_backtesting.meta_label"' in src


def test_live_multirepo_uses_lock_local_paths(tmp_path, monkeypatch) -> None:
    common = tmp_path / "renquant-common"
    pipeline = tmp_path / "renquant-pipeline"
    (common / "src").mkdir(parents=True)
    (pipeline / "src").mkdir(parents=True)
    lock = tmp_path / "subrepos.lock.json"
    lock.write_text(json.dumps({
        "subrepos": [
            {"name": "renquant-common", "local_path": str(common)},
            {"name": "renquant-pipeline", "local_path": str(pipeline)},
        ],
    }))

    mod = _load_bridge_module()
    monkeypatch.setattr(mod, "_PIN_SRCS", ["renquant-common", "renquant-pipeline"])
    monkeypatch.setattr(mod, "LOCK_FILE", lock)
    monkeypatch.setattr(mod, "SIBLINGS", tmp_path / "unused")
    monkeypatch.setenv("RENQUANT_SUBREPO_ROOT", str(tmp_path / "unused-root"))

    roots, missing = mod._subrepo_src_roots()

    assert roots == [common / "src", pipeline / "src"]
    assert missing == []


def test_daily_full_run_defaults_to_orchestrator_daily_bridge() -> None:
    src = (REPO / "scripts" / "daily_104.sh").read_text()
    assert 'RQ_DAILY_RUNNER:-multirepo' in src
    assert 'source "$REPO_DIR/scripts/subrepo_env.sh"' in src
    assert 'renquant_load_subrepo_env "$REPO_DIR"' in src
    assert 'export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"' in src
    assert 'renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-orchestrator' in src
    assert "RENQUANT_OPS_FAIL_CLOSED" in src
    assert 'RUNNER_ARGS=(-m renquant_orchestrator daily-bridge --repo-dir "$REPO_DIR")' in src
    assert "RUNNER_ARGS=(-m live.runner)" in src


def test_live_multirepo_injects_pinned_prod_strategy_config(monkeypatch, tmp_path) -> None:
    mod = _load_bridge_module()
    monkeypatch.setenv("RENQUANT_SUBREPO_ROOT", str(tmp_path))

    argv = mod._with_pinned_strategy_config([
        "--strategy", "renquant_104",
        "--broker", "alpaca",
        "--once",
    ])

    assert "--strategy-config-path" in argv
    cfg = argv[argv.index("--strategy-config-path") + 1]
    assert cfg == str(tmp_path / "renquant-strategy-104" / "configs" / "strategy_config.json")


def test_live_multirepo_injects_pinned_shadow_strategy_config(monkeypatch, tmp_path) -> None:
    mod = _load_bridge_module()
    monkeypatch.setenv("RENQUANT_SUBREPO_ROOT", str(tmp_path))

    argv = mod._with_pinned_strategy_config([
        "--strategy", "renquant_104",
        "--broker", "readonly-alpaca",
        "--once",
    ])

    cfg = argv[argv.index("--strategy-config-path") + 1]
    assert cfg == str(tmp_path / "renquant-strategy-104" / "configs" / "strategy_config.shadow.json")


def test_live_multirepo_converts_explicit_config_name_to_pinned_path(monkeypatch, tmp_path) -> None:
    mod = _load_bridge_module()
    monkeypatch.setenv("RENQUANT_SUBREPO_ROOT", str(tmp_path))

    argv = mod._with_pinned_strategy_config([
        "--strategy", "renquant_104",
        "--strategy-config-name", "strategy_config.golden.json",
    ])

    assert "--strategy-config-name" not in argv
    cfg = argv[argv.index("--strategy-config-path") + 1]
    assert cfg == str(tmp_path / "renquant-strategy-104" / "configs" / "strategy_config.golden.json")


def test_strategy_subrepo_configs_are_available_from_lock_pin() -> None:
    lock = json.loads((REPO / "subrepos.lock.json").read_text())
    strategy_entry = next(
        entry for entry in lock["subrepos"]
        if entry["name"] == "renquant-strategy-104"
    )
    repo = Path(strategy_entry["local_path"])
    commit = strategy_entry["commit"]

    for name in (
        "strategy_config.json",
        "strategy_config.golden.json",
        "strategy_config.shadow.json",
    ):
        pinned_raw = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{commit}:configs/{name}"],
            text=True,
        )
        pinned = json.loads(pinned_raw)
        assert isinstance(pinned.get("watchlist"), list)
        assert isinstance(pinned.get("ranking"), dict)


def _locked_subrepo_source(repo_name: str, path: str) -> str:
    lock = json.loads((REPO / "subrepos.lock.json").read_text())
    entry = next(item for item in lock["subrepos"] if item["name"] == repo_name)
    return subprocess.check_output(
        ["git", "-C", entry["local_path"], "show", f"{entry['commit']}:{path}"],
        text=True,
    )


def test_pinned_orchestrator_rehearsal_plan_threads_native_commit_plan_output() -> None:
    src = _locked_subrepo_source(
        "renquant-orchestrator",
        "src/renquant_orchestrator/live_rehearsal_plan.py",
    )

    assert "native_live_run_candidate" in src
    assert "--commit-plan-output-json" in src
    assert "native_commit_plan" in src
    assert 'f"{mode}-native-commit-plan.json"' in src


def test_pinned_orchestrator_native_live_run_writes_commit_plan_output() -> None:
    src = _locked_subrepo_source(
        "renquant-orchestrator",
        "src/renquant_orchestrator/native_live_run.py",
    )

    assert "commit_plan_output_json" in src
    assert "build_live_commit_plan" in src
    assert "--commit-plan-output-json" in src


def test_pinned_execution_live_commit_plan_preserves_ordering_contract() -> None:
    src = _locked_subrepo_source(
        "renquant-execution",
        "src/renquant_execution/live_commit.py",
    )

    assert "def _intent_priority" in src
    assert "sorted(order_intents, key=_intent_priority)" in src
    assert 'execution_payload["execution_audit"]' in src
    assert 'if "state_mutations" in execution_payload' in src


def test_pinned_pipeline_live_context_snapshot_normalizes_holding_aliases() -> None:
    src = _locked_subrepo_source(
        "renquant-pipeline",
        "src/renquant_pipeline/inference.py",
    )

    assert "class LiveContextSnapshot" in src
    assert "def live_context_snapshot_from_live_context" in src
    assert 'row.pop("qty", None)' in src
    assert 'row.pop("shares", None)' in src


def test_daily_shadow_run_uses_same_multirepo_bridge() -> None:
    src = (REPO / "scripts" / "daily_104.sh").read_text()
    shadow = src[src.find("Step 4: Shadow e2e"):]
    assert 'os.environ.get("RQ_DAILY_RUNNER", "multirepo")' in shadow
    assert (
        'runner = [sys.executable, "-m", "renquant_orchestrator", '
        '"live-bridge", "--repo-dir", "$REPO_DIR"]'
    ) in shadow
    assert 'runner = [sys.executable, "-m", "live.runner"]' in shadow


def test_intraday_sell_only_defaults_to_orchestrator_live_bridge() -> None:
    src = (REPO / "scripts" / "intraday_sell_104.sh").read_text()
    assert 'RQ_DAILY_RUNNER:-multirepo' in src
    assert 'source "$REPO_DIR/scripts/subrepo_env.sh"' in src
    assert 'renquant_load_subrepo_env "$REPO_DIR"' in src
    assert 'export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"' in src
    assert 'renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-orchestrator' in src
    assert 'RUNNER_ARGS=(-m renquant_orchestrator live-bridge --repo-dir "$REPO_DIR")' in src
    assert '"${RUNNER_ARGS[@]}" --strategy renquant_104 --broker alpaca --once' in src
    assert "--sell-only --intraday" in src


def test_live_only_104_delegates_to_intraday_wrapper() -> None:
    src = (REPO / "scripts" / "live_only_104.sh").read_text()
    assert "compatibility entrypoint" in src
    assert 'exec bash "$REPO_DIR/scripts/intraday_sell_104.sh"' in src
    assert "-m live.runner" not in src
    assert "--broker alpaca" not in src
    assert "--no-sell-only" in src


def test_live_multirepo_aliases_critical_lifted_modules() -> None:
    mod = _load_bridge_module()
    mod._bootstrap_multirepo()

    preflight = sys.modules.get("kernel.preflight")
    panel_pipeline = sys.modules.get("kernel.panel_pipeline")
    panel_scoring = sys.modules.get("renquant_pipeline.panel_scoring")

    assert preflight is not None
    assert preflight.__name__ == "renquant_pipeline.kernel.preflight"
    assert panel_pipeline is not None
    assert panel_pipeline.__name__ == "renquant_pipeline.kernel.panel_pipeline"
    assert panel_scoring is not None
    assert panel_scoring.__name__ == (
        "renquant_pipeline.kernel.panel_pipeline.job_panel_scoring"
    )


@pytest.mark.parametrize("loader", [_load_bridge_module, _load_daily_module])
def test_multirepo_force_alias_fails_closed_on_critical_import(monkeypatch, loader) -> None:
    mod = loader()

    def fake_import(name: str):
        raise ImportError(f"blocked {name}")

    monkeypatch.setattr(mod.importlib, "import_module", fake_import)
    with pytest.raises(RuntimeError, match="critical multirepo module unavailable"):
        mod._force_alias(
            "kernel.preflight",
            "renquant_pipeline.kernel.preflight",
            [],
        )


def test_orchestrator_bridge_removed_critical_umbrella_fallbacks() -> None:
    bridge_src = ORCH_BRIDGE.read_text()
    assert "renquant_pipeline.kernel.meta_label<-umbrella" not in bridge_src
    assert "renquant_pipeline.kernel.meta_label←umbrella" not in bridge_src
    assert "Critical production modules must not silently fall back" in bridge_src
    for script in ("daily_multirepo.py", "live_multirepo.py"):
        src = (REPO / "scripts" / script).read_text()
        assert "renquant_pipeline.kernel.meta_label<-umbrella" not in src
        assert "renquant_pipeline.kernel.meta_label←umbrella" not in src
        assert "Critical production modules must not silently fall back" not in src
