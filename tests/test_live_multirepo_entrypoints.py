"""Regression guards for scheduled live.runner multirepo entry points."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ORCH_BRIDGE = (
    REPO.parent
    / "renquant-orchestrator"
    / "src"
    / "renquant_orchestrator"
    / "live_bridge.py"
)


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
    assert 'ORCH_SRC = SIBLINGS / "renquant-orchestrator" / "src"' in src
    assert "from renquant_orchestrator import live_bridge as _bridge" in src
    assert "_PIN_SRCS = list(_bridge.DEFAULT_PIN_SRCS)" in src
    assert "return _bridge._with_pinned_strategy_config(argv, repo_root=REPO)" in src
    assert "return _bridge.bootstrap_multirepo(" in src
    assert f'return _bridge.main(mode="{mode}", repo_root=REPO)' in src

    assert 'importlib.import_module("live.runner")' not in src
    assert "from subrepo_paths import resolve_subrepo_root" not in src
    assert "renquant_pipeline.kernel.panel_pipeline.job_panel_scoring" not in src


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
