"""Regression guards for scheduled live.runner multirepo entry points."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_bridge_module():
    path = REPO / "scripts" / "live_multirepo.py"
    spec = importlib.util.spec_from_file_location("live_multirepo_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_daily_multirepo_keeps_standalone_umbrella_bridge() -> None:
    src = (REPO / "scripts" / "daily_multirepo.py").read_text()
    assert "def _bootstrap_multirepo()" in src
    assert 'importlib.import_module("live.runner")' in src
    assert "from live_multirepo import main" not in src
    assert "umbrella.job_panel_scoring" not in src
    assert 'importlib.import_module(\n            "renquant_pipeline.kernel.panel_pipeline.job_panel_scoring")' in src


def test_live_multirepo_resolves_subrepos_from_lock() -> None:
    src = (REPO / "scripts" / "live_multirepo.py").read_text()
    assert 'LOCK_FILE = REPO / "subrepos.lock.json"' in src
    assert "RENQUANT_SUBREPO_ROOT" in src
    assert "RENQUANT_STRICT_SUBREPO_PATHS" in src
    assert "renquant-orchestrator" not in src


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
    monkeypatch.delenv("RENQUANT_SUBREPO_ROOT", raising=False)

    roots, missing = mod._subrepo_src_roots()

    assert roots == [common / "src", pipeline / "src"]
    assert missing == []


def test_daily_full_run_keeps_existing_daily_multirepo_bridge() -> None:
    src = (REPO / "scripts" / "daily_104.sh").read_text()
    assert 'RQ_DAILY_RUNNER:-multirepo' in src
    assert 'RUNNER_ARGS=("$REPO_DIR/scripts/daily_multirepo.py")' in src
    assert "RUNNER_ARGS=(-m live.runner)" in src


def test_daily_shadow_run_uses_same_multirepo_bridge() -> None:
    src = (REPO / "scripts" / "daily_104.sh").read_text()
    shadow = src[src.find("Step 4: Shadow e2e"):]
    assert 'os.environ.get("RQ_DAILY_RUNNER", "multirepo")' in shadow
    assert 'runner = [sys.executable, "$REPO_DIR/scripts/live_multirepo.py"]' in shadow
    assert 'runner = [sys.executable, "-m", "live.runner"]' in shadow


def test_intraday_sell_only_defaults_to_shared_multirepo_bridge() -> None:
    src = (REPO / "scripts" / "intraday_sell_104.sh").read_text()
    assert 'RQ_DAILY_RUNNER:-multirepo' in src
    assert 'RUNNER_ARGS=("$REPO_DIR/scripts/live_multirepo.py")' in src
    assert '"${RUNNER_ARGS[@]}" --strategy renquant_104 --broker alpaca --once' in src
    assert "--sell-only --intraday" in src


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
