"""Regression tests for the read-only repo hygiene audit."""
from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "audit_repo_hygiene.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_repo_hygiene_under_test", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_classify_local_runtime_and_agent_state() -> None:
    mod = _load_module()

    assert mod.classify_path(".tmp_dagster_home_x/runs.db", "??") == "local_runtime_scratch"
    assert mod.classify_path(".claude/settings.local.json", "M") == "local_agent_settings"
    assert mod.classify_path("data/runs.alpaca.db", "??") == "local_runtime_state"
    assert mod.classify_path("foo.disabled-20260524.json", "??") == "backup_or_disabled_copy"


def test_classify_artifacts_by_operational_risk() -> None:
    mod = _load_module()

    assert (
        mod.classify_path(
            "backtesting/renquant_104/models/AAPL/AAPL-policy-metadata.json",
            "M",
        )
        == "per_ticker_model_artifact"
    )
    assert (
        mod.classify_path(
            "backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json",
            "??",
        )
        == "production_model_artifact"
    )
    assert (
        mod.classify_path(
            "backtesting/renquant_104/artifacts/shadow/panel-rank-calibration.foo.json",
            "??",
        )
        == "shadow_model_artifact"
    )
    assert (
        mod.classify_path(
            "backtesting/renquant_104/artifacts/sim/walkforward_manifest.json",
            "??",
        )
        == "sim_model_artifact"
    )
    assert (
        mod.classify_path("artifacts/wf_trade_forensics_current_contract.md", "??")
        == "experiment_or_diagnostic_artifact"
    )


def test_classify_code_docs_and_config() -> None:
    mod = _load_module()

    assert mod.classify_path("scripts/run_wf_gate.py", "M") == "code"
    assert mod.classify_path("tests/test_repo_hygiene_audit.py", "??") == "code"
    assert mod.classify_path("rust/Cargo.lock", "??") == "code"
    assert mod.classify_path("doc/dashboard.md", "M") == "documentation"
    assert (
        mod.classify_path("backtesting/renquant_104/strategy_config.json", "M")
        == "strategy_config"
    )


def test_report_policy_is_inventory_only(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(
        mod,
        "_git_status",
        lambda: [
            ("M", "scripts/run_wf_gate.py"),
            ("??", "artifacts/wf_trade_forensics_current_contract.md"),
        ],
    )

    report = mod.build_report()

    assert report["total_dirty_entries"] == 2
    assert report["counts"] == {"code": 1, "experiment_or_diagnostic_artifact": 1}
    assert report["policy"] == {
        "delete_files": False,
        "default_action": "inventory_only",
        "archive_requires_review": True,
    }
