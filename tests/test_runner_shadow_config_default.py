"""Regression guards for readonly-Alpaca shadow config routing."""
from __future__ import annotations

from live import runner


def test_readonly_alpaca_104_defaults_to_shadow_config():
    assert runner._resolve_strategy_config_name(
        "renquant_104",
        "readonly-alpaca",
        None,
    ) == "strategy_config.shadow.json"


def test_readonly_alpaca_explicit_config_is_respected():
    assert runner._resolve_strategy_config_name(
        "renquant_104",
        "readonly-alpaca",
        "strategy_config.json",
    ) == "strategy_config.json"


def test_non_shadow_paths_default_to_production_config():
    assert runner._resolve_strategy_config_name(
        "renquant_104",
        "alpaca",
        None,
    ) == "strategy_config.json"
    assert runner._resolve_strategy_config_name(
        "renquant_103",
        "readonly-alpaca",
        None,
    ) == "strategy_config.json"


def test_runner_accepts_external_strategy_config_path():
    src = (runner.REPO_ROOT / "live" / "runner.py").read_text()
    assert "--strategy-config-path" in src
    assert "config[\"_strategy_config_path\"]" in src
    assert "state, data, and artifact paths still resolve" in src
    assert "only the config" in src
