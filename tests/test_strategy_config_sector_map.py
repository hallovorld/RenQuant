"""Production config invariants for sector-aware live decisions.

Regression guard for 2026-05-22 live orders where BAC/WFC/D entered the
buy optimizer without sector metadata, bypassing relative-strength context
and QP sector caps.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"


SECTOR_CONFIGS = [
    "strategy_config.json",
    "strategy_config.golden.json",
    "strategy_config.shadow.json",
    "strategy_config.sim_patchtst_clean_20260522.json",
    "strategy_config.sim_xgb_truly_oos_20260522.json",
]


@pytest.mark.parametrize("name", SECTOR_CONFIGS)
def test_104_buyable_watchlist_has_sector_metadata(name):
    cfg = json.loads((STRATEGY_DIR / name).read_text())
    watchlist = cfg.get("watchlist") or []
    benchmark = cfg.get("benchmark", "SPY")
    sector_map = cfg.get("sector_map") or {}
    missing = sorted(
        ticker
        for ticker in watchlist
        if ticker != benchmark and not sector_map.get(ticker)
    )
    assert not missing, f"{name} missing sector_map entries: {missing}"


@pytest.mark.parametrize("name", SECTOR_CONFIGS)
def test_104_sector_metadata_has_sector_etf_mapping(name):
    cfg = json.loads((STRATEGY_DIR / name).read_text())
    sector_map = cfg.get("sector_map") or {}
    sector_etf_map = cfg.get("sector_etf_map") or {}
    sectors = sorted({sector for sector in sector_map.values() if sector})
    unmapped = [sector for sector in sectors if not sector_etf_map.get(sector)]
    assert not unmapped, f"{name} missing sector_etf_map entries: {unmapped}"


def test_104_active_and_golden_sector_metadata_match():
    active = json.loads((STRATEGY_DIR / "strategy_config.json").read_text())
    golden = json.loads((STRATEGY_DIR / "strategy_config.golden.json").read_text())
    assert active.get("risk", {}).get("require_sector_map_for_buys") is True
    assert golden.get("risk", {}).get("require_sector_map_for_buys") is True
    assert active.get("sector_map") == golden.get("sector_map")
    assert active.get("sector_etf_map") == golden.get("sector_etf_map")


@pytest.mark.parametrize(
    "name",
    ["strategy_config.shadow.json", "strategy_config.sim_patchtst_clean_20260522.json"],
)
def test_104_shadow_configs_require_sector_metadata(name):
    cfg = json.loads((STRATEGY_DIR / name).read_text())
    assert cfg.get("risk", {}).get("require_sector_map_for_buys") is True
