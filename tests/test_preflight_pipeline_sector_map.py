"""Track H — paired tests for SectorMapCoverageTask asserting byte-equivalence
with legacy ``_check_sector_map_coverage`` on every documented branch.

Coverage:
  (a) panel-LTR disabled + require_sector_map_for_buys=False → soft pass
  (b) full coverage (every buyable has sector, every sector has ETF) → HARD pass
  (c) missing sector for some tickers (full run) → HARD fail
  (d) missing sector for some tickers (sell-only run) → soft pass with warning
  (e) sector present but no ETF mapping (full run) → HARD fail
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backtesting/renquant_104"))

from kernel.preflight import _check_sector_map_coverage
from kernel.preflight_pipeline import PreflightContext, SectorMapCoverageTask


def _ctx(config: dict, run_mode: str | None = None,
         tmp_path: Path | None = None) -> PreflightContext:
    return PreflightContext(
        config=config,
        strategy_dir=tmp_path or Path("."),
        run_mode=run_mode,
    )


def _base_config(panel_enabled: bool = True) -> dict:
    """Minimal config with panel-scoring enabled (which forces sector-map
    coverage required by default)."""
    return {
        "ranking": {"panel_scoring": {"enabled": panel_enabled}},
        "watchlist": ["AAPL", "MSFT", "NVDA", "SPY"],
        "benchmark": "SPY",
        "sector_map": {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "NVDA": "Technology",
        },
        "sector_etf_map": {"Technology": "XLK"},
    }


class TestSectorMapCoverageTaskParity:

    def test_panel_disabled_not_required_soft_pass(self, tmp_path):
        cfg = _base_config(panel_enabled=False)
        leg = _check_sector_map_coverage(config=cfg, strategy_dir=tmp_path)
        ctx = _ctx(cfg, tmp_path=tmp_path)
        SectorMapCoverageTask().run(ctx)
        new = ctx.results[-1]
        assert new.name == leg.name == "P-SECTOR-MAP"
        assert new.severity == leg.severity == "soft"
        assert new.ok is leg.ok is True
        assert new.message == leg.message

    def test_full_coverage_hard_pass(self, tmp_path):
        cfg = _base_config()
        leg = _check_sector_map_coverage(config=cfg, strategy_dir=tmp_path)
        ctx = _ctx(cfg, tmp_path=tmp_path)
        SectorMapCoverageTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is True
        assert new.message == leg.message
        assert new.details["watchlist_size"] == leg.details["watchlist_size"] == 4
        assert new.details["buyable_size"] == leg.details["buyable_size"] == 3
        assert new.details["missing_count"] == leg.details["missing_count"] == 0

    def test_missing_sector_full_run_hard_fail(self, tmp_path):
        cfg = _base_config()
        # Drop NVDA from sector_map
        cfg["sector_map"].pop("NVDA")
        leg = _check_sector_map_coverage(config=cfg, strategy_dir=tmp_path)
        ctx = _ctx(cfg, tmp_path=tmp_path)
        SectorMapCoverageTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message
        assert "NVDA" in new.message

    def test_missing_sector_sell_only_soft_pass(self, tmp_path):
        cfg = _base_config()
        cfg["sector_map"].pop("NVDA")
        leg = _check_sector_map_coverage(config=cfg, strategy_dir=tmp_path,
                                          run_mode="sell-only")
        ctx = _ctx(cfg, tmp_path=tmp_path, run_mode="sell-only")
        SectorMapCoverageTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "soft"
        assert new.ok is leg.ok is True
        assert new.message == leg.message

    def test_sector_without_etf_full_run_hard_fail(self, tmp_path):
        cfg = _base_config()
        # Add a ticker in a sector that has no ETF mapping
        cfg["watchlist"].append("JPM")
        cfg["sector_map"]["JPM"] = "Financials"
        # sector_etf_map only has "Technology"
        leg = _check_sector_map_coverage(config=cfg, strategy_dir=tmp_path)
        ctx = _ctx(cfg, tmp_path=tmp_path)
        SectorMapCoverageTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message
        assert "Financials" in new.message
