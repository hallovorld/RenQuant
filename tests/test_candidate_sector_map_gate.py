"""AUDIT REGRESSION GUARD: buy candidates require sector metadata.

The 2026-05-22 live override bought BAC/WFC while both were absent from
``sector_map``. That silently disabled relative-strength context and QP
sector caps for those names. A panel-scored strategy must fail closed here,
even when preflight strict mode is manually disabled.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from kernel.pipeline.context import TickerInferenceContext  # noqa: E402
from kernel.pipeline.task_candidates import SectorMapGateTask  # noqa: E402


def _tc(ticker: str, *, sector_map: dict | None):
    return TickerInferenceContext(
        ticker=ticker,
        ohlcv={},
        model={},
        config={
            "benchmark": "SPY",
            "ranking": {"panel_scoring": {"enabled": True}},
            "sector_map": sector_map or {},
        },
        today=dt.date(2026, 5, 22),
        regime="BULL_CALM",
        regime_params={},
        exit_params={},
    )


class TestSectorMapGateRegressionGuard:
    def test_blocks_missing_sector_when_panel_scoring_enabled(self):
        tc = _tc("BAC", sector_map={})

        out = SectorMapGateTask().run(tc)

        assert out is False

    def test_allows_ticker_with_sector(self):
        tc = _tc("BAC", sector_map={"BAC": "finance"})

        out = SectorMapGateTask().run(tc)

        assert out is None

    def test_allows_benchmark_symbol(self):
        tc = _tc("SPY", sector_map={})

        out = SectorMapGateTask().run(tc)

        assert out is None

    def test_preserves_non_panel_strategies_by_default(self):
        tc = _tc("BAC", sector_map={})
        tc.config["ranking"]["panel_scoring"]["enabled"] = False

        out = SectorMapGateTask().run(tc)

        assert out is None
