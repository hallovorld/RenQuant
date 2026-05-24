from __future__ import annotations

import datetime
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.pipeline.context import TickerInferenceContext  # noqa: E402
from kernel.pipeline.task_candidates import BuildFeaturesTask, SectorMapGateTask  # noqa: E402


def _tc(ticker: str, *, config: dict | None = None) -> TickerInferenceContext:
    return TickerInferenceContext(
        ticker=ticker,
        ohlcv={},
        model=None,
        config=config or {},
        today=datetime.date(2026, 5, 23),
        regime="BULL_CALM",
        regime_params={},
        exit_params={},
    )


def test_sector_map_gate_stamps_blocked_reason():
    tc = _tc(
        "BAC",
        config={
            "benchmark": "SPY",
            "risk": {"require_sector_map_for_buys": True},
            "sector_map": {},
        },
    )

    assert SectorMapGateTask().run(tc) is False
    assert tc.blocked_by == "missing_sector_map"


def test_build_features_stamps_missing_input_reason():
    tc = _tc("AAPL", config={"indicator_spec": {}})

    assert BuildFeaturesTask().run(tc) is False
    assert tc.blocked_by == "missing_input:stock_ohlcv,model,spy_ohlcv"
