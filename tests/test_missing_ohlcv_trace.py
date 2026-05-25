"""Regression: missing OHLCV must not be reported as no model signal."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.decision_trace import build_ticker_daily_state_rows  # noqa: E402
from kernel.pipeline.pp_inference import _buy_universe, _mark_missing_buy_ohlcv  # noqa: E402


def test_missing_ohlcv_loaded_model_gets_precise_block_reason() -> None:
    ctx = SimpleNamespace(
        models={"AAA": object(), "BBB": object(), "HELD": object()},
        holdings={"HELD": object()},
        ohlcv={"BBB": object()},
        config={
            "watchlist": ["AAA", "BBB", "HELD"],
            "benchmark_sleeve": {"enabled": False},
        },
        pending_broker_tickers=set(),
        _blocked_by_ticker={},
        counters={},
        bear_only=False,
    )

    _mark_missing_buy_ohlcv(ctx)

    assert _buy_universe(ctx) == ["BBB"]
    assert ctx._blocked_by_ticker == {"AAA": "missing_ohlcv"}
    assert ctx.counters["missing_ohlcv"] == 1


def test_ticker_daily_state_preserves_missing_ohlcv_reason() -> None:
    ctx = SimpleNamespace(
        _full_candidate_snapshot=[],
        candidates=[],
        holdings={},
        prices={},
        regime="BULL_CALM",
        confidence=0.7,
        portfolio_value=100_000.0,
    )

    rows = build_ticker_daily_state_rows(
        config={"watchlist": ["AAA"], "benchmark_sleeve": {"enabled": False}},
        ctx=ctx,
        selected_tickers=set(),
        blocked_map={"AAA": "missing_ohlcv"},
        model_types={"AAA": "XGBoost"},
    )

    assert rows[0]["ticker"] == "AAA"
    assert rows[0]["blocked_by"] == "missing_ohlcv"
