from __future__ import annotations

import datetime
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.decision_trace import build_ticker_daily_state_rows  # noqa: E402
from kernel.exits import HoldingState  # noqa: E402
from kernel.selection import CandidateResult  # noqa: E402


def test_ticker_daily_state_rows_carry_score_horizons() -> None:
    cand = CandidateResult(
        ticker="AAA",
        raw_score=0.0,
        rank_score=0.62,
        rs_score=0.0,
        detail="",
        expected_return=0.04,
        expected_return_horizon_days=60,
        mu=0.04,
        mu_horizon_days=60,
    )
    holding = HoldingState(
        entry_price=100.0,
        entry_date=datetime.date(2026, 4, 1),
        high_watermark=105.0,
        rank_score=0.58,
        expected_return=0.03,
        expected_return_horizon_days=60,
        mu=0.03,
        mu_horizon_days=60,
    )
    ctx = SimpleNamespace(
        candidates=[cand],
        holdings={"BBB": holding},
        prices={"BBB": 105.0},
        portfolio_value=100_000.0,
        regime="BULL_CALM",
        confidence=0.7,
        _ticker_score_snapshot={},
    )

    rows = build_ticker_daily_state_rows(
        config={"watchlist": ["AAA", "BBB"]},
        ctx=ctx,
        selected_tickers=set(),
        blocked_map={},
        model_types={"AAA": "panel_ltr", "BBB": "panel_ltr"},
    )

    by_ticker = {r["ticker"]: r for r in rows}
    assert by_ticker["AAA"]["expected_return_horizon_days"] == 60
    assert by_ticker["AAA"]["mu_horizon_days"] == 60
    assert by_ticker["BBB"]["expected_return_horizon_days"] == 60
    assert by_ticker["BBB"]["mu_horizon_days"] == 60


def test_benchmark_sleeve_trace_row_is_not_marked_watchlist_member() -> None:
    ctx = SimpleNamespace(
        candidates=[],
        holdings={},
        prices={},
        portfolio_value=100_000.0,
        regime="BULL_CALM",
        confidence=0.7,
        _ticker_score_snapshot={},
    )

    rows = build_ticker_daily_state_rows(
        config={
            "watchlist": ["AAA"],
            "portfolio": {
                "benchmark_sleeve": {
                    "enabled": True,
                    "ticker": "SPY",
                    "exclude_from_alpha_pipeline": True,
                },
            },
        },
        ctx=ctx,
        selected_tickers=set(),
        blocked_map={},
        model_types={},
    )

    by_ticker = {r["ticker"]: r for r in rows}
    assert by_ticker["AAA"]["in_watchlist"] == 1
    assert by_ticker["SPY"]["in_watchlist"] == 0
