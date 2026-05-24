from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from kernel.pipeline.pipeline import ParallelTimeoutError  # noqa: E402
from training_panel import pp_panel_training  # noqa: E402
from training_panel.context import TickerPanelContext  # noqa: E402
from training_panel.pp_panel_training import run_panel_ticker_parallel  # noqa: E402


def _tc(ticker: str, config: dict | None = None) -> TickerPanelContext:
    return TickerPanelContext(
        ticker=ticker,
        ohlcv={},
        sector_momentum={},
        ticker_sectors={},
        config=config or {},
    )


def test_panel_run_ticker_parallel_timeout_is_hard_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ctxs = [_tc("FAST"), _tc("SLOW")]

    def fake_chain(tc: TickerPanelContext) -> None:
        time.sleep(0.20 if tc.ticker == "SLOW" else 0.0)
        tc.feature_frame = object()

    monkeypatch.setattr(pp_panel_training, "_run_panel_ticker_chain", fake_chain)
    caplog.set_level(logging.INFO, logger="training_panel.pipeline")

    with pytest.raises(ParallelTimeoutError) as excinfo:
        run_panel_ticker_parallel(
            ctxs,
            max_workers=2,
            timeout_seconds=0.03,
            progress_log_seconds=0.01,
        )

    assert excinfo.value.job_name == "PanelTickerChain"
    assert "SLOW" in excinfo.value.pending_tickers
    assert any(
        "TIMEOUT" in r.message and "worker may still be running" in r.message
        for r in caplog.records
    )


def test_panel_run_ticker_parallel_progress_logs_pending_tickers(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ctxs = [_tc("FAST"), _tc("SLOW")]

    def fake_chain(tc: TickerPanelContext) -> None:
        time.sleep(0.08 if tc.ticker == "SLOW" else 0.0)
        tc.feature_frame = object()

    monkeypatch.setattr(pp_panel_training, "_run_panel_ticker_chain", fake_chain)
    caplog.set_level(logging.INFO, logger="training_panel.pipeline")

    run_panel_ticker_parallel(
        ctxs,
        max_workers=2,
        timeout_seconds=1.0,
        progress_log_seconds=0.01,
    )

    assert any(
        "PanelTickerChain progress" in r.message
        and "done=1/2" in r.message
        and "SLOW" in r.message
        for r in caplog.records
    )


def test_panel_run_ticker_parallel_config_timeout_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = {
        "parallel_ticker_timeout_seconds": 0.03,
        "parallel_progress_log_seconds": 0.01,
    }
    ctxs = [_tc("SLOW", cfg)]

    def fake_chain(tc: TickerPanelContext) -> None:
        time.sleep(0.20)
        tc.feature_frame = object()

    monkeypatch.setattr(pp_panel_training, "_run_panel_ticker_chain", fake_chain)

    with pytest.raises(ParallelTimeoutError):
        run_panel_ticker_parallel(ctxs, max_workers=1)


def test_panel_run_ticker_parallel_success_preserves_context_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctxs = [_tc("A"), _tc("B")]

    def fake_chain(tc: TickerPanelContext) -> None:
        tc.feature_frame = {"ticker": tc.ticker}
        tc.neutralized_frame = {"ticker": tc.ticker, "neutralized": True}

    monkeypatch.setattr(pp_panel_training, "_run_panel_ticker_chain", fake_chain)

    run_panel_ticker_parallel(
        ctxs,
        max_workers=2,
        timeout_seconds=1.0,
        progress_log_seconds=0.01,
    )

    assert [tc.feature_frame for tc in ctxs] == [{"ticker": "A"}, {"ticker": "B"}]
    assert [tc.neutralized_frame for tc in ctxs] == [
        {"ticker": "A", "neutralized": True},
        {"ticker": "B", "neutralized": True},
    ]
