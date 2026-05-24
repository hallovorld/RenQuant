from __future__ import annotations

import datetime
import logging
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.pipeline.context import TickerInferenceContext  # noqa: E402
from kernel.pipeline.pipeline import ParallelTimeoutError, TickerJob, run_parallel  # noqa: E402


def _tc(ticker: str, config: dict | None = None) -> TickerInferenceContext:
    return TickerInferenceContext(
        ticker=ticker,
        ohlcv={},
        model={},
        config=config or {},
        today=datetime.date(2026, 5, 23),
        regime="BULL_CALM",
        regime_params={},
        exit_params={},
    )


class _SleepJob(TickerJob):
    def __init__(self, delays: dict[str, float]) -> None:
        self.delays = delays

    def run(self, tc: TickerInferenceContext) -> None:
        time.sleep(self.delays.get(tc.ticker, 0.0))
        tc.candidate = {"ticker": tc.ticker}


def test_run_parallel_timeout_is_hard_failure_not_silent(caplog: pytest.LogCaptureFixture):
    ctxs = [_tc("FAST"), _tc("SLOW")]
    caplog.set_level(logging.INFO, logger="kernel.pipeline")

    with pytest.raises(ParallelTimeoutError) as excinfo:
        run_parallel(
            ctxs,
            _SleepJob({"FAST": 0.0, "SLOW": 0.20}),
            max_workers=2,
            timeout_seconds=0.03,
            progress_log_seconds=0.01,
        )

    assert excinfo.value.job_name == "_SleepJob"
    assert "SLOW" in excinfo.value.pending_tickers
    assert any("TIMEOUT" in r.message and "worker may still be running" in r.message
               for r in caplog.records)


def test_run_parallel_progress_logs_pending_tickers(caplog: pytest.LogCaptureFixture):
    ctxs = [_tc("FAST"), _tc("SLOW")]
    caplog.set_level(logging.INFO, logger="kernel.pipeline")

    run_parallel(
        ctxs,
        _SleepJob({"FAST": 0.0, "SLOW": 0.08}),
        max_workers=2,
        timeout_seconds=1.0,
        progress_log_seconds=0.01,
    )

    assert any(
        "_SleepJob progress" in r.message
        and "done=1/2" in r.message
        and "SLOW" in r.message
        for r in caplog.records
    )


def test_run_parallel_config_timeout_is_used():
    cfg = {
        "parallel_ticker_timeout_seconds": 0.03,
        "parallel_progress_log_seconds": 0.01,
    }
    ctxs = [_tc("SLOW", cfg)]

    with pytest.raises(ParallelTimeoutError):
        run_parallel(ctxs, _SleepJob({"SLOW": 0.20}), max_workers=1)


def test_run_parallel_success_preserves_context_mutations():
    ctxs = [_tc("A"), _tc("B")]

    run_parallel(
        ctxs,
        _SleepJob({"A": 0.0, "B": 0.0}),
        max_workers=2,
        timeout_seconds=1.0,
        progress_log_seconds=0.01,
    )

    assert [tc.candidate for tc in ctxs] == [{"ticker": "A"}, {"ticker": "B"}]
