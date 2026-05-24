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
from kernel.pipeline import pp_training  # noqa: E402
from kernel.pipeline.pp_training import TickerTrainingContext, run_ticker_parallel  # noqa: E402


def _tc(ticker: str, config: dict | None = None) -> TickerTrainingContext:
    return TickerTrainingContext(
        ticker=ticker,
        ohlcv={},
        config=config or {},
        strategy_dir=None,
    )


def test_training_run_ticker_parallel_timeout_is_hard_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ctxs = [_tc("FAST"), _tc("SLOW")]

    def fake_chain(tc: TickerTrainingContext) -> None:
        time.sleep(0.20 if tc.ticker == "SLOW" else 0.0)
        tc.exported = True

    monkeypatch.setattr(pp_training, "_run_ticker_chain", fake_chain)
    caplog.set_level(logging.INFO, logger="training.pipeline")

    with pytest.raises(ParallelTimeoutError) as excinfo:
        run_ticker_parallel(
            ctxs,
            max_workers=2,
            timeout_seconds=0.03,
            progress_log_seconds=0.01,
        )

    assert excinfo.value.job_name == "TickerTrainingChain"
    assert "SLOW" in excinfo.value.pending_tickers
    assert any("TIMEOUT" in r.message and "worker may still be running" in r.message
               for r in caplog.records)


def test_training_run_ticker_parallel_progress_logs_pending_tickers(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ctxs = [_tc("FAST"), _tc("SLOW")]

    def fake_chain(tc: TickerTrainingContext) -> None:
        time.sleep(0.08 if tc.ticker == "SLOW" else 0.0)
        tc.exported = True

    monkeypatch.setattr(pp_training, "_run_ticker_chain", fake_chain)
    caplog.set_level(logging.INFO, logger="training.pipeline")

    run_ticker_parallel(
        ctxs,
        max_workers=2,
        timeout_seconds=1.0,
        progress_log_seconds=0.01,
    )

    assert any(
        "TickerTrainingChain progress" in r.message
        and "done=1/2" in r.message
        and "SLOW" in r.message
        for r in caplog.records
    )


def test_training_run_ticker_parallel_config_timeout_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = {
        "parallel_ticker_timeout_seconds": 0.03,
        "parallel_progress_log_seconds": 0.01,
    }
    ctxs = [_tc("SLOW", cfg)]

    def fake_chain(tc: TickerTrainingContext) -> None:
        time.sleep(0.20)
        tc.exported = True

    monkeypatch.setattr(pp_training, "_run_ticker_chain", fake_chain)

    with pytest.raises(ParallelTimeoutError):
        run_ticker_parallel(ctxs, max_workers=1)


def test_training_run_ticker_parallel_success_preserves_context_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctxs = [_tc("A"), _tc("B")]

    def fake_chain(tc: TickerTrainingContext) -> None:
        tc.result = {"ticker": tc.ticker}
        tc.exported = True

    monkeypatch.setattr(pp_training, "_run_ticker_chain", fake_chain)

    run_ticker_parallel(
        ctxs,
        max_workers=2,
        timeout_seconds=1.0,
        progress_log_seconds=0.01,
    )

    assert [tc.result for tc in ctxs] == [{"ticker": "A"}, {"ticker": "B"}]
    assert [tc.exported for tc in ctxs] == [True, True]
