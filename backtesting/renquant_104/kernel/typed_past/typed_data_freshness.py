"""TypedDataFreshnessGate — TypedTask version of DataFreshnessGateTask.

Proof-of-concept migration (M4 track, 2026-05-10):
  * Original: kernel/pipeline/task_data_freshness.py::DataFreshnessGateTask
  * Reads ctx.today + ctx.ohlcv (a dict[ticker -> DataFrame])
  * Raises RuntimeError on stale data

Migrated form reads ONLY from a frozen Past at cursor t. The flattened
``past.ohlcv`` (date-indexed union of all tickers) gives us the freshness
check directly: ``past.ohlcv.index.max() < last_completed_close``.

Per §5.13.10 we do NOT add defensive None checks; the adapter guarantees
``past`` is non-None and DataFrames are real (possibly empty).

Per §5.13.1 the test in tests/test_typed_past.py exercises this through
the adapter using a real ctx-like construction, NOT a synthetic Past.
"""
from __future__ import annotations

import datetime as _dt
import logging

import pandas as pd

from .estimator import TaskResult
from .past import Past

log = logging.getLogger("kernel.typed_past.data_freshness")


class TypedDataFreshnessGate:
    """Hard-fail when past.ohlcv is older than last completed NYSE session.

    TypedTask contract: ``values_in_time(t, past) -> TaskResult``.
    """

    def __init__(self, *, enabled: bool = True):
        self.enabled = enabled

    def values_in_time(self, t: pd.Timestamp, past: Past) -> TaskResult:
        if not self.enabled:
            log.info("TypedDataFreshnessGate: disabled — skipping")
            return TaskResult(continue_chain=True)

        if len(past.ohlcv) == 0:
            log.warning(
                "TypedDataFreshnessGate: past.ohlcv empty — staleness check "
                "skipped (downstream tasks will fail if data really missing)."
            )
            return TaskResult(continue_chain=True)

        ref_date = t.date() if isinstance(t, pd.Timestamp) else _dt.date.today()
        last_close = _last_completed_nyse_close(ref_date)
        if last_close is None:
            log.warning(
                "TypedDataFreshnessGate: no NYSE session in last 14d before %s",
                ref_date,
            )
            return TaskResult(continue_chain=True)

        max_d = past.ohlcv.index.max().date()
        if max_d < last_close:
            msg = (
                f"TypedDataFreshnessGate: PANEL STALE — max date {max_d} < "
                f"last completed NYSE close {last_close}. Refusing to continue."
            )
            log.error(msg)
            raise RuntimeError(msg)

        log.info("TypedDataFreshnessGate: PASS  max_date=%s ≥ %s", max_d, last_close)
        return TaskResult(
            continue_chain=True,
            diagnostics={"max_date": max_d, "last_close": last_close},
        )


def _last_completed_nyse_close(ref: _dt.date) -> "_dt.date | None":
    try:
        import pandas_market_calendars as mcal  # noqa: PLC0415
    except ImportError:
        return ref - _dt.timedelta(days=2)

    cal = mcal.get_calendar("NYSE")
    sched = cal.schedule(
        start_date=ref - _dt.timedelta(days=14),
        end_date=ref,
    )
    sched_before = sched[sched.index.date < ref]
    if sched_before.empty:
        return None
    return sched_before.index[-1].date()
