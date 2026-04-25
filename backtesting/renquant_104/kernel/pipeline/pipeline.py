"""Task, Job, TickerJob ABCs and run_parallel helper.

Self-contained: only stdlib.  No common/ imports.
InferencePipeline and SellOnlyPipeline live in pp_inference.py.
TrainingPipeline and all training jobs live in pp_training.py.
"""
from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from threading import current_thread

from .context import InferenceContext, TickerInferenceContext

log = logging.getLogger("kernel.pipeline")


def resolve_workers(config_value: "int | None", item_count: int) -> int:
    """Resolve worker count: explicit config wins; None/<=0 → cpu_count-2 (min 1)."""
    if config_value is not None and config_value > 0:
        n = int(config_value)
    else:
        n = max(1, (os.cpu_count() or 4) - 2)
    return min(n, item_count)


# ── Task ABC ───────────────────────────────────────────────────────────────────

class Task(ABC):
    """Atomic step within a Job or TickerJob.

    run() returns True (or None) to continue the chain, False to stop early.
    Short-circuit is used by gate tasks (e.g. EarningsFilterTask) to halt
    downstream processing when a condition is not met.
    """

    @abstractmethod
    def run(self, ctx) -> "bool | None": ...

    @property
    def name(self) -> str:
        return type(self).__name__


# ── Job ABCs ───────────────────────────────────────────────────────────────────

class Job(ABC):
    """Global pipeline stage — reads/writes InferenceContext."""

    @property
    def tasks(self) -> "list[Task]":
        return []

    def run(self, ctx: InferenceContext) -> None:
        for task in self.tasks:
            if task.run(ctx) is False:
                log.debug("[%s] chain stopped by %s", type(self).__name__, task.name)
                return

    def should_skip(self, ctx: InferenceContext) -> bool:
        return False


class TickerJob(ABC):
    """Per-ticker pipeline stage — reads/writes TickerInferenceContext."""

    @property
    def tasks(self) -> "list[Task]":
        return []

    def run(self, tc: TickerInferenceContext) -> None:
        for task in self.tasks:
            if task.run(tc) is False:
                log.debug("[%s|%s] chain stopped by %s",
                          tc.ticker, type(self).__name__, task.name)
                return


# ── Parallel executor ──────────────────────────────────────────────────────────

def run_parallel(
    ticker_ctxs: list[TickerInferenceContext],
    job: TickerJob,
    max_workers: "int | None" = None,
    timeout_seconds: "float | None" = None,
) -> None:
    """Run job.run(tc) for each tc in parallel; faults are logged, not raised.

    max_workers=None → auto (cpu_count-2, min 1).
    timeout_seconds=None → no per-ticker timeout. Hung tickers are logged and skipped.
    """
    if not ticker_ctxs:
        return
    if max_workers is None and ticker_ctxs:
        cfg = getattr(ticker_ctxs[0], "config", None)
        cfg_workers = (cfg or {}).get("parallel_workers") if isinstance(cfg, dict) else None
        cfg_timeout = (cfg or {}).get("parallel_ticker_timeout_seconds") if isinstance(cfg, dict) else None
        max_workers = cfg_workers
        if timeout_seconds is None:
            timeout_seconds = cfg_timeout
    job_name = type(job).__name__
    n = resolve_workers(max_workers, len(ticker_ctxs))
    log.info("run_parallel: %s  %d tickers  %d workers  timeout=%s",
             job_name, len(ticker_ctxs), n, timeout_seconds)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n, thread_name_prefix="infer") as ex:
        futures = {ex.submit(_wrapped_run, job, tc): tc.ticker for tc in ticker_ctxs}
        for fut in as_completed(futures, timeout=None):
            ticker = futures[fut]
            try:
                fut.result(timeout=timeout_seconds)
            except TimeoutError:
                # Audit #2: ThreadPoolExecutor can't actually interrupt a
                # running thread; fut.cancel() is a no-op once the worker
                # has started. The hung worker keeps consuming CPU until
                # it returns. Don't pretend we "skipped" it — the result
                # is still pending; we're only abandoning *this* result
                # collection. Log accordingly so operators don't expect
                # the underlying work to stop.
                fut.cancel()   # only effective if worker hasn't started
                log.error("run_parallel [%s] %s TIMEOUT after %ss — abandoning "
                          "result (worker may still be running in background)",
                          ticker, job_name, timeout_seconds)
            except Exception as e:
                log.error("run_parallel [%s] %s ERROR — %s: %s",
                          ticker, job_name, type(e).__name__, e)
    log.info("run_parallel: %s DONE  %.2fs", job_name, time.monotonic() - t0)


def _wrapped_run(job: TickerJob, tc: TickerInferenceContext) -> None:
    log.debug("[%s|%s] %s START", tc.ticker, current_thread().name, type(job).__name__)
    t0 = time.monotonic()
    job.run(tc)
    log.debug("[%s|%s] %s DONE  %.2fs", tc.ticker, current_thread().name,
              type(job).__name__, time.monotonic() - t0)
