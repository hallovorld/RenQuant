"""Task, Job, TickerJob ABCs and run_parallel helper.

Self-contained: only stdlib.  No common/ imports.
InferencePipeline and SellOnlyPipeline live in pp_inference.py.
TrainingPipeline and all training jobs live in pp_training.py.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import current_thread

from .context import InferenceContext, TickerInferenceContext

log = logging.getLogger("kernel.pipeline")


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
    max_workers: int = 8,
) -> None:
    """Run job.run(tc) for each tc in parallel; faults are logged, not raised."""
    if not ticker_ctxs:
        return
    job_name = type(job).__name__
    n = min(max_workers, len(ticker_ctxs))
    log.info("run_parallel: %s  %d tickers  %d workers", job_name, len(ticker_ctxs), n)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n, thread_name_prefix="infer") as ex:
        futures = {ex.submit(_wrapped_run, job, tc): tc.ticker for tc in ticker_ctxs}
        for fut in as_completed(futures):
            ticker = futures[fut]
            exc = fut.exception()
            if exc:
                log.error("run_parallel [%s] %s ERROR — %s: %s",
                          ticker, job_name, type(exc).__name__, exc)
    log.info("run_parallel: %s DONE  %.2fs", job_name, time.monotonic() - t0)


def _wrapped_run(job: TickerJob, tc: TickerInferenceContext) -> None:
    log.debug("[%s|%s] %s START", tc.ticker, current_thread().name, type(job).__name__)
    t0 = time.monotonic()
    job.run(tc)
    log.debug("[%s|%s] %s DONE  %.2fs", tc.ticker, current_thread().name,
              type(job).__name__, time.monotonic() - t0)
