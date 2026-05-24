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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from threading import current_thread

from .context import InferenceContext, TickerInferenceContext

log = logging.getLogger("kernel.pipeline")


class ParallelTimeoutError(RuntimeError):
    """Raised when a per-ticker parallel phase exceeds its wall-clock budget."""

    def __init__(self, job_name: str, elapsed: float, pending_tickers: list[str]) -> None:
        self.job_name = job_name
        self.elapsed = float(elapsed)
        self.pending_tickers = list(pending_tickers)
        super().__init__(
            f"{job_name} timed out after {elapsed:.2f}s with "
            f"{len(pending_tickers)} pending ticker(s): {', '.join(pending_tickers[:20])}"
        )


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
    progress_log_seconds: "float | None" = None,
) -> None:
    """Run job.run(tc) for each tc in parallel; faults are logged, not raised.

    max_workers=None → auto (cpu_count-2, min 1).
    timeout_seconds=None → no wall-clock phase timeout.

    ThreadPoolExecutor cannot safely interrupt a running worker. If a phase
    exceeds timeout_seconds, fail hard before downstream sizing/order emission
    can consume a partial candidate set.
    """
    if not ticker_ctxs:
        return
    if ticker_ctxs:
        cfg = getattr(ticker_ctxs[0], "config", None)
        cfg_workers = (cfg or {}).get("parallel_workers") if isinstance(cfg, dict) else None
        cfg_timeout = (cfg or {}).get("parallel_ticker_timeout_seconds") if isinstance(cfg, dict) else None
        cfg_progress = (cfg or {}).get("parallel_progress_log_seconds") if isinstance(cfg, dict) else None
        if max_workers is None:
            max_workers = cfg_workers
        if timeout_seconds is None:
            timeout_seconds = cfg_timeout
        if progress_log_seconds is None:
            progress_log_seconds = cfg_progress
    if progress_log_seconds is None:
        progress_log_seconds = 30.0
    job_name = type(job).__name__
    n = resolve_workers(max_workers, len(ticker_ctxs))
    log.info("run_parallel: %s  %d tickers  %d workers  timeout=%s",
             job_name, len(ticker_ctxs), n, timeout_seconds)
    t0 = time.monotonic()
    ex = ThreadPoolExecutor(max_workers=n, thread_name_prefix="infer")
    futures = {ex.submit(_wrapped_run, job, tc): tc.ticker for tc in ticker_ctxs}
    pending = set(futures)
    completed = 0
    progress_interval = max(0.01, float(progress_log_seconds or 0.0))
    next_progress = t0 + progress_interval
    abandon_executor = False
    try:
        while pending:
            now = time.monotonic()
            elapsed = now - t0
            if timeout_seconds is not None and elapsed >= float(timeout_seconds):
                pending_tickers = sorted(futures[f] for f in pending)
                for fut in pending:
                    fut.cancel()  # only effective for workers not yet started
                log.error(
                    "run_parallel: %s TIMEOUT after %.2fs — done=%d/%d "
                    "pending=%d tickers=%s; worker may still be running",
                    job_name, elapsed, completed, len(futures), len(pending_tickers),
                    pending_tickers[:20],
                )
                ex.shutdown(wait=False, cancel_futures=True)
                abandon_executor = True
                raise ParallelTimeoutError(job_name, elapsed, pending_tickers)

            wait_timeout = max(0.0, next_progress - now)
            if timeout_seconds is not None:
                wait_timeout = min(wait_timeout, max(0.0, float(timeout_seconds) - elapsed))
            done, pending = wait(pending, timeout=wait_timeout, return_when=FIRST_COMPLETED)

            for fut in done:
                ticker = futures[fut]
                completed += 1
                try:
                    fut.result()
                except Exception as e:
                    log.error("run_parallel [%s] %s ERROR — %s: %s",
                              ticker, job_name, type(e).__name__, e)

            now = time.monotonic()
            if pending and now >= next_progress:
                pending_tickers = sorted(futures[f] for f in pending)
                log.info(
                    "run_parallel: %s progress done=%d/%d pending=%d "
                    "elapsed=%.2fs pending_tickers=%s",
                    job_name, completed, len(futures), len(pending_tickers),
                    now - t0, pending_tickers[:10],
                )
                next_progress = now + progress_interval
    finally:
        if not abandon_executor:
            ex.shutdown(wait=True)
    log.info("run_parallel: %s DONE  %.2fs", job_name, time.monotonic() - t0)


def _wrapped_run(job: TickerJob, tc: TickerInferenceContext) -> None:
    log.debug("[%s|%s] %s START", tc.ticker, current_thread().name, type(job).__name__)
    t0 = time.monotonic()
    job.run(tc)
    log.debug("[%s|%s] %s DONE  %.2fs", tc.ticker, current_thread().name,
              type(job).__name__, time.monotonic() - t0)
