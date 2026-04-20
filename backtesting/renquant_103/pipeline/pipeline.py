"""Job ABC and Pipeline orchestrator."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from .context import PipelineContext

log = logging.getLogger("pipeline")


class Job(ABC):
    """One stage in the pipeline."""

    @abstractmethod
    def run(self, ctx: PipelineContext) -> None:
        """Execute this job, reading from and writing to *ctx*."""

    def should_skip(self, ctx: PipelineContext) -> bool:
        """Return True to skip this job entirely. Override in subclasses."""
        return False


class Pipeline:
    """Run jobs sequentially; each job mutates the shared context."""

    def __init__(self, jobs: list[Job]) -> None:
        self._jobs = jobs

    def run(self, ctx: PipelineContext) -> None:
        job_names = [type(j).__name__ for j in self._jobs]
        log.info("Pipeline START  jobs=%s", job_names)
        t0 = time.monotonic()

        for job in self._jobs:
            name = type(job).__name__
            if job.should_skip(ctx):
                log.info("  %s  SKIPPED", name)
                continue
            t1 = time.monotonic()
            log.info("  %s  START", name)
            job.run(ctx)
            log.info("  %s  DONE  (%.2fs)", name, time.monotonic() - t1)

        log.info("Pipeline DONE  total=%.2fs", time.monotonic() - t0)
