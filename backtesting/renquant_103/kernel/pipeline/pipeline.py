"""Job ABC and InferencePipeline orchestrator.

Self-contained: only stdlib.  No common/ imports.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from .context import InferenceContext

log = logging.getLogger("kernel.pipeline")


class Job(ABC):
    """One stage in the InferencePipeline."""

    @abstractmethod
    def run(self, ctx: InferenceContext) -> None:
        """Execute this job, reading from and writing to *ctx*."""

    def should_skip(self, ctx: InferenceContext) -> bool:
        """Return True to skip this job entirely. Override in subclasses."""
        return False


class InferencePipeline:
    """Full pipeline: Regime → Drawdown → Sell → BuyGates → Candidates → Ranking → Selection."""

    def __init__(self) -> None:
        from .jobs.regime import RegimeJob
        from .jobs.drawdown import DrawdownJob
        from .jobs.sell import SellJob
        from .jobs.gates import BuyGatesJob
        from .jobs.candidates import CandidateJob
        from .jobs.ranking import RankingJob
        from .jobs.selection import SelectionJob

        self._jobs: list[Job] = [
            RegimeJob(),
            DrawdownJob(),
            SellJob(),
            BuyGatesJob(),
            CandidateJob(),
            RankingJob(),
            SelectionJob(),
        ]

    def run(self, ctx: InferenceContext) -> None:
        job_names = [type(j).__name__ for j in self._jobs]
        log.info("InferencePipeline START  jobs=%s", job_names)
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

        log.info("InferencePipeline DONE  total=%.2fs", time.monotonic() - t0)


class SellOnlyPipeline:
    """Sell-only variant: Regime → Drawdown → Sell."""

    def __init__(self) -> None:
        from .jobs.regime import RegimeJob
        from .jobs.drawdown import DrawdownJob
        from .jobs.sell import SellJob

        self._jobs: list[Job] = [
            RegimeJob(),
            DrawdownJob(),
            SellJob(),
        ]

    def run(self, ctx: InferenceContext) -> None:
        log.info("SellOnlyPipeline START")
        t0 = time.monotonic()

        for job in self._jobs:
            name = type(job).__name__
            if job.should_skip(ctx):
                log.info("  %s  SKIPPED", name)
                continue
            job.run(ctx)

        log.info("SellOnlyPipeline DONE  total=%.2fs", time.monotonic() - t0)
