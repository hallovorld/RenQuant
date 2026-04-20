"""Job ABC and Pipeline orchestrator."""
from __future__ import annotations

from abc import ABC, abstractmethod

from .context import PipelineContext


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
        for job in self._jobs:
            if not job.should_skip(ctx):
                job.run(ctx)
