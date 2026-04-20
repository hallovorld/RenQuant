"""Job ABC and Pipeline sequential orchestrator — Docker-safe."""
from __future__ import annotations

from abc import ABC, abstractmethod

from .context import InferenceContext


class Job(ABC):
    @abstractmethod
    def run(self, ctx: InferenceContext) -> None:
        """Execute this job, reading from and writing to ctx."""

    def should_skip(self, ctx: InferenceContext) -> bool:
        return False


class Pipeline:
    def __init__(self, jobs: list[Job]) -> None:
        self._jobs = jobs

    def run(self, ctx: InferenceContext) -> None:
        for job in self._jobs:
            if not job.should_skip(ctx):
                job.run(ctx)
