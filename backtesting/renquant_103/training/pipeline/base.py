"""TrainingJob ABC and TrainingPipeline orchestrator."""
from __future__ import annotations

from abc import ABC, abstractmethod

from .context import TrainingContext


class TrainingJob(ABC):
    @abstractmethod
    def run(self, ctx: TrainingContext) -> None:
        """Execute this training step."""

    def should_skip(self, ctx: TrainingContext) -> bool:
        return False


class TrainingPipeline:
    def __init__(self, jobs: list[TrainingJob] | None = None) -> None:
        self._jobs = jobs or _DEFAULT_JOBS()

    def run(self, ctx: TrainingContext) -> None:
        for job in self._jobs:
            if not job.should_skip(ctx):
                job.run(ctx)


def _DEFAULT_JOBS() -> list[TrainingJob]:
    from .jobs import (
        DataFetchJob, RegimeFitJob, FeatureJob,
        TournamentJob, ExportJob, CorrelationJob, CalibrationJob,
    )
    return [
        DataFetchJob(),
        RegimeFitJob(),
        FeatureJob(),
        TournamentJob(),
        ExportJob(),
        CorrelationJob(),
        CalibrationJob(),
    ]
