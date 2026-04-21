"""TickerCandidateJob — score one candidate ticker for buy eligibility."""
from __future__ import annotations

from .pipeline import TickerJob, Task
from .task_candidates import (
    EarningsFilterTask, WashSaleFilterTask, BuildFeaturesTask,
    ScoreBuyTask, ScoreThresholdTask, RelativeStrengthTask, AssembleCandidateTask,
)


class TickerCandidateJob(TickerJob):
    """Task chain: EarningsFilter → WashSaleFilter → BuildFeatures →
                  ScoreBuy → ScoreThreshold → RelativeStrength → AssembleCandidate"""

    @property
    def tasks(self) -> list[Task]:
        return [
            EarningsFilterTask(),
            WashSaleFilterTask(),
            BuildFeaturesTask(),
            ScoreBuyTask(),
            ScoreThresholdTask(),
            RelativeStrengthTask(),
            AssembleCandidateTask(),
        ]
