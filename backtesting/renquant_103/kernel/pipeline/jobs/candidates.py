"""TickerCandidateJob — score one candidate ticker for buy eligibility.

Task chain (each step can return False to discard the ticker):
    EarningsFilterTask    Skip if earnings within ±buffer days
    WashSaleFilterTask    Skip if sold within wash_sale_days
    BuildFeaturesTask     Build feature frame; skip if unavailable
    ScoreBuyTask          Score model; skip if not a buy signal
    ScoreThresholdTask    Skip if rank_score < min_model_score threshold
    RelativeStrengthTask  Compute RS vs sector ETF → tc.rs_score
    AssembleCandidateTask Package CandidateResult → tc.candidate
"""
from __future__ import annotations

from ..pipeline import TickerJob, Task
from ..tasks.candidates import (
    EarningsFilterTask, WashSaleFilterTask, BuildFeaturesTask,
    ScoreBuyTask, ScoreThresholdTask, RelativeStrengthTask, AssembleCandidateTask,
)


class TickerCandidateJob(TickerJob):
    """Score one ticker and produce a CandidateResult if it qualifies.

    Task chain: EarningsFilter → WashSaleFilter → BuildFeatures →
                ScoreBuy → ScoreThreshold → RelativeStrength → AssembleCandidate
    """

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
