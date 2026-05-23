"""TickerCandidateJob — score one candidate ticker for buy eligibility."""
from __future__ import annotations

from .pipeline import TickerJob, Task
from .task_candidates import (
    EarningsFilterTask, WashSaleFilterTask, SectorMapGateTask, BuildFeaturesTask,
    ScoreBuyTask, ScoreThresholdTask, RelativeStrengthTask,
    AssembleCandidateTask,
)


class TickerCandidateJob(TickerJob):
    """Task chain: EarningsFilter → WashSaleFilter → BuildFeatures →
                  ScoreBuy → ScoreThreshold → RelativeStrength →
                  AssembleCandidate

    Z1 parabolic-exhaustion gate (added 2026-04-28 after NVTS post-mortem)
    was REMOVED on 2026-04-28 after the panel A/A test falsified the
    underlying hypothesis: on this watchlist + period, top-decile
    rel_mom_20d names *outperform* the rest at 1d/2d/5d horizons (paired
    diff +0.30% over 5d, A/A perm |p95|=0.08% — significant in the
    OPPOSITE direction). The NVTS −12% was a tail event, not a panel
    signal. NVTS root cause stays with the open Z8 sample-size penalty
    and stop_loss tightening tasks. See doc/archives/audits/
    2026-04-28-nvts-buy-postmortem.md.
    """

    @property
    def tasks(self) -> list[Task]:
        return [
            EarningsFilterTask(),
            WashSaleFilterTask(),
            SectorMapGateTask(),
            BuildFeaturesTask(),
            ScoreBuyTask(),
            ScoreThresholdTask(),
            RelativeStrengthTask(),
            AssembleCandidateTask(),
        ]
