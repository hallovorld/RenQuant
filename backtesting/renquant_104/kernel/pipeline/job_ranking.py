"""RankingJob — blend rank_score + rs_score into a combined ranking."""
from __future__ import annotations

from .context import InferenceContext
from .pipeline import Job, Task
from .task_ranking import BlendScoresTask, SortCandidatesTask


class RankingJob(Job):
    """Task chain: BlendScores → SortCandidates"""

    def should_skip(self, ctx: InferenceContext) -> bool:
        if not ctx.candidates:
            return True
        return ctx.buy_blocked and not ctx.bear_only

    @property
    def tasks(self) -> list[Task]:
        return [BlendScoresTask(), SortCandidatesTask()]
