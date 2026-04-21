"""RankingJob — blend rank_score + rs_score into a combined ranking.

Task chain:
    BlendScoresTask    Apply normalised blend weights to each candidate
    SortCandidatesTask Sort candidates by blended score descending → ctx.ranked
"""
from __future__ import annotations

from ..context import InferenceContext
from ..pipeline import Job, Task
from ..tasks.ranking import BlendScoresTask, SortCandidatesTask


class RankingJob(Job):
    """Rank candidates by blended (w_rank × rank_score + w_rs × rs_score).

    Task chain: BlendScores → SortCandidates
    """

    def should_skip(self, ctx: InferenceContext) -> bool:
        if not ctx.candidates:
            return True
        return ctx.buy_blocked and not ctx.bear_only

    @property
    def tasks(self) -> list[Task]:
        return [BlendScoresTask(), SortCandidatesTask()]
