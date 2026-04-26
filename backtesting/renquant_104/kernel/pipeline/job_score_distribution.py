"""ScoreDistributionJob — flag-gated wrapper for RecordScoreDistributionTask."""
from __future__ import annotations

from .context import InferenceContext
from .pipeline import Job, Task
from .task_score_distribution import RecordScoreDistributionTask


class ScoreDistributionJob(Job):
    """Single-task wrapper. Default off; opt-in via score_db.enabled."""

    def should_skip(self, ctx: InferenceContext) -> bool:
        cfg = ctx.config.get("score_db") or {}
        if not cfg.get("enabled", False):
            return True
        if not ctx.candidates and not ctx.holdings:
            return True
        return False

    @property
    def tasks(self) -> list[Task]:
        return [RecordScoreDistributionTask()]
