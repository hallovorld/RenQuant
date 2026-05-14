"""ShortCandidateJob — Long-Short Phase 2B.

Single-task Job that wraps ``ShortCandidateSelectionTask``. Lives in the
inference pipeline AFTER PanelScoringJob (which writes
``ctx._panel_scores_all``) and BEFORE JointActionJob (which reads
``ctx.short_candidates`` via ``BuildSourceMapTask``).

OFF by default — entire job no-ops when ``long_short.enabled`` is false.
"""
from __future__ import annotations

from kernel.pipeline.context import InferenceContext
from kernel.pipeline.pipeline import Job, Task

from .task_short_candidates import ShortCandidateSelectionTask


class ShortCandidateJob(Job):
    """Populate ``ctx.short_candidates`` (Phase 2B).

    Skips the entire job when ``long_short.enabled`` is false. Even
    inside the task we re-check the flag, but the Job-level skip avoids
    the task-construction cost.
    """

    name = "ShortCandidateJob"

    def should_skip(self, ctx: InferenceContext) -> bool:
        ls_cfg = (getattr(ctx, "config", None) or {}).get(
            "long_short", {}
        ) or {}
        return not ls_cfg.get("enabled", False)

    @property
    def tasks(self) -> list[Task]:
        return [ShortCandidateSelectionTask()]


__all__ = ["ShortCandidateJob"]
