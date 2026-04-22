"""SelectionJob — greedy slot-filling with tiered thresholds and all guards."""
from __future__ import annotations

from .context import InferenceContext
from .pipeline import Job, Task
from .task_selection import PrepareSelectionTask, RunSelectionTask, SizeAndEmitTask


class SelectionJob(Job):
    """Task chain: PrepareSelection → RunSelection → SizeAndEmit"""

    def should_skip(self, ctx: InferenceContext) -> bool:
        return not ctx.ranked

    @property
    def tasks(self) -> list[Task]:
        return [PrepareSelectionTask(), RunSelectionTask(), SizeAndEmitTask()]
