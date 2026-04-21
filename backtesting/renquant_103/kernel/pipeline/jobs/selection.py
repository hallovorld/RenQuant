"""SelectionJob — greedy slot-filling with tiered thresholds and all guards.

Task chain:
    PrepareSelectionTask  Compute open slots + BEAR cap; build SelectionContext
    RunSelectionTask      Run greedy loop → selected list; update block counters
    SizeAndEmitTask       Compute position size per ticker; emit ctx.orders
"""
from __future__ import annotations

from ..context import InferenceContext
from ..pipeline import Job, Task
from ..tasks.selection import PrepareSelectionTask, RunSelectionTask, SizeAndEmitTask


class SelectionJob(Job):
    """Fill open slots from ctx.ranked, applying all guards and sizing.

    Task chain: PrepareSelection → RunSelection → SizeAndEmit
    """

    def should_skip(self, ctx: InferenceContext) -> bool:
        return not ctx.ranked

    @property
    def tasks(self) -> list[Task]:
        return [PrepareSelectionTask(), RunSelectionTask(), SizeAndEmitTask()]
