"""BuyGatesJob — pre-buy gate checks in strict priority order."""
from __future__ import annotations

from .pipeline import Job, Task
from .task_gates import (
    DrawdownGateTask, TransitionWindowTask, BEARBranchTask,
    VelocityCrashTask, EMA50GateTask,
)


class BuyGatesJob(Job):
    """Task chain: DrawdownGate → TransitionWindow → BEARBranch → VelocityCrash → EMA50"""

    @property
    def tasks(self) -> list[Task]:
        return [
            DrawdownGateTask(),
            TransitionWindowTask(),
            BEARBranchTask(),
            VelocityCrashTask(),
            EMA50GateTask(),
        ]
