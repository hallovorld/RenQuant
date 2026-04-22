"""DrawdownJob — portfolio drawdown circuit breaker."""
from __future__ import annotations

from .pipeline import Job, Task
from .task_drawdown import HWMUpdateTask, DrawdownCircuitTask


class DrawdownJob(Job):
    """Task chain: HWMUpdate → DrawdownCircuit"""

    @property
    def tasks(self) -> list[Task]:
        return [HWMUpdateTask(), DrawdownCircuitTask()]
