"""TickerSellJob — evaluate exit signals for one held position."""
from __future__ import annotations

from .pipeline import TickerJob, Task
from .task_sell import PrepareHoldingTask, ScoreModelTask, EvaluateExitsTask


class TickerSellJob(TickerJob):
    """Task chain: PrepareHolding → ScoreModel → EvaluateExits"""

    @property
    def tasks(self) -> list[Task]:
        return [PrepareHoldingTask(), ScoreModelTask(), EvaluateExitsTask()]
