"""TickerSellJob — evaluate exit signals for one held position."""
from __future__ import annotations

from .pipeline import TickerJob, Task
from .task_sell import (
    PrepareHoldingTask, ScoreModelTask, EvaluateExitsTask,
    PanelConvictionExitTask,
)


class TickerSellJob(TickerJob):
    """Task chain: PrepareHolding → ScoreModel → EvaluateExits → PanelConvictionExit.

    PanelConvictionExit runs LAST so higher-priority rules (trailing,
    stop-loss, single-day loss, max_hold, model-streak) always win.
    It adds a panel/NGBoost-based exit when no other rule fired and
    the cross-sectional panel + μ/σ head both turned bearish.
    """

    @property
    def tasks(self) -> list[Task]:
        return [
            PrepareHoldingTask(),
            ScoreModelTask(),
            EvaluateExitsTask(),
            PanelConvictionExitTask(),   # last — tiebreaker
        ]
