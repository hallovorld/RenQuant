"""TickerSellJob — evaluate exit signals for one held position.

Task chain:
    PrepareHoldingTask  Validate holding + price; attach prev_close
    ScoreModelTask      Build features → score model → tc.model_action
    EvaluateExitsTask   Run 5-exit priority chain → tc.holding + tc.exit_signal
"""
from __future__ import annotations

from ..pipeline import TickerJob, Task
from ..tasks.sell import PrepareHoldingTask, ScoreModelTask, EvaluateExitsTask


class TickerSellJob(TickerJob):
    """Compute exit signal for one held ticker.

    Task chain: PrepareHolding → ScoreModel → EvaluateExits
    """

    @property
    def tasks(self) -> list[Task]:
        return [PrepareHoldingTask(), ScoreModelTask(), EvaluateExitsTask()]
