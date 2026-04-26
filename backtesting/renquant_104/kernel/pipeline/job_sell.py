"""TickerSellJob — evaluate exit signals for one held position."""
from __future__ import annotations

from .pipeline import TickerJob, Task
from .task_sell import (
    PrepareHoldingTask, ScoreModelTask, EvaluateExitsTask,
    SellGateBTask, PanelConvictionExitTask,
)


class TickerSellJob(TickerJob):
    """Task chain: PrepareHolding → ScoreModel → EvaluateExits →
    SellGateB → PanelConvictionExit.

    SellGateB (2026-04-26 round-7) sits between the priority chain and
    the panel-conviction tiebreaker. It can BLOCK a model_sell exit
    (and only a model_sell — path rules pass through) when the latest
    NGBoost μ/σ doesn't agree with a bearish view. PanelConvictionExit
    runs LAST so higher-priority rules (trailing, stop-loss, single-day
    loss, max_hold) always win — it adds a panel/NGBoost-based exit
    only when no other rule fired and both the cross-sectional panel
    and μ/σ head turned bearish.
    """

    @property
    def tasks(self) -> list[Task]:
        return [
            PrepareHoldingTask(),
            ScoreModelTask(),
            EvaluateExitsTask(),
            SellGateBTask(),              # NGBoost μ/σ guard on model_sell
            PanelConvictionExitTask(),    # last — tiebreaker
        ]
