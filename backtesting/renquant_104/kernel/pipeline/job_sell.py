"""TickerSellJob — evaluate exit signals for one held position."""
from __future__ import annotations

from .pipeline import TickerJob, Task
from .task_sell import (
    PrepareHoldingTask, ScoreModelTask, EvaluateExitsTask,
    SellGateBTask, PanelConvictionExitTask, EarningsBlackoutSellTask,
)


class TickerSellJob(TickerJob):
    """Task chain: PrepareHolding → ScoreModel → EvaluateExits →
    SellGateB → PanelConvictionExit → EarningsBlackoutSell.

    SellGateB (2026-04-26 round-7) sits between the priority chain and
    the panel-conviction tiebreaker. It can BLOCK a model_sell exit
    (and only a model_sell — path rules pass through) when the latest
    NGBoost μ/σ doesn't agree with a bearish view. PanelConvictionExit is
    the legacy calibrated-rank tiebreaker; production 104 configs disable it
    with risk.panel_exit.legacy_enabled=false because the post-PanelScoring
    CrossSectionalPanelExitTask owns raw panel/NGBoost exits.

    EarningsBlackoutSell (2026-05-01 trade-audit response) runs LAST so it
    sees the FINAL exit_signal (whichever of model_sell / panel_conviction
    actually won the prior chain) and can veto it when the holding sits
    inside the earnings event-blackout window. Path-action exits are
    exempt — see task docstring for the invariant.
    """

    @property
    def tasks(self) -> list[Task]:
        return [
            PrepareHoldingTask(),
            ScoreModelTask(),
            EvaluateExitsTask(),
            SellGateBTask(),              # NGBoost μ/σ guard on model_sell
            PanelConvictionExitTask(),    # tiebreaker
            EarningsBlackoutSellTask(),   # event-blackout veto on model_*
        ]
