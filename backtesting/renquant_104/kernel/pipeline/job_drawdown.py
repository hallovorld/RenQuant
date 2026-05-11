"""DrawdownJob — portfolio drawdown circuit breaker + rebalance."""
from __future__ import annotations

from .pipeline import Job, Task
from .task_drawdown import HWMUpdateTask, DrawdownCircuitTask
from .task_drawdown_rebalance import DrawdownRebalanceTask


class DrawdownJob(Job):
    """Task chain: HWMUpdate → DrawdownCircuit → DrawdownRebalance.

    The first two tasks update the high-water mark and toggle the buy-side
    circuit breaker (block new buys when DD ≥ halt_pct). The third task
    (Grossman & Zhou 1993, JEEM 19(2):241-276) operationalises drawdown
    control on the SELL side too — liquidating the weakest holdings to
    scale gross exposure down per the Kelly-fraction = max(0, 1 - DD/DD_max)
    rule. Off by default; opt-in via ``risk.drawdown_rebalance.enabled``.
    """

    @property
    def tasks(self) -> list[Task]:
        return [HWMUpdateTask(), DrawdownCircuitTask(), DrawdownRebalanceTask()]
