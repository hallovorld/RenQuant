"""DrawdownJob — portfolio drawdown circuit breaker.

Task chain:
    HWMUpdateTask        Advance high-water mark to max(hwm, portfolio_value)
    DrawdownCircuitTask  Check if drawdown ≥ halt_pct; set ctx.skip_buys
"""
from __future__ import annotations

from ..pipeline import Job, Task
from ..tasks.drawdown import HWMUpdateTask, DrawdownCircuitTask


class DrawdownJob(Job):
    """Update HWM and set skip_buys if the drawdown circuit breaker fires.

    Task chain: HWMUpdate → DrawdownCircuit
    """

    @property
    def tasks(self) -> list[Task]:
        return [HWMUpdateTask(), DrawdownCircuitTask()]
