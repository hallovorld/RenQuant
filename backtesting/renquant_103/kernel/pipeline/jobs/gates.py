"""BuyGatesJob — pre-buy gate checks applied in strict priority order.

Task chain (each returns False to stop the chain and block buys):
    DrawdownGateTask      Gate 0: drawdown circuit breaker already fired
    TransitionWindowTask  Gate 1: CUSUM uncertainty window active
    BEARBranchTask        Gate 2: BEAR regime — restrict to defensives, stop normal scan
    VelocityCrashTask     Gate 3: SPY down > threshold% in last N days
    EMA50GateTask         Gate 4: SPY price below 50-day EMA
"""
from __future__ import annotations

from ..pipeline import Job, Task
from ..tasks.gates import (
    DrawdownGateTask, TransitionWindowTask, BEARBranchTask,
    VelocityCrashTask, EMA50GateTask,
)


class BuyGatesJob(Job):
    """Apply all pre-buy gates in priority order; first failure short-circuits.

    Task chain: DrawdownGate → TransitionWindow → BEARBranch → VelocityCrash → EMA50
    """

    @property
    def tasks(self) -> list[Task]:
        return [
            DrawdownGateTask(),
            TransitionWindowTask(),
            BEARBranchTask(),
            VelocityCrashTask(),
            EMA50GateTask(),
        ]
