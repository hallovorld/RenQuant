"""BuyGatesJob — pre-buy gate checks in strict priority order."""
from __future__ import annotations

from .pipeline import Job, Task
from .task_gates import (
    DrawdownGateTask, TransitionWindowTask, ConfidenceVetoTask,
    BullVolOffensiveBlockTask, BEARBranchTask, VelocityCrashTask,
    EMA50GateTask,
)


class BuyGatesJob(Job):
    """Task chain: DrawdownGate → TransitionWindow → ConfidenceVeto
                  → BullVolOffensiveBlock → BEARBranch → VelocityCrash → EMA50

    BullVolOffensiveBlock sits AFTER ConfidenceVeto (which can already
    force defensives-only on low-confidence regimes) and BEFORE
    BEARBranch (which does the same for BEAR) — BULL_VOL is treated as
    "near-BEAR" when the AA-surfaced IC inversion flag is on.
    """

    @property
    def tasks(self) -> list[Task]:
        return [
            DrawdownGateTask(),
            TransitionWindowTask(),
            ConfidenceVetoTask(),
            BullVolOffensiveBlockTask(),
            BEARBranchTask(),
            VelocityCrashTask(),
            EMA50GateTask(),
        ]
