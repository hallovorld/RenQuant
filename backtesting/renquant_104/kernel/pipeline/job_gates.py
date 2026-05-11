"""BuyGatesJob — pre-buy gate checks in strict priority order."""
from __future__ import annotations

from .pipeline import Job, Task
from .task_gates import (
    FlattenCooldownGateTask,
    DrawdownGateTask, TransitionWindowTask, ConfidenceVetoTask,
    BullVolOffensiveBlockTask, BEARBranchTask, VelocityCrashTask,
    EMA50GateTask,
)


class BuyGatesJob(Job):
    """Task chain: FlattenCooldown → DrawdownGate → TransitionWindow →
                  ConfidenceVeto → BullVolOffensiveBlock → BEARBranch →
                  VelocityCrash → EMA50

    FlattenCooldownGateTask (2026-05-11) sits FIRST so post-flatten
    cooldown overrides DrawdownGate's resume threshold — see task
    docstring for the S-3 death-spiral motivation. No-op when
    ``risk.drawdown_flatten.cooldown_bars`` is unset or 0.

    BullVolOffensiveBlock sits AFTER ConfidenceVeto (which can already
    force defensives-only on low-confidence regimes) and BEFORE
    BEARBranch (which does the same for BEAR) — BULL_VOL is treated as
    "near-BEAR" when the AA-surfaced IC inversion flag is on.
    """

    @property
    def tasks(self) -> list[Task]:
        return [
            FlattenCooldownGateTask(),
            DrawdownGateTask(),
            TransitionWindowTask(),
            ConfidenceVetoTask(),
            BullVolOffensiveBlockTask(),
            BEARBranchTask(),
            VelocityCrashTask(),
            EMA50GateTask(),
        ]
