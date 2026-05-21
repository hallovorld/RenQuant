"""BuyGatesJob — pre-buy gate checks in strict priority order."""
from __future__ import annotations

from .pipeline import Job, Task
from .task_gates import (
    FlattenCooldownGateTask,
    DrawdownGateTask, TransitionWindowTask, ConfidenceVetoTask,
    BullVolOffensiveBlockTask, RegimeAlphaGateTask, BEARBranchTask,
    VelocityCrashTask, EMA50GateTask,
)


class BuyGatesJob(Job):
    """Task chain: FlattenCooldown → DrawdownGate → TransitionWindow →
                  ConfidenceVeto → BullVolOffensiveBlock → RegimeAlphaGate →
                  BEARBranch → VelocityCrash → EMA50

    FlattenCooldownGateTask (2026-05-11) sits FIRST so post-flatten
    cooldown overrides DrawdownGate's resume threshold — see task
    docstring for the S-3 death-spiral motivation. No-op when
    ``risk.drawdown_flatten.cooldown_bars`` is unset or 0.

    BullVolOffensiveBlock sits AFTER ConfidenceVeto (which can already
    force defensives-only on low-confidence regimes) and BEFORE
    BEARBranch (which does the same for BEAR) — BULL_VOL is treated as
    "near-BEAR" when the AA-surfaced IC inversion flag is on.

    RegimeAlphaGateTask (2026-05-20) sits AFTER BullVol and BEFORE BEARBranch:
    per-regime "model has no OOS alpha" block. Sourced from
    artifacts/prod/truly_oos_eval/eval_truly_oos.json — BULL_CALM has
    top-10 OOS alpha −0.045, so we block new buys there.
    """

    @property
    def tasks(self) -> list[Task]:
        return [
            FlattenCooldownGateTask(),
            DrawdownGateTask(),
            TransitionWindowTask(),
            ConfidenceVetoTask(),
            BullVolOffensiveBlockTask(),
            RegimeAlphaGateTask(),
            BEARBranchTask(),
            VelocityCrashTask(),
            EMA50GateTask(),
        ]
