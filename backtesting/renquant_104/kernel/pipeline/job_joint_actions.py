"""JointActionJob — flag-gated unified buy / sell / rotate selector.

When `rotation.joint_actions.enabled = true`, this Job replaces the
RotationJob + SelectionJob pair in the InferencePipeline phase order.
When `false` (default), `should_skip()` returns True and the legacy
chain runs unchanged.

Phase 2 of the rotation algorithm rewrite (2026-04-25). See
task_joint_actions.py for the full algorithm.
"""
from __future__ import annotations

from .context import InferenceContext
from .pipeline import Job, Task
from .task_joint_actions import JointActionTask


class JointActionJob(Job):
    """Single-task wrapper around JointActionTask."""

    def should_skip(self, ctx: InferenceContext) -> bool:
        joint_cfg = (ctx.config.get("rotation", {})
                              .get("joint_actions", {}))
        if not joint_cfg.get("enabled", False):
            return True
        # Joint mode handles offensive regimes; BEAR routing stays in legacy
        # SelectionJob (defensive-only logic is non-trivial to port).
        if ctx.bear_only:
            return True
        # Need at least one candidate or one held to do anything useful.
        if not ctx.ranked and not ctx.holdings:
            return True
        return False

    @property
    def tasks(self) -> list[Task]:
        # JointPortfolioQPTask runs FIRST. It's a no-op unless
        # rotation.joint_actions.solver == "qp" (default "greedy").
        # When active, it owns the bar (returns True) and JointActionTask
        # short-circuits via the same flag check at the top of its run().
        from kernel.portfolio_qp.task_joint_qp import (  # noqa: PLC0415
            JointPortfolioQPTask,
        )
        return [JointPortfolioQPTask(), JointActionTask()]
