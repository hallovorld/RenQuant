"""JointPortfolioQPTask — back-compat shim.

The 2026-04-29 monolith (459 lines, 6 logical steps in one Task) has
been split per CLAUDE.md §1b "Every Logical Unit Is a Task, Job, or
Pipeline" into:

    JointPortfolioQPJob (in kernel/portfolio_qp/job_qp.py)
    ├── PrepareQPVectorsTask     — w_current, μ, σ, Σ, prices
    ├── BuildTaxCostTask         — Brown-Smith dynamic + harvest credit
    ├── BuildQPConstraintsTask   — wash, caps, dw_max
    ├── SolveQPTask              — call solve_portfolio_qp
    └── EmitQPOrdersTask         — Δw → ctx.orders / ctx.exits

User mandate (2026-05-04): "代码短，逻辑清晰，才能bug少". Each new
Task is ≤50 lines, single-responsibility, individually testable.

This shim preserves the old import path so existing callers keep
working. New code should reference `JointPortfolioQPJob` directly.

Reference: ``doc/components/portfolio-qp.md`` for the math.
"""
from __future__ import annotations

import logging

from kernel.pipeline.context import InferenceContext
from kernel.pipeline.pipeline import Task

from .job_qp import JointPortfolioQPJob

log = logging.getLogger("kernel.portfolio_qp.joint_qp")


class JointPortfolioQPTask(Task):
    """Thin wrapper that runs `JointPortfolioQPJob`.

    Kept for back-compat with `pp_inference.py` and any test fixture
    that constructed the old class directly. The Job's `should_skip`
    handles the same bear_only / solver=qp / enabled=true gates the
    old monolith had — so this shim has no extra logic.
    """

    name = "JointPortfolioQPTask"
    _job = JointPortfolioQPJob()

    def run(self, ctx: InferenceContext) -> bool | None:
        if self._job.should_skip(ctx):
            return False
        self._job.run(ctx)
        # Back-compat: old Task returned True on success / False on skip.
        # Job.run is void, but emit-task logs the outcome.
        return True


__all__ = ["JointPortfolioQPTask"]
