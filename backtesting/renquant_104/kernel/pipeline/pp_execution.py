"""ExecutionPipeline — consolidation target for sim / LEAN / live order
placement, holding-state bookkeeping, and wash-sale stamping.

Sits AFTER :class:`InferencePipeline` in every adapter's bar loop:

::

   InferencePipeline.run(ctx)   # decision logic — what to buy / sell
   ExecutionPipeline.run(ctx)   # this module — translate decisions to broker

Per CLAUDE.md §1b: composed of Jobs (ExitsJob → BuysJob), each a sequence
of ≤50-line Tasks. This path is test-backed but not yet the active adapter
execution path; sim / runner / LEAN still use their adapter commit hooks
until the P0 execution consolidation is finished.
"""
from __future__ import annotations

import logging

from .context import InferenceContext
from .pipeline import Job
from .task_execution import (
    DedupeBuysTask,
    DedupeExitsTask,
    ExecuteBuysTask,
    ExecuteExitsTask,
    PrepareExecutionTask,
    PruneFullExitsTask,
    StampWashSaleTask,
    UpsertHoldingsTask,
)

log = logging.getLogger("kernel.pipeline.execution")


class ExitsJob(Job):
    """Place exit orders, stamp wash-sale + post-stop cooldown dates,
    prune fully-liquidated tickers from ``ctx.holdings``."""

    @property
    def tasks(self):
        return [
            DedupeExitsTask(),
            ExecuteExitsTask(),
            StampWashSaleTask(),
            PruneFullExitsTask(),
        ]


class BuysJob(Job):
    """Place buy orders, upsert :class:`HoldingState` (new or vol-weighted topup)."""

    @property
    def tasks(self):
        return [
            DedupeBuysTask(),
            ExecuteBuysTask(),
            UpsertHoldingsTask(),
        ]


class ExecutionPipeline:
    """Orchestrates :class:`ExitsJob` → :class:`BuysJob`.

    Ordering invariant: exits run before buys so same-bar sell proceeds can
    fund same-bar buys. ``PrepareExecutionTask`` resets ``ctx.fills`` so a
    stale entry from the previous bar can't survive.

    Usage::

        ctx = adapter.make_context(today)
        ctx.execution_backend = adapter.backend
        InferencePipeline(...).run(ctx)
        ExecutionPipeline().run(ctx)
        adapter.post_commit(ctx)  # adapter-specific bookkeeping (trade log, etc.)
    """

    @property
    def jobs(self) -> list[Job]:
        return [ExitsJob(), BuysJob()]

    def run(self, ctx: InferenceContext) -> None:
        # Prep is a single Task (not a Job) — keeps the phase explicit
        # without an empty-Job overhead.
        PrepareExecutionTask().run(ctx)
        for job in self.jobs:
            if job.should_skip(ctx):
                log.debug("ExecutionPipeline: skipping %s", type(job).__name__)
                continue
            job.run(ctx)


__all__ = ["ExecutionPipeline", "ExitsJob", "BuysJob"]
