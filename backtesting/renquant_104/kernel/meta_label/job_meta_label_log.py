"""MetaLabelLoggingJob — Job chain for per-bar position-snapshot logging.

Runs as the last Job in InferencePipeline (after all sell/buy/exit
decisions are finalized in ctx). The Job's ``should_skip`` returns True
unless config opts in AND the adapter has set ``ctx.snapshot_logger``.

Config:
    "meta_label_training": {
        "enabled": true,            # default false
        "output_path": "data/position_day_snapshots.parquet"
    }

The adapter (SimAdapter / RunnerAdapter) is responsible for:
  1. Reading config.meta_label_training.enabled
  2. Instantiating SnapshotLogger and stashing it on ctx
  3. On teardown / build_result: calling logger.dump_to_parquet(...)

This Job only EMITS rows; the persist step lives in the adapter
boundary so it integrates with snapshot_artifacts_ctx + the existing
build_result pattern.
"""
from __future__ import annotations

from kernel.pipeline.context import InferenceContext
from kernel.pipeline.pipeline import Job, Task

from .task_snapshot import SnapshotHoldingsTask


class MetaLabelLoggingJob(Job):
    """Single-task job; gated by config + logger presence."""

    @property
    def tasks(self) -> "list[Task]":
        return [SnapshotHoldingsTask()]

    def should_skip(self, ctx: InferenceContext) -> bool:
        cfg = (ctx.config or {}).get("meta_label_training") or {}
        if not cfg.get("enabled", False):
            return True
        if getattr(ctx, "snapshot_logger", None) is None:
            # Defensive: adapter didn't set up the logger, even though
            # config said enabled. Skip silently (cf. §5.13.10 — log
            # path absent → bypass, don't crash prod) — the adapter
            # boot should already have warned in that case.
            return True
        return False
