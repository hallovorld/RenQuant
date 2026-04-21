"""Ranking tasks: blend scores then sort."""
from __future__ import annotations

import logging

from .context import InferenceContext
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.ranking")


class BlendScoresTask(Task):
    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.selection import score_candidates  # noqa: PLC0415
        ranking_cfg = ctx.config.get("ranking", {})
        bw    = ranking_cfg.get("blend_weights", [0.5, 0.5])
        total = float(bw[0]) + float(bw[1])
        w_rank = float(bw[0]) / total if total > 0 else 0.5
        w_rs   = float(bw[1]) / total if total > 0 else 0.5
        ctx._blended = score_candidates(ctx.candidates, w_rank, w_rs)  # noqa: SLF001
        ctx._blend_w = (w_rank, w_rs)                                   # noqa: SLF001
        log.debug("BlendScoresTask: %d candidates  w_rank=%.2f  w_rs=%.2f",
                  len(ctx.candidates), w_rank, w_rs)


class SortCandidatesTask(Task):
    def run(self, ctx: InferenceContext) -> bool | None:
        blended      = getattr(ctx, "_blended", ctx.candidates)
        ctx.ranked   = sorted(blended, key=lambda c: c.rank_score, reverse=True)
        w_rank, w_rs = getattr(ctx, "_blend_w", (0.5, 0.5))
        log.info("SortCandidatesTask: %d ranked (w_rank=%.2f w_rs=%.2f)",
                 len(ctx.ranked), w_rank, w_rs)
