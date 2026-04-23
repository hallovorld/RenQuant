"""Ranking tasks: blend scores then sort."""
from __future__ import annotations

import logging

from .context import InferenceContext
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.ranking")


class BlendScoresTask(Task):
    """Normalize rank_score across today's candidates.

    rs_score used to be blended in via recalibrated `ranking.blend_weights`,
    but the recalibrator has consistently driven the rs_score weight to zero
    in production. The channel is now removed from the ranking math —
    rs_score is still carried on CandidateResult for logging but contributes
    nothing to the blend. This keeps the min-max normalization load-bearing
    behavior (maps raw panel-LTR scores onto the same scale the tiered
    thresholds expect) while dropping the dead channel.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.selection import score_candidates  # noqa: PLC0415
        ranking_cfg = ctx.config.get("ranking", {})
        legacy_bw = ranking_cfg.get("blend_weights")
        if legacy_bw is not None:
            try:
                w_rs_legacy = float(legacy_bw[1]) / (float(legacy_bw[0]) + float(legacy_bw[1]))
            except (IndexError, TypeError, ValueError, ZeroDivisionError):
                w_rs_legacy = 0.0
            if w_rs_legacy > 0.0:
                log.warning(
                    "BlendScoresTask: ignoring legacy ranking.blend_weights=%s "
                    "(rs_score channel is deprecated and no longer blended)",
                    legacy_bw,
                )
        w_rank, w_rs = 1.0, 0.0
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
