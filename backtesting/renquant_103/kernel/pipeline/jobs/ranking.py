"""RankingJob — normalize + blend rank_score and RS score → combined_rank."""
from __future__ import annotations

from ..base import Job
from ..context import InferenceContext
from ...selection import score_candidates


class RankingJob(Job):
    """Blends calibrated rank_score and RS score into combined_rank, sorts descending.

    Reads:  ctx.candidates, ctx.config
    Writes: ctx.ranked
    """

    def should_skip(self, ctx: InferenceContext) -> bool:
        return ctx.skip_buys or not ctx.candidates

    def run(self, ctx: InferenceContext) -> None:
        bw  = ctx.config.get("ranking", {}).get("blend_weights", [0.5, 0.5])
        _s  = float(bw[0]) + float(bw[1]) or 1.0
        w_rank = float(bw[0]) / _s
        w_rs   = float(bw[1]) / _s
        ctx.ranked = score_candidates(ctx.candidates, w_rank, w_rs)
