"""RankingJob — blend rank_score + rs_score into a combined ranking.

Reads:  ctx.candidates, ctx.buy_blocked, ctx.bear_only, ctx.config
Writes: ctx.ranked (sorted list of CandidateResult, descending by blended score)
"""
from __future__ import annotations

import logging

from ..context import InferenceContext
from ..pipeline import Job

log = logging.getLogger("kernel.pipeline.ranking")


class RankingJob(Job):
    """Rank candidates by blended (w_rank × rank_score + w_rs × rs_score)."""

    def should_skip(self, ctx: InferenceContext) -> bool:
        if not ctx.candidates:
            return True
        # Skip if buys are fully blocked (BEAR with bear_only is still allowed)
        return ctx.buy_blocked and not ctx.bear_only

    def run(self, ctx: InferenceContext) -> None:
        from kernel.selection import score_candidates  # noqa: PLC0415

        ranking_cfg = ctx.config.get("ranking", {})
        bw = ranking_cfg.get("blend_weights", [0.5, 0.5])
        total = float(bw[0]) + float(bw[1])
        w_rank = float(bw[0]) / total if total > 0 else 0.5
        w_rs   = float(bw[1]) / total if total > 0 else 0.5

        ctx.ranked = score_candidates(ctx.candidates, w_rank, w_rs)
        log.info("RankingJob: %d ranked (w_rank=%.2f w_rs=%.2f)", len(ctx.ranked), w_rank, w_rs)
