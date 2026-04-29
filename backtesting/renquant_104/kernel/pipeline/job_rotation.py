"""RotationJob — swap held positions for stronger candidates.

Sits between RankingJob and SelectionJob in Phase 3.  Lets held positions
compete with new candidates on the same calibrated rank_score every bar.

Why:
  Without rotation, a held stock with a marginal +score blocks a far-better
  unowned candidate from ever entering the portfolio.  Mainstream quant
  ranks the entire universe each period and lets winners replace laggards.

How:
  Sell-side scoring is already done by TickerSellJob (ScoreModelTask writes
  rank_score onto each HoldingState).  RotationJob:
    1. Pairs eligible held positions with the strongest free candidates,
       requiring the candidate's score to clear a tax-adjusted swap_margin.
    2. Re-validates each pair against wash-sale, sector, and correlation
       guards on the post-swap virtual portfolio.
    3. Emits a "rotation" exit + a sized buy order for each surviving pair
       and removes the bought ticker from ctx.ranked so SelectionJob does
       not double-buy it.
"""
from __future__ import annotations

from .context  import InferenceContext
from .pipeline import Job, Task
from .task_rotation import BuildPairsTask, ValidatePairsTask, EmitRotationsTask


class RotationJob(Job):
    """Task chain: BuildPairs → ValidatePairs → EmitRotations"""

    def should_skip(self, ctx: InferenceContext) -> bool:
        rcfg = ctx.config.get("rotation", {})
        if not rcfg.get("enabled", False) or not ctx.ranked or not ctx.holdings:
            return True
        if ctx.bear_only:
            return True
        # Audit fix 2026-04-29: rotation fires in BULL_VOLATILE causing whipsaw
        # (-2.5 APY per event recorded in CLAUDE.md). Only run in regimes where
        # rotation is declared profitable (default: BULL_CALM only).
        allowed = rcfg.get("enabled_regimes")
        if allowed is not None and ctx.regime not in allowed:
            return True
        return False

    @property
    def tasks(self) -> list[Task]:
        return [BuildPairsTask(), ValidatePairsTask(), EmitRotationsTask()]
