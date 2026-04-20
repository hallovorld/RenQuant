"""TournamentJob — 4-model tournament per ticker, pick best by OOS Sharpe."""
from __future__ import annotations

from ..base import TrainingJob
from ..context import TrainingContext


class TournamentJob(TrainingJob):
    def should_skip(self, ctx: TrainingContext) -> bool:
        return not ctx.feature_frames

    def run(self, ctx: TrainingContext) -> None:
        from training.tournament import run_tournament_all

        cfg = ctx.config
        ctx.tournament_results = run_tournament_all(
            cfg.get("watchlist", []),
            ctx.feature_frames,
            ctx.ohlcv,
            cfg,
        )
        passed = sum(1 for r in ctx.tournament_results.values() if r.get("passes_floor"))
        print(f"TournamentJob: {passed}/{len(ctx.tournament_results)} tickers passed Sharpe floor")
