"""ExportJob — save model artifacts and retrain live models."""
from __future__ import annotations

from ..base import TrainingJob
from ..context import TrainingContext


class ExportJob(TrainingJob):
    def should_skip(self, ctx: TrainingContext) -> bool:
        return not ctx.tournament_results

    def run(self, ctx: TrainingContext) -> None:
        from training.export import export_models, retrain_live_models

        cfg       = ctx.config
        mp        = cfg.get("model_params", {})
        floor     = float(cfg.get("sharpe_floor", 0.8))
        lookahead = mp.get("lookahead", 5)
        strategy  = cfg.get("strategy", "renquant_103")

        exported, skipped = export_models(
            ctx.tournament_results, ctx.strategy_dir,
            ctx.today, floor, lookahead, strategy,
        )
        retrain_live_models(
            ctx.tournament_results, ctx.feature_frames,
            exported, ctx.strategy_dir, mp, cfg, ctx.today,
        )
        ctx.exported = exported
        ctx.skipped  = skipped
        print(f"ExportJob: exported={len(exported)}, skipped={len(skipped)}")
