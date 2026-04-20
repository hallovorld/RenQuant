"""CalibrationJob — refit score calibration and blend weights, write to config."""
from __future__ import annotations

from ..base import TrainingJob
from ..context import TrainingContext


class CalibrationJob(TrainingJob):
    """Delegates to scripts/recalibrate_scores.py logic via training.scoring."""

    def should_skip(self, ctx: TrainingContext) -> bool:
        return not ctx.exported

    def run(self, ctx: TrainingContext) -> None:
        import json
        from training.scoring import recalibrate_scores, fit_blend_weights

        cfg           = ctx.config
        strategy_dir  = ctx.strategy_dir
        config_path   = strategy_dir / "strategy_config.json"

        # Recalibrate per-model score calibration
        recalibrate_scores(ctx.tournament_results, ctx.exported, strategy_dir, ctx.today)

        # Fit blend weights from OOS data
        bw = fit_blend_weights(ctx.tournament_results, ctx.ohlcv, cfg)
        ctx.blend_weights = bw

        # Persist into strategy_config.json
        live_cfg = json.loads(config_path.read_text())
        live_cfg.setdefault("ranking", {})["blend_weights"] = bw
        config_path.write_text(json.dumps(live_cfg, indent=2))
        print(f"CalibrationJob: blend_weights={bw}")
