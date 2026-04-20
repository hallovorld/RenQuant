"""RegimeFitJob — train GMM on SPY features and save spy-gmm-regime.json."""
from __future__ import annotations

from ..base import TrainingJob
from ..context import TrainingContext


class RegimeFitJob(TrainingJob):
    def should_skip(self, ctx: TrainingContext) -> bool:
        return "SPY" not in ctx.ohlcv

    def run(self, ctx: TrainingContext) -> None:
        from training.regime import fit_and_save_gmm

        artifacts_dir = ctx.strategy_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        gmm_path = fit_and_save_gmm(
            ctx.ohlcv["SPY"],
            artifacts_dir / ctx.config["regime"].get("gmm_artifact", "spy-gmm-regime.json"),
        )
        ctx.gmm_artifact_path = gmm_path
        print(f"RegimeFitJob: GMM saved to {gmm_path.name}")
