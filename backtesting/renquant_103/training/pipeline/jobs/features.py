"""FeatureJob — build labelled training feature frames per ticker."""
from __future__ import annotations

from ..base import TrainingJob
from ..context import TrainingContext


class FeatureJob(TrainingJob):
    def should_skip(self, ctx: TrainingContext) -> bool:
        return not ctx.ohlcv

    def run(self, ctx: TrainingContext) -> None:
        from training.features import build_all_training_features

        cfg         = ctx.config
        watchlist   = cfg.get("watchlist", [])
        mp          = cfg.get("model_params", {})
        spec        = cfg.get("indicator_spec", {})
        lookahead   = mp.get("lookahead", 5)
        threshold   = mp.get("threshold", 0.03)

        ctx.feature_frames = build_all_training_features(
            watchlist, ctx.ohlcv, spec, lookahead, threshold,
        )
        print(f"FeatureJob: built frames for {len(ctx.feature_frames)} tickers")
