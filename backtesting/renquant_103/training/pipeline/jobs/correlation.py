"""CorrelationJob — compute pairwise correlation matrix and save artifact."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..base import TrainingJob
from ..context import TrainingContext


class CorrelationJob(TrainingJob):
    def should_skip(self, ctx: TrainingContext) -> bool:
        return not ctx.ohlcv

    def run(self, ctx: TrainingContext) -> None:
        cfg       = ctx.config
        watchlist = cfg.get("watchlist", [])
        oos_start = cfg.get("oos_start", "2024-01-01")

        close_df = pd.DataFrame({
            t: ctx.ohlcv[t]["close"]
            for t in watchlist if t in ctx.ohlcv
        }).loc[oos_start:]

        corr_mat = close_df.pct_change().corr()

        # Save artifact
        artifacts_dir = ctx.strategy_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        corr_path = artifacts_dir / cfg.get("regime", {}).get(
            "correlation_artifact", "watchlist-correlation.json"
        )

        corr_dict: dict[str, dict[str, float]] = {}
        for t in corr_mat.index:
            corr_dict[t] = {
                u: round(float(v), 4) if not np.isnan(v) else 0.0
                for u, v in corr_mat.loc[t].items()
            }

        corr_path.write_text(json.dumps(corr_dict, indent=2))
        ctx.corr_dict = corr_dict
        print(f"CorrelationJob: {len(corr_dict)} tickers, saved to {corr_path.name}")
