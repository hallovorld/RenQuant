"""RegimeJob — 3-layer regime detection (Hurst + CUSUM + GMM).

Reads:  ctx.spy_returns, ctx.ohlcv["SPY"], ctx.gmm, ctx.regime_state, ctx.config
Writes: ctx.regime_state, ctx.regime, ctx.confidence, ctx.regime_counts
"""
from __future__ import annotations

import numpy as np

from ..context import InferenceContext
from ..pipeline import Job


class RegimeJob(Job):
    """Detect market regime via Hurst + CUSUM + GMM and update ctx."""

    def run(self, ctx: InferenceContext) -> None:
        from kernel.regime import detect_regime  # noqa: PLC0415

        spy_df = ctx.ohlcv.get("SPY")
        spy_returns = np.array(ctx.spy_returns)

        ctx.regime_state = detect_regime(
            spy_returns,
            spy_df,
            ctx.gmm,
            ctx.regime_state,
            ctx.config,
        )

        ctx.regime     = ctx.regime_state.regime
        ctx.confidence = ctx.regime_state.confidence

        # Update regime day-count telemetry
        ctx.regime_counts[ctx.regime] = ctx.regime_counts.get(ctx.regime, 0) + 1
