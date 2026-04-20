"""RegimeJob — layers 1-3 regime detection + hard override + resolve + confidence."""
from __future__ import annotations

from ..base import Job
from ..context import InferenceContext
from ...regime import detect_regime


class RegimeJob(Job):
    """Runs 3-layer regime detection (Hurst + CUSUM + GMM) for today's bar.

    Reads:  ctx.spy_returns, ctx.ohlcv["SPY"], ctx.gmm_artifact, ctx.regime_state, ctx.config
    Writes: ctx.regime, ctx.regime_confidence, ctx.in_transition, ctx.regime_params,
            ctx.regime_state (updated in-place)
    """

    def run(self, ctx: InferenceContext) -> None:
        spy_df_window = ctx.ohlcv.get("SPY")
        if spy_df_window is None:
            ctx.regime = "BULL_CALM"
            ctx.regime_confidence = 0.5
            ctx.in_transition = False
        else:
            import pandas as pd
            today_ts = pd.Timestamp(ctx.today)
            spy_df_window = spy_df_window.loc[:today_ts]

            ctx.regime_state = detect_regime(
                ctx.spy_returns,
                spy_df_window,
                ctx.gmm_artifact,
                ctx.regime_state,
                ctx.config,
            )
            ctx.regime = ctx.regime_state.regime
            ctx.regime_confidence = ctx.regime_state.confidence
            ctx.in_transition = ctx.regime_state.in_transition

        rp_table = ctx.config.get("regime_params", {})
        ctx.regime_params = rp_table.get(ctx.regime, rp_table.get("BULL_CALM", {}))
