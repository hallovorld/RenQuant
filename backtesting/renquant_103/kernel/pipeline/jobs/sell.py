"""TickerSellJob — evaluate exit signals for one held position.

Per-ticker job: reads/writes TickerInferenceContext only.
Run in parallel by InferencePipeline for all held tickers.

Reads:  tc.ticker, tc.ohlcv, tc.model, tc.holding, tc.price,
        tc.exit_params, tc.config, tc.today, tc.regime
Writes: tc.holding (updated streak + HWM), tc.exit_signal
"""
from __future__ import annotations

import logging

from ..context import TickerInferenceContext
from ..pipeline import TickerJob

log = logging.getLogger("kernel.pipeline.sell")


class TickerSellJob(TickerJob):
    """Compute exit signal for one held ticker."""

    def run(self, tc: TickerInferenceContext) -> None:
        from kernel.exits      import compute_exits       # noqa: PLC0415
        from kernel.models     import score_artifact      # noqa: PLC0415
        from kernel.indicators import build_feature_frame # noqa: PLC0415

        hs = tc.holding
        if hs is None:
            return

        price = tc.price
        if price <= 0:
            log.warning("TickerSellJob: no price for %s — skipping", tc.ticker)
            return

        stock_df = tc.ohlcv.get(tc.ticker)
        spy_df   = tc.ohlcv.get("SPY")
        if stock_df is None:
            return

        # Attach prev_close
        if len(stock_df) >= 2:
            hs.prev_close = float(stock_df["close"].iloc[-2])
        else:
            hs.prev_close = None

        # Model action for sell-streak check
        model_action = "hold"
        if tc.model is not None and spy_df is not None:
            spec    = tc.config.get("indicator_spec", {})
            vol_win = int(tc.config.get("regime", {}).get("vol_realized_window", 20))
            features = build_feature_frame(stock_df, spy_df, spec, vol_win)
            if features is not None and not features.empty:
                sr = score_artifact(tc.model, features.iloc[-1], qty=1)
                model_action = sr.signal

        sig, hs = compute_exits(price, tc.today, model_action, hs, tc.exit_params)
        tc.holding = hs   # updated streak + HWM

        if sig.should_exit:
            tc.exit_signal = sig
        elif model_action == "sell" and hs.sell_streak > 0:
            # Signal back to orchestrator that a streak is accumulating
            sig._blocked_streak = True   # noqa: SLF001 — lightweight flag
            tc.exit_signal = sig
