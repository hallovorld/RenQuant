"""Per-ticker sell evaluation tasks."""
from __future__ import annotations

import logging

from .context import TickerInferenceContext
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.sell")


class PrepareHoldingTask(Task):
    """Validate holding + price; attach prev_close."""

    def run(self, tc: TickerInferenceContext) -> bool | None:
        if tc.holding is None:
            return False

        if tc.price <= 0:
            log.warning("PrepareHoldingTask: no price for %s — skipping", tc.ticker)
            return False

        stock_df = tc.ohlcv.get(tc.ticker)
        if stock_df is None:
            return False

        if len(stock_df) >= 2:
            tc.holding.prev_close = float(stock_df["close"].iloc[-2])
        else:
            tc.holding.prev_close = None


class ScoreModelTask(Task):
    """Build feature frame and score model → tc.model_action."""

    def run(self, tc: TickerInferenceContext) -> bool | None:
        from kernel.models     import score_artifact       # noqa: PLC0415
        from kernel.indicators import build_feature_frame  # noqa: PLC0415

        spy_df   = tc.ohlcv.get("SPY")
        stock_df = tc.ohlcv.get(tc.ticker)

        if tc.model is None or spy_df is None or stock_df is None:
            tc.model_action = "hold"
            return

        spec    = tc.config.get("indicator_spec", {})
        vol_win = int(tc.config.get("regime", {}).get("vol_realized_window", 20))
        tc.features = build_feature_frame(stock_df, spy_df, spec, vol_win)

        if tc.features is not None and not tc.features.empty:
            sr = score_artifact(tc.model, tc.features.iloc[-1], qty=1)
            tc.model_action = sr.signal
        else:
            tc.model_action = "hold"

        log.debug("ScoreModelTask [%s]: action=%s", tc.ticker, tc.model_action)


class EvaluateExitsTask(Task):
    """Run the 5-exit priority chain; update tc.holding and tc.exit_signal."""

    def run(self, tc: TickerInferenceContext) -> bool | None:
        from kernel.exits import compute_exits  # noqa: PLC0415

        sig, updated_hs = compute_exits(
            tc.price, tc.today, tc.model_action, tc.holding, tc.exit_params
        )
        tc.holding = updated_hs

        if sig.should_exit:
            tc.exit_signal = sig
        elif tc.model_action == "sell" and updated_hs.sell_streak > 0:
            sig._blocked_streak = True   # noqa: SLF001
            tc.exit_signal = sig

        log.debug("EvaluateExitsTask [%s]: should_exit=%s  type=%s",
                  tc.ticker, sig.should_exit, getattr(sig, "exit_type", None))
