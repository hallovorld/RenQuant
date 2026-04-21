"""SellJob — evaluate exit signals for all held positions.

Reads:  ctx.holdings, ctx.prices, ctx.ohlcv, ctx.models, ctx.regime,
        ctx.config, ctx.today
Writes: ctx.exits (list of (ticker, ExitSignal))
        ctx.holdings (updated HoldingState per ticker)
        ctx.counters["blocked_streak"] incremented when streak not met
"""
from __future__ import annotations

import logging

from ..context import InferenceContext
from ..pipeline import Job

log = logging.getLogger("kernel.pipeline.sell")


class SellJob(Job):
    """Run all 5 exit checks for each held ticker; populate ctx.exits."""

    def run(self, ctx: InferenceContext) -> None:
        from kernel.exits import compute_exits          # noqa: PLC0415
        from kernel.models import score_artifact        # noqa: PLC0415
        from kernel.indicators import build_feature_frame  # noqa: PLC0415

        config    = ctx.config
        regime_p  = config.get("regime_params", {}).get(ctx.regime, {})
        spec      = config.get("indicator_spec", {})
        vol_win   = int(config.get("regime", {}).get("vol_realized_window", 20))
        spy_df    = ctx.ohlcv.get("SPY")

        exit_params = _build_exit_params(regime_p, config)

        for ticker, hs in list(ctx.holdings.items()):
            current_price = ctx.prices.get(ticker)
            if current_price is None or current_price <= 0:
                log.warning("SellJob: no price for %s — skipping", ticker)
                continue

            stock_df = ctx.ohlcv.get(ticker)
            if stock_df is None:
                continue

            # Attach prev_close to HoldingState
            if len(stock_df) >= 2:
                hs.prev_close = float(stock_df["close"].iloc[-2])
            else:
                hs.prev_close = None

            # Model signal — needed for sell-streak check
            model_action = "hold"
            artifact = ctx.models.get(ticker)
            if artifact is not None and spy_df is not None:
                features = build_feature_frame(stock_df, spy_df, spec, vol_win)
                if features is not None and not features.empty:
                    qty = 1  # non-zero → held position bucket in Q-learning
                    sr = score_artifact(artifact, features.iloc[-1], qty)
                    model_action = sr.signal

            sig, hs = compute_exits(current_price, ctx.today, model_action, hs, exit_params)
            ctx.holdings[ticker] = hs  # persist updated streak + HWM

            if sig.should_exit:
                ctx.exits.append((ticker, sig))
            elif model_action == "sell" and hs.sell_streak > 0:
                ctx.counters["blocked_streak"] = ctx.counters.get("blocked_streak", 0) + 1
                log.debug("%s sell streak %d — waiting", ticker, hs.sell_streak)


# ── Helper ─────────────────────────────────────────────────────────────────────

def _build_exit_params(regime_p: dict, config: dict) -> dict:
    return {
        "trailing_stop_trigger_pct": regime_p.get("trailing_stop_trigger_pct", 0),
        "trailing_stop_trail_pct":   regime_p.get("trailing_stop_trail_pct",   0),
        "stop_loss_pct":             regime_p.get("stop_loss_pct",             0),
        "max_single_day_loss_pct":   regime_p.get("max_single_day_loss_pct",   0),
        "max_hold_days":             regime_p.get("max_hold_days",             0),
        "consecutive_sell_signals":  int(config.get("consecutive_sell_signals", 3)),
        "min_hold_days":             int(config.get("min_hold_days", 0)),
        "lt_hold_gate_days":         int(config.get("lt_hold_gate_days", 0)),
        "lt_hold_min_gain":          float(config.get("lt_hold_min_gain", 0.10)),
    }
