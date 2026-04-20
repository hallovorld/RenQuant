"""SellJob — all 5 exits in priority order for every held position."""
from __future__ import annotations

import pandas as pd

from ..base import Job
from ..context import InferenceContext
from ...exits import compute_exits, HoldingState
from ...portfolio import compute_trade_tax


class SellJob(Job):
    """Checks every held position for exit conditions and populates ctx.exit_actions.

    Exit priority: trailing_stop → stop_loss → single_day_loss → max_hold
                   → tax_hold_gate → model_sell

    Reads:  ctx.holdings, ctx.pos_shares, ctx.ohlcv, ctx.today, ctx.regime_params,
            ctx.config, ctx.action_fn
    Writes: ctx.exit_actions  (list of sell dicts ready for the adapter to apply)
            ctx.holdings / ctx.pos_shares are NOT mutated here — the adapter
            applies exit_actions after the full pipeline completes (or the
            NotebookAdapter applies them inline before the buy phase).
    """

    def run(self, ctx: InferenceContext) -> None:
        today_ts = pd.Timestamp(ctx.today)
        rp = ctx.regime_params
        cfg = ctx.config

        exit_params = {
            "trailing_stop_trigger_pct": rp.get("trailing_stop_trigger_pct", 0),
            "trailing_stop_trail_pct":   rp.get("trailing_stop_trail_pct", 0),
            "stop_loss_pct":             rp["stop_loss_pct"],
            "max_single_day_loss_pct":   rp.get("max_single_day_loss_pct", 0),
            "max_hold_days":             rp["max_hold_days"],
            "consecutive_sell_signals":  cfg.get("consecutive_sell_signals", 3),
            "min_hold_days":             cfg.get("min_hold_days", 30),
            "lt_hold_gate_days":         cfg.get("lt_hold_gate_days", 330),
            "lt_hold_min_gain":          cfg.get("lt_hold_min_gain", 0.10),
        }

        st_rate  = cfg["tax"]["short_term_rate"]
        lt_rate  = cfg["tax"]["long_term_rate"]
        lt_days  = cfg["tax"]["long_term_threshold_days"]

        exit_actions = []
        updated_states: dict[str, HoldingState] = {}

        for t, state in ctx.holdings.items():
            df = ctx.ohlcv.get(t)
            if df is None or today_ts not in df.index:
                continue

            price = float(df.loc[today_ts, "close"])
            action = ctx.action_fn(t, today_ts)

            sig, new_state = compute_exits(price, ctx.today, action, state, exit_params)
            new_state.prev_close = price
            updated_states[t] = new_state

            if sig.should_exit:
                shares    = ctx.pos_shares[t]
                hold_days = (ctx.today - state.entry_date).days
                gross_pnl = shares * (price - state.entry_price)
                tax       = compute_trade_tax(gross_pnl, hold_days, st_rate, lt_rate, lt_days)
                exit_actions.append({
                    "ticker":    t,
                    "price":     price,
                    "shares":    shares,
                    "hold_days": hold_days,
                    "pnl_pct":   (price - state.entry_price) / state.entry_price,
                    "tax":       tax,
                    "exit_type": sig.exit_type,
                })

        # Write updated HoldingState back (prev_close updated even if no exit)
        for t, st in updated_states.items():
            ctx.holdings[t] = st

        ctx.exit_actions = exit_actions
