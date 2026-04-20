"""SelectionJob — tiered threshold, sector guard, correlation guard, sizing, buy orders."""
from __future__ import annotations

import pandas as pd

from ..base import Job
from ..context import InferenceContext
from ...selection import SelectionContext, run_selection_loop
from ...sizing import compute_position_size


class SelectionJob(Job):
    """Runs the greedy selection loop over ctx.ranked and appends buy orders to ctx.orders.

    Applies: tiered threshold → wash-sale (2nd check) → sector guard → correlation guard
             → position sizing → order creation.

    Reads:  ctx.ranked, ctx.holdings, ctx.portfolio_value, ctx.cash, ctx.regime_params,
            ctx.regime_confidence, ctx.config, ctx.last_sell_dates, ctx.earnings_cal,
            ctx.corr_dict, ctx.today
    Writes: ctx.orders (buy orders appended)
    """

    def should_skip(self, ctx: InferenceContext) -> bool:
        return ctx.skip_buys or not ctx.ranked

    def run(self, ctx: InferenceContext) -> None:
        cfg      = ctx.config
        rp       = ctx.regime_params
        conf     = ctx.regime_confidence
        today_ts = pd.Timestamp(ctx.today)

        max_pos    = cfg.get("max_concurrent_positions", 8)
        open_slots = max_pos - len(ctx.holdings)
        if open_slots <= 0:
            return

        # Confidence-scaled sizing params
        max_pos_pct  = float(rp.get("max_position_pct", 0.15)) * conf
        cash_res_pct = float(rp.get("cash_reserve_pct", 0.0)) * conf

        sel_ctx = SelectionContext(
            today=ctx.today,
            held_tickers=list(ctx.holdings.keys()),
            last_sell_dates=ctx.last_sell_dates,
            earnings_calendar=ctx.earnings_cal,
            corr_matrix=ctx.corr_dict,
            sector_map=cfg.get("sector_map", {}),
            defensive_set=set(cfg.get("defensive_tickers", [])),
            wash_sale_days=cfg.get("wash_sale_days", 30),
            earnings_buffer=cfg.get("regime", {}).get("earnings_buffer_days", 3),
            corr_threshold=cfg.get("regime", {}).get("correlation_guard_threshold", 0.70),
            max_per_sector=cfg.get("max_positions_per_sector", 3),
            tiered_thresholds=cfg.get("tiered_thresholds", [{"min_model_score": 0.10}]),
            open_slots=open_slots,
        )

        selected, _ = run_selection_loop(ctx.ranked, sel_ctx)

        for t in selected:
            df = ctx.ohlcv.get(t)
            if df is None or today_ts not in df.index:
                continue
            price = float(df.loc[today_ts, "close"])
            _, shares = compute_position_size(
                ctx.portfolio_value, ctx.cash, max_pos_pct, cash_res_pct, price
            )
            if shares < 1:
                continue
            invest      = shares * price
            ctx.cash   -= invest   # pre-decrement so next iteration respects budget
            ctx.orders.append({
                "ticker":  t,
                "price":   price,
                "shares":  shares,
                "invest":  invest,
                "regime":  ctx.regime,
            })
