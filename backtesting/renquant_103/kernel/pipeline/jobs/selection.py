"""SelectionJob — greedy slot-filling with tiered thresholds and all guards.

Reads:  ctx.ranked, ctx.holdings, ctx.last_sell_dates, ctx.portfolio_value,
        ctx.cash, ctx.prices, ctx.regime, ctx.confidence, ctx.bear_only,
        ctx.corr_matrix, ctx.earnings_calendar, ctx.config, ctx.today
Writes: ctx.orders (list of order dicts)
        ctx.counters["wash_sale", "sector_blocks", "corr_blocks"] incremented
"""
from __future__ import annotations

import logging

from ..context import InferenceContext
from ..pipeline import Job

log = logging.getLogger("kernel.pipeline.selection")


class SelectionJob(Job):
    """Fill open slots from ctx.ranked, applying all guards and sizing."""

    def should_skip(self, ctx: InferenceContext) -> bool:
        return not ctx.ranked

    def run(self, ctx: InferenceContext) -> None:
        from kernel.selection import (  # noqa: PLC0415
            SelectionContext, run_selection_loop, is_wash_sale_blocked
        )
        from kernel.sizing import compute_position_size  # noqa: PLC0415

        config        = ctx.config
        regime_p      = config.get("regime_params", {}).get(ctx.regime, {})
        regime_cfg    = config.get("regime", {})
        max_positions = int(config.get("max_concurrent_positions", 8))
        wash_days     = int(config.get("wash_sale_days", 0))
        earnings_buf  = int(regime_cfg.get("earnings_buffer_days", 3))
        corr_threshold = float(regime_cfg.get("correlation_guard_threshold", 0.70))
        max_per_sector = int(config.get("max_positions_per_sector", 0))
        sector_map    = config.get("sector_map", {})
        defensive_set = set(config.get("defensive_tickers", []))
        tiered        = config.get("tiered_thresholds", [])

        held = list(ctx.holdings.keys())
        open_slots = max_positions - len(held)

        if open_slots <= 0:
            log.info("SelectionJob: no open slots")
            return

        # BEAR: max 1 defensive slot
        if ctx.bear_only:
            open_slots = min(open_slots, 1)

        sel_ctx = SelectionContext(
            today             = ctx.today,
            held_tickers      = held,
            last_sell_dates   = ctx.last_sell_dates,
            earnings_calendar = ctx.earnings_calendar or {},
            corr_matrix       = ctx.corr_matrix,
            sector_map        = sector_map,
            defensive_set     = defensive_set,
            wash_sale_days    = wash_days,
            earnings_buffer   = earnings_buf,
            corr_threshold    = corr_threshold,
            max_per_sector    = max_per_sector,
            tiered_thresholds = tiered,
            open_slots        = open_slots,
        )

        selected, blocks = run_selection_loop(ctx.ranked, sel_ctx)

        ctx.counters["blocked_wash"]   = ctx.counters.get("blocked_wash", 0)   + blocks.get("wash_sale",    0)
        ctx.counters["sector_blocks"]  = ctx.counters.get("sector_blocks", 0)  + blocks.get("sector",       0)
        ctx.counters["corr_blocks"]    = ctx.counters.get("corr_blocks", 0)    + blocks.get("correlation",  0)

        # Confidence-scaled sizing params
        max_pct    = float(regime_p.get("max_position_pct", 0.15)) * ctx.confidence
        reserve_pct = float(regime_p.get("cash_reserve_pct", 0.0)) * ctx.confidence
        override_pct = 0.15 if ctx.bear_only else None  # BEAR defensive uses fixed 15%

        for ticker in selected:
            price = ctx.prices.get(ticker)
            if price is None or price <= 0:
                log.warning("SelectionJob: no price for %s — skipping", ticker)
                continue

            _, shares = compute_position_size(
                ctx.portfolio_value, ctx.cash,
                max_pct, reserve_pct, price,
                override_pct=override_pct,
            )
            if shares < 1:
                log.info("SelectionJob: %s insufficient cash — skip", ticker)
                continue

            invest     = shares * price
            target_pct = invest / ctx.portfolio_value if ctx.portfolio_value > 0 else 0.0

            c = next((c for c in ctx.ranked if c.ticker == ticker), None)
            ctx.orders.append({
                "ticker":     ticker,
                "shares":     shares,
                "price":      price,
                "invest":     invest,
                "target_pct": target_pct,
                "regime":     ctx.regime,
                "confidence": ctx.confidence,
                "rank_score": c.rank_score if c else 0.0,
                "rs_score":   c.rs_score   if c else 0.0,
                "detail":     c.detail     if c else "",
            })
            log.info("SelectionJob: %s BUY %d shares @ %.2f (%.1f%%)",
                     ticker, shares, price, target_pct * 100)

        log.info("SelectionJob: %d orders placed, blocks=%s", len(ctx.orders), blocks)
