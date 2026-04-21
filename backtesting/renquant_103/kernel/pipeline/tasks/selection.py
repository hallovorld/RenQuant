"""Greedy slot-filling tasks: prepare context → run selection loop → size and emit.

Reads:  ctx.ranked, ctx.holdings, ctx.last_sell_dates, ctx.portfolio_value,
        ctx.cash, ctx.prices, ctx.regime, ctx.confidence, ctx.bear_only,
        ctx.corr_matrix, ctx.earnings_calendar, ctx.config, ctx.today
Writes: ctx.orders (list of order dicts)
        ctx.counters["blocked_wash", "sector_blocks", "corr_blocks"] incremented
"""
from __future__ import annotations

import logging

from ..context import InferenceContext
from ..pipeline import Task

log = logging.getLogger("kernel.pipeline.selection")


class PrepareSelectionTask(Task):
    """Compute open slots, apply BEAR cap, build SelectionContext → ctx._sel_ctx."""

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.selection import SelectionContext  # noqa: PLC0415

        config         = ctx.config
        regime_cfg     = config.get("regime", {})
        max_positions  = int(config.get("max_concurrent_positions", 8))
        wash_days      = int(config.get("wash_sale_days", 0))
        earnings_buf   = int(regime_cfg.get("earnings_buffer_days", 3))
        corr_threshold = float(regime_cfg.get("correlation_guard_threshold", 0.70))
        max_per_sector = int(config.get("max_positions_per_sector", 0))
        sector_map     = config.get("sector_map", {})
        defensive_set  = set(config.get("defensive_tickers", []))
        tiered         = config.get("tiered_thresholds", [])

        held       = list(ctx.holdings.keys())
        open_slots = max_positions - len(held)

        if open_slots <= 0:
            log.info("PrepareSelectionTask: no open slots")
            return False

        if ctx.bear_only:
            open_slots = min(open_slots, 1)

        ctx._sel_ctx = SelectionContext(  # noqa: SLF001
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
        log.debug("PrepareSelectionTask: open_slots=%d  bear_only=%s", open_slots, ctx.bear_only)


class RunSelectionTask(Task):
    """Run the greedy selection loop → ctx._selected, ctx._blocks; update counters."""

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.selection import run_selection_loop  # noqa: PLC0415

        sel_ctx = ctx._sel_ctx  # noqa: SLF001
        selected, blocks = run_selection_loop(ctx.ranked, sel_ctx)

        ctx._selected = selected  # noqa: SLF001
        ctx._blocks   = blocks    # noqa: SLF001

        ctx.counters["blocked_wash"]  = ctx.counters.get("blocked_wash",  0) + blocks.get("wash_sale",   0)
        ctx.counters["sector_blocks"] = ctx.counters.get("sector_blocks", 0) + blocks.get("sector",      0)
        ctx.counters["corr_blocks"]   = ctx.counters.get("corr_blocks",   0) + blocks.get("correlation", 0)

        log.debug("RunSelectionTask: %d selected  blocks=%s", len(selected), blocks)


class SizeAndEmitTask(Task):
    """Size each selected ticker and emit buy orders → ctx.orders."""

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.sizing import compute_position_size  # noqa: PLC0415

        regime_p     = ctx.config.get("regime_params", {}).get(ctx.regime, {})
        max_pct      = float(regime_p.get("max_position_pct", 0.15)) * ctx.confidence
        reserve_pct  = float(regime_p.get("cash_reserve_pct", 0.0))  * ctx.confidence
        override_pct = 0.15 if ctx.bear_only else None  # BEAR defensive uses fixed 15%

        for ticker in ctx._selected:  # noqa: SLF001
            price = ctx.prices.get(ticker)
            if price is None or price <= 0:
                log.warning("SizeAndEmitTask: no price for %s — skipping", ticker)
                continue

            _, shares = compute_position_size(
                ctx.portfolio_value, ctx.cash,
                max_pct, reserve_pct, price,
                override_pct=override_pct,
            )
            if shares < 1:
                log.info("SizeAndEmitTask: %s insufficient cash — skip", ticker)
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
            log.info("SizeAndEmitTask: %s BUY %d shares @ %.2f (%.1f%%)",
                     ticker, shares, price, target_pct * 100)

        log.info("SizeAndEmitTask: %d orders placed  blocks=%s",
                 len(ctx.orders), getattr(ctx, "_blocks", {}))
