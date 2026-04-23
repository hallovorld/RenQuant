"""Selection tasks: prepare context → run greedy loop → size and emit orders."""
from __future__ import annotations

import logging

from .context import InferenceContext
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.selection")


class PrepareSelectionTask(Task):
    """Compute open slots, apply BEAR cap, build SelectionContext → ctx._sel_ctx."""

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.selection import SelectionContext  # noqa: PLC0415

        config         = ctx.config
        regime_cfg     = config.get("regime", {})
        regime_params  = config.get("regime_params", {}).get(ctx.regime, {})
        max_positions  = int(regime_params.get(
            "max_concurrent_positions",
            config.get("max_concurrent_positions", 8),
        ))
        wash_days      = int(config.get("wash_sale_days", 0))
        earnings_buf   = int(regime_cfg.get("earnings_buffer_days", 3))
        corr_threshold = float(regime_cfg.get("correlation_guard_threshold", 0.70))
        max_per_sector = int(config.get("max_positions_per_sector", 0))
        sector_map     = config.get("sector_map", {})
        defensive_set  = set(config.get("defensive_tickers", []))
        tiered         = config.get("tiered_thresholds", [])

        # Account for rotations already emitted by RotationJob: the sells will
        # be liquidated this bar (so they don't count as held for guards) and
        # the buys are already booked (so they do count as held for guards).
        rotation_sells = {p.sell_ticker for p in (ctx.rotations or [])}
        rotation_buys  = {p.buy_ticker  for p in (ctx.rotations or [])}
        effective_held = (set(ctx.holdings.keys()) - rotation_sells) | rotation_buys

        held       = list(effective_held)
        open_slots = max_positions - len(held)

        if open_slots <= 0:
            log.info("PrepareSelectionTask: no open slots")
            return False

        if ctx.bear_only:
            bear_slots     = int(config.get("bear_defensive_slots", 1))
            defensive_held = sum(1 for t in held if t in defensive_set)
            remaining      = max(bear_slots - defensive_held, 0)
            open_slots     = min(open_slots, remaining)

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


class RunSelectionTask(Task):
    """Run the greedy selection loop → ctx._selected, ctx._blocks; update counters."""

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.selection import run_selection_loop  # noqa: PLC0415

        selected, blocks = run_selection_loop(ctx.ranked, ctx._sel_ctx)  # noqa: SLF001
        ctx._selected = selected  # noqa: SLF001
        ctx._blocks   = blocks    # noqa: SLF001

        ctx.counters["blocked_wash"]  = ctx.counters.get("blocked_wash",  0) + blocks.get("wash_sale",   0)
        ctx.counters["sector_blocks"] = ctx.counters.get("sector_blocks", 0) + blocks.get("sector",      0)
        ctx.counters["corr_blocks"]   = ctx.counters.get("corr_blocks",   0) + blocks.get("correlation", 0)


class SizeAndEmitTask(Task):
    """Size each selected ticker and emit buy orders → ctx.orders."""

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.sizing import (  # noqa: PLC0415
            compute_position_size,
            conviction_multiplier,
            sigma_multiplier,
            universe_sigma_median,
        )

        regime_p      = ctx.config.get("regime_params", {}).get(ctx.regime, {})
        base_max_pct  = float(regime_p.get("max_position_pct", 0.15)) * ctx.confidence
        reserve_pct   = float(regime_p.get("cash_reserve_pct", 0.0))  * ctx.confidence
        bear_def_pct  = float(ctx.config.get("bear_defensive_pct", 0.15))
        override_pct  = bear_def_pct if ctx.bear_only else None
        sizing_cfg    = (ctx.config.get("ranking", {})
                          .get("panel_scoring", {}).get("sizing", {}))
        sigma_cfg     = (ctx.config.get("ranking", {})
                          .get("panel_scoring", {})
                          .get("sigma_sizing", {}))

        # Universe σ median over all ranked candidates (σ written by ApplyNGBoostTask).
        sigma_median = universe_sigma_median(
            [getattr(c, "sigma", None) for c in ctx.ranked]
        )

        for ticker in ctx._selected:  # noqa: SLF001
            price = ctx.prices.get(ticker)
            if price is None or price <= 0:
                log.warning("SizeAndEmitTask: no price for %s — skipping", ticker)
                continue

            c = next((c for c in ctx.ranked if c.ticker == ticker), None)
            conv = conviction_multiplier(
                getattr(c, "panel_score", None) if c else None, sizing_cfg,
            )
            sig_m = sigma_multiplier(
                getattr(c, "sigma", None) if c else None,
                sigma_median, sigma_cfg,
            )
            max_pct = base_max_pct * conv * sig_m

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
            ctx.orders.append({
                "ticker":     ticker,
                "shares":     shares,
                "price":      price,
                "invest":     invest,
                "target_pct": target_pct,
                "regime":     ctx.regime,
                "confidence": ctx.confidence,
                "conviction": conv,
                "sigma_mult": sig_m,
                "rank_score": c.rank_score  if c else 0.0,
                "rs_score":   c.rs_score    if c else 0.0,
                "panel_score": getattr(c, "panel_score", None) if c else None,
                "sigma":      getattr(c, "sigma", None)        if c else None,
                "mu":         getattr(c, "mu", None)           if c else None,
                "detail":     c.detail      if c else "",
            })
            log.info(
                "SizeAndEmitTask: %s BUY %d shares @ %.2f "
                "(%.1f%% conv=%.2f σ_mult=%.2f)",
                ticker, shares, price, target_pct * 100, conv, sig_m,
            )

        log.info("SizeAndEmitTask: %d orders placed", len(ctx.orders))
