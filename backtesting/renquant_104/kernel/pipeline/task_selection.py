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
            bear_only         = bool(ctx.bear_only),
        )


class RunSelectionTask(Task):
    """Run the greedy selection loop → ctx._selected, ctx._blocks; update counters.

    Also populates ctx._blocked_by_ticker (Plan P): per-ticker rejection
    reason, fed to candidate_scores.blocked_by in the decision-trace DB.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.selection import run_selection_loop  # noqa: PLC0415

        blocked_by_ticker: dict[str, str] = {}
        selected, blocks = run_selection_loop(
            ctx.ranked, ctx._sel_ctx,  # noqa: SLF001
            blocked_by_ticker=blocked_by_ticker,
        )
        ctx._selected          = selected            # noqa: SLF001
        ctx._blocks            = blocks              # noqa: SLF001
        ctx._blocked_by_ticker = blocked_by_ticker   # noqa: SLF001

        ctx.counters["blocked_wash"]  = ctx.counters.get("blocked_wash",  0) + blocks.get("wash_sale",   0)
        ctx.counters["sector_blocks"] = ctx.counters.get("sector_blocks", 0) + blocks.get("sector",      0)
        ctx.counters["corr_blocks"]   = ctx.counters.get("corr_blocks",   0) + blocks.get("correlation", 0)
        # Plan O — non-BEAR defensive rejections (e.g. XLU in BULL_VOLATILE).
        ctx.counters["defensive_non_bear_blocks"] = (
            ctx.counters.get("defensive_non_bear_blocks", 0)
            + blocks.get("defensive_non_bear", 0)
        )


class SizeAndEmitTask(Task):
    """Size each selected ticker and emit buy orders → ctx.orders."""

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.sizing import (  # noqa: PLC0415
            compute_position_size,
            conviction_multiplier,
            sigma_multiplier,
            universe_sigma_median,
        )

        # Audit fix CONF-MULT (2026-04-25): use floored confidence multiplier
        # so low confidence (e.g. 0.0041 from a Hurst/GMM disagreement) doesn't
        # collapse position size to ~$0. See kernel/regime.py::confidence_to_size_multiplier.
        from kernel.regime import confidence_to_size_multiplier  # noqa: PLC0415
        _conf_mult    = confidence_to_size_multiplier(ctx.confidence)
        regime_p      = ctx.config.get("regime_params", {}).get(ctx.regime, {})
        base_max_pct  = float(regime_p.get("max_position_pct", 0.15)) * _conf_mult

        # CUSUM-v2 Design C (user-locked 2026-04-24): when
        # `regime.cusum_cooldown_mode == "wall_time"`, scale max_pct by
        # cooldown_progress (0→1 over cusum_cooldown_days). Default mode
        # "bar_count" preserves v4 behaviour (hard transition block via
        # TransitionWindowTask; this path is a no-op).
        cooldown_mult = 1.0
        _regime_cfg = ctx.config.get("regime", {})
        if str(_regime_cfg.get("cusum_cooldown_mode", "bar_count")) == "wall_time":
            from kernel.regime import cusum_cooldown_progress  # noqa: PLC0415
            cd_start = getattr(ctx.regime_state, "cooldown_start", None) \
                       if ctx.regime_state is not None else None
            cd_days  = float(_regime_cfg.get("cusum_cooldown_days", 3.0))
            cooldown_mult = cusum_cooldown_progress(ctx.today, cd_start, cd_days)
            if cooldown_mult < 1.0:
                log.info("SizeAndEmitTask: CUSUM cooldown active — "
                         "scaling max_pct × %.3f", cooldown_mult)
        base_max_pct *= cooldown_mult
        reserve_pct   = float(regime_p.get("cash_reserve_pct", 0.0))  * _conf_mult
        bear_def_pct  = float(ctx.config.get("bear_defensive_pct", 0.15))
        override_pct  = bear_def_pct if ctx.bear_only else None
        sizing_cfg    = (ctx.config.get("ranking", {})
                          .get("panel_scoring", {}).get("sizing", {}))
        sigma_cfg     = (ctx.config.get("ranking", {})
                          .get("panel_scoring", {})
                          .get("sigma_sizing", {}))
        kelly_cfg     = ctx.config.get("ranking", {}).get("kelly_sizing", {})
        kelly_on      = bool(kelly_cfg.get("enabled", False))
        # When Kelly is primary sizer, conviction_multiplier (derived from
        # panel_score) and sigma_multiplier (inverse of σ) approximately
        # re-scale the SAME quantities Kelly already encodes (μ and σ²).
        # Flag lets us test the pure-Kelly hypothesis — no stacked
        # multipliers. Default False preserves v4 behaviour.
        kelly_pure    = bool(kelly_cfg.get("disable_extra_multipliers", False))

        # Universe σ median over all ranked candidates (σ written by ApplyNGBoostTask).
        sigma_median = universe_sigma_median(
            [getattr(c, "sigma", None) for c in ctx.ranked]
        )

        # Audit fix SE-1 (Round 2 deep audit, 2026-04-25): pre-fix,
        # `if price is None or price <= 0` let NaN slip through (NaN<=0
        # is False), then `int(invest / NaN_price)` propagated NaN into
        # share counts and order dicts. Fail-SAFE: treat non-finite price
        # the same as None — skip the ticker, log a warning so operators
        # see WHICH ticker had bad data.
        import math as _math
        for ticker in ctx._selected:  # noqa: SLF001
            price = ctx.prices.get(ticker)
            if price is None or not _math.isfinite(price) or price <= 0:
                log.warning("SizeAndEmitTask: bad price (%s) for %s — skipping",
                            price, ticker)
                continue

            c = next((c for c in ctx.ranked if c.ticker == ticker), None)
            if kelly_on and kelly_pure:
                # Pure-Kelly mode — neutralise extra multipliers that
                # overlap with Kelly's μ / σ² inputs.
                conv, sig_m = 1.0, 1.0
            else:
                conv = conviction_multiplier(
                    getattr(c, "panel_score", None) if c else None, sizing_cfg,
                )
                sig_m = sigma_multiplier(
                    getattr(c, "sigma", None) if c else None,
                    sigma_median, sigma_cfg,
                )
            # Plan C: when kelly_sizing.enabled, the target position
            # weight is the Kelly number precomputed by
            # ApplyKellySizingTask (f* = μ/σ², capped at max_pct +
            # max_concentration). Otherwise: legacy max_pct × conv × σ.
            if kelly_on and c is not None and getattr(c, "kelly_target_pct", None) is not None:
                max_pct = float(c.kelly_target_pct) * conv * sig_m
                if max_pct <= 0:
                    log.info("SizeAndEmitTask: %s Kelly=0 — skip", ticker)
                    continue
            else:
                max_pct = base_max_pct * conv * sig_m

            # Multi-entry accumulation (user-requested 2026-04-24):
            # "65% OK, but not from one session — allow model to buy same
            # stock multiple times". When `per_session_buy_cap` is set,
            # cap any ONE order's target fraction at that value even if
            # kelly_target is higher. Over multiple sessions, top-up and
            # new-buy orders can still build up to the full kelly_target
            # via TopUpHeldTask. Default None = unchanged behaviour.
            per_session_cap = kelly_cfg.get("per_session_buy_cap")
            if per_session_cap is not None:
                cap = float(per_session_cap)
                if cap > 0 and max_pct > cap:
                    log.info("SizeAndEmitTask: %s max_pct %.3f capped to "
                              "per_session %.3f (multi-entry mode)",
                              ticker, max_pct, cap)
                    max_pct = cap

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
                # Thesis-degradation baseline (Approach A) — carry the
                # Kelly target THE MODEL COMPUTED for this candidate so
                # adapters can stamp it as entry_kelly_target_pct. Distinct
                # from `target_pct` (the actually-sized fraction).
                "kelly_target_pct": getattr(c, "kelly_target_pct", None) if c else None,
                "detail":     c.detail      if c else "",
            })
            log.info(
                "SizeAndEmitTask: %s BUY %d shares @ %.2f "
                "(%.1f%% conv=%.2f σ_mult=%.2f)",
                ticker, shares, price, target_pct * 100, conv, sig_m,
            )

        log.info("SizeAndEmitTask: %d orders placed", len(ctx.orders))
