"""JointActionTask — unified buy / sell / rotate action selection.

Phase 2 of the rotation algorithm rewrite (2026-04-25). When
`rotation.joint_actions.enabled = true`, this Task replaces the
traditional RotationJob + SelectionJob pipeline with a single greedy
selector over a unified action menu where buys, sells, and rotations
all compete for the same slot budget.

Algorithm:

  1. Build action menu:
       * BUY    — for each cand with rank_score >= panel_buy_floor:
                     net_alpha = cand.expected_return - fee - slippage
       * SELL   — for each held with rank_score <= panel_sell_floor:
                     net_alpha = -held.expected_return - fee - slippage
                                 - tax_drag(held)
       * ROTATE — for each (held, cand) where both pass their floors:
                     net_alpha = (cand.ER - held.ER) - 2*(fee + slippage)
                                 - tax_drag(held)

  2. Sort actions by net_alpha desc.

  3. Greedy fill:
       slot_budget = max(open_slots, max_rotations_per_bar)  ("shared" mode)
       cash_remaining, sectors_used, used_holds, used_cands = …
       For each action in sorted order:
         skip if any of:
           - slot_budget exceeded (BUY consumes 1; SELL frees 1; ROTATE = +1−1 = 0)
           - cash insufficient
           - sector cap violated
           - correlation guard violated
           - wash-sale (cand sold within wash_sale_days)
           - ticker already used (one action per held; one per cand)
         else: select; update budgets/used sets.
       Stop when budget exhausted or no remaining action passes guards.

  4. Emit:
       BUY    → ctx.orders
       SELL   → ctx.exits  (ExitSignal exit_type="joint_sell")
       ROTATE → ctx.exits + ctx.orders (atomic pair, exit_type="rotation")

Design choices:
- Tie-breaking: stable sort by (net_alpha desc, action-type-order, ticker).
  Action-type order: ROTATE > BUY > SELL when net_alpha ties — rotations
  pre-emptively swap a weak hold for a strong cand even if absolute
  net_alpha matches a fresh buy.
- Slot budget mode "shared": rotations + new buys share one cap. Mode
  "separate" preserves current behaviour (rotation uses
  max_rotations_per_bar, selection uses open_slots) — included for
  forward compat but the JointActionJob is currently flag-gated off so
  it's not the default path.
- Reuses kernel.selection guard helpers, kernel.sizing for position
  sizing, kernel.regime.confidence_to_size_multiplier, and
  kernel.rotation.tax_drag for tax drag — no duplicated logic.
- Counters: ctx.counters["rotations"] still incremented per emitted
  rotation pair; new counters["joint_buys"], ["joint_sells"],
  ["joint_blocked_*"] for telemetry.

NOTE: When `rotation.joint_actions.enabled = false` (default), this
task short-circuits via should_skip → existing RotationJob +
SelectionJob run unchanged.
"""
from __future__ import annotations

import datetime
import logging
import math
from dataclasses import dataclass
from typing import Any

from .context import InferenceContext
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.joint_actions")


# ── Action records ─────────────────────────────────────────────────────────────

@dataclass
class _Action:
    """One candidate decision for the joint selector."""
    kind:        str               # "buy" | "sell" | "rotate"
    net_alpha:   float
    cand_ticker: str | None = None
    held_ticker: str | None = None
    cand_obj:    Any = None        # CandidateResult-like
    held_obj:    Any = None        # HoldingState-like


# ── Helpers ───────────────────────────────────────────────────────────────────

def _eligible_held_for_swap(
    holding: Any,
    cur_price: float,
    today: datetime.date,
    min_hold_days: int,
    lt_threshold_days: int,
    lt_protect_days: int,
) -> bool:
    """Replicate the rotation eligibility checks (min_hold + LT-protected).

    cur_price comes from `ctx.prices.get(ticker)` — the HoldingState
    dataclass doesn't carry today's mark-to-market price.
    """
    from kernel.rotation import is_lt_protected  # noqa: PLC0415

    entry_date  = getattr(holding, "entry_date", None)
    entry_price = float(getattr(holding, "entry_price", 0.0) or 0.0)
    if entry_date is None or entry_price <= 0:
        return False
    hold_days = (today - entry_date).days
    if hold_days < min_hold_days:
        return False
    if not math.isfinite(cur_price) or cur_price <= 0:
        return False
    unreal_pct = (cur_price - entry_price) / entry_price
    if is_lt_protected(unreal_pct, hold_days, lt_threshold_days, lt_protect_days):
        return False
    return True


def _held_tax_drag(
    holding: Any,
    cur_price: float,
    today: datetime.date,
    tax_cfg: dict,
) -> float:
    from kernel.rotation import tax_drag  # noqa: PLC0415

    entry_date  = getattr(holding, "entry_date", None)
    entry_price = float(getattr(holding, "entry_price", 0.0) or 0.0)
    if entry_date is None or entry_price <= 0:
        return 0.0
    if not math.isfinite(cur_price) or cur_price <= 0:
        return 0.0
    hold_days    = (today - entry_date).days
    unreal_pct   = (cur_price - entry_price) / entry_price
    st_rate      = float(tax_cfg.get("short_term_rate", 0.50))
    lt_rate      = float(tax_cfg.get("long_term_rate", 0.32))
    lt_threshold = int(tax_cfg.get("long_term_threshold_days", 365))
    return tax_drag(unreal_pct, hold_days, st_rate, lt_rate, lt_threshold)


# ── The main task ─────────────────────────────────────────────────────────────

class JointActionTask(Task):
    """Build the unified action menu and greedy-fill into orders/exits.

    Reads:
      ctx.ranked, ctx.holdings, ctx.prices, ctx.cash, ctx.portfolio_value,
      ctx.last_sell_dates, ctx.regime, ctx.confidence, ctx.bear_only
      ctx.config["rotation"], ctx.config["regime_params"], ctx.config["tax"],
      ctx.config["sector_map"], ctx.config["max_positions_per_sector"],
      ctx.config["wash_sale_days"]

    Writes:
      ctx.orders          — all BUY + ROTATE buy legs
      ctx.exits           — all SELL + ROTATE sell legs
      ctx.rotations       — list of RotationPair records (compat with downstream)
      ctx.counters["rotations"], ["joint_buys"], ["joint_sells"]
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.rotation import RotationPair                              # noqa: PLC0415
        from kernel.exits    import ExitSignal                                # noqa: PLC0415
        from kernel.selection import (                                        # noqa: PLC0415
            is_wash_sale_blocked, passes_sector_guard, passes_correlation_guard,
        )
        from kernel.sizing   import (                                         # noqa: PLC0415
            compute_position_size, conviction_multiplier, sigma_multiplier,
            universe_sigma_median,
        )
        from kernel.regime   import confidence_to_size_multiplier            # noqa: PLC0415

        joint_cfg = (ctx.config.get("rotation", {})
                              .get("joint_actions", {}))
        if not joint_cfg.get("enabled", False):
            return False  # short-circuit: the legacy chain owns this bar

        rotation_cfg = ctx.config.get("rotation", {})
        if ctx.bear_only:
            # Spec: keep BEAR routing in the legacy SelectionJob so we don't
            # duplicate the defensive-only logic. Joint mode only runs in
            # offensive regimes (matches RotationJob.should_skip behaviour).
            log.info("JointActionJob: BEAR — defer to legacy SelectionJob")
            return False

        # ── Configuration ────────────────────────────────────────────────
        fee_pct      = float(joint_cfg.get("fee_pct", 0.0005))
        slip_pct     = float(joint_cfg.get("slippage_pct", 0.0005))
        budget_mode  = str(joint_cfg.get("slot_budget_mode", "shared"))

        # Reuse the same floors Phase 1 introduced
        _bf_raw = rotation_cfg.get("panel_buy_floor")
        _sf_raw = rotation_cfg.get("panel_sell_floor")
        buy_floor  = float(_bf_raw) if _bf_raw is not None else None
        sell_floor = float(_sf_raw) if _sf_raw is not None else None

        min_hold     = int(rotation_cfg.get("min_rotation_hold_days", 30))
        lt_protect   = int(rotation_cfg.get("lt_protection_days", 30))
        max_rot_bar  = int(rotation_cfg.get("max_rotations_per_bar", 2))
        horizon      = int(rotation_cfg.get("target_horizon_days", 20))

        tax_cfg      = ctx.config.get("tax", {})
        lt_threshold = int(tax_cfg.get("long_term_threshold_days", 365))

        regime_cfg     = ctx.config.get("regime", {})
        regime_params  = ctx.config.get("regime_params", {}).get(ctx.regime, {})
        max_positions  = int(regime_params.get(
            "max_concurrent_positions",
            ctx.config.get("max_concurrent_positions", 8),
        ))
        wash_days      = int(ctx.config.get("wash_sale_days", 0))
        corr_threshold = float(regime_cfg.get("correlation_guard_threshold", 0.70))
        max_per_sector = int(ctx.config.get("max_positions_per_sector", 0))
        sector_map     = ctx.config.get("sector_map", {})
        defensive_set  = set(ctx.config.get("defensive_tickers", []))
        tiered         = ctx.config.get("tiered_thresholds", [])

        # Existing exits (e.g. stop-loss already emitted by SellJob) free a slot.
        prior_exit_tickers = {t for t, _ in ctx.exits}
        held_set           = set(ctx.holdings.keys())
        effective_held     = held_set - prior_exit_tickers

        open_slots = max_positions - len(effective_held)
        # Slot budget — "shared" lets rotations and new buys share the cap.
        # Cap the effective budget so a single bar never exceeds
        # max_concurrent_positions on net.
        if budget_mode == "shared":
            slot_budget = max(open_slots, 0) + max_rot_bar
        else:  # "separate" — preserves legacy quotas (rare path)
            slot_budget = max(open_slots, 0) + max_rot_bar

        log.info(
            "JointActionJob: open_slots=%d  rot_quota=%d  budget=%d  mode=%s",
            open_slots, max_rot_bar, slot_budget, budget_mode,
        )

        # ── Build action menu ───────────────────────────────────────────
        eligible_cands = [c for c in ctx.ranked
                          if c.ticker not in held_set]

        # Sizing helpers (computed once, used per BUY / ROTATE leg)
        _conf_mult    = confidence_to_size_multiplier(ctx.confidence)
        base_max_pct  = float(regime_params.get("max_position_pct", 0.15)) * _conf_mult
        reserve_pct   = float(regime_params.get("cash_reserve_pct", 0.0))  * _conf_mult
        sizing_cfg    = (ctx.config.get("ranking", {})
                          .get("panel_scoring", {}).get("sizing", {}))
        sigma_cfg     = (ctx.config.get("ranking", {})
                          .get("panel_scoring", {})
                          .get("sigma_sizing", {}))
        sigma_median  = universe_sigma_median(
            [getattr(c, "sigma", None) for c in ctx.ranked]
        )

        def _passes_tier(cand) -> bool:
            """Approximate the SelectionJob tier_idx=0 baseline.

            We can't know the final slot index until we run the greedy
            loop, so apply the loosest tier (tier 0) here as a pre-filter.
            The greedy loop still re-checks per-slot tier later.
            """
            if not tiered:
                return True
            tier_min = float(tiered[0].get("min_model_score", 0.0))
            rs = getattr(cand, "rank_score", None)
            if rs is None or not math.isfinite(rs):
                return False
            return rs >= tier_min

        actions: list[_Action] = []

        # BUY actions — candidate must clear panel_buy_floor (when set)
        for c in eligible_cands:
            cand_score = float(getattr(c, "rank_score", 0.0) or 0.0)
            if buy_floor is not None and cand_score < buy_floor:
                continue
            # Plan O — no defensives in non-BEAR offensive regimes
            if c.ticker in defensive_set:
                continue
            if not _passes_tier(c):
                continue
            cand_er = float(getattr(c, "expected_return", 0.0) or 0.0)
            if not math.isfinite(cand_er):
                continue
            net = cand_er - fee_pct - slip_pct
            actions.append(_Action(
                kind="buy", net_alpha=net,
                cand_ticker=c.ticker, cand_obj=c,
            ))

        # SELL actions — held must be weak enough to cross sell_floor
        for ticker, h in ctx.holdings.items():
            if ticker in prior_exit_tickers:
                continue
            held_score = getattr(h, "rank_score", None)
            if held_score is None:
                continue
            if sell_floor is not None and float(held_score) > sell_floor:
                continue
            held_er = float(getattr(h, "expected_return", 0.0) or 0.0)
            if not math.isfinite(held_er):
                continue
            cur_p = float(ctx.prices.get(ticker, 0.0) or 0.0)
            tax_d = _held_tax_drag(h, cur_p, ctx.today, tax_cfg)
            net = -held_er - fee_pct - slip_pct - tax_d
            actions.append(_Action(
                kind="sell", net_alpha=net,
                held_ticker=ticker, held_obj=h,
            ))

        # ROTATE actions — both floors must pass; held must be swap-eligible
        for h_t, h in ctx.holdings.items():
            if h_t in prior_exit_tickers:
                continue
            held_score = getattr(h, "rank_score", None)
            if held_score is None:
                continue
            if sell_floor is not None and float(held_score) > sell_floor:
                continue
            cur_p = float(ctx.prices.get(h_t, 0.0) or 0.0)
            if not _eligible_held_for_swap(
                h, cur_p, ctx.today, min_hold, lt_threshold, lt_protect,
            ):
                continue
            held_er = float(getattr(h, "expected_return", 0.0) or 0.0)
            if not math.isfinite(held_er):
                continue
            tax_d = _held_tax_drag(h, cur_p, ctx.today, tax_cfg)
            for c in eligible_cands:
                cand_score = float(getattr(c, "rank_score", 0.0) or 0.0)
                if buy_floor is not None and cand_score < buy_floor:
                    continue
                if c.ticker in defensive_set:
                    continue
                if not _passes_tier(c):
                    continue
                cand_er = float(getattr(c, "expected_return", 0.0) or 0.0)
                if not math.isfinite(cand_er):
                    continue
                net = (cand_er - held_er) - 2.0 * (fee_pct + slip_pct) - tax_d
                actions.append(_Action(
                    kind="rotate", net_alpha=net,
                    cand_ticker=c.ticker, cand_obj=c,
                    held_ticker=h_t, held_obj=h,
                ))

        log.info(
            "JointActionJob: menu sizes — buys=%d  sells=%d  rotates=%d",
            sum(1 for a in actions if a.kind == "buy"),
            sum(1 for a in actions if a.kind == "sell"),
            sum(1 for a in actions if a.kind == "rotate"),
        )

        if not actions:
            return

        # Tie-breaking — net_alpha desc; ROTATE before BUY before SELL on ties;
        # then ticker for full determinism.
        _kind_priority = {"rotate": 0, "buy": 1, "sell": 2}
        actions.sort(key=lambda a: (
            -a.net_alpha,
            _kind_priority[a.kind],
            (a.held_ticker or "") + "|" + (a.cand_ticker or ""),
        ))

        # ── Greedy fill ─────────────────────────────────────────────────
        cash_remaining = float(ctx.cash)
        sectors_used: dict[str, int] = {}
        for t in effective_held:
            sec = sector_map.get(t, "other")
            sectors_used[sec] = sectors_used.get(sec, 0) + 1
        used_holds: set[str] = set()
        used_cands: set[str] = set()
        slots_consumed = 0
        rot_consumed   = 0

        # Mutable virtual holdings list for sector + correlation guard
        virtual_held: list[str] = list(effective_held)

        accepted: list[_Action] = []

        for a in actions:
            if slots_consumed >= slot_budget:
                log.debug("JointActionJob: budget exhausted")
                break

            # Per-ticker dedupe — one action per held, one per cand
            if a.held_ticker is not None and a.held_ticker in used_holds:
                ctx.counters["joint_blocked_dedup"] = (
                    ctx.counters.get("joint_blocked_dedup", 0) + 1
                )
                continue
            if a.cand_ticker is not None and a.cand_ticker in used_cands:
                ctx.counters["joint_blocked_dedup"] = (
                    ctx.counters.get("joint_blocked_dedup", 0) + 1
                )
                continue

            # ROTATE quota — separate cap on rotation count even in shared mode
            if a.kind == "rotate" and rot_consumed >= max_rot_bar:
                ctx.counters["joint_blocked_rot_quota"] = (
                    ctx.counters.get("joint_blocked_rot_quota", 0) + 1
                )
                continue

            # Wash-sale check (cand side; SELL has no cand)
            if a.kind in ("buy", "rotate"):
                if is_wash_sale_blocked(
                    a.cand_ticker, ctx.today, ctx.last_sell_dates, wash_days,
                ):
                    ctx.counters["joint_blocked_wash"] = (
                        ctx.counters.get("joint_blocked_wash", 0) + 1
                    )
                    continue

            # Sector + correlation — virtual_held reflects post-action portfolio
            if a.kind in ("buy", "rotate"):
                # If rotation, the held seat opens up; treat as removed for
                # the guard check.
                tmp_held = virtual_held[:]
                if a.kind == "rotate":
                    try:
                        tmp_held.remove(a.held_ticker)
                    except ValueError:
                        pass
                if not passes_sector_guard(
                    a.cand_ticker, tmp_held, sector_map,
                    max_per_sector, defensive_set,
                ):
                    ctx.counters["joint_blocked_sector"] = (
                        ctx.counters.get("joint_blocked_sector", 0) + 1
                    )
                    continue
                if not passes_correlation_guard(
                    a.cand_ticker, tmp_held, ctx.corr_matrix, corr_threshold,
                ):
                    ctx.counters["joint_blocked_corr"] = (
                        ctx.counters.get("joint_blocked_corr", 0) + 1
                    )
                    continue

            # Sizing & cash check (BUY + ROTATE only)
            shares = 0
            invest = 0.0
            price = 0.0
            conv = 1.0
            sig_m = 1.0
            if a.kind in ("buy", "rotate"):
                price = float(ctx.prices.get(a.cand_ticker, 0.0) or 0.0)
                if not math.isfinite(price) or price <= 0:
                    ctx.counters["joint_blocked_price"] = (
                        ctx.counters.get("joint_blocked_price", 0) + 1
                    )
                    continue
                conv = conviction_multiplier(
                    getattr(a.cand_obj, "panel_score", None), sizing_cfg,
                )
                sig_m = sigma_multiplier(
                    getattr(a.cand_obj, "sigma", None), sigma_median, sigma_cfg,
                )
                max_pct = base_max_pct * conv * sig_m
                _, shares = compute_position_size(
                    ctx.portfolio_value, cash_remaining,
                    max_pct, reserve_pct, price,
                )
                if shares < 1:
                    ctx.counters["joint_blocked_cash"] = (
                        ctx.counters.get("joint_blocked_cash", 0) + 1
                    )
                    continue
                invest = shares * price

            # ── Accept ──────────────────────────────────────────────────
            accepted.append(a)
            if a.kind == "buy":
                slots_consumed += 1
                cash_remaining -= invest
                used_cands.add(a.cand_ticker)
                virtual_held.append(a.cand_ticker)
                # Emit BUY order
                target_pct = invest / ctx.portfolio_value if ctx.portfolio_value > 0 else 0.0
                ctx.orders.append({
                    "ticker":     a.cand_ticker,
                    "shares":     shares,
                    "price":      price,
                    "invest":     invest,
                    "target_pct": target_pct,
                    "regime":     ctx.regime,
                    "confidence": ctx.confidence,
                    "conviction": conv,
                    "sigma_mult": sig_m,
                    "rank_score": getattr(a.cand_obj, "rank_score", 0.0),
                    "rs_score":   getattr(a.cand_obj, "rs_score",   0.0),
                    "panel_score": getattr(a.cand_obj, "panel_score", None),
                    "mu":         getattr(a.cand_obj, "mu", None),
                    "sigma":      getattr(a.cand_obj, "sigma", None),
                    "kelly_target_pct": getattr(a.cand_obj, "kelly_target_pct", None),
                    "detail":     getattr(a.cand_obj, "detail", "") + " (joint_buy)",
                    "order_type": "JOINT_BUY",
                })
                ctx.counters["joint_buys"] = ctx.counters.get("joint_buys", 0) + 1
                log.info(
                    "JOINT_BUY    %-6s  shares=%d  net_alpha=%+.4f",
                    a.cand_ticker, shares, a.net_alpha,
                )
            elif a.kind == "sell":
                # SELL frees a slot — nets to -1 consumption
                slots_consumed -= 1
                used_holds.add(a.held_ticker)
                if a.held_ticker in virtual_held:
                    virtual_held.remove(a.held_ticker)
                ctx.exits.append((
                    a.held_ticker,
                    ExitSignal(
                        should_exit = True,
                        reason      = f"joint_sell net_alpha={a.net_alpha:+.4f}",
                        exit_type   = "joint_sell",
                    ),
                ))
                ctx.counters["joint_sells"] = ctx.counters.get("joint_sells", 0) + 1
                log.info(
                    "JOINT_SELL   %-6s  net_alpha=%+.4f",
                    a.held_ticker, a.net_alpha,
                )
            else:  # rotate
                # Rotate is net-zero on slots (free 1, take 1); count only the
                # rotation quota.
                rot_consumed += 1
                cash_remaining -= invest
                used_holds.add(a.held_ticker)
                used_cands.add(a.cand_ticker)
                if a.held_ticker in virtual_held:
                    virtual_held.remove(a.held_ticker)
                virtual_held.append(a.cand_ticker)
                # Build a RotationPair so downstream telemetry / state stays
                # parity-compatible with the legacy chain.
                _hp = float(ctx.prices.get(a.held_ticker, 0.0) or 0.0)
                pair = RotationPair(
                    sell_ticker      = a.held_ticker,
                    buy_ticker       = a.cand_ticker,
                    sell_score       = float(getattr(a.held_obj, "rank_score", 0.0) or 0.0),
                    buy_score        = float(getattr(a.cand_obj, "rank_score", 0.0) or 0.0),
                    sell_er          = float(getattr(a.held_obj, "expected_return", 0.0) or 0.0),
                    buy_er           = float(getattr(a.cand_obj, "expected_return", 0.0) or 0.0),
                    horizon_days     = horizon,
                    raw_advantage    = (float(getattr(a.cand_obj, "expected_return", 0.0) or 0.0)
                                        - float(getattr(a.held_obj, "expected_return", 0.0) or 0.0)),
                    tax_drag         = _held_tax_drag(a.held_obj, _hp, ctx.today, tax_cfg),
                    transaction_cost = 2.0 * (fee_pct + slip_pct),
                    net_advantage    = a.net_alpha,
                    threshold        = 0.0,    # joint mode uses net_alpha sort, not a fixed threshold
                    margin_realized  = a.net_alpha,
                )
                ctx.rotations.append(pair)
                ctx.exits.append((
                    a.held_ticker,
                    ExitSignal(
                        should_exit = True,
                        reason      = (f"joint_rotation→{a.cand_ticker} "
                                       f"net_alpha={a.net_alpha:+.4f}"),
                        exit_type   = "rotation",
                    ),
                ))
                target_pct = invest / ctx.portfolio_value if ctx.portfolio_value > 0 else 0.0
                ctx.orders.append({
                    "ticker":     a.cand_ticker,
                    "shares":     shares,
                    "price":      price,
                    "invest":     invest,
                    "target_pct": target_pct,
                    "regime":     ctx.regime,
                    "confidence": ctx.confidence,
                    "conviction": conv,
                    "sigma_mult": sig_m,
                    "rank_score": getattr(a.cand_obj, "rank_score", 0.0),
                    "rs_score":   0.0,
                    "panel_score": getattr(a.cand_obj, "panel_score", None),
                    "mu":         getattr(a.cand_obj, "mu", None),
                    "sigma":      getattr(a.cand_obj, "sigma", None),
                    "kelly_target_pct": getattr(a.cand_obj, "kelly_target_pct", None),
                    "detail":     (f"joint_rotation←{a.held_ticker} "
                                   f"net_alpha={a.net_alpha:+.4f}"),
                    "order_type": "ROTATION",
                })
                ctx.counters["rotations"] = ctx.counters.get("rotations", 0) + 1
                log.info(
                    "JOINT_ROT    %-6s→%-6s  shares=%d  net_alpha=%+.4f",
                    a.held_ticker, a.cand_ticker, shares, a.net_alpha,
                )

        # Prune ranked so any post-job task (e.g. TopUpHeldTask) doesn't
        # double-buy a ticker we just placed.
        if used_cands:
            ctx.ranked = [c for c in ctx.ranked if c.ticker not in used_cands]

        log.info(
            "JointActionJob: accepted %d action(s)  buys=%d  sells=%d  rotates=%d",
            len(accepted),
            sum(1 for a in accepted if a.kind == "buy"),
            sum(1 for a in accepted if a.kind == "sell"),
            sum(1 for a in accepted if a.kind == "rotate"),
        )
