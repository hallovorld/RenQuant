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
from .order_attribution import stamp_order_attribution
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
        # When solver=qp, JointPortfolioQPTask already ran and emitted
        # orders/exits. Skip the greedy path entirely.
        solver = str(joint_cfg.get("solver", "greedy")).lower()
        if solver == "qp":
            log.info("JointActionTask: solver=qp — already handled by QP task")
            return False

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
        # BROKER-PRECHECK (2026-04-26): exclude tickers with pending
        # orders at the broker — pre-fix these were sized in then
        # rejected at submit time, distorting cash budget.
        pending_at_broker: set[str] = set(
            getattr(ctx, "pending_broker_tickers", None) or []
        )
        eligible_cands = [c for c in ctx.ranked
                          if c.ticker not in held_set
                          and c.ticker not in pending_at_broker]
        if pending_at_broker:
            log.info(
                "JointActionJob: BROKER-PRECHECK excluded %d cand(s) "
                "with pending broker orders",
                len([c for c in ctx.ranked if c.ticker in pending_at_broker]),
            )

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

        # Audit fix BUY-FLOOR-RANK-FALLBACK (2026-04-26 round-5):
        # absolute panel_buy_floor (default 0.45) is too aggressive when
        # calibrator's pool_ic is low — score distribution compresses near
        # base_rate (~0.27), few/no cands cross 0.45, BUY menu always
        # empty, portfolio bleeds via sells-only. Per user spec: "结合
        # rank based" — use TOP-N rank fallback.
        #
        # Logic:
        #   1. Sort eligible_cands by rank_score DESC.
        #   2. For each cand: ABSOLUTE pass requires rank_score >= buy_floor.
        #   3. RANK fallback: if cand is in top-N AND rank_score >= rank_floor
        #      (a looser absolute floor, default 0.20 ≈ base_rate-ish), allow.
        #   4. Both gates use the SAME tier escalation downstream — quality
        #      filter at greedy_loop is unchanged.
        #
        # Default: top_n=3 (one slot of guaranteed quality), rank_floor=0.20.
        # When buy_floor is None, only rank fallback gates the menu.
        rotation_cfg_full = ctx.config.get("rotation", {})
        rank_top_n_raw  = rotation_cfg_full.get("panel_buy_top_n", 3)
        rank_floor_raw  = rotation_cfg_full.get("panel_buy_rank_floor", 0.20)
        rank_top_n  = max(0, int(rank_top_n_raw)) if rank_top_n_raw is not None else 0
        rank_floor  = float(rank_floor_raw) if rank_floor_raw is not None else 0.0

        # Pre-sort eligible_cands by rank_score desc for the rank-fallback gate.
        ranked_cands = sorted(
            eligible_cands,
            key=lambda c: float(getattr(c, "rank_score", 0.0) or 0.0),
            reverse=True,
        )
        cand_index = {id(c): i for i, c in enumerate(ranked_cands)}

        n_pass_floor = 0
        n_pass_rank  = 0

        # BUY actions — candidate must clear panel_buy_floor OR be in top-N
        for c in eligible_cands:
            cand_score = float(getattr(c, "rank_score", 0.0) or 0.0)
            # Absolute floor pass
            passes_absolute = (
                buy_floor is None or cand_score >= buy_floor
            )
            # Rank fallback pass: top-N AND above rank_floor
            cand_rank = cand_index.get(id(c), len(ranked_cands))
            passes_rank = (
                rank_top_n > 0
                and cand_rank < rank_top_n
                and cand_score >= rank_floor
            )
            if not (passes_absolute or passes_rank):
                continue
            if passes_absolute:
                n_pass_floor += 1
            elif passes_rank:
                n_pass_rank += 1
                log.info(
                    "RANK-FALLBACK admitted %-6s rank=%d/N=%d score=%.3f "
                    "(below buy_floor=%.2f but in top-%d above rank_floor=%.2f)",
                    c.ticker, cand_rank + 1, len(ranked_cands), cand_score,
                    buy_floor or 0.0, rank_top_n, rank_floor,
                )
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
            # Audit fix JOINT-NEW-2 (2026-04-26 round-3): explicit NaN
            # guard. Pre-fix, NaN score passed `is None` check + `> sell_floor`
            # comparison (NaN > X is False), creating SELL/ROTATE entries
            # with NaN net_alpha. Filter explicitly.
            if held_score is None or not math.isfinite(float(held_score)):
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

        # ROTATE actions — both floors must pass; held must be swap-eligible.
        # Audit fix JOINT-ROT-QUOTA-ZERO (2026-04-25, edge from Bug MM):
        # when max_rot_bar=0, no rotation can fire. Skip menu generation
        # entirely so Bug MM's defer-for-rotate logic doesn't strand a
        # SELL waiting on an impossible rotation.
        rotate_holdings = ctx.holdings.items() if max_rot_bar > 0 else []
        for h_t, h in rotate_holdings:
            if h_t in prior_exit_tickers:
                continue
            held_score = getattr(h, "rank_score", None)
            # Audit fix JOINT-NEW-2 (2026-04-26 round-3): explicit NaN
            # guard. Pre-fix, NaN score passed `is None` check + `> sell_floor`
            # comparison (NaN > X is False), creating SELL/ROTATE entries
            # with NaN net_alpha. Filter explicitly.
            if held_score is None or not math.isfinite(float(held_score)):
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

        # Audit fix JOINT-NET-NEG (Bug Q, 2026-04-25): drop BUY/ROTATE
        # actions with negative net_alpha — accepting them is a
        # guaranteed loss after fees + slippage + tax.
        #
        # SELLs are EXEMPT from this filter: per user spec "被替换的
        # portfolio 里的 stock 的 score 要低于一个值" the sell trigger
        # is the score floor, not net_alpha. A weak-score held with
        # mildly positive expected_return (so net_alpha < 0 after fees)
        # can still be the right action to exit on conviction grounds —
        # the model says it's no longer alpha-positive ENOUGH to retain
        # the slot, and the score-floor gate already filtered out
        # acceptably-strong holds in the menu-build phase.
        actions = [a for a in actions
                   if (a.kind == "sell") or (a.net_alpha > 0.0)]
        if not actions:
            return

        # Tie-breaking — net_alpha desc; ROTATE before BUY before SELL on ties;
        # then RAW PANEL SCORE desc (tiebreaker for calibrator-saturated ties);
        # then NGBoost μ desc (further tiebreaker when panel_score also tied);
        # then ticker for full determinism.
        #
        # Audit fix JOINT-NET-ALPHA-SAT (2026-04-26): pre-fix, tied net_alpha
        # (extremely common when calibrator's isotonic top-bin saturates 4-N
        # candidates to identical rank_score) made the choice fall through to
        # the alphabetical ticker tiebreaker — choice driven by alphabet, not
        # signal quality. Live e2e showed JNJ/NET/NVDA/RTX all picked at
        # net_alpha=+0.1446. New cascade uses raw `panel_score` (pre-calibration,
        # full granularity) and then NGBoost μ — both bypass the calibrator
        # bottleneck and provide signal-driven tiebreaks.
        _kind_priority = {"rotate": 0, "buy": 1, "sell": 2}

        def _tiebreak_score(a: _Action) -> float:
            """Signal-driven tiebreak. Higher = better."""
            # Sells don't have a candidate object; tiebreak on held's panel_score.
            obj = a.cand_obj if a.cand_obj is not None else a.held_obj
            if obj is None:
                return 0.0
            ps = getattr(obj, "panel_score", None)
            if ps is not None and math.isfinite(float(ps)):
                return float(ps)
            return 0.0

        def _mu_tiebreak(a: _Action) -> float:
            obj = a.cand_obj if a.cand_obj is not None else a.held_obj
            if obj is None:
                return 0.0
            mu = getattr(obj, "mu", None)
            if mu is not None and math.isfinite(float(mu)):
                return float(mu)
            return 0.0

        actions.sort(key=lambda a: (
            -a.net_alpha,
            _kind_priority[a.kind],
            -_tiebreak_score(a),     # primary tiebreak: raw panel_score desc
            -_mu_tiebreak(a),        # secondary tiebreak: NGBoost μ desc
            (a.held_ticker or "") + "|" + (a.cand_ticker or ""),
        ))

        # ── Two-pass greedy fill ───────────────────────────────────────
        # Audit fix JOINT-GREEDY-SELL-LATE (Bug L, 2026-04-25): pre-fix,
        # sort by net_alpha asc put SELLs at the end (negative ER on the
        # held side dominates), so a SELL that would free a slot for a
        # high-net_alpha BUY/ROTATE was processed AFTER all BUYs/ROTATEs
        # had already been blocked by the slot budget. Two-pass fix:
        #   Pass 1: accept all SELLs that pass per-action guards. They
        #           free slots in `net_position_consumed` for Pass 2.
        #   Pass 2: process BUY+ROTATE actions in net_alpha-desc order
        #           against the freshly-freed budget.
        # Audit fix JOINT-NET-POSITIONS (Bug F, 2026-04-25): track
        # NET position delta separately from rotation quota. Pre-fix
        # `slot_budget = open_slots + max_rot_bar` allowed BUYs to
        # over-fill past max_concurrent_positions when rotations didn't
        # materialize — a bar with held=8, max_pos=8 would have
        # slot_budget=2, then 2 BUYs (no offsetting SELLs) ended at
        # 10 holdings. New bookkeeping:
        #   net_position_consumed: BUY +1, SELL -1, ROTATE 0 (net-zero
        #                          swap). Capped by `open_slots`.
        #   rot_consumed:          ROTATE only. Capped by max_rot_bar.
        # Both caps respected → portfolio cannot exceed max_concurrent_positions.
        cash_remaining = float(ctx.cash)
        sectors_used: dict[str, int] = {}
        for t in effective_held:
            sec = sector_map.get(t, "other")
            sectors_used[sec] = sectors_used.get(sec, 0) + 1
        used_holds: set[str] = set()
        used_cands: set[str] = set()
        net_position_consumed = 0  # +1 per BUY, -1 per SELL; ROTATE = 0
        rot_consumed   = 0
        slots_consumed = 0  # tier-escalation index proxy (BUY+ROTATE only)

        # Mutable virtual holdings list for sector + correlation guard
        virtual_held: list[str] = list(effective_held)

        accepted: list[_Action] = []

        # ── Pass 1: SELLs only (with Bug MM joint-optimal deferral) ──
        # Audit fix JOINT-PASS1-SELL-VS-ROTATE-CONFLICT (Bug MM,
        # 2026-04-25): when a held has BOTH a SELL action AND a ROTATE
        # action with HIGHER net_alpha, accepting SELL in Pass 1 would
        # dedup the ROTATE in Pass 2 — we'd lose the rotation's alpha.
        # Per user "make it perfect" spec, we defer such SELLs to Pass 3
        # and let the higher-net_alpha ROTATE try first. If the ROTATE
        # fails downstream guards (cash, sector, corr), the deferred
        # SELL fires retroactively in Pass 3 — so we never miss a
        # legitimate exit signal.
        sell_actions = [a for a in actions if a.kind == "sell"]
        non_sell_actions = [a for a in actions if a.kind != "sell"]

        # Build best-rotate-net_alpha per held ticker
        best_rot_alpha_per_held: dict[str, float] = {}
        for a in actions:
            if a.kind == "rotate" and a.held_ticker is not None:
                cur = best_rot_alpha_per_held.get(a.held_ticker, float("-inf"))
                if a.net_alpha > cur:
                    best_rot_alpha_per_held[a.held_ticker] = a.net_alpha

        deferred_sells: list[_Action] = []  # Bug MM — re-tried in Pass 3

        def _accept_sell(a: _Action) -> None:
            """Common SELL accept body — used by Pass 1 and Pass 3."""
            nonlocal net_position_consumed
            accepted.append(a)
            net_position_consumed -= 1
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

        for a in sell_actions:
            if a.held_ticker in used_holds:
                ctx.counters["joint_blocked_dedup"] = (
                    ctx.counters.get("joint_blocked_dedup", 0) + 1
                )
                continue
            # Bug MM: defer SELL only if a ROTATE for same held has
            # higher net_alpha AND swapping (vs selling) wouldn't keep
            # the portfolio above max_positions. When overfilled
            # (len(virtual_held) > max_positions) we ALWAYS prefer SELL
            # because rotation is net-zero on position count and would
            # leave us still overfilled. Net-zero swap only helps when
            # we have room to absorb the new ticker.
            best_rot = best_rot_alpha_per_held.get(a.held_ticker, float("-inf"))
            can_defer_for_rotate = len(virtual_held) <= max_positions
            if best_rot > a.net_alpha and can_defer_for_rotate:
                deferred_sells.append(a)
                ctx.counters["joint_deferred_sells"] = (
                    ctx.counters.get("joint_deferred_sells", 0) + 1
                )
                log.info(
                    "JOINT_DEFER  %-6s  sell_net=%+.4f  best_rot_net=%+.4f  → Pass 3",
                    a.held_ticker, a.net_alpha, best_rot,
                )
                continue
            _accept_sell(a)

        # ── Pass 2: BUYs and ROTATEs ─────────────────────────────────
        # Audit fix JOINT-NET-POSITIONS (Bug F) + JOINT-OVERFILL-EDGE
        # (Bug Y, 2026-04-25):
        #
        # Capacity constraint: len(virtual_held_after_pass2) <= max_positions.
        # Equivalently:  net_position_consumed <= open_slots
        # where open_slots = max_positions - len(effective_held).
        #
        # `open_slots` may be negative (over-filled by external path,
        # e.g. legacy ledger). The cap still holds — over-filled portfolio
        # can only BUY when prior sells outpace the overflow.
        #
        # ROTATE has its own quota (max_rot_bar) and is net-zero on
        # net_position_consumed, so does NOT count toward open_slots.

        new_buys_consumed = 0  # buys accepted in Pass 2 (= net_position_consumed delta vs Pass-1 end state)

        for a in non_sell_actions:
            # Per-ticker dedupe
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

            # Per-action position-budget gate (Bug F + Bug Y)
            if a.kind == "buy":
                if (net_position_consumed + 1) > open_slots:
                    ctx.counters["joint_blocked_budget"] = (
                        ctx.counters.get("joint_blocked_budget", 0) + 1
                    )
                    continue
            else:  # rotate — net-zero on positions, capped only by max_rot_bar
                if rot_consumed >= max_rot_bar:
                    ctx.counters["joint_blocked_rot_quota"] = (
                        ctx.counters.get("joint_blocked_rot_quota", 0) + 1
                    )
                    continue

            # Wash-sale check (cost-aware: gain sales pass, loss sales blocked
            # unless caller has μ̂ to compare against NPV cost — see §1091)
            from kernel.selection import is_wash_sale_blocked_with_cost  # noqa: PLC0415
            blocked, _, _ = is_wash_sale_blocked_with_cost(
                a.cand_ticker, ctx.today, ctx.last_sell_dates,
                getattr(ctx, "last_sell_pls", None) or {}, wash_days,
            )
            if blocked:
                ctx.counters["joint_blocked_wash"] = (
                    ctx.counters.get("joint_blocked_wash", 0) + 1
                )
                continue

            # Sector + correlation — virtual_held reflects post-action portfolio
            tmp_held = virtual_held[:]
            if a.kind == "rotate":
                # held seat opens up via the swap → exclude from guard check
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

            # Audit fix JOINT-TIER-ESC (Bug C) + JOINT-TIER-NEGATIVE
            # (Bug S, 2026-04-25): per-slot tier escalation indexed on
            # NEW BUYS accepted so far (rotations don't escalate the
            # tier — they replace an existing position rather than
            # filling a fresh slot, and our economic constraint is on
            # net new positions). Clamp index to >=0 so any future
            # accounting bug that produces a negative index can't
            # silently wrap around to the toughest tier.
            if tiered:
                tier_idx = min(max(new_buys_consumed, 0), len(tiered) - 1)
                tier_min = float(tiered[tier_idx].get("min_model_score", 0.0))
                rs = float(getattr(a.cand_obj, "rank_score", 0.0) or 0.0)
                if not math.isfinite(rs) or rs < tier_min:
                    ctx.counters["joint_blocked_tier"] = (
                        ctx.counters.get("joint_blocked_tier", 0) + 1
                    )
                    continue

            # ── Sizing ──────────────────────────────────────────────────
            price = float(ctx.prices.get(a.cand_ticker, 0.0) or 0.0)
            if not math.isfinite(price) or price <= 0:
                ctx.counters["joint_blocked_price"] = (
                    ctx.counters.get("joint_blocked_price", 0) + 1
                )
                continue

            # Audit fix JOINT-ROTATE-CASH (Bug M, 2026-04-25): credit
            # the sell-leg proceeds to the buy-leg sizing budget. A
            # rotation IS a paired sell-then-buy executed atomically by
            # the broker (Alpaca settles the sell-leg's cash to the
            # account immediately under RegT margin, available for the
            # buy-leg in the same bar). Pre-fix, the buy-leg only saw
            # `cash_remaining` (post-prior-buys), so a rotation that
            # should have funded itself from the held's market value
            # was undersized — e.g. swap a $20k held for a $20k cand
            # with $1k cash on hand → buy-leg sized at $1k = $19k of
            # signal lost. Now: cash_for_sizing includes the held's
            # mark-to-market value net of fees/slippage on the sell.
            sell_proceeds = 0.0
            if a.kind == "rotate":
                h_shares = int(getattr(a.held_obj, "shares", 0) or 0)
                h_price  = float(ctx.prices.get(a.held_ticker, 0.0) or 0.0)
                if h_shares > 0 and math.isfinite(h_price) and h_price > 0:
                    sell_proceeds = h_shares * h_price * (1.0 - fee_pct - slip_pct)
            cash_for_sizing = cash_remaining + sell_proceeds

            conv = conviction_multiplier(
                getattr(a.cand_obj, "panel_score", None), sizing_cfg,
            )
            sig_m = sigma_multiplier(
                getattr(a.cand_obj, "sigma", None), sigma_median, sigma_cfg,
            )
            max_pct = base_max_pct * conv * sig_m
            _, shares = compute_position_size(
                ctx.portfolio_value, cash_for_sizing,
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
                net_position_consumed += 1
                new_buys_consumed     += 1
                cash_remaining -= invest
                used_cands.add(a.cand_ticker)
                virtual_held.append(a.cand_ticker)
                target_pct = invest / ctx.portfolio_value if ctx.portfolio_value > 0 else 0.0
                ctx.orders.append(stamp_order_attribution({
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
                }, ctx=ctx, source_job="JointActionJob",
                    source_task="JointActionTask",
                    acceptance_reason="joint_action_buy_net_alpha_ranked",
                    source_obj=a.cand_obj,
                    decision_inputs={
                        "net_alpha": a.net_alpha,
                        "rank_score": getattr(a.cand_obj, "rank_score", None),
                        "cash_remaining_before": cash_remaining + invest,
                        "fee_pct": fee_pct,
                        "slippage_pct": slip_pct,
                    }))
                ctx.counters["joint_buys"] = ctx.counters.get("joint_buys", 0) + 1
                log.info(
                    "JOINT_BUY    %-6s  shares=%d  net_alpha=%+.4f  cash_after=%.0f",
                    a.cand_ticker, shares, a.net_alpha, cash_remaining,
                )
            else:  # rotate
                # Net-zero on net_position_consumed (sell-leg already
                # debited in Pass 1 logic if applicable; here the held
                # was NOT in Pass 1 used_holds so it's still virtually
                # in the portfolio — we remove it now).
                rot_consumed += 1
                # Credit the sell-leg proceeds, debit the buy-leg invest.
                cash_remaining = cash_remaining + sell_proceeds - invest
                used_holds.add(a.held_ticker)
                used_cands.add(a.cand_ticker)
                if a.held_ticker in virtual_held:
                    virtual_held.remove(a.held_ticker)
                virtual_held.append(a.cand_ticker)
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
                    threshold        = 0.0,
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
                ctx.orders.append(stamp_order_attribution({
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
                }, ctx=ctx, source_job="JointActionJob",
                    source_task="JointActionTask",
                    acceptance_reason="joint_action_rotation_net_alpha_ranked",
                    source_obj=a.cand_obj,
                    decision_inputs={
                        "sell_ticker": a.held_ticker,
                        "buy_ticker": a.cand_ticker,
                        "net_alpha": a.net_alpha,
                        "sell_score": getattr(a.held_obj, "rank_score", None),
                        "buy_score": getattr(a.cand_obj, "rank_score", None),
                        "tax_drag": pair.tax_drag,
                        "transaction_cost": pair.transaction_cost,
                        "horizon_days": horizon,
                    }))
                ctx.counters["rotations"] = ctx.counters.get("rotations", 0) + 1
                log.info(
                    "JOINT_ROT    %-6s→%-6s  shares=%d  net_alpha=%+.4f  "
                    "sell_proceeds=%.0f  cash_after=%.0f",
                    a.held_ticker, a.cand_ticker, shares, a.net_alpha,
                    sell_proceeds, cash_remaining,
                )

        # ── Pass 3: deferred SELLs whose ROTATE didn't materialize ──
        # Bug MM (cont'd): if Pass 2 didn't fire the rotate that beat a
        # SELL in net_alpha, fire the SELL retroactively so we don't
        # silently drop a legitimate exit signal.
        for a in deferred_sells:
            if a.held_ticker in used_holds:
                # ROTATE materialized — held already exiting. Skip.
                continue
            _accept_sell(a)
            log.info(
                "JOINT_PASS3  %-6s  retroactive sell after rotate failed",
                a.held_ticker,
            )

        # Audit fix JOINT-PRUNE-USED-HOLDS (Bug DD, 2026-04-25): prune
        # ranked of any used cand AND used held (the latter was already
        # excluded from `eligible_cands` but defense in depth — a future
        # ranked refresh between joint mode and downstream tasks could
        # re-introduce them, and we don't want TopUpHeldTask to top up a
        # held we just queued an exit for in this bar).
        if used_cands or used_holds:
            ctx.ranked = [
                c for c in ctx.ranked
                if c.ticker not in used_cands and c.ticker not in used_holds
            ]

        log.info(
            "JointActionJob: accepted %d action(s)  buys=%d  sells=%d  rotates=%d",
            len(accepted),
            sum(1 for a in accepted if a.kind == "buy"),
            sum(1 for a in accepted if a.kind == "sell"),
            sum(1 for a in accepted if a.kind == "rotate"),
        )
