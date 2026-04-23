"""Rotation tasks — swap held positions for stronger candidates.

Three tasks compose RotationJob:

  BuildPairsTask     gather scores + expected returns, run kernel.rotation
                     and emit a structured decision-tree log per pair
                     considered (whether or not it survives)
  ValidatePairsTask  re-check wash-sale + sector + correlation guards on the
                     virtual post-swap holdings set
  EmitRotationsTask  convert each surviving pair into exit + buy order;
                     prune the rotated-in ticker from ctx.ranked so
                     SelectionJob does not double-buy
"""
from __future__ import annotations

import datetime
import logging
import math

from .context  import InferenceContext
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.rotation")


def _log_decision_tree(
    *,
    cand_ticker: str,
    cand_er: float,
    cand_score: float,
    held_table: list[dict],   # one row per eligible held with fields below
    threshold: float,
    txn_cost: float,
    horizon: int,
    chosen: str | None,
) -> None:
    """Emit a structured per-candidate decision log.

    held_table rows: {ticker, score, er, unreal_pct, hold_days, tax_drag,
                      raw_adv, net_adv, decision}
    decision ∈ {"swap", "below_threshold", "lt_protected", "min_hold",
                "no_score", "no_er", "used"}
    """
    log.info(
        "ROTATION_TREE  cand=%s  cand_er=%+.4f  cand_rank=%.3f  "
        "horizon=%dd  threshold=%+.4f  cost=%.4f  chosen=%s",
        cand_ticker, cand_er, cand_score, horizon, threshold, txn_cost,
        chosen or "NONE",
    )
    for row in held_table:
        log.info(
            "  ↳ held=%-5s  er=%+.4f  rank=%.3f  hold=%dd  pnl=%+.3f  "
            "tax=%.4f  raw_adv=%+.4f  net_adv=%+.4f  → %s",
            row["ticker"], row["er"], row["score"], row["hold_days"],
            row["unreal_pct"], row["tax_drag"], row["raw_adv"], row["net_adv"],
            row["decision"],
        )


class BuildPairsTask(Task):
    """Score holdings, call kernel rotation primitive, log decision tree."""

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.rotation import (  # noqa: PLC0415
            find_rotation_pairs, is_lt_protected, tax_drag,
        )

        rotation_cfg = ctx.config.get("rotation", {})
        if not rotation_cfg.get("enabled", False):
            return False
        if not ctx.ranked or not ctx.holdings:
            return False
        if ctx.bear_only:
            return False

        threshold   = float(rotation_cfg.get("min_expected_advantage_pct", 0.03))
        horizon     = int(rotation_cfg.get("target_horizon_days", 20))
        txn_cost    = float(rotation_cfg.get("transaction_cost_pct", 0.0))
        min_hold    = int(rotation_cfg.get("min_rotation_hold_days", 30))
        lt_protect  = int(rotation_cfg.get("lt_protection_days", 30))

        # Cross-sectional panel gate — candidate panel_score must beat held
        # panel_score by this fraction. 0.0 disables the gate (default).
        panel_cfg           = ctx.config.get("ranking", {}).get("panel_scoring", {})
        panel_rot_advantage = float(panel_cfg.get("rotation_advantage", 0.0))

        tax_cfg     = ctx.config.get("tax", {})
        st_rate     = float(tax_cfg.get("short_term_rate", 0.37))
        lt_rate     = float(tax_cfg.get("long_term_rate", 0.20))
        lt_threshold = int(tax_cfg.get("long_term_threshold_days", 365))

        # Holdings already exiting today are not eligible to rotate.
        exit_tickers = {t for t, _ in ctx.exits}

        held_scores: dict = {}
        held_er:     dict = {}
        held_meta:   dict = {}
        # For decision-tree log: track per-held context independent of eligibility
        held_diag:   dict = {}

        for ticker, hs in ctx.holdings.items():
            if ticker in exit_tickers:
                continue
            score   = getattr(hs, "rank_score", None)
            er      = getattr(hs, "expected_return", None)
            entry_p = float(getattr(hs, "entry_price", 0.0) or 0.0)
            cur_p   = ctx.prices.get(ticker, entry_p)
            entry_d = getattr(hs, "entry_date", None)

            unreal_pct = ((cur_p - entry_p) / entry_p) if entry_p > 0 else 0.0
            hold_days  = (ctx.today - entry_d).days if entry_d is not None else 0
            drag       = tax_drag(unreal_pct, hold_days,
                                  st_rate, lt_rate, lt_threshold)

            decision = None
            if score is None:
                decision = "no_score"
            elif er is None or not math.isfinite(float(er)):
                decision = "no_er"
            elif entry_d is None or entry_p <= 0:
                decision = "no_meta"
            elif hold_days < min_hold:
                decision = "min_hold"
            elif is_lt_protected(unreal_pct, hold_days, lt_threshold, lt_protect):
                decision = "lt_protected"

            held_diag[ticker] = {
                "ticker":     ticker,
                "score":      float(score) if score is not None else float("nan"),
                "er":         float(er) if er is not None else float("nan"),
                "unreal_pct": unreal_pct,
                "hold_days":  hold_days,
                "tax_drag":   drag,
                "raw_adv":    float("nan"),     # filled per-candidate below
                "net_adv":    float("nan"),
                "decision":   decision,         # None means eligible
            }

            if decision is None:
                held_scores[ticker] = float(score)
                held_er[ticker]     = float(er)
                held_meta[ticker]   = {
                    "entry_date":    entry_d,
                    "entry_price":   entry_p,
                    "current_price": cur_p,
                }

        held_set = set(ctx.holdings.keys())
        eligible_candidates = [c for c in ctx.ranked if c.ticker not in held_set]

        pairs = find_rotation_pairs(
            held_scores  = held_scores,
            held_er      = held_er,
            held_meta    = held_meta,
            candidates   = eligible_candidates,
            today        = ctx.today,
            rotation_cfg = rotation_cfg,
            tax_cfg      = tax_cfg,
        )

        # Cross-sectional panel gate: require cand.panel_score to beat
        # held.panel_score by panel_rot_advantage (both populated by
        # PanelScoringJob.ApplyScoresTask). Pairs with missing panel scores
        # on either side skip the gate (fall back to ER-only rule).
        if panel_rot_advantage > 0.0 and pairs:
            cand_ps = {c.ticker: getattr(c, "panel_score", None)
                       for c in eligible_candidates}
            held_ps = {t: getattr(hs, "panel_score", None)
                       for t, hs in ctx.holdings.items()}
            kept: list = []
            rejected = 0
            for p in pairs:
                c_ps = cand_ps.get(p.buy_ticker)
                h_ps = held_ps.get(p.sell_ticker)
                if c_ps is None or h_ps is None or (c_ps - h_ps) >= panel_rot_advantage:
                    kept.append(p)
                else:
                    rejected += 1
                    log.info("ROTATION_REJECT  swap=%s→%s  reason=panel_advantage "
                             "cand_ps=%.3f  held_ps=%.3f  need=%+.3f",
                             p.sell_ticker, p.buy_ticker, c_ps, h_ps, panel_rot_advantage)
            if rejected:
                ctx.counters["panel_rotation_rejects"] = (
                    ctx.counters.get("panel_rotation_rejects", 0) + rejected
                )
            pairs = kept

        ctx.rotations = pairs

        # Decision-tree log: one block per candidate considered.  We replay
        # the comparisons for the top-K candidates so the log is auditable
        # without re-running the whole pipeline.
        chosen_pairs = {p.buy_ticker: p for p in pairs}
        topk = eligible_candidates[: max(5, len(pairs) + 2)]
        for c in topk:
            cand_er = float(getattr(c, "expected_return", 0.0) or 0.0)
            rows: list[dict] = []
            for ht, info in held_diag.items():
                row = dict(info)   # shallow copy
                if row["decision"] is None:
                    raw_adv = cand_er - info["er"]
                    net_adv = raw_adv - info["tax_drag"] - txn_cost
                    row["raw_adv"] = raw_adv
                    row["net_adv"] = net_adv
                    row["decision"] = (
                        "swap" if (chosen_pairs.get(c.ticker)
                                   and chosen_pairs[c.ticker].sell_ticker == ht)
                        else "below_threshold" if net_adv < threshold
                        else "available"
                    )
                rows.append(row)
            chosen = chosen_pairs.get(c.ticker)
            _log_decision_tree(
                cand_ticker = c.ticker,
                cand_er     = cand_er,
                cand_score  = float(c.rank_score),
                held_table  = rows,
                threshold   = threshold,
                txn_cost    = txn_cost,
                horizon     = horizon,
                chosen      = chosen.sell_ticker if chosen else None,
            )

        log.info("BuildPairsTask: %d rotation pair(s) proposed", len(pairs))


class ValidatePairsTask(Task):
    """Drop pairs whose buy ticker fails wash-sale, sector, or corr guards."""

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.selection import (  # noqa: PLC0415
            is_wash_sale_blocked,
            passes_sector_guard,
            passes_correlation_guard,
        )

        if not ctx.rotations:
            return False

        cfg            = ctx.config
        regime_cfg     = cfg.get("regime", {})
        wash_days      = int(cfg.get("wash_sale_days", 0))
        corr_threshold = float(regime_cfg.get("correlation_guard_threshold", 0.70))
        max_per_sector = int(cfg.get("max_positions_per_sector", 0))
        sector_map     = cfg.get("sector_map", {})
        defensive_set  = set(cfg.get("defensive_tickers", []))

        validated = []
        for pair in ctx.rotations:
            if is_wash_sale_blocked(pair.buy_ticker, ctx.today,
                                    ctx.last_sell_dates or {}, wash_days):
                log.info("ROTATION_REJECT  swap=%s→%s  reason=wash_sale",
                         pair.sell_ticker, pair.buy_ticker)
                continue

            virtual_held = (
                set(ctx.holdings.keys())
                - {p.sell_ticker for p in validated} - {pair.sell_ticker}
                | {p.buy_ticker for p in validated}
            )

            if not passes_sector_guard(
                pair.buy_ticker, list(virtual_held),
                sector_map, max_per_sector, defensive_set,
            ):
                log.info("ROTATION_REJECT  swap=%s→%s  reason=sector_cap",
                         pair.sell_ticker, pair.buy_ticker)
                continue

            if not passes_correlation_guard(
                pair.buy_ticker, list(virtual_held),
                ctx.corr_matrix, corr_threshold,
            ):
                log.info("ROTATION_REJECT  swap=%s→%s  reason=correlation_guard",
                         pair.sell_ticker, pair.buy_ticker)
                continue

            validated.append(pair)

        ctx.rotations = validated
        log.info("ValidatePairsTask: %d pair(s) survived guards", len(validated))


class EmitRotationsTask(Task):
    """Append rotation exits, sized buy orders; prune ranked to avoid double-buy."""

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.exits  import ExitSignal                              # noqa: PLC0415
        from kernel.sizing import (  # noqa: PLC0415
            compute_position_size,
            conviction_multiplier,
            sigma_multiplier,
            universe_sigma_median,
        )

        if not ctx.rotations:
            return

        regime_p     = ctx.config.get("regime_params", {}).get(ctx.regime, {})
        base_max_pct = float(regime_p.get("max_position_pct", 0.15)) * ctx.confidence
        reserve_pct  = float(regime_p.get("cash_reserve_pct", 0.0))  * ctx.confidence
        sizing_cfg   = (ctx.config.get("ranking", {})
                         .get("panel_scoring", {}).get("sizing", {}))
        sigma_cfg    = (ctx.config.get("ranking", {})
                         .get("panel_scoring", {})
                         .get("sigma_sizing", {}))

        sigma_median = universe_sigma_median(
            [getattr(c, "sigma", None) for c in ctx.ranked]
        )

        rotated_buys: set[str] = set()
        for pair in ctx.rotations:
            ctx.exits.append((
                pair.sell_ticker,
                ExitSignal(
                    should_exit = True,
                    reason      = (f"rotation→{pair.buy_ticker} "
                                   f"net_adv={pair.net_advantage:+.4f} "
                                   f"horizon={pair.horizon_days}d"),
                    exit_type   = "rotation",
                ),
            ))

            price = ctx.prices.get(pair.buy_ticker, 0.0)
            if price <= 0:
                log.warning("EmitRotationsTask: no price for %s — skipping buy",
                            pair.buy_ticker)
                continue

            buy_cand = next((c for c in ctx.ranked if c.ticker == pair.buy_ticker), None)
            conv = conviction_multiplier(
                getattr(buy_cand, "panel_score", None) if buy_cand else None,
                sizing_cfg,
            )
            sig_m = sigma_multiplier(
                getattr(buy_cand, "sigma", None) if buy_cand else None,
                sigma_median, sigma_cfg,
            )
            max_pct = base_max_pct * conv * sig_m

            _, shares = compute_position_size(
                ctx.portfolio_value, ctx.cash,
                max_pct, reserve_pct, price,
            )
            if shares < 1:
                log.info("EmitRotationsTask: %s insufficient cash — skip rotation buy",
                         pair.buy_ticker)
                continue

            invest     = shares * price
            target_pct = invest / ctx.portfolio_value if ctx.portfolio_value > 0 else 0.0
            ctx.orders.append({
                "ticker":     pair.buy_ticker,
                "shares":     shares,
                "price":      price,
                "invest":     invest,
                "target_pct": target_pct,
                "regime":     ctx.regime,
                "confidence": ctx.confidence,
                "conviction": conv,
                "sigma_mult": sig_m,
                "rank_score": pair.buy_score,
                "rs_score":   0.0,
                "mu":         getattr(buy_cand, "mu", None)    if buy_cand else None,
                "sigma":      getattr(buy_cand, "sigma", None) if buy_cand else None,
                "detail":     (f"rotation←{pair.sell_ticker} "
                               f"net_adv={pair.net_advantage:+.4f} "
                               f"horizon={pair.horizon_days}d"),
            })
            rotated_buys.add(pair.buy_ticker)
            ctx.counters["rotations"] = ctx.counters.get("rotations", 0) + 1
            log.info(
                "ROTATION_EXEC  swap=%s→%s  shares=%d  net_adv=%+.4f  "
                "raw_adv=%+.4f  tax=%.4f  cost=%.4f  threshold=%+.4f  horizon=%dd",
                pair.sell_ticker, pair.buy_ticker, shares,
                pair.net_advantage, pair.raw_advantage,
                pair.tax_drag, pair.transaction_cost,
                pair.threshold, pair.horizon_days,
            )

        if rotated_buys:
            ctx.ranked = [c for c in ctx.ranked if c.ticker not in rotated_buys]
