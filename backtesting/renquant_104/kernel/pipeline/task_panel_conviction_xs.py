"""CrossSectionalPanelExitTask — model-bearish exit, calibrator-free.

Replaces the legacy PanelConvictionExitTask in TickerSellJob which has a
0/12336 historical fire rate due to calibrator saturation:

  Audit 2026-05-11:
    * 5983 alpaca-live sells: 0 panel_conviction
    * 12336 held bar-instances had rank<0.20 AND mu<0
    * Yet ZERO exits fired through that mechanism
    * Root cause: PanelConvictionExit gates on `rank_score` (calibrator
      output, saturates above panel ~ 0.5 → rank always ≥ 0.345). The
      legacy task IS reachable but its trigger condition is structurally
      unreachable in production.

This task fixes the root cause:

  * Bypass calibrator entirely. Use raw cross-sectional rank of
    TODAY's `panel_score` across the live candidate set.
  * Runs at PIPELINE LEVEL (not inside TickerSellJob) so it sees the
    full cross-section AFTER PanelScoringJob has finalized candidate
    + holding panel scores.
  * Fires when a held position is in the bottom N% of today's panel
    distribution AND NGBoost μ predicts negative return.

Architectural placement: pp_inference.py post-PanelScoringJob,
pre-RotationJob/SelectionJob/JointActionJob — so rotation logic sees
the updated ctx.exits before deciding swaps.

Config:
    risk:
      panel_exit:
        enabled: true
        # AND-rule: in bottom %ile AND mu ≤ ceiling. Both required.
        xs_panel_percentile_floor: 0.20  # bottom 20% of today's panel scores
        mu_sell_ceiling: 0.0             # NGBoost μ must be ≤ this
        # OR-rule (independent bypass): strong-mu alone fires regardless
        # of percentile. Captures cases like BA where mu=-0.12 but panel
        # is only 32%ile — model strongly says "this will lose money".
        mu_strong_sell_ceiling: -0.05    # μ ≤ -5% predicted 60d return → exit
        min_universe: 5                  # need at least this many scored to fire

References
----------
* 2026-05-11 audit (this commit) — Issue #1 + BA case study
* CLAUDE.md §5.13.10 — `if optional_field is not None defaults to dead
  code unless verified`. Sibling case: numerical condition structurally
  unreachable post-calibrator-saturation.
"""
from __future__ import annotations

import logging
import math

from kernel.exits import ExitSignal
from .context import InferenceContext
from .pipeline import Task
from .soft_exit_guards import (
    lt_gate_suppression,
    resolve_current_price,
    soft_exit_horizon_suppression,
    tax_adjusted_soft_exit_suppression,
)

log = logging.getLogger("kernel.pipeline.panel_conviction_xs")


class CrossSectionalPanelExitTask(Task):
    """Emit exit signals for held positions that are in the bottom N% of
    today's raw panel_score cross-section AND have NGBoost μ ≤ ceiling.

    Idempotent: skips tickers already exiting in ctx.exits.
    Fail-safe: NaN / None inputs → skip the ticker (no false exit).
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        cfg = (ctx.config.get("risk") or {}).get("panel_exit") or {}
        if not cfg.get("enabled", False):
            return None
        if not ctx.holdings:
            return None

        try:
            pct_floor   = float(cfg.get("xs_panel_percentile_floor", 0.20))
            mu_ceiling  = float(cfg.get("mu_sell_ceiling", 0.0))
            min_universe = int(cfg.get("min_universe", 5))
            # OR-bypass: strong negative mu fires regardless of percentile.
            # None disables (only the AND-rule fires).
            mu_strong_raw = cfg.get("mu_strong_sell_ceiling", None)
            mu_strong = (float(mu_strong_raw)
                         if mu_strong_raw is not None
                         and math.isfinite(float(mu_strong_raw))
                         else None)
        except (TypeError, ValueError):
            return None
        if not (0.0 < pct_floor < 1.0):
            return None
        if not math.isfinite(mu_ceiling):
            return None

        # ── Build cross-section of today's panel_score ───────────────
        all_scores: list[float] = []
        for c in (ctx.candidates or []):
            ps = getattr(c, "panel_score", None)
            if ps is not None:
                try:
                    f = float(ps)
                    if math.isfinite(f):
                        all_scores.append(f)
                except (TypeError, ValueError):
                    pass
        for h in ctx.holdings.values():
            ps = getattr(h, "panel_score", None)
            if ps is not None:
                try:
                    f = float(ps)
                    if math.isfinite(f):
                        all_scores.append(f)
                except (TypeError, ValueError):
                    pass

        if len(all_scores) < min_universe:
            return None

        # Bottom-percentile threshold on today's cross-section
        sorted_scores = sorted(all_scores)
        idx = int(round(len(sorted_scores) * pct_floor))
        idx = max(0, min(idx, len(sorted_scores) - 1))
        threshold = sorted_scores[idx]

        # Already-exiting tickers (skip — don't duplicate path-rule exits)
        already_exiting = {
            t for (t, sig) in (ctx.exits or [])
            if sig is not None and getattr(sig, "should_exit", False)
        }

        n_fires = 0
        for ticker, hs in ctx.holdings.items():
            if ticker in already_exiting:
                continue
            panel = getattr(hs, "panel_score", None)
            mu    = getattr(hs, "mu",          None)
            if panel is None or mu is None:
                continue
            try:
                pf = float(panel); mf = float(mu)
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(pf) and math.isfinite(mf)):
                continue

            # Trigger logic:
            #   AND-rule:    bottom-percentile  AND  mu ≤ mu_ceiling
            #   OR-bypass:   mu ≤ mu_strong_sell_ceiling (alone)
            fires_xs     = (pf <= threshold and mf <= mu_ceiling)
            fires_strong = (mu_strong is not None and mf <= mu_strong)
            if not (fires_xs or fires_strong):
                continue
            trigger_kind = "xs+mu" if fires_xs and fires_strong \
                           else ("xs" if fires_xs else "strong_mu")

            suppress, why = soft_exit_horizon_suppression(
                panel_cfg=cfg,
                regime=getattr(ctx, "regime", None),
                today=getattr(ctx, "today", None),
                holding=hs,
            )
            if suppress:
                ctx.counters["xs_panel_exit_horizon_suppressed"] = (
                    ctx.counters.get("xs_panel_exit_horizon_suppressed", 0) + 1
                )
                log.info(
                    "CrossSectionalPanelExit [%s]: SUPPRESSED by horizon gate "
                    "(%s panel=%+.3f mu=%+.4f)",
                    ticker, why, pf, mf,
                )
                continue

            cur_price = resolve_current_price(ctx, hs, ticker)
            suppress, why = lt_gate_suppression(
                config=ctx.config,
                today=getattr(ctx, "today", None),
                holding=hs,
                current_price=cur_price,
            )
            if suppress:
                log.info(
                    "CrossSectionalPanelExit [%s]: SUPPRESSED by LT tax gate "
                    "(%s panel=%+.3f mu=%+.4f)",
                    ticker, why, pf, mf,
                )
                continue

            suppress, why = tax_adjusted_soft_exit_suppression(
                panel_cfg=cfg,
                tax_cfg=ctx.config.get("tax") or {},
                today=getattr(ctx, "today", None),
                holding=hs,
                current_price=cur_price,
                mu=mf,
            )
            if suppress:
                ctx.counters["xs_panel_exit_tax_suppressed"] = (
                    ctx.counters.get("xs_panel_exit_tax_suppressed", 0) + 1
                )
                log.info(
                    "CrossSectionalPanelExit [%s]: SUPPRESSED by tax-adjusted gate "
                    "(%s panel=%+.3f mu=%+.4f)",
                    ticker, why, pf, mf,
                )
                continue

            sig = ExitSignal(
                should_exit = True,
                reason      = (
                    f"panel_conviction[{trigger_kind}] "
                    f"panel={pf:+.3f} (thr={threshold:+.3f} of {len(all_scores)}) "
                    f"mu={mf:+.4f}"
                ),
                exit_type   = "panel_conviction",
            )
            ctx.exits.append((ticker, sig))
            n_fires += 1
            log.info(
                "CrossSectionalPanelExit [%s]: EXIT (%s)  panel=%+.3f thr=%+.3f "
                "mu=%+.4f (mu_ceiling=%.4f mu_strong=%s)",
                ticker, trigger_kind, pf, threshold, mf, mu_ceiling,
                f"{mu_strong:+.4f}" if mu_strong is not None else "off",
            )

        if n_fires:
            ctx.counters["xs_panel_exit"] = (
                ctx.counters.get("xs_panel_exit", 0) + n_fires
            )
        return None
