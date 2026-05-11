"""LimitSellsPerBarTask — portfolio-level cap on model_sell exits per bar.

User spec 2026-04-26 round-7: "把我有的股票全卖了？这他妈的合理吗？"
Pre-fix, a single bar could exit 3-of-6 holdings simultaneously when
multiple per-ticker models all spiked sell signals on the same day.
Per-ticker rules can't see the portfolio-level effect — this task is
the portfolio manager's safety brake.

References:
  Almgren, R. & Chriss, N. (2000). "Optimal Execution of Portfolio
    Transactions", J. Risk 3(2): 5-39. — temporary market impact grows
    with execution rate; concentrated same-bar liquidations incur
    super-linear cost penalty (motivates spreading sells across bars).
  Bertsimas, D. & Lo, A.W. (1998). "Optimal Control of Execution
    Costs", J. Financial Markets 1: 1-50. — formal cost-of-haste model
    for unwinding multiple positions.
  Markowitz, H. (1952). "Portfolio Selection", J. Finance 7(1): 77-91.
    — diversification rationale; mass-exit destroys variance benefit
    accumulated through prior position-building.

Behavior:
  * Counts `model_sell` AND `panel_conviction` exits in ctx.exits — both
    are "soft" signal-driven exits (audit fix 2026-04-29: panel_conviction
    is a MODEL signal, not a price-action stop, so it shares the cap).
  * If combined count exceeds `risk.max_sells_per_bar`, sort by NGBoost
    μ ascending (most-bearish first), keep the top N, drop the rest.
  * Hard risk exits (stop_loss / trailing_stop / single_day_loss /
    max_hold / rotation / kelly_trim / gap_down / joint_sell) are EXEMPT
    — they always fire because their triggers are deterministic price
    events, not signal.
  * Default OFF (max_sells_per_bar=0 means uncapped).

Wired into both InferencePipeline and SellOnlyPipeline AFTER the
parallel sell-aggregation step, BEFORE PanelRankVetoJob.
"""
from __future__ import annotations

import logging
import math

from .context  import InferenceContext
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.limit_sells")


# Canonical exit-type taxonomy (CLAUDE.md §5.13.5 — one source).
# Refactored 2026-05-11 from inline frozenset to module import.
#
# Audit history (2026-04-29): `panel_conviction` was moved OUT of the
# exempt set and INTO the soft set — conviction loss is a MODEL signal
# not a hard price stop, so it's subject to the per-bar cap. Hard
# risk-class exits (stop_loss, trailing, gap, max_hold) stay exempt
# because their triggers are deterministic price events.
from kernel.exit_types import PER_BAR_CAP_EXEMPT as _RISK_EXIT_TYPES  # noqa: E402
from kernel.exit_types import PER_BAR_CAP_SUBJECT as _SOFT_SELL_TYPES  # noqa: E402


class LimitSellsPerBarTask(Task):
    """Cap model_sell exits per bar; risk exits exempt."""

    name = "LimitSellsPerBarTask"

    def run(self, ctx: InferenceContext) -> bool | None:
        max_n = int((ctx.config.get("risk", {}) or {})
                    .get("max_sells_per_bar", 0))
        if max_n <= 0:
            return False   # disabled
        if not ctx.exits:
            return False

        # Partition: risk-exits always pass; model_sells + panel_conviction
        # go through the cap (they're "soft" signal-driven exits).
        risk_kept: list = []
        model_sells: list = []   # list of (ticker, sig, mu_for_sort)
        for ticker, sig in ctx.exits:
            exit_type = str(getattr(sig, "exit_type", "") or "")
            if exit_type in _RISK_EXIT_TYPES:
                risk_kept.append((ticker, sig))
                continue
            if exit_type not in _SOFT_SELL_TYPES:
                # Unknown type — preserve (fail-open).
                risk_kept.append((ticker, sig))
                continue
            # model_sell or panel_conviction — collect with μ for ranking.
            held = (ctx.holdings or {}).get(ticker)
            mu_raw = getattr(held, "mu", None) if held is not None else None
            try:
                mu = float(mu_raw) if mu_raw is not None else None
            except (TypeError, ValueError):
                mu = None
            # Sort key: most-bearish μ first. Missing/NaN μ → treat as
            # +inf (least urgent → first to drop). Conservative: when
            # we can't measure conviction, drop it.
            if mu is None or not math.isfinite(mu):
                sort_mu = float("inf")
            else:
                sort_mu = mu
            model_sells.append((ticker, sig, sort_mu))

        if len(model_sells) <= max_n:
            return True   # under cap — no-op

        # Sort by μ ascending (most-bearish first), keep top N.
        model_sells.sort(key=lambda x: x[2])
        kept_model_sells = model_sells[:max_n]
        dropped         = model_sells[max_n:]

        # Diagnostic: surface what was dropped for ops visibility.
        if not hasattr(ctx, "exits_throttled"):
            ctx.exits_throttled = []
        for ticker, sig, mu_used in dropped:
            ctx.exits_throttled.append({
                "ticker":   ticker,
                "exit_type": "model_sell",
                "reason":   getattr(sig, "reason", ""),
                "mu":       mu_used if math.isfinite(mu_used) else None,
                "cap":      max_n,
                "n_total":  len(model_sells),
            })

        ctx.counters["model_sell_throttled"] = (
            ctx.counters.get("model_sell_throttled", 0) + len(dropped)
        )
        log.warning(
            "LimitSellsPerBarTask: %d model_sell candidates, cap=%d → "
            "kept %s, dropped %s (sorted by μ ascending; risk-exits exempt)",
            len(model_sells), max_n,
            ", ".join(t for t, _, _ in kept_model_sells),
            ", ".join(t for t, _, _ in dropped),
        )

        # Reassemble final exits list (order preserved within partitions
        # to match prior rotation/joint-sell expectations downstream).
        ctx.exits = risk_kept + [(t, s) for t, s, _ in kept_model_sells]
        return True
