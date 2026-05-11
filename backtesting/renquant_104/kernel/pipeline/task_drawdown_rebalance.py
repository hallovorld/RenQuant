"""L3 — Grossman-Zhou drawdown-constrained portfolio rebalance.

When the live drawdown approaches the investor's tolerance, scale gross
exposure DOWN by liquidating the weakest holdings (lowest cross-sectional
panel score). Operationalises Grossman & Zhou 1993, "Optimal Investment
Strategies for Controlling Drawdowns," JEEM 19(2):241-276:

    f*(DD_t) = f_Kelly × max(0, 1 - DD_t / DD_max)              (Eq. 8)

At the portfolio level with N held names, ``target_count = round(N × f*)``;
the (N - target_count) weakest by ``panel_score`` are liquidated.

Architectural inspiration: cvxportfolio (Boyd et al, Cambridge 2024)
"Multi-period optimization" §4.2 — risk-aware optimizer trims gross
exposure as forecast drawdown breaches the soft cap. Direct port of
their constraint isn't feasible here (we don't run a full CVXPY solve
each bar) but the IDEA — "weakest first, scaled by DD" — translates
directly.

Per CLAUDE.md §5.13.10: the optional config block ``risk.drawdown_rebalance``
defaults to disabled. Production opt-in via setting ``enabled=true``.
"""
from __future__ import annotations

import logging
import math

from kernel.exits import ExitSignal
from .context import InferenceContext
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.drawdown_rebalance")


def compute_portfolio_drawdown(hwm: float, portfolio_value: float) -> float:
    """Current portfolio drawdown ``(HWM - PV) / HWM``, clipped to [0, 1].

    Single source of truth for the canonical drawdown ratio used by both
    :class:`DrawdownCircuitTask` (buy-side halt) and
    :class:`DrawdownRebalanceTask` (sell-side Grossman-Zhou rebalance).
    Returns 0 on any non-finite or zero-HWM input so callers can compare
    against thresholds without further NaN guards. Mirrors the canonical
    drawdown formula used in ``quantstats.stats.max_drawdown`` and
    ``ffn.calc_max_drawdown`` (both compute per-bar DD relative to
    running max, then take the min).
    """
    if not math.isfinite(hwm) or not math.isfinite(portfolio_value):
        return 0.0
    if hwm <= 0:
        return 0.0
    dd = (hwm - portfolio_value) / hwm
    if not math.isfinite(dd):
        return 0.0
    # Negative DD (PV above HWM, pre-HWM-update) → 0.
    return max(0.0, dd)


def compute_kelly_scaling(drawdown: float, dd_max: float) -> float:
    """Grossman-Zhou 1993 Eq. 8 evaluated at the current drawdown.

    Args:
        drawdown: current portfolio drawdown, ``(HWM - PV) / HWM`` (≥ 0).
        dd_max: maximum drawdown tolerance, e.g. 0.30 for a 30% cap.

    Returns:
        Kelly fraction multiplier in [0, 1]:
          * 1.0 when drawdown ≤ 0 (no scaling — at or above HWM).
          * Linear in (drawdown, dd_max).
          * 0.0 when drawdown ≥ dd_max.

    Closed-form (Grossman-Zhou Eq. 8 reformulation):
        f* / f_Kelly = max(0, 1 - drawdown / dd_max)
    """
    if dd_max <= 0 or not math.isfinite(dd_max):
        return 1.0
    if not math.isfinite(drawdown) or drawdown <= 0:
        return 1.0
    return max(0.0, 1.0 - drawdown / dd_max)


class DrawdownRebalanceTask(Task):
    """Liquidate the weakest holdings when drawdown breaches the trigger.

    Reads
        ctx.config["risk"]["drawdown_rebalance"]:
            enabled: bool
            trigger_drawdown: float (e.g. 0.20 = 20%)
            max_drawdown: float (e.g. 0.30 = 30%, Grossman-Zhou DD_max)
        ctx.hwm, ctx.portfolio_value
        ctx.holdings (dict[ticker, HoldingState] with panel_score)
        ctx.exits (list[(ticker, ExitSignal)] — used to skip already-exiting)

    Writes
        ctx.exits — appends ExitSignal(exit_type="drawdown_rebalance",
                                       quantity=None) for the
                    (N - target_count) weakest still-open positions.
    """

    def run(self, ctx: InferenceContext) -> "bool | None":
        cfg = (
            ctx.config.get("risk", {}).get("drawdown_rebalance", {})
            if isinstance(ctx.config, dict) else {}
        )
        if not cfg.get("enabled", False):
            return None
        trigger = float(cfg.get("trigger_drawdown", 0.20))
        dd_max = float(cfg.get("max_drawdown", 0.30))

        # Use the single-source-of-truth DD helper (NaN-guarded internally).
        hwm = float(ctx.hwm)
        pv = float(ctx.portfolio_value)
        if not (math.isfinite(pv) and pv > 0):
            return None
        drawdown = compute_portfolio_drawdown(hwm, pv)
        if drawdown < trigger:
            return None  # not yet armed

        # Filter to open positions (not already exiting).
        already_exiting = {t for t, _ in (ctx.exits or [])}
        open_positions = [
            (t, hs) for t, hs in (ctx.holdings or {}).items()
            if t not in already_exiting
        ]
        if not open_positions:
            return None

        # Grossman-Zhou Kelly scaling.
        f_kelly = compute_kelly_scaling(drawdown, dd_max)
        n_open = len(open_positions)
        target_count = int(math.floor(n_open * f_kelly))
        to_liquidate = n_open - target_count
        if to_liquidate <= 0:
            return None

        # Rank by panel_score ASC — weakest first. Treat None as -inf
        # (weakest, prioritised for liquidation).
        def _key(item):
            _, hs = item
            ps = getattr(hs, "panel_score", None)
            return (ps if (ps is not None and math.isfinite(ps)) else float("-inf"))

        ranked = sorted(open_positions, key=_key)
        targets = ranked[:to_liquidate]

        reason_tpl = (
            f"drawdown_rebalance DD={drawdown:.1%} ≥ trigger={trigger:.1%} "
            f"f_kelly={f_kelly:.2f}"
        )
        for ticker, _hs in targets:
            sig = ExitSignal(
                should_exit=True,
                reason=reason_tpl,
                exit_type="drawdown_rebalance",
                quantity=None,   # full liquidate
            )
            ctx.exits.append((ticker, sig))

        log.info(
            "DrawdownRebalanceTask: DD=%.1f%% f_kelly=%.2f n_open=%d "
            "liquidated=%d (weakest first)",
            drawdown * 100, f_kelly, n_open, to_liquidate,
        )
        return None


__all__ = ["DrawdownRebalanceTask", "compute_kelly_scaling"]
