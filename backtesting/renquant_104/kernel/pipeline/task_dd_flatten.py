"""S-2 (2026-05-11) — HARD FLATTEN at portfolio drawdown threshold.

The existing :class:`DrawdownCircuitTask` only sets ``ctx.skip_buys=True``
when drawdown ≥ ``drawdown_halt_pct`` — buys halt but **existing positions
continue to fall**. This is exactly the gap that let MaxDD reach 44.4%
on the 27-mo OOS sim while ``drawdown_halt_pct`` was set to 0.35.

This Task is the portfolio-level **kill switch**: when drawdown crosses
a configurable hard threshold (``flatten_pct``), augment ``ctx.exits``
with full-liquidation signals for every holding that doesn't already
have a path-rule exit this bar. The synthetic signals propagate through
the standard selection / execution path, so no special-case rewiring
downstream.

Architectural placement: runs in pp_inference.py **after** the parallel
Phase-2a TickerSellJob completes (so path-rule exits — trailing_stop /
stop_loss / single_day_loss / max_hold / model_sell — are already
appended to ctx.exits and we don't double-emit). The flatten signal
fills the gap for every untouched holding.

Reference: see cvxportfolio's ``RiskLimit`` constraint and Almgren-Chriss
2000 §5 — a hard upper bound on realised loss is the only mechanism with
a *deterministic* upper bound; everything else is statistical.

Config block (under ``risk``)::

    "risk": {
        "drawdown_flatten": {
            "enabled":     true,
            "flatten_pct": 0.25
        }
    }

Default: disabled. Golden behaviour preserved unless opted in.

Fail-open contract: non-finite ``hwm`` / ``portfolio_value`` →
no-op (DrawdownCircuitTask already fail-SAFEs the buy halt).
"""
from __future__ import annotations

import logging
import math

from kernel.exits import ExitSignal
from .context import InferenceContext
from .pipeline import Task
from .task_drawdown_rebalance import compute_portfolio_drawdown

log = logging.getLogger("kernel.pipeline.dd_flatten")


class DrawdownFlattenTask(Task):
    """Emit a full-liquidation ``ExitSignal`` for every held position
    NOT already in ``ctx.exits`` when portfolio drawdown ≥
    ``risk.drawdown_flatten.flatten_pct``.

    Path-rule exits already in ``ctx.exits`` are preserved (same
    semantics — full liquidation). Idempotent across re-runs in a bar.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        cfg = (ctx.config.get("risk") or {}).get("drawdown_flatten") or {}
        if not cfg.get("enabled", False):
            return None
        try:
            flatten_pct = float(cfg.get("flatten_pct", 0.0))
        except (TypeError, ValueError):
            return None
        if flatten_pct <= 0:
            return None
        if not (math.isfinite(ctx.hwm) and math.isfinite(ctx.portfolio_value)):
            return None
        if ctx.hwm <= 0 or not ctx.holdings:
            return None

        drawdown = compute_portfolio_drawdown(ctx.hwm, ctx.portfolio_value)
        if drawdown < flatten_pct:
            return None

        already_exiting = {t for (t, sig) in ctx.exits
                           if sig is not None and sig.should_exit}
        added = 0
        for tkr in ctx.holdings.keys():
            if tkr in already_exiting:
                continue
            sig = ExitSignal(
                should_exit = True,
                reason      = (f"drawdown_flatten dd={drawdown:.1%} "
                                f"≥ flatten_pct={flatten_pct:.1%}"),
                exit_type   = "drawdown_flatten",
            )
            ctx.exits.append((tkr, sig))
            added += 1

        log.info(
            "DrawdownFlattenTask: HARD FLATTEN — drawdown=%.1f%% ≥ "
            "flatten_pct=%.1f%%; added %d flatten exits (path rules "
            "already covered %d).",
            drawdown * 100, flatten_pct * 100,
            added, len(already_exiting),
        )
        # Also halt buys for this bar regardless of DrawdownCircuit's
        # halt_pct — once we flatten we shouldn't immediately rebuy.
        # DrawdownCircuit resume logic still controls future bars.
        ctx.skip_buys = True
        return None
