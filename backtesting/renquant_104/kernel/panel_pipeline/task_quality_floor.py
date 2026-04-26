"""Quality-gate filtering for candidate buys (Stage 0 — flag OFF by default).

Three gates, each grounded in a different established framework:

  Gate A  Distribution-relative floor (cross-sectional percentile)
          → ranking.panel_scoring.quality_floor.distribution_floor
  Gate B  Edge-Sharpe floor (Lo 2002 / Grinold-Kahn 1999)
          → ranking.panel_scoring.quality_floor.edge_sharpe_floor
  Gate C  No-trade region (Constantinides 1986 / Davis-Norman 1990)
          → ranking.panel_scoring.quality_floor.no_trade_band

A candidate must pass ALL enabled gates. Disabled gates are skipped.
With every gate disabled (the Stage-0 default) ctx.candidates is left
untouched — bit-for-bit parity with current behaviour.

Reference: ``doc/buy_logic_redesign_2026-04-26.md``.
"""
from __future__ import annotations

import logging
from typing import Any

from kernel.pipeline.context import InferenceContext
from kernel.pipeline.pipeline import Task

log = logging.getLogger("kernel.panel_pipeline.quality_floor")


_BLOCKED_REASON_PREFIX = "quality_floor:"


def _gate_b_edge_sharpe(
    cand: Any,
    threshold: float,
) -> tuple[bool, str | None]:
    """Lo 2002 — predicted instantaneous Sharpe of the edge.

    edge_sharpe = μ / σ. Reject when below threshold or σ ≤ 0 or
    μ NaN. Returns (passes, reject_reason).
    """
    mu    = getattr(cand, "mu",    None)
    sigma = getattr(cand, "sigma", None)
    if mu is None or sigma is None:
        return True, None  # no NGBoost → no signal to gate; pass
    try:
        mu_f    = float(mu)
        sigma_f = float(sigma)
    except (TypeError, ValueError):
        return True, None
    if sigma_f <= 0.0:
        return False, "sigma_nonpositive"
    if mu_f != mu_f:   # NaN
        return False, "mu_nan"
    edge_sharpe = mu_f / sigma_f
    if edge_sharpe < threshold:
        return False, f"edge_sharpe={edge_sharpe:+.3f}<{threshold:.3f}"
    return True, None


class QualityFloorTask(Task):
    """Filter ctx.candidates by quality gates A/B/C (each flag-controlled).

    Stage 0: only Gate B (Edge-Sharpe) is implemented. Gates A and C are
    placeholders for future commits — the flag schema is in place so
    enabling them later doesn't churn config files.

    Doesn't touch ctx.holdings — quality floors are buy-side gates.
    Sells / rotations have their own (path-dependent) controls.
    """

    name = "QualityFloorTask"

    def run(self, ctx: InferenceContext) -> bool | None:
        cfg = (ctx.config.get("ranking", {})
                          .get("panel_scoring", {})
                          .get("quality_floor", {}))
        if not cfg.get("enabled", False):
            return True
        if not ctx.candidates:
            return True

        # Gate B — Edge Sharpe ----------------------------------------
        gate_b_cfg = cfg.get("edge_sharpe_floor", {})
        gate_b_enabled = bool(gate_b_cfg.get("enabled", False))
        gate_b_threshold = float(gate_b_cfg.get("threshold", 0.20))

        if not gate_b_enabled:
            return True

        kept: list[Any] = []
        rejected: list[tuple[str, str]] = []
        for c in ctx.candidates:
            ticker = getattr(c, "ticker", "?")
            ok, reason = _gate_b_edge_sharpe(c, gate_b_threshold)
            if ok:
                kept.append(c)
            else:
                rejected.append((ticker, reason or "unknown"))

        if rejected:
            blocked = getattr(ctx, "_blocked_by_ticker", None) or {}
            for ticker, reason in rejected:
                blocked[ticker] = f"{_BLOCKED_REASON_PREFIX}gate_b:{reason}"
            ctx._blocked_by_ticker = blocked  # noqa: SLF001
            log.info(
                "QualityFloorTask: Gate-B rejected %d/%d cand(s) "
                "(τ_S=%.3f): %s",
                len(rejected), len(ctx.candidates),
                gate_b_threshold,
                ", ".join(f"{t}({r})" for t, r in rejected[:5])
                + ("…" if len(rejected) > 5 else ""),
            )
        ctx.candidates = kept
        return True
