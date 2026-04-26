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


def _gate_a_distribution_floor(
    cand: Any,
    threshold: float | None,
) -> tuple[bool, str | None]:
    """Distribution-relative floor (cross-sectional percentile lookup).

    Reject when the candidate's `panel_score` is below the trailing N-day
    p_X cutoff retrieved from `score_percentiles_daily`. Returns
    (passes, reject_reason). When threshold is None (no history yet),
    skip the gate.
    """
    if threshold is None:
        return True, None
    panel = getattr(cand, "panel_score", None)
    if panel is None:
        return True, None
    try:
        panel_f = float(panel)
    except (TypeError, ValueError):
        return True, None
    if panel_f != panel_f:    # NaN
        return False, "panel_nan"
    if panel_f < threshold:
        return False, f"panel_score={panel_f:+.4f}<{threshold:+.4f}"
    return True, None


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

        # Gate A — Distribution-relative floor (cross-sectional pct) -----
        gate_a_cfg = cfg.get("distribution_floor", {})
        gate_a_enabled = bool(gate_a_cfg.get("enabled", False))
        gate_a_threshold: float | None = None
        if gate_a_enabled:
            gate_a_threshold = self._gate_a_threshold(ctx, gate_a_cfg)

        # Gate B — Edge Sharpe -------------------------------------------
        gate_b_cfg = cfg.get("edge_sharpe_floor", {})
        gate_b_enabled = bool(gate_b_cfg.get("enabled", False))
        gate_b_threshold = float(gate_b_cfg.get("threshold", 0.20))

        if not gate_a_enabled and not gate_b_enabled:
            return True

        kept: list[Any] = []
        rejected: list[tuple[str, str]] = []
        for c in ctx.candidates:
            ticker = getattr(c, "ticker", "?")
            reason: str | None = None
            if gate_a_enabled:
                ok_a, reason_a = _gate_a_distribution_floor(
                    c, gate_a_threshold,
                )
                if not ok_a:
                    reason = f"gate_a:{reason_a}"
            if reason is None and gate_b_enabled:
                ok_b, reason_b = _gate_b_edge_sharpe(
                    c, gate_b_threshold,
                )
                if not ok_b:
                    reason = f"gate_b:{reason_b}"
            if reason is None:
                kept.append(c)
            else:
                rejected.append((ticker, reason))

        if rejected:
            blocked = getattr(ctx, "_blocked_by_ticker", None) or {}
            for ticker, reason in rejected:
                blocked[ticker] = f"{_BLOCKED_REASON_PREFIX}{reason}"
            ctx._blocked_by_ticker = blocked  # noqa: SLF001
            log.info(
                "QualityFloorTask: rejected %d/%d cand(s) "
                "(gate_a=%s, gate_b_τ=%.3f): %s",
                len(rejected), len(ctx.candidates),
                f"{gate_a_threshold:+.4f}" if gate_a_threshold is not None
                else "off",
                gate_b_threshold if gate_b_enabled else float("nan"),
                ", ".join(f"{t}({r})" for t, r in rejected[:5])
                + ("…" if len(rejected) > 5 else ""),
            )
        ctx.candidates = kept
        return True

    @staticmethod
    def _gate_a_threshold(
        ctx: InferenceContext,
        gate_a_cfg: dict,
    ) -> float | None:
        """Look up trailing-N-day percentile cutoff from score DB.

        Returns None if there's no DB attached or insufficient history,
        in which case Gate A no-ops (defensive).
        """
        db = getattr(ctx, "_db", None)
        if db is None:
            return None
        try:
            from kernel.pipeline.task_score_distribution import (  # noqa: PLC0415
                get_score_percentile_threshold,
            )
        except Exception:
            return None
        percentile = int(gate_a_cfg.get("percentile", 85))
        lookback   = int(gate_a_cfg.get("lookback_days", 20))
        min_history = int(gate_a_cfg.get("min_history_days", 5))
        try:
            today_iso = ctx.today.isoformat()
        except Exception:
            return None
        try:
            cur = db.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM score_percentiles_daily WHERE date < ?",
                (today_iso,),
            )
            row = cur.fetchone()
            n_rows = int(row[0]) if row else 0
        except Exception:
            return None
        if n_rows < min_history:
            return None
        return get_score_percentile_threshold(
            db, today_iso, percentile=percentile, lookback_days=lookback,
        )
