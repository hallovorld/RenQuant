"""Sizing task cluster — realized-vol fallback + Kelly sizing.

EXTRACTED 2026-06-13 from job_panel_scoring.py (eng plan S2 item 5,
decomposition slice 5; behavior-identical move, DRPH-gated with
pre-change baselines). Symbols re-exported from job_panel_scoring.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from kernel.pipeline.context import InferenceContext
from kernel.pipeline.pipeline import Task

log = logging.getLogger("kernel.panel_pipeline.scoring")


class ApplyRealizedVolFallbackTask(Task):
    """Fill c.sigma with trailing realized vol when NGBoost OFF.

    Background: NGBoost is the only task that writes `c.sigma` today.
    When NGBoost is disabled (current prod since 2026-05-09), every
    candidate's sigma is None → Kelly skips with `kelly_zero:sigma_none`.
    This task provides a fallback: annualized stdev of trailing 60-day
    daily returns from ctx.ohlcv[ticker]['close'].

    OPT-IN via `ranking.kelly_sizing.use_realized_vol_fallback=true`.
    Disabled by default so prod behavior is unchanged. Pairs with the
    Phase-3 `use_calibrator_mu` flag — both must be on to re-enable
    Kelly sizing with proper μ/σ via the calibrator + realized-vol path.

    Runs AFTER ApplyGlobalCalibrationTask (so c.mu is set) and BEFORE
    ApplyKellySizingTask (so Kelly sees the populated sigma).

    Reuses the same helper logic as RealizedVolGateTask, kept local
    here to avoid a kernel.pipeline import cycle.
    """

    def run(self, ctx: "InferenceContext") -> "bool | None":
        kelly_cfg = ctx.config.get("ranking", {}).get("kelly_sizing", {})
        if not bool(kelly_cfg.get("use_realized_vol_fallback", False)):
            return
        window = int(kelly_cfg.get("realized_vol_window_days", 60))
        floor = float(kelly_cfg.get("realized_vol_floor", 0.05))     # 5% σ floor
        ceiling = float(kelly_cfg.get("realized_vol_ceiling", 1.50)) # 150% σ cap

        ohlcv = getattr(ctx, "ohlcv", None) or {}
        n_filled = 0
        for c in ctx.candidates:
            if getattr(c, "sigma", None) is not None and math.isfinite(c.sigma):
                continue  # already populated by NGBoost
            sig = _realized_vol_annualized(ohlcv.get(c.ticker), window)
            if sig is not None:
                c.sigma = float(np.clip(sig, floor, ceiling))
                n_filled += 1

        for ticker, hs in ctx.holdings.items():
            if getattr(hs, "sigma", None) is not None and math.isfinite(hs.sigma):
                continue
            sig = _realized_vol_annualized(ohlcv.get(ticker), window)
            if sig is not None:
                hs.sigma = float(np.clip(sig, floor, ceiling))

        if n_filled:
            log.info(
                "ApplyRealizedVolFallbackTask: filled c.sigma from realized "
                "vol (window=%dd, clip=[%.2f, %.2f]) for %d/%d candidates",
                window, floor, ceiling, n_filled, len(ctx.candidates),
            )


def _realized_vol_annualized(df, window: int):
    """Return annualized stdev of daily returns over last `window` bars,
    or None if df is missing / has insufficient history.

    Pure function — mirrors RealizedVolGateTask._realized_vol_annualized
    so we don't create a kernel.pipeline → kernel.panel_pipeline cycle.
    """
    if df is None:
        return None
    try:
        close = df["close"]
    except (KeyError, TypeError):
        return None
    if len(close) < max(window, 5):
        return None
    rets = close.pct_change().tail(window).dropna()
    if len(rets) < max(window // 2, 5):
        return None
    std = float(rets.std())
    if not math.isfinite(std):
        return None
    return std * math.sqrt(252.0)


def _kelly_sigma_horizon_days(kelly_cfg: dict) -> float:
    raw = kelly_cfg.get("sigma_horizon_days", 252.0)
    try:
        days = float(raw)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(days) or days <= 0:
        return float("nan")
    return days


def _rescale_annualized_sigma_for_kelly(sigma: float, horizon_days: float) -> float:
    if horizon_days == 252.0:
        return sigma
    return sigma * math.sqrt(horizon_days / 252.0)


# ── Kelly sizing (Plan C — the smart part) ───────────────────────────────────

class ApplyKellySizingTask(Task):
    """Populate `kelly_target_pct` on every candidate AND holding using
    the classical continuous-returns Kelly: f* = μ/σ².

    Runs LAST in PanelScoringJob — after ApplyNGBoostTask writes μ,σ
    and ApplyGlobalCalibrationTask settles rank_score. The Kelly
    target is then consumed by three downstream layers:

      SizeAndEmitTask  — caps new-buy size at `kelly_target_pct`.
      TopUpHeldTask    — emits a BUY if held.kelly_target exceeds
                         current weight by `top_up_threshold`.
      RotationJob      — (future) rotation advantage test in Kelly
                         units rather than raw rank_score.

    One math, one place, one field. See `kernel/kelly.py` for the
    full formula + safety discussion.
    """

    def run(self, ctx: "InferenceContext") -> "bool | None":
        kelly_cfg = ctx.config.get("ranking", {}).get("kelly_sizing", {})
        if not kelly_cfg.get("enabled", False):
            return   # no-op — golden behaviour preserved

        from kernel.kelly import kelly_target_pct      # noqa: PLC0415

        fractional        = float(kelly_cfg.get("fractional",        0.25))
        min_edge          = float(kelly_cfg.get("min_edge",          0.0))
        max_concentration = float(kelly_cfg.get("max_concentration", 0.35))
        # Realized-vol fallback writes annualized σ. Default 252 keeps that
        # legacy unit; opt-in 60 aligns σ with the 60d calibrator μ horizon.
        sigma_horizon_days = _kelly_sigma_horizon_days(kelly_cfg)

        # Audit fix CONF-MULT (2026-04-25): floored confidence multiplier.
        from kernel.regime import confidence_to_size_multiplier  # noqa: PLC0415
        _conf_mult = confidence_to_size_multiplier(ctx.confidence)
        regime_p = ctx.config.get("regime_params", {}).get(ctx.regime, {})
        max_pct  = float(regime_p.get("max_position_pct", 0.15)) * _conf_mult

        # 2026-05-15 P0 cleanup: vol-target + DD-Kelly scaling REMOVED
        # from this local-variable path. They previously modified `max_pct`
        # (a function-scope variable that QP never reads) — see
        # doc/AUDIT_2026-05-12_dead_paths.md. The live implementation
        # lives in kernel.portfolio_qp.tasks.ApplyExposureScalingTask
        # which writes ctx._vol_target_scale / ctx._dd_kelly_scale and
        # multiplies them into ctx._qp_w_upper inside the QP job. That
        # is the architecturally correct location: all exposure-cap
        # modifiers compose at the QP bound, not inside a Kelly local
        # that may be unused when mu is None.

        # 2026-05-04 instrumentation (user mandate: explainable funnel,
        # decision-tree DB persistence). Per-candidate skip-reason
        # counters + write to ctx._blocked_by_ticker so SQL queries on
        # candidate_scores.blocked_by show exactly why each ticker was
        # filtered. Without this, the funnel stage "n_cands=48 →
        # kelly=0 non-zero" was opaque.
        import math   # noqa: PLC0415
        skip_counts = {
            "kelly_zero:mu_none":        0,
            "kelly_zero:mu_nonfinite":   0,
            "kelly_zero:sigma_none":     0,
            "kelly_zero:sigma_nonfinite":0,
            "kelly_zero:sigma_nonpos":   0,
            "kelly_zero:sigma_horizon_invalid": 0,
            "kelly_zero:mu_le_min_edge": 0,
            "kelly_zero:capped_zero":    0,
        }
        blocked = getattr(ctx, "_blocked_by_ticker", None) or {}

        def _kelly_with_reason(obj):
            mu_v = getattr(obj, "mu",    None)
            sg_v = getattr(obj, "sigma", None)
            if mu_v is None:
                return 0.0, "kelly_zero:mu_none"
            if sg_v is None:
                return 0.0, "kelly_zero:sigma_none"
            try:
                mu_f = float(mu_v)
                sg_f = float(sg_v)
            except (TypeError, ValueError):
                return 0.0, "kelly_zero:mu_nonfinite"
            if not math.isfinite(mu_f):
                return 0.0, "kelly_zero:mu_nonfinite"
            if not math.isfinite(sg_f):
                return 0.0, "kelly_zero:sigma_nonfinite"
            if sg_f <= 0:
                return 0.0, "kelly_zero:sigma_nonpos"
            if not math.isfinite(sigma_horizon_days):
                return 0.0, "kelly_zero:sigma_horizon_invalid"
            if mu_f <= min_edge:
                return 0.0, "kelly_zero:mu_le_min_edge"
            sg_f = _rescale_annualized_sigma_for_kelly(sg_f, sigma_horizon_days)
            target = kelly_target_pct(
                mu_f, sg_f,
                max_pct           = max_pct,
                max_concentration = max_concentration,
                fractional        = fractional,
                min_edge          = min_edge,
            )
            if target <= 0:
                return 0.0, "kelly_zero:capped_zero"
            return target, None

        for cand in ctx.candidates:
            target, reason = _kelly_with_reason(cand)
            cand.kelly_target_pct = target
            if reason is not None:
                skip_counts[reason] += 1
                # Don't clobber a more upstream block (e.g. ngb_skipped)
                blocked.setdefault(cand.ticker, reason)

        for hs in ctx.holdings.values():
            target, _ = _kelly_with_reason(hs)
            hs.kelly_target_pct = target

        ctx._blocked_by_ticker = blocked  # noqa: SLF001

        # Audit summary — most informative when live.
        cand_targets = [c.kelly_target_pct for c in ctx.candidates
                         if c.kelly_target_pct]
        held_targets = [h.kelly_target_pct for h in ctx.holdings.values()
                         if h.kelly_target_pct]
        # Compact skip-reason summary: only emit non-zero counts.
        skip_str = " ".join(f"{r.split(':',1)[1]}={c}"
                              for r, c in skip_counts.items() if c > 0)
        # 2026-06-02 daily decision-tree audit Finding B: the previous log
        # only printed ``holdings=N non-zero`` which the operator read as
        # "N holdings exist" — but it was actually "of M holdings, N had a
        # non-zero Kelly target" (the rest were zero'd by wash-sale / churn
        # / saturation / etc.). When the broker holds 7 positions but only 6
        # got a non-zero Kelly, the log said ``holdings=6`` and the audit
        # spent time chasing the missing holding. Surface BOTH the total
        # ``ctx.holdings`` count AND the non-zero subset so the gap is
        # readable. Same for candidates.
        log.info(
            "ApplyKellySizingTask: fractional=%.2f max_conc=%.2f  "
            "cands=%d/%d non-zero (avg=%.1f%%)  "
            "holdings=%d/%d non-zero (avg=%.1f%%)"
            "%s",
            fractional, max_concentration,
            len(cand_targets), len(ctx.candidates),
            (sum(cand_targets) / len(cand_targets) * 100) if cand_targets else 0,
            len(held_targets), len(ctx.holdings),
            (sum(held_targets) / len(held_targets) * 100) if held_targets else 0,
            f"  zero_reasons[{skip_str}]" if skip_str else "",
        )


# ── Job ──────────────────────────────────────────────────────────────────────
