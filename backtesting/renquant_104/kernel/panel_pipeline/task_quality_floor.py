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

Reference: ``doc/components/buy-logic-design.md``.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
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

    Reject when the candidate's CALIBRATED ``rank_score`` is below the
    trailing N-day p_X cutoff retrieved from ``score_percentiles_daily``.

    2026-05-03 P0 fix (same scale-mismatch class as VetoWeakBuysTask):
    pre-fix this read raw ``cand.panel_score`` (XGB margin ~ [0, 0.05])
    and compared to a threshold drawn from ``score_percentiles_daily``,
    which actually stores percentiles of CALIBRATED ``rank_score``
    (range [0, 1]). Currently dormant — production has
    ``distribution_floor.enabled=false`` — but still a latent bug.
    Fixed alongside the active VetoWeakBuysTask scale fix to keep the
    codebase coherent.
    """
    if threshold is None:
        return True, None
    score = getattr(cand, "rank_score", None)
    if score is None:
        return True, None
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        return True, None
    if score_f != score_f:    # NaN
        return False, "rank_score_nan"
    if score_f < threshold:
        return False, f"rank_score={score_f:+.4f}<{threshold:+.4f}"
    return True, None


def _gate_c_no_trade_band(
    cand: Any,
    *,
    risk_aversion: float,
    round_trip_cost: float,
    band_constant: float,
    current_weight: float,
) -> tuple[bool, str | None]:
    """Constantinides 1986 / Davis-Norman 1990 no-trade region.

    Davis-Norman closed form for log-utility w/ proportional cost τ:
        band_i = c · (γ · σ_i² · τ)^(1/3)

    Reject when |target_weight - current_weight| < band. The target
    weight is approximated as `μ_i / (γ · σ_i²)` (single-asset Kelly-
    equivalent). The combined check:

        admit ⇔ |μ / (γ σ²) - w_current| > c · (γ σ² τ)^(1/3)

    For our parameters (γ=3, σ=0.08, τ=0.001): band ≈ 4% NAV. So a
    candidate at current weight=0 with target weight 3% (would-be
    deviation of 3%) is REJECTED — natural "no-trade region" behaviour
    that prevents fill-empty-slots-with-weak-signal.

    Returns (passes, reject_reason).
    """
    mu    = getattr(cand, "mu",    None)
    sigma = getattr(cand, "sigma", None)
    if mu is None or sigma is None:
        return True, None
    try:
        mu_f    = float(mu)
        sigma_f = float(sigma)
    except (TypeError, ValueError):
        return True, None
    # 2026-05-04 audit Issue 23: NaN sigma slips past `<= 0.0` (NaN <= 0
    # is False) → sigma_sq = NaN → target_w = NaN → `deviation < band` is
    # False → CANDIDATE PASSES the gate (fail-OPEN). The mu NaN-check
    # below was already there (mu_f != mu_f) but the symmetric sigma
    # check was missing. Add explicit isfinite guard.
    import math as _math
    if not _math.isfinite(sigma_f) or not _math.isfinite(mu_f):
        return False, "sigma_or_mu_nonfinite"
    if sigma_f <= 0.0 or mu_f != mu_f:
        return False, "sigma_zero_or_mu_nan"
    sigma_sq = sigma_f * sigma_f
    target_w  = mu_f / (risk_aversion * sigma_sq)
    band      = band_constant * (
        (risk_aversion * sigma_sq * round_trip_cost) ** (1.0 / 3.0)
    )
    deviation = abs(target_w - current_weight)
    if deviation < band:
        return False, (
            f"deviation={deviation:.4f}<band={band:.4f}"
        )
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
    # 2026-05-04 audit Issue 24: NaN sigma slips past `<= 0.0` (NaN <= 0
    # is False) → edge_sharpe = mu/NaN = NaN → `edge_sharpe < threshold`
    # is False → CANDIDATE PASSES (fail-OPEN). Same class as Issue 23
    # in _gate_c. Explicit isfinite first.
    import math as _math
    if not _math.isfinite(sigma_f) or not _math.isfinite(mu_f):
        return False, "sigma_or_mu_nonfinite"
    if sigma_f <= 0.0:
        return False, "sigma_nonpositive"
    if mu_f != mu_f:   # NaN (now redundant, kept for explicit symmetry)
        return False, "mu_nan"
    edge_sharpe = mu_f / sigma_f
    if edge_sharpe < threshold:
        return False, f"edge_sharpe={edge_sharpe:+.3f}<{threshold:.3f}"
    return True, None


class QualityFloorTask(Task):
    """Filter ctx.candidates by quality gates A/B/C (each flag-controlled).

    All three gates are implemented (Stage 1 complete, 2026-04-26):
      Gate A — Distribution-relative percentile floor (reads score_db)
      Gate B — Edge-Sharpe floor (Lo 2002 → μ/σ > τ_S)
      Gate C — No-trade band (Constantinides 1986 / Davis-Norman 1990)

    Each gate is independently flag-controlled. With every gate disabled
    (the default) ctx.candidates is left bit-for-bit untouched.

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
        # M3 (2026-04-28): regime-conditional conformal-fitted τ overrides
        # the static config value when an artifact is present. Falls back
        # to config threshold for unfit regimes (e.g. BEAR with no live
        # history yet). Disable via gate_b.use_conformal=false.
        gate_b_threshold = self._gate_b_static_threshold(ctx, gate_b_cfg)
        if gate_b_enabled and gate_b_cfg.get("use_conformal", True):
            tau = self._gate_b_conformal_tau(ctx, getattr(ctx, "regime", None))
            if tau is not None:
                gate_b_threshold = float(tau)

        # Gate C — Constantinides no-trade band --------------------------
        gate_c_cfg = cfg.get("no_trade_band", {})
        gate_c_enabled = bool(gate_c_cfg.get("enabled", False))
        gate_c_gamma   = float(gate_c_cfg.get("risk_aversion", 3.0))
        gate_c_tau     = float(gate_c_cfg.get("round_trip_cost", 0.001))
        gate_c_const   = float(gate_c_cfg.get("band_constant", 1.5))

        if (not gate_a_enabled and not gate_b_enabled and
                not gate_c_enabled):
            return True

        # Pre-compute holdings weights for Gate C deviation calc.
        portfolio_value = float(getattr(ctx, "portfolio_value", 0.0) or 0.0)
        holdings_weights: dict[str, float] = {}
        if gate_c_enabled and portfolio_value > 0.0:
            for tk, hs in (ctx.holdings or {}).items():
                shares = float(getattr(hs, "shares", 0.0) or 0.0)
                px     = float((ctx.prices or {}).get(tk, 0.0) or 0.0)
                if px > 0.0 and shares > 0.0:
                    holdings_weights[tk] = (shares * px) / portfolio_value

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
            if reason is None and gate_c_enabled:
                w_curr = holdings_weights.get(ticker, 0.0)
                ok_c, reason_c = _gate_c_no_trade_band(
                    c,
                    risk_aversion = gate_c_gamma,
                    round_trip_cost = gate_c_tau,
                    band_constant = gate_c_const,
                    current_weight = w_curr,
                )
                if not ok_c:
                    reason = f"gate_c:{reason_c}"
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
                "(gate_a=%s, gate_b_τ=%.3f, regime=%s): %s",
                len(rejected), len(ctx.candidates),
                f"{gate_a_threshold:+.4f}" if gate_a_threshold is not None
                else "off",
                gate_b_threshold if gate_b_enabled else float("nan"),
                getattr(ctx, "regime", None),
                ", ".join(f"{t}({r})" for t, r in rejected[:5])
                + ("…" if len(rejected) > 5 else ""),
            )
        ctx.candidates = kept
        return True

    @staticmethod
    def _gate_b_static_threshold(
        ctx: InferenceContext,
        gate_b_cfg: dict,
    ) -> float:
        """Resolve static Gate-B τ with regime config before global fallback."""
        default = float(gate_b_cfg.get("threshold", 0.20))
        regime = getattr(ctx, "regime", None)
        if regime is None:
            return default
        regime_params = ctx.config.get("regime_params", {})
        if not isinstance(regime_params, dict):
            return default
        regime_cfg = regime_params.get(regime, {})
        if not isinstance(regime_cfg, dict):
            return default

        nested = (
            regime_cfg.get("quality_floor", {})
            if isinstance(regime_cfg.get("quality_floor", {}), dict)
            else {}
        )
        nested_gate_b = (
            nested.get("edge_sharpe_floor", {})
            if isinstance(nested.get("edge_sharpe_floor", {}), dict)
            else {}
        )
        raw = nested_gate_b.get(
            "threshold",
            regime_cfg.get("edge_sharpe_floor_threshold", default),
        )
        try:
            threshold = float(raw)
        except (TypeError, ValueError):
            log.warning(
                "Gate B regime threshold invalid for regime=%s: %r; using global %.3f",
                regime, raw, default,
            )
            return default
        if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
            log.warning(
                "Gate B regime threshold outside [0,1] for regime=%s: %r; "
                "using global %.3f",
                regime, raw, default,
            )
            return default
        return threshold

    @staticmethod
    def _gate_b_conformal_tau(
        ctx: InferenceContext, regime: str | None,
    ) -> float | None:
        """Read regime-keyed τ from a config-driven gate_b artifact path.

        Path source (priority): ``ranking.panel_scoring.quality_floor.
        gate_b_artifact_path`` → falls back to ``prod/gate_b_thresholds.json``
        relative to ``<strategy_dir>/artifacts/``.

        Produced by ``scripts/fit_conformal_gate_b.py``. Returns None when
        any of the following — caller then falls back to the static config
        threshold:

          * regime is None / missing in artifact
          * artifact file absent / unreadable
          * artifact too stale (older than ``conformal_max_age_days``)
            — STALE-1 self-audit fix 2026-04-28
          * τ is non-finite / outside [0.0, 1.0]
          * thresholds dict is the wrong type
        """
        if regime is None:
            return None
        try:
            import datetime as _dt    # noqa: PLC0415
            import json as _j         # noqa: PLC0415
            import math as _m         # noqa: PLC0415

            strategy_dir = Path(ctx.config.get("_strategy_dir", ""))
            if not strategy_dir.is_absolute():
                return None
            # 2026-05-11 sim/prod isolation: artifact lives under prod/ or sim/.
            qfloor_cfg = (
                ctx.config.get("ranking", {})
                .get("panel_scoring", {})
                .get("quality_floor", {})
            )
            rel_path = qfloor_cfg.get(
                "gate_b_artifact_path", "prod/gate_b_thresholds.json"
            )
            path = strategy_dir / "artifacts" / rel_path
            if not path.exists():
                return None
            data = _j.loads(path.read_text())
            if not isinstance(data, dict):
                log.warning("Conformal Gate B: artifact is not a dict — skip")
                return None

            # STALE-1: max-age check. Default 7 days; operator can extend or
            # disable via gate_b.conformal_max_age_days = 0.
            max_age_days = int(
                ctx.config.get("ranking", {})
                          .get("panel_scoring", {})
                          .get("quality_floor", {})
                          .get("edge_sharpe_floor", {})
                          .get("conformal_max_age_days", 7)
            )
            if max_age_days > 0:
                fitted_at = data.get("fitted_at")
                if fitted_at:
                    try:
                        # Accept ISO with or without timezone
                        dt = _dt.datetime.fromisoformat(fitted_at.replace("Z", "+00:00"))
                        # Strip tz for naive subtraction
                        if dt.tzinfo is not None:
                            dt = dt.replace(tzinfo=None)
                        age_days = (_dt.datetime.utcnow() - dt).total_seconds() / 86400
                        if age_days > max_age_days:
                            log.warning(
                                "Conformal Gate B artifact is %.1f days old "
                                "(max %d). Falling back to config τ. "
                                "Refresh: scripts/fit_conformal_gate_b.py",
                                age_days, max_age_days,
                            )
                            return None
                    except (ValueError, TypeError):
                        log.warning("Conformal Gate B: unparseable fitted_at=%r", fitted_at)
                        return None

            thresholds = data.get("thresholds")
            if not isinstance(thresholds, dict):
                log.warning("Conformal Gate B: thresholds is not a dict — skip")
                return None
            tau = thresholds.get(regime)
            if tau is None:
                return None
            tau_f = float(tau)
            if not _m.isfinite(tau_f) or tau_f < 0.0 or tau_f > 1.0:
                log.warning(
                    "Conformal Gate B: τ=%r for regime=%s outside [0,1] — skip",
                    tau, regime,
                )
                return None
            log.info(
                "Conformal Gate B: regime=%s τ=%.4f (artifact age %.1fd)",
                regime, tau_f,
                ((_dt.datetime.utcnow() - _dt.datetime.fromisoformat(
                    data.get("fitted_at", "1970-01-01T00:00:00").replace("Z", "+00:00")
                ).replace(tzinfo=None)).total_seconds() / 86400)
                if data.get("fitted_at") else float("nan"),
            )
            return tau_f
        except Exception as exc:
            log.warning("Conformal Gate B read failed: %s — using config τ", exc)
            return None

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
        run_type = getattr(ctx, "_run_type", None) or getattr(ctx, "run_type", None)
        try:
            cur = db.cursor()
            if run_type:
                cur.execute(
                    """SELECT COUNT(*) FROM score_percentiles_daily
                       WHERE date < ? AND run_type = ?""",
                    (today_iso, run_type),
                )
            else:
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
            run_type=run_type if isinstance(run_type, str) and run_type else None,
            include_today=False,
        )
