"""Portfolio-QP pipeline Tasks — atom-composed.

User mandate (2026-05-04 §1c): Tasks are reusable atoms; domain Tasks
glue them with QP-specific math. This file holds the QP-specific
domain Tasks; reusable building blocks live in
`kernel/pipeline/atoms/`.

Job composition (in `job_qp.py`):

    JointPortfolioQPJob
    ├── SkipIfConfigDisabledTask("rotation.joint_actions.enabled")     [atom]
    ├── SkipIfFieldEqualsTask("bear_only", True)                        [atom]
    ├── StableTickerOrderTask("holdings", "candidates", "_qp_tickers")  [atom]
    ├── BuildWeightVectorTask                                           [domain]
    ├── BuildVectorFromMappingTask × N (mu, sigma)                      [atom]
    ├── ComputeFullSigmaTask                                            [domain]
    ├── ComputeBrownSmithTaxCostTask                                    [domain]
    ├── ComputeWashSaleMaskTask                                         [domain — uses BuildMaskFromConditionTask atom]
    ├── ComputeQPConstraintsTask                                        [domain]
    ├── SolveMarkowitzQPTask                                            [domain]
    ├── EmitOrdersFromQPSolutionTask                                    [domain]
    ├── IncrementCounterTask × 2                                        [atom]
    └── LogSummaryTask                                                  [atom]

Each domain Task here is ≤30 lines body, single-responsibility.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from kernel.pipeline.atoms.ctx_ops import _get_path, _set_path
from kernel.pipeline.context import InferenceContext
from kernel.pipeline.pipeline import Task

log = logging.getLogger("kernel.portfolio_qp.tasks")


# ── 1. Build w_current from shares × prices / NAV ────────────────────────────

class BuildWeightVectorTask(Task):
    """Compute current portfolio weight vector from holdings.

    Reads:  ctx._qp_tickers (list[str]), ctx.holdings (dict),
             ctx.prices (dict), ctx.portfolio_value (float)
    Writes: ctx._qp_w_current (np.ndarray)
    """
    name = "BuildWeightVectorTask"

    def run(self, ctx) -> bool | None:
        tickers = _get_path(ctx, "_qp_tickers") or []
        if not tickers:
            return False
        nav = float(_get_path(ctx, "portfolio_value", 0.0) or 0.0)
        if nav <= 0:
            return False
        prices = _get_path(ctx, "prices") or {}
        holdings = _get_path(ctx, "holdings") or {}
        w = np.zeros(len(tickers))
        for i, t in enumerate(tickers):
            hs = holdings.get(t)
            if hs is None:
                continue
            shares = float(getattr(hs, "shares", 0.0) or 0.0)
            px = float(prices.get(t, 0.0) or 0.0)
            if px > 0:
                w[i] = shares * px / nav
        ctx._qp_w_current = w  # noqa: SLF001


# ── 2. Build full Σ from a cached correlation matrix ────────────────────────

class ComputeFullSigmaTask(Task):
    """Build n×n Σ_full = ρ × σ_i × σ_j from `watchlist-correlation.json`.

    Reads:  ctx._qp_tickers, ctx._qp_sigma, ctx.config['_strategy_dir'],
             ctx.config['rotation']['joint_actions']['qp_use_full_sigma']
    Writes: ctx._qp_Sigma_full (np.ndarray | None — None falls back to
             diagonal Σ in the solver)
    """
    name = "ComputeFullSigmaTask"

    def run(self, ctx) -> bool | None:
        cfg = _qp_cfg(ctx)
        if not bool(cfg.get("qp_use_full_sigma", True)):
            ctx._qp_Sigma_full = None  # noqa: SLF001
            return
        sd = (ctx.config or {}).get("_strategy_dir", "")
        path = Path(sd) / "artifacts" / "watchlist-correlation.json" if sd else None
        if not path or not path.exists():
            ctx._qp_Sigma_full = None  # noqa: SLF001
            return
        try:
            raw = json.loads(path.read_text())
        except Exception as exc:
            log.warning("ComputeFullSigmaTask: corr load failed (%s)", exc)
            ctx._qp_Sigma_full = None  # noqa: SLF001
            return
        # 2026-05-10 audit §5.13.5: unwrap v1/v2 schema. SimAdapter
        # has already enforced the as-of-date guard at __init__ time,
        # so we only unwrap here (no second guard call).
        from kernel.walk_forward import parse_correlation_artifact  # noqa: PLC0415
        corr, _ = parse_correlation_artifact(raw)
        tickers = _get_path(ctx, "_qp_tickers") or []
        sig = _get_path(ctx, "_qp_sigma")
        n = len(tickers)
        Sigma = np.zeros((n, n))
        for i, ti in enumerate(tickers):
            for j, tj in enumerate(tickers):
                if i == j:
                    Sigma[i, j] = sig[i] ** 2
                    continue
                rho = corr.get(ti, {}).get(tj) or corr.get(tj, {}).get(ti, 0.0)
                try:
                    rho_f = max(-0.99, min(0.99, float(rho)))
                except (TypeError, ValueError):
                    rho_f = 0.0
                Sigma[i, j] = rho_f * sig[i] * sig[j]
        ctx._qp_Sigma_full = Sigma + 1e-8 * np.eye(n)  # noqa: SLF001


# ── 2b. Ledoit-Wolf 2004 Σ shrinkage (post-step on full Σ) ──────────────────

class ShrinkSigmaLedoitWolfTask(Task):
    """Apply Ledoit-Wolf 2004 shrinkage to Σ_full toward scalar identity.

        Σ_shrunk = (1 - λ) · Σ_full + λ · F     with F = (trace(Σ)/n) · I

    Effect: pulls off-diagonal correlation toward zero AND equalises
    diagonal variances toward the average — reducing noise on small-n
    correlation estimates. λ=0 → no change; λ=1 → identity·avg_var
    (no correlation, equal variance).

    **2026-05-10 default bumped 0.0 → 0.2** (Track C3). λ=0.2 is the
    industry-standard mid-of-range from Ledoit & Wolf 2004 ("Honey, I
    Shrunk the Sample Covariance Matrix", J. Portfolio Management 30(4):
    110-119): they show on a 169-stock universe (matching ours) the OAS
    (oracle approximating shrinkage) optimum sits in [0.13, 0.27].
    Choosing λ=0.2 (mid of that range) is conservative, robust, and
    config-overridable. Set 0.0 to disable; 1.0 → diagonal.

    Eigenvalue floor: post-shrinkage we clip Σ's eigenvalues to ≥1e-8
    (per CLAUDE.md §5.13.12) — guarantees CLARABEL/OSQP/SCS see a strict
    PSD matrix and do not stall on numerical near-singularity (a real
    failure mode pre-fix when correlation_artifact NaN cells leak into
    Σ_full and the LW blend doesn't fully wash them out).

    Reads:  ctx._qp_Sigma_full,
             ctx.config['rotation']['joint_actions']['qp_ledoit_wolf_lambda']
    Writes: ctx._qp_Sigma_full (in place; None if upstream produced None
             — diagonal-Σ fallback in solver is unaffected)
    """
    name = "ShrinkSigmaLedoitWolfTask"

    # Default λ=0.2: ledoit-wolf 2004, see class docstring. Override via
    # config['rotation']['joint_actions']['qp_ledoit_wolf_lambda'].
    DEFAULT_LAMBDA = 0.2
    EIGEN_FLOOR    = 1e-8

    def run(self, ctx) -> bool | None:
        cfg = _qp_cfg(ctx)
        lam = float(cfg.get("qp_ledoit_wolf_lambda", self.DEFAULT_LAMBDA))
        if not math.isfinite(lam) or lam <= 0.0:
            return                                      # off
        lam = min(lam, 1.0)
        S = _get_path(ctx, "_qp_Sigma_full")
        if S is None:
            return                                      # diagonal-Σ path
        n = S.shape[0]
        if n == 0:
            return
        avg_var = float(np.trace(S)) / max(n, 1)
        F = avg_var * np.eye(n)
        S_blend = (1.0 - lam) * S + lam * F
        # §5.13.12 — clamp eigenvalues so the solver always sees a sane
        # PSD matrix. Symmetrize first to absorb any asymmetric float
        # noise before eigh (which assumes Hermitian input).
        S_sym = 0.5 * (S_blend + S_blend.T)
        eigvals, eigvecs = np.linalg.eigh(S_sym)
        if (eigvals < self.EIGEN_FLOOR).any():
            eigvals = np.maximum(eigvals, self.EIGEN_FLOOR)
            S_blend = eigvecs @ np.diag(eigvals) @ eigvecs.T
            # Re-symmetrize: V·diag·V^T floats can drift ~1e-16 off-symmetric.
            S_blend = 0.5 * (S_blend + S_blend.T)
        ctx._qp_Sigma_full = S_blend  # noqa: SLF001


# ── 3. Brown-Smith dynamic tax + Berkin-Jeffrey loss-harvest ────────────────

class ComputeBrownSmithTaxCostTask(Task):
    """Per-asset tax cost vector. Brown-Smith (2011) LT-bridge for
    winners; Berkin-Jeffrey (1990) loss-harvest credit (negative cost)
    for losers when ctx.ytd_realized_gain_dollar > 0.

    Reads:  ctx._qp_tickers, ctx._qp_w_current, ctx.holdings, ctx.prices,
             ctx.portfolio_value, ctx.today, ctx.ytd_realized_gain_dollar,
             ctx.config['rotation']['joint_actions']['qp_tax_*']
    Writes: ctx._qp_tax_cost (np.ndarray)
    """
    name = "ComputeBrownSmithTaxCostTask"

    def run(self, ctx) -> bool | None:
        tickers = _get_path(ctx, "_qp_tickers") or []
        n = len(tickers)
        cost = np.zeros(n)
        cfg = _qp_cfg(ctx)
        if not cfg.get("qp_tax_aware", True):
            ctx._qp_tax_cost = cost  # noqa: SLF001
            return
        st_rate = float(cfg.get("qp_tax_rate_st", 0.30))
        lt_rate = float(cfg.get("qp_tax_rate_lt", 0.15))
        lt_days = int(cfg.get("qp_lt_threshold_days", 365))
        bridge_w = int(cfg.get("qp_lt_bridge_window_days", 30))
        # G7: tax-lot disposal method. "fifo"/"hifo" → per-lot accounting;
        # "avg" → legacy single-cost-basis path (kill-switch).
        lot_method = str(cfg.get("qp_tax_lot_method", "fifo")).lower()
        offset = max(0.0, float(getattr(ctx, "ytd_realized_gain_dollar", 0.0) or 0.0))
        nav = float(_get_path(ctx, "portfolio_value", 0.0) or 0.0)
        w_current = _get_path(ctx, "_qp_w_current")
        prices = _get_path(ctx, "prices") or {}
        holdings = _get_path(ctx, "holdings") or {}
        today = ctx.today
        for i, t in enumerate(tickers):
            hs = holdings.get(t)
            if hs is None or w_current[i] <= 0:
                continue
            if lot_method == "avg":
                cost[i], offset = _per_asset_tax(
                    hs, prices.get(t, 0.0), w_current[i], nav, today,
                    st_rate, lt_rate, lt_days, bridge_w, offset,
                )
            else:
                cost[i], offset = _per_asset_tax_lots(
                    hs, prices.get(t, 0.0), w_current[i], nav, today,
                    st_rate, lt_rate, lt_days, bridge_w, offset, lot_method,
                )
        ctx._qp_tax_cost = cost  # noqa: SLF001


# ── 4. Wash-sale mask (uses atom + predicate) ───────────────────────────────

class ComputeWashSaleMaskTask(Task):
    """Wash-sale mask: tickers sold within wash_sale_days where §1091 BLOCKS
    the re-entry get Δw_i ≤ 0 in the QP.

    Cost-aware per IRC §1091 (mirrors `WashSaleFilterTask` in candidate path):
      - Sale outside the wash_sale_days window → not blocked
      - Sale was a GAIN (or unknown — fail-conservative) → §1091 N/A → not blocked
      - Sale was a LOSS → §1091 applies → blocked (forces Δw ≤ 0)

    2026-05-09 audit Phase 2.2 fix: pre-fix this task ignored
    `ctx.last_sell_pls` and applied a binary 30-day block. The candidate
    filter (`WashSaleFilterTask`) was correctly cost-aware, but tickers
    that passed the filter (e.g. just sold for a gain) hit the binary QP
    mask and were silently locked from increases. Result: post-gain re-
    entries were architecturally impossible despite §1091 not applying.

    Reads:  ctx._qp_tickers, ctx.last_sell_dates, ctx.last_sell_pls,
             ctx.config['wash_sale_days']
    Writes: ctx._qp_wash_mask (np.ndarray of bool)

    References:
      - IRC §1091 wash-sale; §1091(d) basis adjustment; §1223(3) holding period
      - kernel/selection.py::is_wash_sale_blocked_with_cost (single-source-of-truth)
    """
    name = "ComputeWashSaleMaskTask"

    def run(self, ctx) -> bool | None:
        wash_days = int((ctx.config or {}).get("wash_sale_days", 0))
        tickers = _get_path(ctx, "_qp_tickers") or []
        if wash_days <= 0 or not tickers:
            ctx._qp_wash_mask = np.zeros(len(tickers), dtype=bool)  # noqa: SLF001
            return
        last_sells_d = _get_path(ctx, "last_sell_dates") or {}
        last_sells_p = _get_path(ctx, "last_sell_pls") or {}
        today = ctx.today

        # Use the SAME helper as the upstream candidate filter so the two
        # wash-sale paths can never silently diverge. Cost-aware: gains
        # are not blocked, only losses (or unknown-and-conservative).
        from kernel.selection import is_wash_sale_blocked_with_cost  # noqa: PLC0415
        mask = np.zeros(len(tickers), dtype=bool)
        for i, t in enumerate(tickers):
            blocked, _, _ = is_wash_sale_blocked_with_cost(
                ticker=t,
                today=today,
                last_sell_dates=last_sells_d,
                last_sell_pls=last_sells_p,
                wash_sale_days=wash_days,
            )
            mask[i] = bool(blocked)
        ctx._qp_wash_mask = mask  # noqa: SLF001


# ── 5. Position caps + scalar constraints ──────────────────────────────────

class ComputeQPConstraintsTask(Task):
    """Per-asset weight caps (regime × confidence-scaled) + scalar limits.

    Reads:  ctx._qp_tickers, ctx.regime, ctx.confidence, ctx.regime_state,
             ctx.config (regime_params, regime, rotation.joint_actions)
    Writes: ctx._qp_w_upper (np.ndarray), ctx._qp_w_lower (float),
             ctx._qp_dw_max (np.ndarray), ctx._qp_cash_reserve (float),
             ctx._qp_drawdown (float), ctx._qp_drawdown_limit (float),
             ctx._qp_turnover_max (float | None)
    """
    name = "ComputeQPConstraintsTask"

    def run(self, ctx) -> bool | None:
        from kernel.regime import confidence_to_size_multiplier
        cfg = _qp_cfg(ctx)
        tickers = _get_path(ctx, "_qp_tickers") or []
        n = len(tickers)
        rp = (ctx.config.get("regime_params", {})
                          .get(getattr(ctx, "regime", None), {}))
        max_pct = float(rp.get("max_position_pct",
                                ctx.config.get("max_position_pct", 0.20)))
        scale = confidence_to_size_multiplier(getattr(ctx, "confidence", None))
        ctx._qp_w_upper = np.full(n, max_pct * scale)  # noqa: SLF001
        # 2026-05-13 Long-Short Phase 2A: allow w_lower < 0 when enabled.
        # Config: long_short.enabled (bool), max_short_pct (per-name short cap),
        #         max_gross_exposure (Σ|w| cap).
        # Default 0 → long-only (current behavior preserved).
        #
        # 2026-05-14 SAFETY (user mandate, no leverage authorized):
        # Hard-cap max_gross_exposure at 1.0. The QP can still use shorts
        # to net to (long + abs(short)) ≤ NAV, so the strategy is
        # dollar-neutral capped, NOT leveraged. Any config value > 1.0
        # is silently clamped to 1.0. Re-enabling leverage requires:
        # (1) explicit user authorization, (2) editing this hard-cap
        # constant, (3) updating tests/test_no_leverage_invariant.py.
        _LEVERAGE_HARDCAP = 1.0
        ls_cfg = ctx.config.get("long_short", {}) or {}
        if ls_cfg.get("enabled", False):
            max_short_pct = float(ls_cfg.get("max_short_pct", 0.05))
            # Bear regime: also block shorts to keep risk symmetric
            if getattr(ctx, "regime", None) == "BEAR":
                ctx._qp_w_lower = 0.0  # noqa: SLF001
                ctx._qp_gross_max = None  # noqa: SLF001 — no shorts, no gross cap
            else:
                ctx._qp_w_lower = -float(max_short_pct) * scale  # noqa: SLF001
                _cfg_gross = float(ls_cfg.get("max_gross_exposure", _LEVERAGE_HARDCAP))
                ctx._qp_gross_max = min(_cfg_gross, _LEVERAGE_HARDCAP)  # noqa: SLF001
        else:
            ctx._qp_w_lower = 0.0  # noqa: SLF001
            ctx._qp_gross_max = None  # noqa: SLF001 — long-only path; cp.sum(wp)≤1 binds
        ctx._qp_dw_max = np.full(n, float(cfg.get("qp_dw_max", 0.50)))  # noqa: SLF001
        ctx._qp_cash_reserve = float(rp.get(  # noqa: SLF001
            "cash_reserve_pct",
            ctx.config.get("cash_reserve_pct", 0.0),
        ))
        rs = getattr(ctx, "regime_state", None)
        ctx._qp_drawdown = (  # noqa: SLF001
            0.0 if rs is None
            else float(rs.get("drawdown", 0.0) or 0.0) if isinstance(rs, dict)
            else float(getattr(rs, "drawdown", 0.0) or 0.0)
        )
        ctx._qp_drawdown_limit = float(cfg.get(  # noqa: SLF001
            "qp_drawdown_limit",
            ctx.config.get("regime", {}).get("drawdown_halt_pct", 0.20),
        ))
        tm = cfg.get("qp_turnover_max", 0.30)
        try:
            ctx._qp_turnover_max = float(tm) if tm else None  # noqa: SLF001
        except (TypeError, ValueError):
            ctx._qp_turnover_max = None  # noqa: SLF001


_BuildADVVectorTask = None  # lazy class, defined below


def _resolve_regime_override(base_cfg: dict, ctx) -> dict:
    """P1 (2026-05-12): if `base_cfg` has `regime_overrides` AND ctx.spy_regime
    is set, return base_cfg merged with the regime-specific override.

    Resolution order (highest precedence first):
      1. regime_overrides[ctx.spy_regime]  (if regime label exists in overrides)
      2. base_cfg                          (fallback)

    Keys in the override block fully override base_cfg keys (shallow merge);
    this means an override CAN flip `enabled: true → false` to disable a
    feature in toxic regimes.

    Returns base_cfg unmodified if:
      - no `regime_overrides` block
      - ctx.spy_regime is None (SpyRegimeLabelTask disabled)
      - current regime not in overrides

    Fail-open: any error → return base_cfg (no override).
    """
    if not isinstance(base_cfg, dict):
        return {}
    overrides = base_cfg.get("regime_overrides")
    if not isinstance(overrides, dict) or not overrides:
        return base_cfg
    regime = getattr(ctx, "spy_regime", None)
    if regime is None or regime not in overrides:
        return base_cfg
    override = overrides.get(regime)
    if not isinstance(override, dict):
        return base_cfg
    # Shallow merge — override wins on key collision
    merged = dict(base_cfg)
    merged.update(override)
    return merged


# ── 4a. Force μ_QP source (Option A: validate NGBoost theory) ───────────────

class ForceMuSourceTask(Task):
    """Override `_qp_mu` from a specific candidate attribute, independent
    of the NGBoost mu/panel_score fallback chain in BuildMuVectorTask.

    Enables Option A validation (CLAUDE.md §2b NGBoost audit, 2026-05-12):
    when `ngboost.enabled=true` (so σ flows through to Kelly + risk), we
    can still force μ_QP = panel_score so the QP's risk/return tradeoff
    stays in its calibrated z-score scale.

    This isolates the contribution of NGBoost σ from the destructive
    μ-scale mismatch that broke E55 (455 trades vs 303, APY +2.99% vs
    +6.77%).

    Config (off by default):
        ranking.qp_mu_source = "panel_score"  (or "rank_score" / "mu")
                            default: "mu"  → no-op, preserves baseline

    Stage A → if NGB-on + force panel_score beats NGB-off baseline,
    NGBoost σ is contributing real value. Then we know option C
    (Grinold-Kahn normalization of NGB μ) is the right architecture.
    """
    name = "ForceMuSourceTask"

    def run(self, ctx) -> bool | None:
        source = str((ctx.config or {}).get("ranking", {}).get("qp_mu_source", "mu")).lower()
        if source == "mu":
            return None  # no-op: keep mu from BuildMuVectorTask
        if source not in ("panel_score", "rank_score"):
            log.warning("ForceMuSource: unknown source '%s' — no-op", source)
            return None
        tickers   = _get_path(ctx, "_qp_tickers") or []
        src_map   = _get_path(ctx, "_qp_mu_source_map") or {}
        new_mu    = np.zeros(len(tickers))
        n_set     = 0
        for i, t in enumerate(tickers):
            obj = src_map.get(t)
            if obj is None:
                continue
            val = getattr(obj, source, None)
            if val is None:
                # fallback: try the other rank field
                alt = "rank_score" if source == "panel_score" else "panel_score"
                val = getattr(obj, alt, None)
            try:
                v = float(val) if val is not None else 0.0
            except (TypeError, ValueError):
                v = 0.0
            if math.isfinite(v):
                new_mu[i] = v
                n_set += 1
        ctx._qp_mu = new_mu  # noqa: SLF001
        log.info(
            "ForceMuSource: μ_QP ← %s for %d/%d tickers (μ̄=%.4f, σ_μ=%.4f)",
            source, n_set, len(tickers),
            float(new_mu.mean()) if len(new_mu) else 0.0,
            float(new_mu.std(ddof=1)) if len(new_mu) > 1 else 0.0,
        )


# ── 4b. Grinold-Kahn α→μ transform (scale-normalizes any score) ─────────────

class ApplyGrinoldKahnTransformTask(Task):
    """Convert raw `_qp_mu` into σ-scale natural units via Grinold-Kahn.

    Reference: Grinold 1989 "The Fundamental Law of Active Management"
    (*J. Portfolio Management*); Grinold-Kahn 1999 *Active Portfolio
    Management* ch.5. Formula:

        μ_i  =  IC  ×  σ_i  ×  z(score_i)

    Where z(score) is the cross-sectional z-score of the raw score, σ_i
    the asset volatility (from `_qp_sigma`), IC the information
    coefficient (use calibrator's pool_ic).

    Fixes §5.13.10 NGBoost μ-scale-mismatch bug class: swapping between
    LTR `panel_score` (~±2 z-units) and NGBoost μ (~1e-3 raw return) used
    to silently change the QP's risk/return tradeoff because λ_risk and
    transaction-cost weights are anchored to one input scale.

    With this transform, μ is always in σ-units (the natural scale of the
    quadratic risk term), so swapping signal sources is safe.

    Config (off by default — opt-in to preserve baseline):
        ranking.alpha_to_mu.enabled = true
        ranking.alpha_to_mu.ic      = 0.094   (default: calibrator pool_ic)
        ranking.alpha_to_mu.regime_overrides = {                # P1 2026-05-12
            "HIGH_CALM":   {"enabled": true, "ic": 0.094},
            "HIGH_SPIKED": {"enabled": false},                  # disable in toxic regime
            "LOW_SPIKED":  {"enabled": true, "ic": 0.15},       # different IC per regime
        }

    When `regime_overrides` is set AND `ctx.spy_regime` is non-None
    (set by SpyRegimeLabelTask, off by default), this task selects the
    override for the current regime. Falls back to global if regime
    not in overrides. Allows regime-conditional deployment per
    doc/research/2026-05-12-findings-and-next.md.
    """
    name = "ApplyGrinoldKahnTransformTask"

    def run(self, ctx) -> bool | None:
        base_cfg = (ctx.config or {}).get("ranking", {}).get("alpha_to_mu", {})
        cfg = _resolve_regime_override(base_cfg, ctx)
        if not cfg.get("enabled", False):
            return None
        ic = float(cfg.get("ic", 0.094))
        if not math.isfinite(ic):
            return None
        mu_arr = _get_path(ctx, "_qp_mu")
        sigma_arr = _get_path(ctx, "_qp_sigma")
        if mu_arr is None or sigma_arr is None:
            return None
        mu_arr = np.asarray(mu_arr, dtype=float)
        sigma_arr = np.asarray(sigma_arr, dtype=float)
        if len(mu_arr) < 2 or len(mu_arr) != len(sigma_arr):
            return None
        finite = np.isfinite(mu_arr)
        if int(finite.sum()) < 2:
            return None
        m  = float(mu_arr[finite].mean())
        sd = float(mu_arr[finite].std(ddof=1))
        if not math.isfinite(sd) or sd <= 0:
            return None
        z = np.zeros_like(mu_arr)
        z[finite] = (mu_arr[finite] - m) / sd
        ctx._qp_mu = ic * sigma_arr * z  # noqa: SLF001
        log.info(
            "ApplyGrinoldKahnTransform: IC=%.3f raw_μ̄=%.4f raw_σ_μ=%.4f → μ̄_QP=%.4f",
            ic, m, sd, float(np.abs(ctx._qp_mu).mean()),
        )


# ── 5a. Exposure scaling (vol-target + DD-Kelly) ────────────────────────────

class ApplyExposureScalingTask(Task):
    """Scale per-asset `_qp_w_upper` by basket-level exposure modifiers.

    Composes Moskowitz-Ooi-Pedersen 2012 volatility-targeting and
    Grossman-Zhou 1993 drawdown-conditioned Kelly scaling at the QP
    upper-bound level, INDEPENDENT of the Kelly sizing path (which is
    dead when NGB is off — see doc/AUDIT_2026-05-12_dead_paths.md).

    Invariant pinned:
        _qp_w_upper ≡ max_pos × confidence × vol_target_scale × dd_scale

    Config (read from BOTH legacy and new locations for backward compat):
        ranking.kelly_sizing.vol_target.{enabled,target_vol,window_days,...}
        ranking.kelly_sizing.drawdown_scaling.{enabled,dd_max,exponent}
        exposure_scaling.vol_target.*        (new top-level path)
        exposure_scaling.drawdown_scaling.*  (new top-level path)

    Both helpers fail-open (return 1.0 on malformed input).
    """
    name = "ApplyExposureScalingTask"

    def run(self, ctx) -> bool | None:
        w_upper = _get_path(ctx, "_qp_w_upper")
        if w_upper is None or len(w_upper) == 0:
            ctx._vol_target_scale = 1.0  # noqa: SLF001
            ctx._dd_kelly_scale = 1.0    # noqa: SLF001
            return None
        cfg = ctx.config or {}
        legacy = cfg.get("ranking", {}).get("kelly_sizing", {})
        topl   = cfg.get("exposure_scaling", {})
        vt_cfg = topl.get("vol_target")        or legacy.get("vol_target")        or {}
        dd_cfg = topl.get("drawdown_scaling")  or legacy.get("drawdown_scaling")  or {}
        # P1 (2026-05-12): regime-conditional override per ctx.spy_regime
        vt_cfg = _resolve_regime_override(vt_cfg, ctx)
        dd_cfg = _resolve_regime_override(dd_cfg, ctx)
        vt_scale = _compute_vt_scale(ctx, vt_cfg) if vt_cfg.get("enabled", False) else 1.0
        dd_scale = _compute_dd_scale(ctx, dd_cfg) if dd_cfg.get("enabled", False) else 1.0
        ctx._vol_target_scale = float(vt_scale)  # noqa: SLF001
        ctx._dd_kelly_scale   = float(dd_scale)  # noqa: SLF001
        combined = vt_scale * dd_scale
        if combined != 1.0:
            ctx._qp_w_upper = np.asarray(w_upper) * float(combined)  # noqa: SLF001
            log.info(
                "ApplyExposureScalingTask: w_upper scaled by vt=%.3f × dd=%.3f = %.3f",
                vt_scale, dd_scale, combined,
            )


def _compute_vt_scale(ctx, vt_cfg: dict) -> float:
    from kernel.vol_target import compute_vol_target_scale  # noqa: PLC0415
    return compute_vol_target_scale(
        getattr(ctx, "spy_returns", None) or [],
        target_vol  = float(vt_cfg.get("target_vol",  0.15)),
        window_days = int  (vt_cfg.get("window_days", 60)),
        floor       = float(vt_cfg.get("floor",       0.30)),
        ceiling     = float(vt_cfg.get("ceiling",     1.50)),
    )


def _compute_dd_scale(ctx, dd_cfg: dict) -> float:
    from kernel.kelly import compute_kelly_dd_scale  # noqa: PLC0415
    from kernel.pipeline.task_drawdown_rebalance import compute_portfolio_drawdown  # noqa: PLC0415
    hwm = float(getattr(ctx, "hwm", 0.0) or 0.0)
    pv  = float(getattr(ctx, "portfolio_value", 0.0) or 0.0)
    dd  = compute_portfolio_drawdown(hwm, pv)
    return compute_kelly_dd_scale(
        dd,
        dd_max   = float(dd_cfg.get("dd_max",   0.30)),
        exponent = float(dd_cfg.get("exponent", 1.0)),
    )


# ── 5b. Conviction-scaled per-name cap ──────────────────────────────────────

class ApplyConvictionCapTask(Task):
    """Shrink per-ticker `_qp_w_upper` by conviction multiplier.

    Parity with greedy paths (`task_selection`, `task_rotation`,
    `task_joint_actions`) which multiply position size by
    `conviction_multiplier(panel_score, sizing_cfg)`. Without this, the
    QP path treats every name with the same regime+confidence cap
    regardless of model conviction — high- and low-rank candidates can
    both saturate at `max_position_pct`.

    Wiring: runs AFTER `ComputeQPConstraintsTask` (which writes
    `_qp_w_upper` as a uniform vector) and BEFORE the sector/correlation
    constraint Tasks (which anchor their caps on `_qp_w_upper.max()`).

    Reads:  ctx._qp_tickers, ctx._qp_mu_source_map, ctx._qp_w_upper,
             ctx.config["rotation"]["joint_actions"]["qp_conviction_cap_enabled"],
             ctx.config["ranking"]["panel_scoring"]["sizing"]
    Writes: ctx._qp_w_upper (in-place per-ticker scaling)
             ctx._qp_conviction_caps (list[float] for diagnostics)

    Default: disabled. Opt-in via `qp_conviction_cap_enabled=true`. No
    promotion until sim shows positive APY delta (CLAUDE.md §2a).
    """
    name = "ApplyConvictionCapTask"

    def run(self, ctx) -> bool | None:
        cfg = _qp_cfg(ctx)
        if not bool(cfg.get("qp_conviction_cap_enabled", False)):
            return None
        sizing_cfg = ((ctx.config or {}).get("ranking", {})
                       .get("panel_scoring", {})
                       .get("sizing", {}))
        if not sizing_cfg or not sizing_cfg.get("enabled", False):
            return None

        # Local import to keep qp module decoupled from kernel.sizing.
        from kernel.sizing import conviction_multiplier

        tickers = _get_path(ctx, "_qp_tickers") or []
        w_upper = _get_path(ctx, "_qp_w_upper")
        src     = _get_path(ctx, "_qp_mu_source_map") or {}
        if w_upper is None or len(tickers) == 0 or len(w_upper) != len(tickers):
            return None

        caps: list[float] = []
        for i, t in enumerate(tickers):
            obj = src.get(t)
            ps = getattr(obj, "panel_score", None) if obj is not None else None
            mult = conviction_multiplier(ps, sizing_cfg)
            # Defensive: conviction_multiplier returns 1.0 on bad input
            # (None / NaN / inf / malformed cfg). Clip to [0, 1] in case
            # of future-config changes — w_upper must remain ≤ original.
            try:
                m = float(mult)
            except (TypeError, ValueError):
                m = 1.0
            if not math.isfinite(m):
                m = 1.0
            m = max(0.0, min(1.0, m))
            w_upper[i] = float(w_upper[i]) * m
            caps.append(m)

        ctx._qp_conviction_caps = caps  # noqa: SLF001
        return None


# ── 5a. Sector cap → per-sector indicator matrix + cap vector ───────────────

class BuildSectorConstraintMatrixTask(Task):
    """Construct hard linear sector-cap constraint inputs for the QP.

    Per CLAUDE.md §5.13.5 (single source of truth), sector_map and
    `max_positions_per_sector` come from THE SAME config keys the buy-side
    `passes_sector_guard` uses (`config['sector_map']`,
    `config['max_positions_per_sector']`). The QP enforcing the same caps
    closes the audit gap: once a holding is in the book, the buy-side
    filter never sees it again, but a stress reallocation could still pile
    weight on top of it. The solver constraint catches that.

    Per-sector weight cap = max_per_sector × max_position_pct × confidence.
    Defensive tickers (`config['defensive_tickers']`) are included in the
    indicator (they get the same cap) — divergence from buy-side which
    *bypasses* the count-of-positions cap is intentional: the QP
    constraint is on *weight*, not count, and an unbounded defensive
    sleeve would defeat the diversification goal.

    Reads:  ctx._qp_tickers, ctx.config['sector_map'],
             ctx.config['max_positions_per_sector'],
             ctx._qp_w_upper (anchors per-name cap × sector_count),
             ctx.config['rotation']['joint_actions']['qp_sector_cap_enabled']
    Writes: ctx._qp_sector_indicator (m × n np.ndarray, 0/1 ints) — None
             when constraint disabled / no sectors mapped,
             ctx._qp_sector_cap_vec (m-length np.ndarray of weight caps),
             ctx._qp_sector_names (list[str]) — for diagnostics.
    """
    name = "BuildSectorConstraintMatrixTask"

    def run(self, ctx) -> bool | None:
        cfg = _qp_cfg(ctx)
        if not bool(cfg.get("qp_sector_cap_enabled", True)):
            ctx._qp_sector_indicator = None  # noqa: SLF001
            ctx._qp_sector_cap_vec   = None  # noqa: SLF001
            ctx._qp_sector_names     = []    # noqa: SLF001
            return
        tickers = _get_path(ctx, "_qp_tickers") or []
        n = len(tickers)
        sector_map = (ctx.config or {}).get("sector_map", {}) or {}
        max_per_sector = int((ctx.config or {}).get("max_positions_per_sector", 0))
        if n == 0 or not sector_map or max_per_sector <= 0:
            ctx._qp_sector_indicator = None  # noqa: SLF001
            ctx._qp_sector_cap_vec   = None  # noqa: SLF001
            ctx._qp_sector_names     = []    # noqa: SLF001
            return
        # Per-name cap (post-confidence scaling) is in ctx._qp_w_upper.
        # Use the max element as the per-position anchor — all entries are
        # the same scalar today, but keeping max-based makes this robust
        # if/when ComputeQPConstraintsTask becomes per-asset.
        w_upper = _get_path(ctx, "_qp_w_upper")
        per_name_cap = (
            float(np.max(w_upper)) if (w_upper is not None and len(w_upper))
            else float((ctx.config or {}).get("max_position_pct", 0.20))
        )
        sector_to_idx = self._build_sector_index(tickers, sector_map)
        if not sector_to_idx:
            ctx._qp_sector_indicator = None  # noqa: SLF001
            ctx._qp_sector_cap_vec   = None  # noqa: SLF001
            ctx._qp_sector_names     = []    # noqa: SLF001
            return
        sector_names = sorted(sector_to_idx.keys())
        m = len(sector_names)
        S = np.zeros((m, n), dtype=float)
        for row, name in enumerate(sector_names):
            for j in sector_to_idx[name]:
                S[row, j] = 1.0
        cap_vec = np.full(m, max_per_sector * per_name_cap, dtype=float)
        ctx._qp_sector_indicator = S            # noqa: SLF001
        ctx._qp_sector_cap_vec   = cap_vec      # noqa: SLF001
        ctx._qp_sector_names     = sector_names # noqa: SLF001

    @staticmethod
    def _build_sector_index(tickers, sector_map) -> dict[str, list[int]]:
        """Return {sector_name: [ticker_indices]} for sectors with ≥1 member."""
        out: dict[str, list[int]] = {}
        for j, t in enumerate(tickers):
            sec = sector_map.get(t)
            if not sec or not isinstance(sec, str):
                continue
            out.setdefault(sec, []).append(j)
        return out


# ── 5a-bis. High-correlation pair group cap ────────────────────────────────

class BuildCorrelationGroupConstraintTask(Task):
    """Build (i, j, group_cap) triples for high-correlation pairs.

    For every pair (i, j) where |corr[i, j]| ≥ correlation_guard_threshold,
    add a linear constraint `wp[i] + wp[j] ≤ 2 × per_name_cap` (group
    bound). This is the convex linear approximation of the non-convex
    `wp[i] · wp[j] ≤ pair_cap`. Tradeoff documented in qp_solver.py.

    §5.13.5 single-source-of-truth: the `correlation_guard_threshold` is
    read from `config['regime']['correlation_guard_threshold']` — same key
    `passes_correlation_guard` uses in selection.py. Behaviour-equivalent
    when the candidate filter and QP both fire (no double-blocking; the
    QP just ensures any *internal* re-shuffling can't recreate the pair
    concentration).

    Reads:  ctx._qp_tickers, ctx.corr_matrix (pre-loaded by SimAdapter),
             ctx.config['regime']['correlation_guard_threshold'],
             ctx._qp_w_upper, ctx.config['rotation']['joint_actions']
                 ['qp_correlation_cap_enabled']
    Writes: ctx._qp_corr_group_pairs (list[tuple[int, int, float]] | None)
    """
    name = "BuildCorrelationGroupConstraintTask"

    def run(self, ctx) -> bool | None:
        cfg = _qp_cfg(ctx)
        if not bool(cfg.get("qp_correlation_cap_enabled", True)):
            ctx._qp_corr_group_pairs = None  # noqa: SLF001
            return
        tickers = _get_path(ctx, "_qp_tickers") or []
        n = len(tickers)
        if n < 2:
            ctx._qp_corr_group_pairs = None  # noqa: SLF001
            return
        corr_matrix = getattr(ctx, "corr_matrix", None) or {}
        if not corr_matrix:
            ctx._qp_corr_group_pairs = None  # noqa: SLF001
            return
        thr = float(((ctx.config or {}).get("regime", {}) or {}).get(
            "correlation_guard_threshold", 0.70,
        ))
        if not math.isfinite(thr) or thr <= 0.0 or thr >= 1.0:
            ctx._qp_corr_group_pairs = None  # noqa: SLF001
            return
        w_upper = _get_path(ctx, "_qp_w_upper")
        per_name_cap = (
            float(np.max(w_upper)) if (w_upper is not None and len(w_upper))
            else float((ctx.config or {}).get("max_position_pct", 0.20))
        )
        # Group-cap = 2 × per-name (linear relaxation). Two co-linear holdings
        # can each individually hit the cap; the constraint binds when both
        # try to be near-cap simultaneously and the realized portfolio looks
        # like a single concentrated bet.
        group_cap = 2.0 * per_name_cap
        pairs = self._collect_pairs(tickers, corr_matrix, thr, group_cap)
        ctx._qp_corr_group_pairs = pairs if pairs else None  # noqa: SLF001

    @staticmethod
    def _collect_pairs(tickers, corr_matrix, thr, group_cap):
        """Walk the upper-triangle of the corr matrix; return (i, j, cap)."""
        pairs: list[tuple[int, int, float]] = []
        for i in range(len(tickers)):
            ti = tickers[i]
            row = corr_matrix.get(ti)
            for j in range(i + 1, len(tickers)):
                tj = tickers[j]
                rho = None
                if isinstance(row, dict):
                    rho = row.get(tj)
                if rho is None:
                    other = corr_matrix.get(tj)
                    if isinstance(other, dict):
                        rho = other.get(ti)
                if rho is None:
                    continue
                try:
                    rho_f = float(rho)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(rho_f):
                    # Fail-conservative: NaN correlation → treat as high.
                    rho_f = 1.0
                if abs(rho_f) >= thr:
                    pairs.append((i, j, group_cap))
        return pairs


# ── 5b. Per-asset 20-day ADV (Almgren-Chriss participation) ─────────────────

class BuildADVVectorTask(Task):
    """Per-asset average daily dollar volume (ADV) over `qp_adv_window` days.

    ADV_i = mean(close_t × volume_t) over the last `window` rows of the
    asset's OHLCV frame. Used by Stage G3 sqrt-impact: missing or
    too-short data → NaN entry → solver disables impact for that asset.

    Reads:  ctx._qp_tickers, ctx.ohlcv,
             ctx.config['rotation']['joint_actions']['qp_adv_window']
    Writes: ctx._qp_v_daily_dollar (np.ndarray, $; NaN for unavailable)
    """
    name = "BuildADVVectorTask"

    def run(self, ctx) -> bool | None:
        cfg = _qp_cfg(ctx)
        window = max(1, int(cfg.get("qp_adv_window", 20)))
        tickers = _get_path(ctx, "_qp_tickers") or []
        ohlcv = _get_path(ctx, "ohlcv") or {}
        v = np.full(len(tickers), np.nan)
        for i, t in enumerate(tickers):
            df = ohlcv.get(t)
            if df is None or len(df) == 0:
                continue
            try:
                tail = df.tail(window)
                cv = (tail["close"] * tail["volume"]).mean()
                v[i] = float(cv) if math.isfinite(float(cv)) else math.nan
            except (KeyError, AttributeError, ValueError, TypeError):
                continue
        ctx._qp_v_daily_dollar = v  # noqa: SLF001


# ── 6. Solve the QP ─────────────────────────────────────────────────────────

class SolveMarkowitzQPTask(Task):
    """Call solve_portfolio_qp with the prepared inputs.

    Reads:  every ctx._qp_* field built by upstream Tasks
    Writes: ctx._qp_solution (QPSolution dataclass)
    """
    name = "SolveMarkowitzQPTask"

    def run(self, ctx) -> bool | None:
        cfg = _qp_cfg(ctx)
        # 2026-05-06 — choose backend at runtime. Both backends accept the
        # same kwargs; cvxportfolio variant uses Boyd's reference policy
        # classes directly. Default is cvxpy (faster, supports tax-aware
        # sells + soft cash-drag). cvxportfolio backend is opt-in for
        # users who want Boyd's class hierarchy verbatim.
        backend = str(cfg.get("qp_solver_backend", "cvxpy")).lower()
        if backend == "cvxportfolio":
            from kernel.portfolio_qp.cvxportfolio_backend import (  # noqa: PLC0415
                solve_portfolio_qp_cvxportfolio as _solve,
            )
        else:
            from kernel.portfolio_qp.qp_solver import (  # noqa: PLC0415
                solve_portfolio_qp as _solve,
            )

        kwargs = dict(
            w_current=_get_path(ctx, "_qp_w_current"),
            mu=_get_path(ctx, "_qp_mu"),
            sigma=_get_path(ctx, "_qp_sigma"),
            Sigma=_get_path(ctx, "_qp_Sigma_full"),
            risk_aversion=float(cfg.get("qp_risk_aversion", 3.0)),
            cost_kappa=float(cfg.get("qp_cost_kappa",
                                       cfg.get("fee_pct", 0.0005))),
            cash_reserve=_get_path(ctx, "_qp_cash_reserve"),
            w_upper=_get_path(ctx, "_qp_w_upper"),
            w_lower=_get_path(ctx, "_qp_w_lower"),
            dw_max=_get_path(ctx, "_qp_dw_max"),
            wash_sale_mask=_get_path(ctx, "_qp_wash_mask"),
            signal_decay=float(cfg.get("qp_signal_decay", 0.0)),
            drawdown=_get_path(ctx, "_qp_drawdown"),
            drawdown_limit=_get_path(ctx, "_qp_drawdown_limit"),
            robust_mu_kappa=float(cfg.get("qp_robust_mu_kappa", 0.0)),
            tax_cost_per_sell=_get_path(ctx, "_qp_tax_cost"),
            turnover_max=_get_path(ctx, "_qp_turnover_max"),
            cvar_lambda=float(cfg.get("qp_cvar_lambda", 0.0)),
            cvar_alpha=float(cfg.get("qp_cvar_alpha", 0.05)),
            impact_coef=float(cfg.get("qp_impact_coef", 0.0)),
            v_daily_dollar=_get_path(ctx, "_qp_v_daily_dollar"),
            nav_dollar=float(_get_path(ctx, "portfolio_value", 0.0) or 0.0),
            fixed_cost_per_trade=float(cfg.get("qp_fixed_cost_per_trade", 0.0)),
            fixed_cost_beta=float(cfg.get("qp_fixed_cost_beta", 200.0)),
            budget_mode=str(cfg.get("qp_budget_mode", "inequality")),
            min_invested_pct=float(cfg.get("qp_min_invested_pct", 0.0)),
            # NEW (2026-05-06): soft cash-drag penalty replaces the
            # pre-2026-05-06 hard `Σwp ≥ min_invested_pct` floor that was
            # mathematically infeasible from cash + tight turnover.
            cash_drag_lambda=float(cfg.get("qp_cash_drag_lambda", 0.05)),
            # NEW (2026-05-10): C2 — hard sector + correlation pair caps.
            # cvxportfolio backend doesn't see these (no parity yet); the
            # cvxpy core solver takes them and emits diagnostic counters.
            sector_indicator=_get_path(ctx, "_qp_sector_indicator"),
            sector_cap_vec=_get_path(ctx, "_qp_sector_cap_vec"),
            corr_group_pairs=_get_path(ctx, "_qp_corr_group_pairs"),
            # 2026-05-13 Long-Short Phase 2A: gross-exposure cap (Σ|wp|).
            # Default None → no constraint (long-only path; sum(wp) ≤ 1 binds).
            # When long_short.enabled, set to max_gross_exposure (≤ 1.5 by
            # Reg-T spirit). Read from ctx so ComputeQPConstraintsTask can
            # override.
            gross_max=_get_path(ctx, "_qp_gross_max"),
        )
        # cvxportfolio backend additionally takes a `tickers` kwarg for
        # pandas-Series labelling; cvxpy backend ignores it.
        if backend == "cvxportfolio":
            kwargs["tickers"] = _get_path(ctx, "_qp_tickers")
            # cvxportfolio backend doesn't accept the new linear constraints
            # yet — strip them so it doesn't TypeError. Fall back to soft
            # diversification via Σ shrinkage only.
            kwargs.pop("sector_indicator", None)
            kwargs.pop("sector_cap_vec", None)
            kwargs.pop("corr_group_pairs", None)
            kwargs.pop("gross_max", None)  # Phase 2A: long-short not supported in cvxportfolio backend
        sol = _solve(**kwargs)
        sol = _retry_with_relaxed_c2_caps(sol, kwargs, _solve)
        ctx._qp_solution = sol  # noqa: SLF001
        ctx._qp_n_buys = 0  # noqa: SLF001
        ctx._qp_n_sells = 0  # noqa: SLF001


# ── Soft-fallback for C2 hard constraints (sector + corr pair caps) ───────

def _retry_with_relaxed_c2_caps(sol, kwargs, solve_fn):
    """If the QP went infeasible with C2 caps active, relax + retry.

    Retry sequence (per task spec — never silently drop, always log):
      1. Multiply sector_cap_vec and corr_pair caps by 1.5 → re-solve.
      2. If still infeasible → drop both C2 caps entirely → physics-only solve.

    Returns the final QPSolution. Status carries `infeasible:*` only when
    even the physics-only fallback failed (unusual — points to deeper Σ
    or μ corruption).
    """
    if not sol.status.startswith("infeasible"):
        return sol
    has_c2 = (kwargs.get("sector_indicator") is not None
              or kwargs.get("corr_group_pairs"))
    if not has_c2:
        return sol
    log.warning("QP infeasible with C2 caps — retrying with caps relaxed ×1.5")
    relaxed = dict(kwargs)
    cap_v = relaxed.get("sector_cap_vec")
    if cap_v is not None:
        relaxed["sector_cap_vec"] = np.asarray(cap_v) * 1.5
    pairs = relaxed.get("corr_group_pairs")
    if pairs:
        relaxed["corr_group_pairs"] = [
            (i, j, float(c) * 1.5) for (i, j, c) in pairs
        ]
    sol = solve_fn(**relaxed)
    if not sol.status.startswith("infeasible"):
        return sol
    log.warning(
        "QP still infeasible after relax — dropping C2 caps for this bar "
        "(sector + corr-pair constraints removed)",
    )
    last_resort = dict(kwargs)
    last_resort["sector_indicator"] = None
    last_resort["sector_cap_vec"]   = None
    last_resort["corr_group_pairs"] = None
    return solve_fn(**last_resort)


# ── 7. Translate Δw → orders / exits ───────────────────────────────────────

# ── Helper functions for EmitOrdersFromQPSolutionTask (split per §1c) ──────

def _passes_no_trade_band(
    dw: float, sig_i: float, min_dw: float, no_trade_factor: float,
    band_cap: float = 0.05,
) -> tuple[bool, bool]:
    """Davis-Norman 1990 / Constantinides 1979 no-trade band, capped.

    Original: skip trades inside max(min_dw, no_trade_factor × σ_i).

    2026-05-09 BUG #7 fix: with NGB now emitting σ̂ ≈ 0.10-0.30, the
    raw `no_trade_factor × σ_i` produces bands of 10-30% of equity for
    high-σ holdings, making them structurally uncoverable even when μ̂
    strongly disagrees with the position. Discovered when BA at
    edge_sharpe = -0.51 (12% expected underperformance) failed to sell
    because σ̂_BA = 0.24 → effective band = 24% > BA's 6.7% weight.

    Cap the σ-derived band at `band_cap` (default 5% of equity) so
    high-σ holdings remain reachable. The cap is per-ticker — assets
    with σ < band_cap are unaffected.

    Returns (pass, was_in_band).
    """
    sigma_band = min(band_cap, no_trade_factor * sig_i)
    threshold = max(min_dw, sigma_band)
    if abs(dw) < threshold:
        return False, abs(dw) >= min_dw
    return True, False


def _gate_buy_or_block(
    t: str, dw: float, today, earnings_cal, earn_buf: int,
    buys_gated: bool,
) -> str | None:
    """If dw>0 (buy/top-up): return blocked_reason if any gate fires.
    Returns None if buy is allowed."""
    if dw <= 0:
        return None
    if buys_gated:
        return "buys_gated"
    from kernel.selection import is_earnings_blocked  # noqa: PLC0415
    if today is not None and is_earnings_blocked(t, today, earnings_cal, earn_buf):
        return "earnings"
    return None


def _shares_from_dw(dw: float, nav: float, px: float) -> int:
    """Convert Δw fraction into integer share count, with finite checks."""
    import math as _m  # noqa: PLC0415
    if not (_m.isfinite(dw) and _m.isfinite(px) and _m.isfinite(nav)):
        return 0
    if px <= 0 or nav <= 0:
        return 0
    return int(abs(dw) * nav / px)


class EmitOrdersFromQPSolutionTask(Task):
    """Translate Δw → ctx.orders (buys/top-ups) + ctx.exits (closes/trims).

    Reads:  ctx._qp_solution, ctx._qp_tickers, ctx.prices, ctx.holdings,
             ctx.portfolio_value, ctx.candidates,
             ctx.config['rotation']['joint_actions']['qp_min_dw_pct']
    Writes: ctx.orders (append), ctx.exits (append),
             ctx._qp_n_buys, ctx._qp_n_sells (counters for atom-side LogSummary)

    Logic split per CLAUDE.md §1c (2026-05-06):
      1. Setup gate flags + helper closures
      2. Per-ticker loop calls _passes_no_trade_band, _gate_buy_or_block,
         _shares_from_dw, then _emit_qp_buy / _emit_qp_sell
      3. Log summary of blocked/skipped counters

    Bug fixes pinned by tests (do NOT regress):
      - Bug 3 (wl183 2026-05-05): buy_blocked / skip_buys suppress top-ups
      - Bug 4 (wl183 2026-05-05): earnings blackout suppresses top-ups
      - Bug 9 (2026-05-05): non-finite Δw skipped instead of crashing
      - Davis-Norman no-trade band (2026-05-05 cash-drag fix)
    """
    name = "EmitOrdersFromQPSolutionTask"

    def run(self, ctx) -> bool | None:
        sol = _get_path(ctx, "_qp_solution")
        if sol is None or sol.status != "optimal":
            log.warning("EmitOrdersFromQPSolutionTask: status=%s — skip",
                         sol.status if sol else "none")
            return False
        tickers = _get_path(ctx, "_qp_tickers") or []
        prices = _get_path(ctx, "prices") or {}
        nav = float(_get_path(ctx, "portfolio_value", 0.0) or 0.0)
        cfg = _qp_cfg(ctx)
        min_dw = float(cfg.get("qp_min_dw_pct", 0.005))
        no_trade_factor = float(cfg.get("qp_no_trade_band_factor", 0.0))
        band_cap = float(cfg.get("qp_no_trade_band_cap", 0.05))
        sigma_vec = _get_path(ctx, "_qp_sigma")
        cands = {c.ticker: c for c in (ctx.candidates or [])}
        buy_blocked = bool(getattr(ctx, "buy_blocked", False))
        skip_buys   = bool(getattr(ctx, "skip_buys",   False))
        buys_gated  = buy_blocked or skip_buys
        earnings_cal = getattr(ctx, "earnings_calendar", None) or {}
        earn_buf = int((ctx.config.get("regime", {}) or {})
                          .get("earnings_buffer_days", 3))
        today = getattr(ctx, "today", None)
        import math as _m  # noqa: PLC0415

        nb = ns = 0
        n_blocked_buys = n_blocked_earnings = 0
        n_skipped_nonfinite = n_skipped_band = 0

        # 2026-05-09 BA QP audit: log every holding's per-asset solution
        # so we can see why BA (high negative μ̂) wasn't sold even after
        # BUG #7 band-cap fix. Holdings only — buys are visible via QP_BUY.
        holdings_set = set((ctx.holdings or {}).keys())
        for i, t in enumerate(tickers):
            if t in holdings_set:
                tw = float(sol.target_w[i]) if hasattr(sol, "target_w") else float("nan")
                dw_h = float(sol.delta_w[i]) if hasattr(sol, "delta_w") else float("nan")
                sig_h = float(sigma_vec[i]) if (sigma_vec is not None and i < len(sigma_vec)) else float("nan")
                eff_band = max(min_dw, min(band_cap, no_trade_factor * (sig_h if _m.isfinite(sig_h) else 0)))
                will_skip = (abs(dw_h) < eff_band) if _m.isfinite(dw_h) else None
                log.info(
                    "QP_HOLDING_SOLVE %s: target_w=%+.4f Δw=%+.4f σ=%.3f "
                    "eff_band=%.4f will_skip=%s",
                    t, tw, dw_h, sig_h, eff_band, will_skip,
                )

        for i, t in enumerate(tickers):
            dw = float(sol.delta_w[i])
            if not _m.isfinite(dw):
                n_skipped_nonfinite += 1
                continue
            sig_i = 0.0
            if sigma_vec is not None and i < len(sigma_vec):
                s = float(sigma_vec[i])
                if _m.isfinite(s) and s > 0:
                    sig_i = s
            ok, in_band = _passes_no_trade_band(dw, sig_i, min_dw, no_trade_factor, band_cap=band_cap)
            if not ok:
                if in_band:
                    n_skipped_band += 1
                continue
            shares = _shares_from_dw(dw, nav, prices.get(t, 0.0))
            if shares <= 0:
                continue
            if dw > 0:
                blocked = _gate_buy_or_block(
                    t, dw, today, earnings_cal, earn_buf, buys_gated,
                )
                if blocked == "buys_gated":
                    n_blocked_buys += 1
                    continue
                if blocked == "earnings":
                    n_blocked_earnings += 1
                    continue
                _emit_qp_buy(ctx, t, shares, prices.get(t, 0.0), sol, i, cands)
                nb += 1
            elif _emit_qp_sell(ctx, t, shares, dw, sol, i):
                ns += 1

        self._log_summary(
            n_blocked_buys=n_blocked_buys, buy_blocked=buy_blocked,
            n_blocked_earnings=n_blocked_earnings, earn_buf=earn_buf,
            n_skipped_nonfinite=n_skipped_nonfinite,
            n_skipped_band=n_skipped_band, min_dw=min_dw,
            no_trade_factor=no_trade_factor,
        )
        ctx._qp_n_buys = nb  # noqa: SLF001
        ctx._qp_n_sells = ns  # noqa: SLF001

    @staticmethod
    def _log_summary(
        *, n_blocked_buys, buy_blocked, n_blocked_earnings, earn_buf,
        n_skipped_nonfinite, n_skipped_band, min_dw, no_trade_factor,
    ) -> None:
        if n_blocked_buys:
            reason = ("buy_blocked=True" if buy_blocked
                      else "skip_buys=True (drawdown halt)")
            log.info(
                "EmitOrdersFromQPSolutionTask: %s — suppressed %d QP top-ups",
                reason, n_blocked_buys,
            )
        if n_blocked_earnings:
            log.info(
                "EmitOrdersFromQPSolutionTask: suppressed %d top-ups within "
                "±%d earnings days", n_blocked_earnings, earn_buf,
            )
        if n_skipped_nonfinite:
            log.warning(
                "EmitOrdersFromQPSolutionTask: skipped %d non-finite Δw "
                "(investigate Σ conditioning)", n_skipped_nonfinite,
            )
        if n_skipped_band:
            log.info(
                "EmitOrdersFromQPSolutionTask: skipped %d trades by "
                "no-trade band (min_dw=%.2f%%, factor=%.1fσ — Davis-Norman)",
                n_skipped_band, min_dw * 100, no_trade_factor,
            )


# ── helpers ────────────────────────────────────────────────────────────────

def _qp_cfg(ctx) -> dict:
    return (ctx.config.get("rotation", {}).get("joint_actions", {})) or {}


def _per_asset_tax(hs, price, w_i, nav, today, st_rate, lt_rate,
                    lt_days, bridge_w, offset_left) -> tuple[float, float]:
    """Brown-Smith dynamic tax + Berkin-Jeffrey loss-harvest credit (legacy).

    Uses a single average entry_price/entry_date — kept for back-compat
    when `qp_tax_lot_method == "avg"`. Lot-aware path is `_per_asset_tax_lots`.
    """
    entry_p = float(getattr(hs, "entry_price", 0.0) or 0.0)
    entry_d = getattr(hs, "entry_date", None)
    if entry_p <= 0 or entry_d is None or price <= 0:
        return 0.0, offset_left
    gain = (price - entry_p) / entry_p
    try:
        days_held = (today - entry_d).days
    except Exception:
        days_held = 0
    if gain > 0:
        if days_held >= lt_days:
            return gain * lt_rate, offset_left
        days_to_lt = max(0, lt_days - days_held)
        if days_to_lt <= bridge_w:
            amp = (st_rate - lt_rate) * (1.0 - days_to_lt / max(1, bridge_w))
            return gain * (st_rate + amp), offset_left
        return gain * st_rate, offset_left
    if gain < 0 and offset_left > 0:
        est_loss = w_i * abs(gain) * nav
        used = min(est_loss, offset_left)
        if used > 0:
            savings = used * st_rate
            cost = -(savings / max(nav, 1.0) / max(w_i, 1e-6))
            return cost, offset_left - used
    return 0.0, offset_left


def _bridge_rate(st_rate, lt_rate, lt_days, days_held, bridge_w):
    """ST/LT bridge: between (lt_days - bridge_w) and lt_days, rate
    decays linearly from ST toward LT. Outside the bridge: pure ST or LT.
    """
    if days_held >= lt_days:
        return lt_rate
    days_to_lt = max(0, lt_days - days_held)
    if days_to_lt <= bridge_w:
        amp = (st_rate - lt_rate) * (1.0 - days_to_lt / max(1, bridge_w))
        return st_rate + amp
    return st_rate


def _per_asset_tax_lots(hs, price, w_i, nav, today, st_rate, lt_rate,
                         lt_days, bridge_w, offset_left, lot_method
                         ) -> tuple[float, float]:
    """Lot-aware Brown-Smith tax cost.

    Iterates `hs.lots` in disposal order (HIFO → highest-cost lot first
    minimises realized gain; FIFO → oldest first, broker default), and
    accumulates dollar tax across the lots that would be touched to fund
    a 1-NAV-fraction sell of asset i. Returns (cost_per_unit_w, offset_left).

    Loss harvest: same Berkin-Jeffrey credit as legacy — when a lot has
    gain_per_share < 0 AND offset_left > 0, the harvested loss reduces
    `offset_left` and credits a NEGATIVE cost component (savings).
    """
    from kernel.exits import ensure_lots
    if hs is None or price <= 0 or w_i <= 0:
        return 0.0, offset_left
    ensure_lots(hs)
    lots = hs.lots or []
    if not lots:
        return 0.0, offset_left
    method = (lot_method or "fifo").lower()
    if method == "hifo":
        order = sorted(lots, key=lambda L: -L.price)
    else:   # FIFO — preserve insertion order (older first)
        order = list(lots)
    target_shares = (w_i * nav) / max(price, 1e-9)
    cost_dollar = 0.0
    consumed = 0.0
    for L in order:
        if consumed >= target_shares - 1e-12:
            break
        take = min(float(L.shares), target_shares - consumed)
        if take <= 0:
            continue
        gain_per_share = price - float(L.price)
        try:
            held_days = (today - L.date).days
        except Exception:
            held_days = 0
        if gain_per_share > 0:
            rate = _bridge_rate(st_rate, lt_rate, lt_days, held_days, bridge_w)
            cost_dollar += take * gain_per_share * rate
        elif gain_per_share < 0 and offset_left > 0:
            harvest = take * abs(gain_per_share)
            used = min(harvest, offset_left)
            if used > 0:
                cost_dollar += -used * st_rate          # savings (negative)
                offset_left -= used
        consumed += take
    if not math.isfinite(cost_dollar):
        return 0.0, offset_left
    cost_per_unit_w = cost_dollar / max(w_i * nav, 1.0)
    return cost_per_unit_w, offset_left


def _emit_qp_buy(ctx, ticker, shares, px, sol, i, cands):
    ctx.orders.append({
        "ticker": ticker, "shares": shares, "price": px,
        "invest": shares * px,
        "target_pct": float(sol.target_w[i]),
        "rank_score": getattr(cands.get(ticker), "rank_score", None),
        "source": "qp",
    })
    log.info("QP_BUY  %-6s  Δw=%+.4f  shares=%d  px=%.2f  invest=$%.0f",
             ticker, float(sol.delta_w[i]), shares, px, shares * px)


def _emit_qp_sell(ctx, ticker, shares, dw, sol, i) -> bool:
    """Emit SELL signal, including SHORT-OPEN when target_w < 0.

    Three cases:
    1. Closing a long (current shares > 0, target_w ≥ 0): emit qp_sell
       up to held shares, capped at held.
    2. Closing-and-flipping a long to short (current > 0, target_w < 0):
       emit qp_close for the full long portion (held shares), THEN
       emit qp_short_open for the remaining magnitude needed to reach
       target_w.
    3. Opening fresh short (current = 0 or None, target_w < 0): emit
       qp_short_open with magnitude |shares|.

    Phase 2A wiring fix (2026-05-14): pre-fix this function bailed when
    holdings.get(ticker) was None or when qty went negative, so even
    when the QP requested negative target weights, no short orders were
    ever generated. Sim and live ran long-only regardless.
    """
    from kernel.exits import ExitSignal
    target_w = float(sol.target_w[i])
    hs = (ctx.holdings or {}).get(ticker)
    held = int(getattr(hs, "shares", 0) or 0) if hs is not None else 0
    requested = int(shares)  # always positive; sign comes from target_w

    # Case A: target ≥ 0 → just close-down/no-op of existing long
    if target_w >= -1e-9:
        if held <= 0:
            return False
        qty = min(requested, held)
        if qty <= 0:
            return False
        exit_type = "qp_sell" if target_w > 1e-4 else "qp_close"
        ctx.exits.append((ticker, ExitSignal(
            should_exit=True, exit_type=exit_type,
            quantity=float(qty), reason=f"qp_dw={dw:+.4f}",
        )))
        log.info("QP_SELL %-6s  Δw=%+.4f  shares=%d  reason=%s",
                 ticker, dw, qty, exit_type)
        return True

    # Case B/C: target_w < 0 → final position is short.
    # Total |Δshares| comes from QP's |delta_w[i]| × NAV / price, which
    # caller already converted to `shares`. We split into close-long
    # and short-open portions.
    long_close = min(held, requested) if held > 0 else 0
    short_open = max(0, requested - long_close)

    emitted = False
    if long_close > 0:
        ctx.exits.append((ticker, ExitSignal(
            should_exit=True, exit_type="qp_close",
            quantity=float(long_close), reason=f"qp_dw={dw:+.4f}",
        )))
        log.info("QP_SELL %-6s  Δw=%+.4f  shares=%d  reason=qp_close",
                 ticker, dw, long_close)
        emitted = True
    if short_open > 0:
        # Append a SHORT-OPEN order. SimAdapter.commit reads ctx.orders
        # for buys; for shorts we use ctx.exits with a special exit_type
        # so the downstream consumer can route to a short-open code path
        # in _apply_sell when shares > held.
        ctx.exits.append((ticker, ExitSignal(
            should_exit=True, exit_type="qp_short_open",
            quantity=float(short_open), reason=f"qp_dw={dw:+.4f} target_w={target_w:+.4f}",
        )))
        log.info("QP_SHORT_OPEN %-6s  Δw=%+.4f  shares=%d  target_w=%+.4f",
                 ticker, dw, short_open, target_w)
        emitted = True
    return emitted


__all__ = [
    "BuildWeightVectorTask",
    "ComputeFullSigmaTask",
    "ShrinkSigmaLedoitWolfTask",
    "ComputeBrownSmithTaxCostTask",
    "ComputeWashSaleMaskTask",
    "BuildADVVectorTask",
    "ComputeQPConstraintsTask",
    "ApplyConvictionCapTask",
    "BuildSectorConstraintMatrixTask",
    "BuildCorrelationGroupConstraintTask",
    "SolveMarkowitzQPTask",
    "EmitOrdersFromQPSolutionTask",
]
