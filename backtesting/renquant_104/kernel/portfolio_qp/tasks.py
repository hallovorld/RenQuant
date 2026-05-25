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
from kernel.pipeline.order_attribution import stamp_order_attribution
from kernel.pipeline.pipeline import Task

log = logging.getLogger("kernel.portfolio_qp.tasks")


def _ensure_blocked_map(ctx) -> dict:
    blocked_map = getattr(ctx, "_blocked_by_ticker", None)
    if blocked_map is None:
        blocked_map = {}
        ctx._blocked_by_ticker = blocked_map  # noqa: SLF001
    return blocked_map


def _stamp_qp_ticker_block(ctx, ticker: str, reason: str) -> None:
    if not ticker:
        return
    _ensure_blocked_map(ctx).setdefault(str(ticker), reason)


def _stamp_all_qp_blocks(ctx, reason: str) -> None:
    for ticker in (_get_path(ctx, "_qp_tickers") or []):
        _stamp_qp_ticker_block(ctx, str(ticker), reason)


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
    """Build n×n Σ_full = ρ × σ_i × σ_j from loaded/configured correlations.

    Reads:  ctx._qp_tickers, ctx._qp_sigma, ctx.corr_matrix,
             ctx.config['_strategy_dir'], ctx.config['regime']['correlation_artifact'],
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
        corr = getattr(ctx, "corr_matrix", None)
        if not corr:
            corr = self._load_corr_from_artifact(ctx)
        if not corr:
            log.warning(
                "ComputeFullSigmaTask: qp_use_full_sigma=true but no "
                "correlation matrix was loaded; falling back to diagonal Σ."
            )
            ctx._qp_Sigma_full = None  # noqa: SLF001
            return
        tickers = _get_path(ctx, "_qp_tickers") or []
        sig = _get_path(ctx, "_qp_sigma")
        n = len(tickers)
        Sigma = np.zeros((n, n))
        for i in range(n):
            Sigma[i, i] = sig[i] ** 2
        for i, ti in enumerate(tickers):
            for j in range(i + 1, n):
                tj = tickers[j]
                rho = _lookup_corr_explicit_none(corr, ti, tj, default=0.0)
                try:
                    rho_f = max(-0.99, min(0.99, float(rho)))
                except (TypeError, ValueError):
                    rho_f = 0.0
                cov = rho_f * sig[i] * sig[j]
                Sigma[i, j] = cov
                Sigma[j, i] = cov
        ctx._qp_Sigma_full = Sigma + 1e-8 * np.eye(n)  # noqa: SLF001

    @staticmethod
    def _load_corr_from_artifact(ctx) -> dict | None:
        sd = (ctx.config or {}).get("_strategy_dir", "")
        if not sd:
            return None
        rel = (
            (ctx.config or {})
            .get("regime", {})
            .get("correlation_artifact", "prod/watchlist-correlation.json")
        )
        rel_path = Path(str(rel))
        path = rel_path if rel_path.is_absolute() else Path(sd) / "artifacts" / rel_path
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text())
            from kernel.walk_forward import (  # noqa: PLC0415
                assert_correlation_no_leakage,
                parse_correlation_artifact,
            )
            corr, as_of = parse_correlation_artifact(raw)
            config = ctx.config or {}
            assert_correlation_no_leakage(
                as_of,
                config.get("backtest_start"),
                is_live_mode=bool(config.get("_is_live_mode", False)),
                allow_legacy_without_as_of=bool(
                    (config.get("regime", {}) or {})
                    .get("allow_legacy_correlation_without_as_of", False)
                ),
                context="ComputeFullSigmaTask corr",
            )
            return corr
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("ComputeFullSigmaTask: corr load failed from %s (%s)", path, exc)
            return None


def _lookup_corr_explicit_none(corr: dict, left: str, right: str, *, default: float = 0.0):
    """Symmetric corr lookup that treats 0.0 as real data, not missing."""
    row = corr.get(left)
    if isinstance(row, dict):
        value = row.get(right)
        if value is not None:
            return value
    row = corr.get(right)
    if isinstance(row, dict):
        value = row.get(left)
        if value is not None:
            return value
    return default


class AlignQPHorizonUnitsTask(Task):
    """Align QP σ to the same single-period horizon as μ.

    Markowitz 1952 and Boyd-Vandenberghe 2004 portfolio objectives assume
    μ and Σ describe the same rebalance period. In 104, calibrator μ is a
    forward-return estimate over `panel_ltr.lookahead_days`, while the
    realized-vol fallback is explicitly annualized. This task converts σ
    before Σ is built so risk and expected-return units match.
    """
    name = "AlignQPHorizonUnitsTask"
    TRADING_DAYS_PER_YEAR = 252.0

    def run(self, ctx) -> bool | None:
        cfg = _qp_cfg(ctx)
        mode = str(cfg.get("qp_sigma_horizon_mode", "none")).lower()
        sigma = _get_path(ctx, "_qp_sigma")
        if sigma is None or mode in {"none", "off", "disabled"}:
            return
        horizon = _resolve_qp_mu_horizon_days(ctx, cfg)
        unit = str(cfg.get("qp_sigma_unit", "horizon")).lower()
        if horizon is None:
            return _record_qp_horizon_issue(ctx, cfg, "missing_mu_horizon")
        if getattr(ctx, "_qp_sigma_horizon_scaled", False):
            return
        scale = _qp_sigma_horizon_scale(unit, horizon)
        if scale is None:
            return _record_qp_horizon_issue(ctx, cfg, f"unknown_sigma_unit:{unit}")
        sig = np.asarray(sigma, dtype=float)
        if not np.isfinite(sig).all() or (sig <= 0).any():
            return _record_qp_horizon_issue(ctx, cfg, "non_positive_sigma")
        ctx._qp_sigma_raw = sig.copy()  # noqa: SLF001
        ctx._qp_sigma = sig * scale     # noqa: SLF001
        ctx._qp_sigma_horizon_scaled = True  # noqa: SLF001
        ctx._qp_horizon_contract = {  # noqa: SLF001
            "ok": True, "sigma_unit": unit, "mu_horizon_days": int(horizon),
            "scale": float(scale),
        }


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
        if not cfg.get("qp_tax_aware", False):
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

    2026-05-18 ANTI-CHURN: `min_reentry_days` additionally blocks recent
    sold tickers regardless of gain/loss, preventing immediate same-name
    QP rebuys unless enough time has passed for new information.

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
        min_reentry = int((ctx.config or {}).get("min_reentry_days", 0))
        tickers = _get_path(ctx, "_qp_tickers") or []
        if (wash_days <= 0 and min_reentry <= 0) or not tickers:
            ctx._qp_wash_mask = np.zeros(len(tickers), dtype=bool)  # noqa: SLF001
            return
        held_tickers = set(ctx.holdings.keys()) if getattr(ctx, "holdings", None) else set()
        mask, n_wash, n_churn, n_sat = _compute_qp_wash_mask(
            tickers=tickers,
            today=ctx.today,
            last_sell_dates=_get_path(ctx, "last_sell_dates") or {},
            last_sell_pls=_get_path(ctx, "last_sell_pls") or {},
            wash_days=wash_days,
            min_reentry=min_reentry,
            held_tickers=held_tickers,
            calibrator_saturated=bool(getattr(ctx, "_calibrator_saturated", False)),
        )
        ctx._qp_wash_mask = mask  # noqa: SLF001
        if n_wash or n_churn or n_sat:
            import logging
            logging.getLogger("kernel.portfolio_qp.tasks").info(
                "ComputeWashSaleMaskTask: blocked %d wash + %d churn + "
                "%d calibrator-saturation-abstain (min_reentry=%dd) of %d tickers",
                n_wash, n_churn, n_sat, min_reentry, len(tickers))


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
        self._resolve_short_constraints(ctx, scale)
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

    def _resolve_short_constraints(self, ctx, scale: float) -> None:
        """Set ``_qp_w_lower`` and ``_qp_gross_max`` per the long/short policy.

        PRIME DIRECTIVE: every knob resolves through regime overlay first,
        global second. See CLAUDE.md + doc/roadmap.md P1.

        Resolution: ``regime_params.<regime>.long_short_enabled``
                    > ``long_short.enabled``
                    > False

        BEAR hybrid (option γ, 2026-05-14 LOCKED):
          * shorts disabled globally → long-only (w_lower=0, gross unlimited)
          * regime=BEAR + hard_bear=False → DEFENSIVE: still no shorts
            (bear_defensive_slots picks up GLD/TLT)
          * regime=BEAR + hard_bear=True  → OFFENSIVE: shorts allowed
            (longs already blocked by max_position_pct=0)
          * otherwise (BULL_*, CHOPPY)    → shorts at -max_short_pct

        SAFETY: ``max_gross_exposure`` hard-capped at 1.0 (no leverage
        authorized — see tests/test_no_leverage_invariant.py).
        """
        from kernel.regime_resolver import resolve_regime_knob
        _LEVERAGE_HARDCAP = 1.0

        shorts_enabled = bool(resolve_regime_knob(
            ctx, "long_short", "enabled", default=False,
        ))
        regime = getattr(ctx, "regime", None)
        hard_bear = bool(getattr(getattr(ctx, "regime_state", None),
                                 "hard_bear", False))
        if not shorts_enabled or (regime == "BEAR" and not hard_bear):
            ctx._qp_w_lower = 0.0  # noqa: SLF001
            ctx._qp_gross_max = None  # noqa: SLF001
            return

        max_short_pct = float(resolve_regime_knob(
            ctx, "long_short", "max_short_pct", default=0.05,
        ))
        ctx._qp_w_lower = -float(max_short_pct) * scale  # noqa: SLF001
        _cfg_gross = float(resolve_regime_knob(
            ctx, "long_short", "max_gross_exposure",
            default=_LEVERAGE_HARDCAP,
        ))
        ctx._qp_gross_max = min(_cfg_gross, _LEVERAGE_HARDCAP)  # noqa: SLF001


class ApplySectorMetadataGuardTask(Task):
    """Prevent QP from adding risk to tickers without sector metadata.

    The sector cap matrix can only constrain tickers that have a sector row.
    A missing sector therefore used to be an implicit exemption from sector
    diversification. Upstream candidate gates now block most missing-sector
    entries, but the QP must defend its own contract because holdings can
    enter through broker state and future paths may pass candidates directly.

    Invariant: when sector caps are enabled, an unmapped ticker may be held
    or reduced, but QP cannot increase its post-trade weight.
    """
    name = "ApplySectorMetadataGuardTask"

    def run(self, ctx) -> bool | None:
        cfg = _qp_cfg(ctx)
        if not bool(cfg.get("qp_sector_cap_enabled", True)):
            return None
        tickers = _get_path(ctx, "_qp_tickers") or []
        sector_map = (ctx.config or {}).get("sector_map", {}) or {}
        w_upper = _get_path(ctx, "_qp_w_upper")
        w_current = _get_path(ctx, "_qp_w_current")
        if not tickers or w_upper is None or w_current is None:
            return None
        missing = [
            i for i, t in enumerate(tickers)
            if not isinstance(sector_map.get(t), str) or not sector_map.get(t)
        ]
        if not missing:
            ctx._qp_missing_sector_tickers = []  # noqa: SLF001
            return None
        w_upper_arr = np.asarray(w_upper, dtype=float).copy()
        w_current_arr = np.asarray(w_current, dtype=float)
        blocked: list[str] = []
        blocked_map = getattr(ctx, "_blocked_by_ticker", None)
        if blocked_map is None:
            blocked_map = {}
            ctx._blocked_by_ticker = blocked_map  # noqa: SLF001
        candidate_tickers = {
            getattr(c, "ticker", None)
            for c in (getattr(ctx, "candidates", None) or [])
        }
        for i in missing:
            if i >= len(w_upper_arr) or i >= len(w_current_arr):
                continue
            ticker = tickers[i]
            w_upper_arr[i] = min(float(w_upper_arr[i]), max(float(w_current_arr[i]), 0.0))
            blocked.append(ticker)
            if ticker in candidate_tickers:
                blocked_map.setdefault(ticker, "missing_sector_map")
        ctx._qp_w_upper = w_upper_arr  # noqa: SLF001
        ctx._qp_missing_sector_tickers = blocked  # noqa: SLF001
        _inc_counter(ctx, "qp_missing_sector_guard", len(blocked))
        log.warning(
            "ApplySectorMetadataGuardTask: capped %d missing-sector ticker(s) "
            "at current weight: %s",
            len(blocked), blocked[:10],
        )


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


def _has_finite_attr(obj: Any, attr: str) -> bool:
    if obj is None:
        return False
    try:
        return math.isfinite(float(getattr(obj, attr)))
    except (AttributeError, TypeError, ValueError):
        return False


def _inc_counter(ctx, key: str, amount: int = 1) -> None:
    counters = getattr(ctx, "counters", None)
    if not isinstance(counters, dict):
        counters = {}
        ctx.counters = counters
    counters[key] = int(counters.get(key, 0)) + int(amount)


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
        source_attr = {
            "panel_score": "panel_score",
            "panel": "panel_score",
            "rank_score": "rank_score",
            "rank": "rank_score",
            "rs_score": "rs_score",
            "rs": "rs_score",
            "ranking_composite": "_ranking_composite",
            "composite": "_ranking_composite",
            "blend": "_ranking_composite",
            "blended": "_ranking_composite",
        }.get(source)
        if source_attr is None:
            log.warning("ForceMuSource: unknown source '%s' — no-op", source)
            return None
        tickers   = _get_path(ctx, "_qp_tickers") or []
        src_map   = _get_path(ctx, "_qp_mu_source_map") or {}
        new_mu    = np.full(len(tickers), np.nan)
        n_set     = 0
        missing: list[str] = []
        for i, t in enumerate(tickers):
            obj = src_map.get(t)
            if obj is None:
                missing.append(str(t))
                continue
            val = getattr(obj, source_attr, None)
            try:
                v = float(val) if val is not None else math.nan
            except (TypeError, ValueError):
                v = math.nan
            if math.isfinite(v):
                new_mu[i] = v
                n_set += 1
            else:
                missing.append(str(t))
        ctx._qp_mu = new_mu  # noqa: SLF001
        ctx._qp_forced_mu_source = source  # noqa: SLF001
        ctx._qp_forced_mu_missing_tickers = missing  # noqa: SLF001
        log.info(
            "ForceMuSource: μ_QP ← %s for %d/%d tickers (missing=%d, μ̄=%.4f, σ_μ=%.4f)",
            source, n_set, len(tickers),
            len(missing),
            float(np.nanmean(new_mu)) if n_set else 0.0,
            float(np.nanstd(new_mu, ddof=1)) if n_set > 1 else 0.0,
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
        ctx._qp_mu_transformed = True  # noqa: SLF001
        log.info(
            "ApplyGrinoldKahnTransform: IC=%.3f raw_μ̄=%.4f raw_σ_μ=%.4f → μ̄_QP=%.4f",
            ic, m, sd, float(np.abs(ctx._qp_mu).mean()),
        )


class ValidateQPMuContractTask(Task):
    """Guard QP μ semantics.

    The QP objective expects μ to be an expected-return-like quantity. A
    raw ranking score is acceptable only after the Grinold-Kahn
    ``alpha_to_mu`` transform normalizes it to volatility units. Default
    mode is ``strict``: sim/WF results are invalid if QP falls back to raw
    score semantics.
    """
    name = "ValidateQPMuContractTask"

    def run(self, ctx) -> bool | None:
        cfg = _qp_cfg(ctx)
        mode = str(cfg.get("qp_mu_contract", "strict")).lower()
        if mode in {"off", "disabled", "none"}:
            return None

        alpha_cfg = (ctx.config or {}).get("ranking", {}).get("alpha_to_mu", {})
        alpha_cfg = _resolve_regime_override(alpha_cfg, ctx)
        alpha_applied = bool(getattr(ctx, "_qp_mu_transformed", False))
        forced = str((ctx.config or {}).get("ranking", {}).get("qp_mu_source", "mu")).lower()
        tickers = _get_path(ctx, "_qp_tickers") or []
        src = _get_path(ctx, "_qp_mu_source_map") or {}
        forced_missing = list(getattr(ctx, "_qp_forced_mu_missing_tickers", []) or [])
        missing_mu = [
            t for t in tickers
            if not _has_finite_attr(src.get(t), "mu")
        ]
        missing_sigma = [
            t for t in tickers
            if not _has_finite_attr(src.get(t), "sigma")
        ]
        forced_raw = forced not in {"", "none", "mu"}
        ok = (not forced_missing) and (
            alpha_applied or (not missing_mu and not forced_raw)
        ) and not missing_sigma
        ctx._qp_mu_contract = {  # noqa: SLF001
            "ok": ok,
            "mode": mode,
            "alpha_to_mu_enabled": bool(alpha_cfg.get("enabled", False)),
            "alpha_to_mu_applied": alpha_applied,
            "forced_source": forced,
            "forced_source_missing_count": len(forced_missing),
            "forced_source_missing_sample": forced_missing[:10],
            "missing_mu_count": len(missing_mu),
            "missing_mu_sample": missing_mu[:10],
            "missing_sigma_count": len(missing_sigma),
            "missing_sigma_sample": missing_sigma[:10],
        }
        if ok:
            return None

        affected = sorted({
            str(t) for t in (missing_mu + missing_sigma + forced_missing)
        })
        affected_count = len(affected) or int(forced_raw)
        _inc_counter(ctx, "qp_mu_contract_fallback", affected_count)
        msg = (
            "ValidateQPMuContract: QP μ contract failed "
            f"(missing_mu={len(missing_mu)}, forced_source={forced}, "
            f"forced_missing={len(forced_missing)}, "
            f"missing_sigma={len(missing_sigma)}, "
            f"alpha_to_mu_applied={alpha_applied})"
        )
        if mode in {"strict", "hard", "error", "enforce"}:
            _inc_counter(ctx, "qp_mu_contract_block", 1)
            if affected:
                for ticker in affected:
                    reason = (
                        "qp_mu_contract_block"
                        if ticker in {str(t) for t in missing_mu + forced_missing}
                        else "qp_sigma_contract_block"
                    )
                    _stamp_qp_ticker_block(ctx, str(ticker), reason)
            else:
                for ticker in tickers:
                    _stamp_qp_ticker_block(ctx, str(ticker), "qp_mu_contract_block")
            log.error("%s — stopping QP job", msg)
            return False
        log.warning("%s — continuing in warn mode", msg)
        return None


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
        sector_to_idx = self._build_sector_index(tickers, sector_map)
        if not sector_to_idx:
            ctx._qp_sector_indicator = None  # noqa: SLF001
            ctx._qp_sector_cap_vec   = None  # noqa: SLF001
            ctx._qp_sector_names     = []    # noqa: SLF001
            return
        # Per-name cap (post-confidence/scaling/conviction) is in
        # ctx._qp_w_upper. Anchor only on names that actually belong to a
        # mapped sector row; an unmapped broker holding capped at current
        # weight must not inflate every mapped sector's group limit.
        w_upper = _get_path(ctx, "_qp_w_upper")
        mapped_idx = [j for idxs in sector_to_idx.values() for j in idxs]
        per_name_cap = _max_upper_for_indices(
            w_upper, mapped_idx,
            fallback=float((ctx.config or {}).get("max_position_pct", 0.20)),
        )
        sector_names = sorted(sector_to_idx.keys())
        m = len(sector_names)
        S = np.zeros((m, n), dtype=float)
        for row, name in enumerate(sector_names):
            for j in sector_to_idx[name]:
                S[row, j] = 1.0
        legacy_cap = max_per_sector * per_name_cap
        cap, source = _resolve_sector_weight_cap(ctx, legacy_cap)
        cap_vec = np.full(m, cap, dtype=float)
        ctx._qp_sector_indicator = S            # noqa: SLF001
        ctx._qp_sector_cap_vec   = cap_vec      # noqa: SLF001
        ctx._qp_sector_names     = sector_names # noqa: SLF001
        ctx._qp_sector_cap_source = source       # noqa: SLF001

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


def _resolve_sector_weight_cap(ctx, legacy_cap: float) -> tuple[float, str]:
    """Return QP sector cap with regime override support.

    Resolution:
      regime_params.<regime>.max_sector_weight_pct
        > config.max_sector_weight_pct
        > max_positions_per_sector * per_name_cap

    The final cap is min(configured, legacy_count_cap), so count-based
    diversification remains a hard ceiling while regime-level exposure
    tightening can reduce concentration in dominant regimes.
    """
    from kernel.regime_resolver import resolve_regime_knob  # noqa: PLC0415
    cap = legacy_cap
    source = "count_x_per_name"
    configured = resolve_regime_knob(
        ctx, None, "max_sector_weight_pct", default=None,
    )
    try:
        cfg_cap = float(configured) if configured is not None else float("nan")
    except (TypeError, ValueError):
        cfg_cap = float("nan")
    if math.isfinite(cfg_cap) and cfg_cap > 0:
        cap = min(float(legacy_cap), cfg_cap)
        source = "regime_or_global_max_sector_weight_pct"
    return float(max(0.0, cap)), source


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
        corr_matrix = getattr(ctx, "corr_matrix", None)
        if not corr_matrix:
            self._cap_missing_corr_tickers(
                ctx, tickers, set(tickers), reason="missing_correlation_matrix",
            )
            ctx._qp_corr_group_pairs = None  # noqa: SLF001
            return
        thr = float(((ctx.config or {}).get("regime", {}) or {}).get(
            "correlation_guard_threshold", 0.70,
        ))
        if not math.isfinite(thr) or thr <= 0.0 or thr >= 1.0:
            ctx._qp_corr_group_pairs = None  # noqa: SLF001
            return
        w_upper = _get_path(ctx, "_qp_w_upper")
        fallback_cap = float((ctx.config or {}).get("max_position_pct", 0.20))
        pairs, missing_tickers = self._collect_pairs(
            tickers, corr_matrix, thr, w_upper, fallback_cap,
        )
        if missing_tickers:
            self._cap_missing_corr_tickers(
                ctx, tickers, missing_tickers, reason="missing_correlation_pair",
            )
        ctx._qp_corr_group_pairs = pairs if pairs else None  # noqa: SLF001

    @staticmethod
    def _collect_pairs(tickers, corr_matrix, thr, w_upper, fallback_cap):
        """Walk the upper-triangle of the corr matrix; return (i, j, cap)."""
        pairs: list[tuple[int, int, float]] = []
        missing_tickers: set[str] = set()
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
                    missing_tickers.update({ti, tj})
                    continue
                try:
                    rho_f = float(rho)
                except (TypeError, ValueError):
                    missing_tickers.update({ti, tj})
                    continue
                if not math.isfinite(rho_f):
                    # Fail-conservative: NaN correlation → treat as high.
                    rho_f = 1.0
                if abs(rho_f) >= thr:
                    group_cap = _pair_upper_cap(w_upper, i, j, fallback_cap)
                    pairs.append((i, j, group_cap))
        return pairs, missing_tickers

    @staticmethod
    def _cap_missing_corr_tickers(ctx, tickers, missing_tickers: set[str], reason: str):
        w_upper = _get_path(ctx, "_qp_w_upper")
        w_current = _get_path(ctx, "_qp_w_current")
        if w_upper is None or w_current is None:
            return
        w_upper_arr = np.asarray(w_upper, dtype=float).copy()
        w_current_arr = np.asarray(w_current, dtype=float)
        blocked: list[str] = []
        blocked_map = getattr(ctx, "_blocked_by_ticker", None)
        if blocked_map is None:
            blocked_map = {}
            ctx._blocked_by_ticker = blocked_map  # noqa: SLF001
        candidate_tickers = {
            getattr(c, "ticker", None)
            for c in (getattr(ctx, "candidates", None) or [])
        }
        for i, ticker in enumerate(tickers):
            if ticker not in missing_tickers:
                continue
            if i >= len(w_upper_arr) or i >= len(w_current_arr):
                continue
            w_upper_arr[i] = min(
                float(w_upper_arr[i]), max(float(w_current_arr[i]), 0.0),
            )
            blocked.append(ticker)
            if ticker in candidate_tickers:
                blocked_map.setdefault(ticker, reason)
        if not blocked:
            return
        ctx._qp_w_upper = w_upper_arr  # noqa: SLF001
        ctx._qp_missing_correlation_tickers = blocked  # noqa: SLF001
        _inc_counter(ctx, "qp_missing_correlation_guard", len(blocked))
        log.warning(
            "BuildCorrelationGroupConstraintTask: capped %d ticker(s) "
            "at current weight due to incomplete correlation metadata: %s",
            len(blocked), blocked[:10],
        )


def _max_upper_for_indices(w_upper, indices: list[int], *, fallback: float) -> float:
    """Max finite positive upper bound over a constrained group."""
    if w_upper is None:
        return float(fallback)
    arr = np.asarray(w_upper, dtype=float)
    vals = [
        float(arr[i]) for i in indices
        if 0 <= i < len(arr) and math.isfinite(float(arr[i])) and float(arr[i]) >= 0
    ]
    return max(vals) if vals else float(fallback)


def _pair_upper_cap(w_upper, i: int, j: int, fallback: float) -> float:
    """Linear high-correlation cap from the two assets' own upper bounds."""
    if w_upper is None:
        return 2.0 * float(fallback)
    arr = np.asarray(w_upper, dtype=float)
    vals: list[float] = []
    for idx in (i, j):
        if 0 <= idx < len(arr):
            val = float(arr[idx])
            vals.append(val if math.isfinite(val) and val >= 0 else float(fallback))
        else:
            vals.append(float(fallback))
    return float(vals[0] + vals[1])


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
        backend, _solve = self._pick_backend(cfg)
        kwargs = self._build_solver_kwargs(ctx, cfg)
        if backend == "cvxportfolio":
            self._strip_kwargs_for_cvxportfolio(kwargs, ctx)
        sol = _solve(**kwargs)
        sol = _retry_with_relaxed_c2_caps(sol, kwargs, _solve)
        ctx._qp_solution = sol  # noqa: SLF001
        ctx._qp_status = str(getattr(sol, "status", "missing_solution"))  # noqa: SLF001
        ctx._qp_diagnostics = dict(getattr(sol, "diagnostics", {}) or {})  # noqa: SLF001
        if sol.status != "optimal":
            reason = "qp_no_signal" if sol.status == "optimal_no_signal" else f"qp_global:{sol.status}"
            ctx._qp_failure_reason = reason  # noqa: SLF001
            _stamp_all_qp_blocks(ctx, reason)
        ctx._qp_n_buys = 0  # noqa: SLF001
        ctx._qp_n_sells = 0  # noqa: SLF001

    @staticmethod
    def _pick_backend(cfg: dict):
        """Choose cvxpy (default) vs cvxportfolio (opt-in, Boyd ref).

        2026-05-06: both backends accept the same kwargs; cvxportfolio
        uses Boyd's reference policy classes verbatim.
        """
        backend = str(cfg.get("qp_solver_backend", "cvxpy")).lower()
        if backend == "cvxportfolio":
            from kernel.portfolio_qp.cvxportfolio_backend import (  # noqa: PLC0415
                solve_portfolio_qp_cvxportfolio as _solve,
            )
        else:
            from kernel.portfolio_qp.qp_solver import (  # noqa: PLC0415
                solve_portfolio_qp as _solve,
            )
        return backend, _solve

    @staticmethod
    def _build_solver_kwargs(ctx, cfg: dict) -> dict:
        """Marshal ctx + cfg into solve_portfolio_qp's kwargs.

        Single source of truth for every QP knob; future additions go here.
        See `qp_solver.solve_portfolio_qp` docstring for parameter semantics.
        """
        return dict(
            w_current=_get_path(ctx, "_qp_w_current"),
            mu=_get_path(ctx, "_qp_mu"),
            sigma=_get_path(ctx, "_qp_sigma"),
            Sigma=_get_path(ctx, "_qp_Sigma_full"),
            risk_aversion=float(cfg.get("qp_risk_aversion", 3.0)),
            cost_kappa=_effective_qp_cost_kappa(cfg),
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
            min_invested_pct=_effective_min_invested_pct(ctx, cfg),
            cash_drag_lambda=float(cfg.get("qp_cash_drag_lambda", 0.05)),
            sector_indicator=_get_path(ctx, "_qp_sector_indicator"),
            sector_cap_vec=_get_path(ctx, "_qp_sector_cap_vec"),
            corr_group_pairs=_get_path(ctx, "_qp_corr_group_pairs"),
            gross_max=_get_path(ctx, "_qp_gross_max"),
        )

    @staticmethod
    def _strip_kwargs_for_cvxportfolio(kwargs: dict, ctx) -> None:
        """cvxportfolio backend doesn't accept the post-2026-05-10 linear
        constraints (sector/corr/gross_max) — strip them so it doesn't
        TypeError. Falls back to soft diversification via Σ shrinkage."""
        kwargs["tickers"] = _get_path(ctx, "_qp_tickers")
        kwargs.pop("sector_indicator", None)
        kwargs.pop("sector_cap_vec", None)
        kwargs.pop("corr_group_pairs", None)
        kwargs.pop("gross_max", None)


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


def _qp_max_positions(ctx) -> int:
    regime_params = (
        (ctx.config.get("regime_params", {}) or {})
        .get(getattr(ctx, "regime", None), {})
        or {}
    )
    return int(regime_params.get(
        "max_concurrent_positions",
        ctx.config.get("max_concurrent_positions", 8),
    ))


def _qp_buy_admission_block_reason(ctx, env: dict, ticker: str) -> str | None:
    """Fail closed when QP tries to add risk without alpha admission.

    QP solves portfolio weights; it must not become the model-selection layer.
    The gate is intentionally applied at order emission so the solver can still
    trim or close holdings, while new risk additions require finite calibrated
    score evidence and optional raw-panel support.
    """
    gate = (env.get("cfg", {}) or {}).get("qp_admission_gate", {}) or {}
    if not bool(gate.get("enabled", False)):
        return None

    is_held = ticker in env.get("holdings_set", set())
    if (
        not is_held
        and bool(gate.get("respect_open_slots", True))
        and not bool(env.get("ignore_slots", False))
    ):
        held_after_exits = set(env.get("holdings_set", set())) - set(
            env.get("preexisting_exit_tickers", set())
        )
        admitted_new = set(env.get("admitted_new_tickers", set()) or set())
        emitted_new = set(env.get("emitted_new_tickers", set()) or set())
        used_slots = len(held_after_exits | admitted_new | emitted_new)
        if used_slots >= int(env.get("max_positions", 0) or 0):
            return "qp_admission_no_slot"

    source = (
        (env.get("score_sources") or {}).get(ticker)
        or (env.get("cands") or {}).get(ticker)
        or (env.get("holdings") or {}).get(ticker)
    )
    if source is None:
        return "qp_admission_missing_score"

    rank_floor = gate.get(
        "topup_min_rank_score" if is_held else "min_rank_score",
        gate.get("min_rank_score"),
    )
    rank = _source_float(source, "rank_score")
    if rank_floor is not None:
        floor = float(rank_floor)
        if not math.isfinite(rank) or rank < floor:
            return "qp_admission_rank"

    panel_floor = gate.get(
        "topup_min_panel_score" if is_held else "min_panel_score",
        gate.get("min_panel_score"),
    )
    panel = _source_float(source, "panel_score")
    if panel_floor is not None:
        floor = float(panel_floor)
        if not math.isfinite(panel) or panel < floor:
            return "qp_admission_panel"

    sigma_cap = _qp_admission_gate_value(
        gate,
        "topup_max_sigma" if is_held else "max_sigma",
        getattr(ctx, "regime", None),
    )
    if sigma_cap is not None:
        cap = float(sigma_cap)
        sigma = _source_float(source, "sigma")
        if not math.isfinite(sigma) or sigma > cap:
            return "qp_admission_sigma"

    er_floor = _qp_admission_expected_return_floor(gate, is_held, getattr(ctx, "regime", None))
    if er_floor is not None:
        floor = float(er_floor)
        expected_return = _source_float(source, "expected_return")
        if not math.isfinite(expected_return):
            expected_return = _source_float(source, "mu")
        if not math.isfinite(expected_return) or expected_return < floor:
            return "qp_admission_expected_return"

    er_over_sigma_floor = _qp_admission_expected_return_over_sigma_floor(
        gate,
        is_held,
        getattr(ctx, "regime", None),
    )
    if er_over_sigma_floor is not None:
        floor = float(er_over_sigma_floor)
        expected_return = _source_float(source, "expected_return")
        if not math.isfinite(expected_return):
            expected_return = _source_float(source, "mu")
        sigma = _source_float(source, "sigma")
        ratio = (
            expected_return / sigma
            if math.isfinite(expected_return) and math.isfinite(sigma) and sigma > 0
            else float("nan")
        )
        if not math.isfinite(ratio) or ratio < floor:
            return "qp_admission_expected_return_over_sigma"

    return None


def _qp_admission_gate_value(gate: dict, key: str, regime: str | None):
    by_regime = gate.get(f"{key}_by_regime")
    if isinstance(by_regime, dict) and regime in by_regime:
        return by_regime[regime]
    return gate.get(key)


def _qp_admission_expected_return_floor(
    gate: dict,
    is_held: bool,
    regime: str | None,
):
    keys = (
        (
            "topup_min_expected_return",
            "topup_min_expected_excess_return",
            "min_expected_return",
            "min_expected_excess_return",
        )
        if is_held else
        (
            "min_expected_return",
            "min_expected_excess_return",
        )
    )
    for key in keys:
        value = _qp_admission_gate_value(gate, key, regime)
        if value is not None:
            return value
    return None


def _qp_admission_expected_return_over_sigma_floor(
    gate: dict,
    is_held: bool,
    regime: str | None,
):
    keys = (
        (
            "topup_min_expected_return_over_sigma",
            "topup_min_mu_over_sigma",
            "topup_min_edge_over_sigma",
            "min_expected_return_over_sigma",
            "min_mu_over_sigma",
            "min_edge_over_sigma",
        )
        if is_held else
        (
            "min_expected_return_over_sigma",
            "min_mu_over_sigma",
            "min_edge_over_sigma",
        )
    )
    for key in keys:
        value = _qp_admission_gate_value(gate, key, regime)
        if value is not None:
            return value
    return None


def _source_float(source: object, name: str) -> float:
    value = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _buy_cost_multiplier(config: dict) -> float:
    """Return conservative cash multiplier for a buy order."""
    exec_cfg = (config or {}).get("execution", {}) or {}
    if bool(exec_cfg.get("legacy_no_fees", False)):
        return 1.0
    if not bool(exec_cfg.get("enabled", True)):
        return 1.0
    bps = (
        float(exec_cfg.get("half_spread_bps", 2.0) or 0.0)
        + float(exec_cfg.get("commission_bps", 0.0) or 0.0)
        + float(exec_cfg.get("qp_buy_cash_buffer_bps", 1.0) or 0.0)
    )
    return 1.0 + max(0.0, bps) / 10000.0


def _cap_buy_shares_to_cash(
    shares: int,
    px: float,
    cash_left: float,
    cost_multiplier: float,
) -> tuple[int, float]:
    """Cap buy shares so emitted QP orders fit free cash."""
    if shares <= 0 or px <= 0 or cash_left <= 0:
        return 0, 0.0
    unit_cost = px * max(1.0, float(cost_multiplier))
    capped = min(int(shares), int(cash_left // unit_cost))
    return capped, capped * unit_cost


def _actual_qp_buy_target_pct(ctx, ticker: str, shares: int, px: float) -> float:
    """Return post-fill target weight implied by emitted shares.

    QP's solver target_w is the desired total weight before integer share and
    cash caps. LEAN executes BUY orders through SetHoldings(target_pct), so the
    order target must match the shares actually emitted after those caps.
    """
    nav = float(getattr(ctx, "portfolio_value", 0.0) or 0.0)
    if nav <= 0 or px <= 0 or shares <= 0:
        return 0.0
    hs = (getattr(ctx, "holdings", None) or {}).get(ticker)
    held_shares = float(getattr(hs, "shares", 0.0) or 0.0) if hs is not None else 0.0
    return max(0.0, (held_shares + float(shares)) * float(px) / nav)


def _qp_soft_sell_block_reason(ctx, ticker: str, sol, i: int) -> str | None:
    """Apply model-soft-exit guards to QP long trims/closes.

    QP sells are optimizer-driven, not hard risk exits, so they respect the
    same thesis-age horizon gate as panel-conviction exits. Tax-aware soft
    sell gates are different: the production contract says `qp_tax_aware=false`
    means no QP tax-driven sell/hold logic, including the order-emission stage.
    """
    cfg = _qp_cfg(ctx)
    guard_cfg = cfg.get("qp_soft_sell_guard", {})
    if isinstance(guard_cfg, dict) and guard_cfg.get("enabled") is False:
        return None
    target_w = float(sol.target_w[i])
    if target_w < -1e-9:
        return None
    hs = (getattr(ctx, "holdings", None) or {}).get(ticker)
    if hs is None:
        return None
    panel_cfg = _qp_soft_sell_effective_panel_cfg(
        ((getattr(ctx, "config", {}) or {}).get("risk", {}) or {}).get("panel_exit", {}) or {},
        guard_cfg,
    )
    from kernel.pipeline.soft_exit_guards import (  # noqa: PLC0415
        configured_soft_exit_min_days,
        lt_gate_suppression,
        resolve_current_price,
        soft_exit_horizon_suppression,
        tax_adjusted_soft_exit_suppression,
    )
    suppress, why = soft_exit_horizon_suppression(
        panel_cfg=panel_cfg,
        regime=getattr(ctx, "regime", None),
        today=getattr(ctx, "today", None),
        holding=hs,
    )
    if suppress:
        return "qp_soft_sell_horizon:" + why
    min_days = configured_soft_exit_min_days(panel_cfg, getattr(ctx, "regime", None))
    if min_days > 0:
        pending_shares = (_get_path(ctx, "_qp_pending_sell_shares") or {}).get(ticker)
        lot_days = _disposed_lot_min_holding_days(
            holding=hs,
            shares=pending_shares,
            today=getattr(ctx, "today", None),
            lot_method=str(cfg.get("qp_tax_lot_method", "fifo")).lower(),
        )
        if lot_days is not None and lot_days < min_days:
            return (
                "qp_soft_sell_lot_horizon:"
                f"lot_days={lot_days} < {min_days} "
                f"regime={getattr(ctx, 'regime', None)} "
                f"method={str(cfg.get('qp_tax_lot_method', 'fifo')).lower()}"
            )
    current_price = resolve_current_price(ctx, hs, ticker)
    if not _qp_soft_sell_tax_gates_enabled(cfg, guard_cfg):
        return None
    suppress, why = lt_gate_suppression(
        config=getattr(ctx, "config", {}) or {},
        today=getattr(ctx, "today", None),
        holding=hs,
        current_price=current_price,
    )
    if suppress:
        return "qp_soft_sell_lt_gate:" + why
    mu_vec = _get_path(ctx, "_qp_mu")
    mu_i = None
    if mu_vec is not None and i < len(mu_vec):
        mu_i = float(mu_vec[i])
    suppress, why = tax_adjusted_soft_exit_suppression(
        panel_cfg=panel_cfg,
        tax_cfg=(getattr(ctx, "config", {}) or {}).get("tax") or {},
        today=getattr(ctx, "today", None),
        holding=hs,
        current_price=current_price,
        mu=mu_i,
    )
    if suppress:
        return "qp_soft_sell_tax:" + why
    return None


def _qp_soft_sell_effective_panel_cfg(
    panel_cfg: dict[str, Any],
    guard_cfg: Any,
) -> dict[str, Any]:
    """QP-specific soft-sell guard config.

    QP trims are optimizer-driven soft exits but they do not need to share
    every threshold with the cross-sectional panel-conviction exit. Let the
    QP guard override the thesis-age horizon while inheriting the shared
    panel-exit defaults for LT/tax helpers.
    """
    merged = dict(panel_cfg or {})
    if not isinstance(guard_cfg, dict):
        return merged
    for key in ("min_holding_days", "min_holding_days_by_regime"):
        if key in guard_cfg:
            merged[key] = guard_cfg[key]
    return merged


def _disposed_lot_min_holding_days(
    *,
    holding: Any,
    shares: Any,
    today: Any,
    lot_method: str,
) -> int | None:
    """Minimum age among lots a QP soft sell would actually dispose."""
    if not isinstance(today, _dt.date):
        return None
    try:
        target = float(shares)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(target) or target <= 0:
        return None
    lots = list(getattr(holding, "lots", None) or [])
    if not lots:
        entry_date = getattr(holding, "entry_date", None)
        if isinstance(entry_date, _dt.date):
            return max(0, (today - entry_date).days)
        return None

    method = str(lot_method or "fifo").lower()
    if method == "hifo":
        ordered = sorted(lots, key=lambda lot: -float(getattr(lot, "price", 0.0) or 0.0))
    elif method == "avg":
        entry_date = getattr(holding, "entry_date", None)
        if isinstance(entry_date, _dt.date):
            return max(0, (today - entry_date).days)
        ordered = lots
    else:
        ordered = lots

    consumed = 0.0
    min_days: int | None = None
    for lot in ordered:
        if consumed >= target - 1e-12:
            break
        try:
            lot_shares = float(getattr(lot, "shares", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(lot_shares) or lot_shares <= 0:
            continue
        lot_date = getattr(lot, "date", None)
        if not isinstance(lot_date, _dt.date):
            continue
        take = min(lot_shares, target - consumed)
        if take <= 0:
            continue
        age = max(0, (today - lot_date).days)
        min_days = age if min_days is None else min(min_days, age)
        consumed += take
    return min_days


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
            reason = (
                "qp_missing_solution" if sol is None
                else ("qp_no_signal" if sol.status == "optimal_no_signal"
                      else f"qp_global:{sol.status}")
            )
            ctx._qp_status = str(sol.status if sol else "missing_solution")  # noqa: SLF001
            ctx._qp_failure_reason = reason  # noqa: SLF001
            if sol is not None:
                ctx._qp_diagnostics = dict(getattr(sol, "diagnostics", {}) or {})  # noqa: SLF001
            _stamp_all_qp_blocks(ctx, reason)
            log.warning("EmitOrdersFromQPSolutionTask: status=%s — skip",
                         sol.status if sol else "none")
            return False
        env = self._build_env(ctx, sol)
        self._log_holding_solves(env)
        nb, ns, counters = self._emit_orders_loop(ctx, env)
        for key, value in counters.items():
            if value:
                ckey = f"qp_{key}"
                ctx.counters[ckey] = ctx.counters.get(ckey, 0) + int(value)
        self._log_summary(
            n_blocked_buys=counters["blocked_buys"], buy_blocked=env["buy_blocked"],
            n_blocked_earnings=counters["blocked_earnings"], earn_buf=env["earn_buf"],
            n_defensive_non_bear=counters["defensive_non_bear"],
            n_skipped_nonfinite=counters["skipped_nonfinite"],
            n_skipped_band=counters["skipped_band"], min_dw=env["min_dw"],
            no_trade_factor=env["no_trade_factor"],
            n_delta_below_min_dw=counters["delta_below_min_dw"],
            n_zero_shares=counters["zero_shares"],
            n_no_buy_delta=counters["no_buy_delta"],
            n_not_selected=counters["not_selected"],
            n_cash_capped=counters["cash_capped"],
            n_cash_exhausted=counters["cash_exhausted"],
            n_soft_sell_blocked=counters["soft_sell_blocked"],
            n_preexisting_exit=counters["preexisting_exit"],
            n_admission_blocked=counters["admission_blocked"],
        )
        ctx._qp_n_buys = nb  # noqa: SLF001
        ctx._qp_n_sells = ns  # noqa: SLF001

    @staticmethod
    def _build_env(ctx, sol) -> dict:
        """Snapshot the per-run gates + thresholds in one dict so each
        downstream helper sees a coherent view."""
        cfg = _qp_cfg(ctx)
        buy_blocked = bool(getattr(ctx, "buy_blocked", False))
        skip_buys = bool(getattr(ctx, "skip_buys", False))
        from kernel.pipeline.task_benchmark_sleeve import (  # noqa: PLC0415
            benchmark_sleeve_alpha_funding_capacity,
            benchmark_sleeve_cash_reserve_credit,
        )
        cash = float(_get_path(
            ctx, "cash", _get_path(ctx, "portfolio_value", 0.0),
        ) or 0.0)
        alpha_funding_cash = float(benchmark_sleeve_alpha_funding_capacity(ctx))
        cash_reserve = float(_get_path(ctx, "_qp_cash_reserve", 0.0) or 0.0)
        reserve_credit = float(benchmark_sleeve_cash_reserve_credit(ctx))
        effective_cash_reserve = max(0.0, cash_reserve - reserve_credit)
        ctx._qp_alpha_funding_cash = alpha_funding_cash  # noqa: SLF001
        ctx._qp_cash_reserve_effective = effective_cash_reserve  # noqa: SLF001
        return dict(
            cfg=cfg,
            sol=sol,
            tickers=_get_path(ctx, "_qp_tickers") or [],
            prices=_get_path(ctx, "prices") or {},
            nav=float(_get_path(ctx, "portfolio_value", 0.0) or 0.0),
            cash=cash + alpha_funding_cash,
            cash_actual=cash,
            alpha_funding_cash=alpha_funding_cash,
            cash_reserve=effective_cash_reserve,
            cash_reserve_configured=cash_reserve,
            cash_reserve_credit=reserve_credit,
            buy_cost_multiplier=_buy_cost_multiplier(ctx.config or {}),
            min_dw=float(cfg.get("qp_min_dw_pct", 0.005)),
            no_trade_factor=float(cfg.get("qp_no_trade_band_factor", 0.0)),
            band_cap=float(cfg.get("qp_no_trade_band_cap", 0.05)),
            sigma_vec=_get_path(ctx, "_qp_sigma"),
            cands={c.ticker: c for c in (ctx.candidates or [])},
            score_sources=_get_path(ctx, "_qp_mu_source_map") or {},
            buy_blocked=buy_blocked,
            buys_gated=buy_blocked or skip_buys,
            earnings_cal=getattr(ctx, "earnings_calendar", None) or {},
            earn_buf=int((ctx.config.get("regime", {}) or {})
                          .get("earnings_buffer_days", 3)),
            today=getattr(ctx, "today", None),
            holdings_set=set((ctx.holdings or {}).keys()),
            holdings=(ctx.holdings or {}),
            max_positions=_qp_max_positions(ctx),
            # 2026-05-17 min_share_floor for high-price stocks. See
            # _emit_orders_loop comment for rationale. Defaults: floor 5%,
            # ceiling 15%. Disable by setting floor=0.0.
            min_share_floor_pct=float(cfg.get("qp_min_share_floor_pct", 0.05)),
            min_share_ceiling_pct=float(cfg.get("qp_min_share_ceiling_pct", 0.15)),
            defensive_set=set((ctx.config or {}).get("defensive_tickers", []) or []),
            bear_only=bool(getattr(ctx, "bear_only", False)),
            preexisting_exit_tickers={
                t for t, _ in (getattr(ctx, "exits", None) or [])
            },
            emitted_new_tickers=set(),
        )

    @staticmethod
    def _log_holding_solves(env: dict) -> None:
        """2026-05-09 BA QP audit: log every holding's per-asset solution
        so we can see why a name (e.g. high-negative-μ̂ BA) wasn't sold
        even after BUG #7 band-cap fix. Holdings only — buys are visible
        via QP_BUY. Diagnostic-only; no behavior change."""
        import math as _m  # noqa: PLC0415
        sol = env["sol"]; sigma_vec = env["sigma_vec"]
        for i, t in enumerate(env["tickers"]):
            if t not in env["holdings_set"]:
                continue
            tw = float(sol.target_w[i]) if hasattr(sol, "target_w") else float("nan")
            dw_h = float(sol.delta_w[i]) if hasattr(sol, "delta_w") else float("nan")
            sig_h = float(sigma_vec[i]) if (sigma_vec is not None and i < len(sigma_vec)) else float("nan")
            eff_band = max(env["min_dw"], min(env["band_cap"],
                                                env["no_trade_factor"] * (sig_h if _m.isfinite(sig_h) else 0)))
            will_skip = (abs(dw_h) < eff_band) if _m.isfinite(dw_h) else None
            log.info(
                "QP_HOLDING_SOLVE %s: target_w=%+.4f Δw=%+.4f σ=%.3f "
                "eff_band=%.4f will_skip=%s",
                t, tw, dw_h, sig_h, eff_band, will_skip,
            )

    @staticmethod
    def _emit_orders_loop(ctx, env: dict) -> tuple[int, int, dict]:
        """Iterate tickers, apply no-trade-band + earnings/halt gates,
        emit buys/sells. Returns (n_buys, n_sells, counters)."""
        import math as _m  # noqa: PLC0415
        sol = env["sol"]; sigma_vec = env["sigma_vec"]
        nb = ns = 0
        candidate_tickers = set(env["cands"].keys())
        emitted_candidates: set[str] = set()
        blocked_map = getattr(ctx, "_blocked_by_ticker", None)
        if blocked_map is None:
            blocked_map = {}
            ctx._blocked_by_ticker = blocked_map  # noqa: SLF001

        def stamp(ticker: str, reason: str) -> None:
            if ticker in candidate_tickers or ticker in env["holdings_set"]:
                blocked_map.setdefault(ticker, reason)

        c = dict(blocked_buys=0, blocked_earnings=0, defensive_non_bear=0,
                 skipped_nonfinite=0, skipped_band=0,
                 delta_below_min_dw=0, zero_shares=0,
                 no_buy_delta=0, not_selected=0,
                 cash_capped=0, cash_exhausted=0, soft_sell_blocked=0,
                 preexisting_exit=0, admission_blocked=0)
        buy_cash_left = max(0.0, env["cash"] - env["nav"] * env["cash_reserve"])
        pending_sell_shares: dict[str, float] = {}
        _set_path(ctx, "_qp_pending_sell_shares", pending_sell_shares)
        for i, t in enumerate(env["tickers"]):
            dw = float(sol.delta_w[i])
            if not _m.isfinite(dw):
                c["skipped_nonfinite"] += 1
                stamp(t, "qp_nonfinite_delta")
                continue
            if t in env["preexisting_exit_tickers"]:
                c["preexisting_exit"] += 1
                stamp(t, "qp_preexisting_exit")
                log.info(
                    "QP_TRADE_SUPPRESSED %-6s preexisting_exit "
                    "(QP must not double-act on an already exiting ticker)",
                    t,
                )
                continue
            sig_i = 0.0
            if sigma_vec is not None and i < len(sigma_vec):
                s = float(sigma_vec[i])
                if _m.isfinite(s) and s > 0:
                    sig_i = s
            ok, in_band = _passes_no_trade_band(dw, sig_i, env["min_dw"],
                                                  env["no_trade_factor"], band_cap=env["band_cap"])
            if not ok:
                if in_band:
                    c["skipped_band"] += 1
                    stamp(t, "qp_no_trade_band")
                else:
                    c["delta_below_min_dw"] += 1
                    stamp(t, "qp_delta_below_min_dw")
                continue
            px = env["prices"].get(t, 0.0)
            shares = _shares_from_dw(dw, env["nav"], px)
            if shares <= 0 and dw > 0 and px > 0 and env["nav"] > 0:
                # 2026-05-17 min_share_floor for high-price stocks (EQIX/META class).
                # Without this, any candidate whose share price exceeds the QP's
                # dollar budget (target_w × NAV) gets silently dropped — for a
                # $10k account this blocks EQIX ($1059), BKNG ($5k), NVR ($8k),
                # etc. entirely, biasing the strategy toward low-price names.
                # Fix: if 1 share's weight is in [floor, ceiling], allow buying
                # 1 share. ceiling caps the over-allocation vs intended target.
                # Default range [5%, 15%]: floor 5% = minimum-conviction
                # opening threshold; ceiling 15% < the `max_position_pct=20%`
                # golden cap leaves safety buffer. Portfolio-construction
                # engineering choices, not academic — exploratory per
                # CLAUDE.md §5.12. (Proper fix is fractional-share support
                # via Alpaca; out of today's scope.)
                floor   = env["min_share_floor_pct"]
                ceiling = env["min_share_ceiling_pct"]
                if floor > 0:
                    one_share_pct = px / env["nav"]
                    if floor <= one_share_pct <= ceiling:
                        shares = 1
                        log.info(
                            "QP_MIN_SHARE_FLOOR %s: dw=%+.4f → 0 shares "
                            "(px=$%.2f > target $%.0f) — buy 1 share "
                            "(1 share = %.1f%% NAV, floor=%.1f%%, ceil=%.1f%%)",
                            t, dw, px, abs(dw) * env["nav"],
                            one_share_pct * 100, floor * 100, ceiling * 100,
                        )
            if shares <= 0:
                if t in candidate_tickers:
                    if dw > 0:
                        c["zero_shares"] += 1
                        stamp(t, "qp_zero_shares")
                    else:
                        c["no_buy_delta"] += 1
                        stamp(t, "qp_no_buy_delta")
                continue
            if dw > 0:
                admission_block = _qp_buy_admission_block_reason(ctx, env, t)
                if admission_block:
                    c["admission_blocked"] += 1
                    stamp(t, admission_block)
                    log.info(
                        "QP_BUY_SUPPRESSED %-6s %s "
                        "(QP only sizes pre-qualified alpha)",
                        t, admission_block,
                    )
                    continue
                if t in env["defensive_set"] and not env["bear_only"]:
                    c["defensive_non_bear"] += 1
                    stamp(t, "defensive_non_bear")
                    log.info(
                        "QP_BUY_SUPPRESSED %-6s defensive_non_bear "
                        "(regime=%s)",
                        t, getattr(ctx, "regime", None),
                    )
                    continue
                blocked = _gate_buy_or_block(
                    t, dw, env["today"], env["earnings_cal"], env["earn_buf"],
                    env["buys_gated"],
                )
                if blocked == "buys_gated":
                    c["blocked_buys"] += 1
                    stamp(t, "buy_blocked" if env["buy_blocked"] else "skip_buys")
                    continue
                if blocked == "earnings":
                    c["blocked_earnings"] += 1
                    stamp(t, "earnings")
                    continue
                capped_shares, used_cash = _cap_buy_shares_to_cash(
                    shares, px, buy_cash_left, env["buy_cost_multiplier"],
                )
                if capped_shares <= 0:
                    c["cash_exhausted"] += 1
                    stamp(t, "qp_cash_exhausted")
                    continue
                if capped_shares < shares:
                    c["cash_capped"] += 1
                    stamp(t, "qp_cash_capped")
                    shares = capped_shares
                buy_cash_left = max(0.0, buy_cash_left - used_cash)
                _emit_qp_buy(
                    ctx, t, shares, env["prices"].get(t, 0.0),
                    sol, i, env["score_sources"],
                )
                emitted_candidates.add(t)
                if t not in env["holdings_set"]:
                    env["emitted_new_tickers"].add(t)
                nb += 1
            else:
                pending_sell_shares[t] = float(shares)
                soft_block = _qp_soft_sell_block_reason(ctx, t, sol, i)
                pending_sell_shares.pop(t, None)
                if soft_block:
                    c["soft_sell_blocked"] += 1
                    stamp(t, soft_block)
                    log.info("QP_SELL_SUPPRESSED %-6s  Δw=%+.4f  %s",
                             t, dw, soft_block)
                    continue
                if _emit_qp_sell(ctx, t, shares, dw, sol, i):
                    ns += 1
                else:
                    stamp(t, "qp_no_sell_position")
        for ticker in candidate_tickers - emitted_candidates:
            if ticker not in blocked_map:
                c["not_selected"] += 1
                blocked_map[ticker] = "qp_not_selected"
        return nb, ns, c

    @staticmethod
    def _log_summary(
        *, n_blocked_buys, buy_blocked, n_blocked_earnings, earn_buf,
        n_defensive_non_bear,
        n_skipped_nonfinite, n_skipped_band, min_dw, no_trade_factor,
        n_delta_below_min_dw, n_zero_shares, n_no_buy_delta, n_not_selected,
        n_cash_capped, n_cash_exhausted, n_soft_sell_blocked,
        n_preexisting_exit, n_admission_blocked,
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
        if n_defensive_non_bear:
            log.info(
                "EmitOrdersFromQPSolutionTask: suppressed %d defensive "
                "QP buy/top-up(s) outside BEAR regime",
                n_defensive_non_bear,
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
        if n_delta_below_min_dw:
            log.info(
                "EmitOrdersFromQPSolutionTask: skipped %d trades below "
                "minimum Δw %.2f%%",
                n_delta_below_min_dw, min_dw * 100,
            )
        if n_zero_shares:
            log.info(
                "EmitOrdersFromQPSolutionTask: skipped %d candidate buy(s) "
                "because Δw rounded to 0 shares",
                n_zero_shares,
            )
        if n_no_buy_delta:
            log.info(
                "EmitOrdersFromQPSolutionTask: skipped %d candidate buy(s) "
                "because QP assigned no positive buy delta",
                n_no_buy_delta,
            )
        if n_not_selected:
            log.info(
                "EmitOrdersFromQPSolutionTask: %d candidate(s) received no "
                "QP allocation reason after solve",
                n_not_selected,
            )
        if n_cash_capped:
            log.info(
                "EmitOrdersFromQPSolutionTask: capped %d QP buy(s) to available cash",
                n_cash_capped,
            )
        if n_cash_exhausted:
            log.info(
                "EmitOrdersFromQPSolutionTask: skipped %d QP buy(s) because cash was exhausted",
                n_cash_exhausted,
            )
        if n_soft_sell_blocked:
            log.info(
                "EmitOrdersFromQPSolutionTask: suppressed %d QP soft sell(s) "
                "by horizon/LT/tax guards",
                n_soft_sell_blocked,
            )
        if n_preexisting_exit:
            log.info(
                "EmitOrdersFromQPSolutionTask: suppressed %d QP trade(s) "
                "for ticker(s) already carrying an exit intent",
                n_preexisting_exit,
            )
        if n_admission_blocked:
            log.info(
                "EmitOrdersFromQPSolutionTask: suppressed %d QP buy/top-up(s) "
                "by alpha-admission gate",
                n_admission_blocked,
            )


# ── helpers ────────────────────────────────────────────────────────────────

# Keys that support per-regime override (2026-05-16 B-track):
# Reading order:
#   regime_params.<ctx.regime>.<KEY>  →  rotation.joint_actions.<KEY>
# Pattern matches CLAUDE.md PRIME DIRECTIVE (regime-conditional strategy).
# Test pin: tests/test_qp_cfg_per_regime_override.py
_QP_PER_REGIME_KEYS = (
    "qp_cvar_lambda",
    "qp_cvar_alpha",
    "qp_turnover_max",
    "qp_risk_aversion",
    "qp_cost_kappa",
    "qp_cost_kappa_floor_round_trip",
    "qp_dw_max",
    "qp_min_dw_pct",
    "qp_no_trade_band_factor",
    "qp_no_trade_band_cap",
    "qp_min_invested_pct",
    "qp_cash_drag_lambda",
    "qp_min_invested_requires_positive_edge",
    "qp_min_invested_edge_floor",
    "qp_mu_horizon_days",
    "qp_sigma_unit",
    "qp_sigma_horizon_mode",
    "qp_horizon_contract",
    "qp_admission_gate",
)


def _qp_cfg(ctx) -> dict:
    base = dict((ctx.config.get("rotation", {}).get("joint_actions", {})) or {})
    regime = getattr(ctx, "regime", None)
    if regime:
        regime_p = (ctx.config.get("regime_params", {}) or {}).get(regime, {}) or {}
        for key in _QP_PER_REGIME_KEYS:
            if key in regime_p:
                base[key] = regime_p[key]
    return base


def _resolve_qp_mu_horizon_days(ctx, cfg: dict) -> int | None:
    raw = cfg.get("qp_mu_horizon_days")
    if raw is None:
        raw = (ctx.config.get("panel_ltr", {}) or {}).get("lookahead_days")
    if raw is None:
        raw = ctx.config.get("lookahead_days")
    try:
        horizon = int(raw)
    except (TypeError, ValueError):
        return None
    return horizon if horizon > 0 else None


def _qp_sigma_horizon_scale(unit: str, horizon_days: int) -> float | None:
    if unit in {"horizon", "period", "matched"}:
        return 1.0
    if unit in {"annual", "annualized", "ann"}:
        return math.sqrt(float(horizon_days) / 252.0)
    if unit == "daily":
        return math.sqrt(float(horizon_days))
    return None


def _record_qp_horizon_issue(ctx, cfg: dict, reason: str) -> bool | None:
    contract = str(cfg.get("qp_horizon_contract", cfg.get("qp_mu_contract", "warn"))).lower()
    report = {"ok": False, "reason": reason}
    ctx._qp_horizon_contract = report  # noqa: SLF001
    counters = getattr(ctx, "counters", None)
    if counters is not None:
        key = "qp_horizon_contract_block" if contract == "strict" else "qp_horizon_contract_warn"
        counters[key] = counters.get(key, 0) + 1
    log.warning("AlignQPHorizonUnitsTask: %s", reason)
    return False if contract == "strict" else None


def _effective_min_invested_pct(ctx, cfg: dict) -> float:
    base = float(cfg.get("qp_min_invested_pct", 0.0) or 0.0)
    if base <= 0.0 or not bool(cfg.get("qp_min_invested_requires_positive_edge", False)):
        return base
    mu = np.asarray(_get_path(ctx, "_qp_mu"), dtype=float)
    finite = mu[np.isfinite(mu)]
    best_mu = float(np.max(finite)) if finite.size else float("-inf")
    floor = float(cfg.get("qp_min_invested_edge_floor", _round_trip_cost(cfg)))
    blocked = best_mu <= floor
    ctx._qp_min_invested_contract = {  # noqa: SLF001
        "base": base, "effective": 0.0 if blocked else base,
        "best_mu": best_mu, "edge_floor": floor, "blocked": blocked,
    }
    return 0.0 if blocked else base


def _round_trip_cost(cfg: dict) -> float:
    fee = float(cfg.get("fee_pct", cfg.get("qp_cost_kappa", 0.0)) or 0.0)
    slip = float(cfg.get("slippage_pct", 0.0) or 0.0)
    return 2.0 * (fee + slip)


def _effective_qp_cost_kappa(cfg: dict) -> float:
    """L1 turnover penalty used by the QP objective.

    Gârleanu-Pedersen 2013 shows proportional transaction costs create a
    no-trade region around the current portfolio. In this single-period QP,
    the convex proxy is the L1 turnover penalty. When the floor flag is on,
    stale configs cannot underprice trading below explicit fee+slippage.
    """
    raw = float(cfg.get("qp_cost_kappa", cfg.get("fee_pct", 0.0005)) or 0.0)
    if bool(cfg.get("qp_cost_kappa_floor_round_trip", False)):
        return max(raw, _round_trip_cost(cfg))
    return raw


def _qp_soft_sell_tax_gates_enabled(cfg: dict, guard_cfg: object) -> bool:
    """Return whether QP order emission may suppress sells for tax reasons."""
    if isinstance(guard_cfg, dict) and "apply_tax_gates" in guard_cfg:
        return bool(guard_cfg.get("apply_tax_gates"))
    return bool(cfg.get("qp_tax_aware", False))


def _compute_qp_wash_mask(
    *,
    tickers: list[str],
    today,
    last_sell_dates: dict,
    last_sell_pls: dict,
    wash_days: int,
    min_reentry: int,
    held_tickers: set[str],
    calibrator_saturated: bool,
) -> tuple[np.ndarray, int, int, int]:
    """Build QP block mask for wash-sale, anti-churn, and saturation abstain."""
    from kernel.selection import is_wash_sale_blocked_with_cost  # noqa: PLC0415
    mask = np.zeros(len(tickers), dtype=bool)
    n_wash = n_churn = n_sat = 0
    for i, t in enumerate(tickers):
        if wash_days > 0:
            blocked, _, _ = is_wash_sale_blocked_with_cost(
                ticker=t,
                today=today,
                last_sell_dates=last_sell_dates,
                last_sell_pls=last_sell_pls,
                wash_sale_days=wash_days,
            )
            if blocked:
                mask[i] = True
                n_wash += 1
                continue
        if min_reentry > 0:
            last = last_sell_dates.get(t)
            if last is not None:
                if isinstance(last, str):
                    try:
                        last = _dt.date.fromisoformat(last[:10])
                    except (ValueError, TypeError):
                        continue
                days_since = (today - last).days
                if 0 <= days_since < min_reentry:
                    mask[i] = True
                    n_churn += 1
                    continue
        if calibrator_saturated and t not in held_tickers:
            mask[i] = True
            n_sat += 1
    return mask, n_wash, n_churn, n_sat


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


def _emit_qp_buy(ctx, ticker, shares, px, sol, i, score_sources):
    cand = score_sources.get(ticker)
    actual_target_pct = _actual_qp_buy_target_pct(ctx, ticker, shares, px)
    ctx.orders.append(stamp_order_attribution({
        "ticker": ticker, "shares": shares, "price": px,
        "invest": shares * px,
        "target_pct": actual_target_pct,
        "regime": getattr(ctx, "regime", None),
        "confidence": getattr(ctx, "confidence", None),
        "rank_score": getattr(cand, "rank_score", None),
        "rs_score": getattr(cand, "rs_score", None),
        "panel_score": getattr(cand, "panel_score", None),
        "mu": getattr(cand, "mu", None),
        "sigma": getattr(cand, "sigma", None),
        "kelly_target_pct": getattr(cand, "kelly_target_pct", None),
        "detail": getattr(cand, "detail", ""),
        "order_type": "QP_BUY",
        "source": "qp",
    }, ctx=ctx, source_job="JointPortfolioQPJob",
        source_task="EmitOrdersFromQPSolutionTask",
        acceptance_reason="qp_target_weight_increase",
        source_obj=cand,
        decision_inputs={
            "delta_w": float(sol.delta_w[i]),
            "target_w": float(sol.target_w[i]),
            "actual_target_w": float(actual_target_pct),
            "solver_status": getattr(sol, "status", None),
            "expected_return_horizon_days": getattr(
                cand, "expected_return_horizon_days", None,
            ),
            "mu_horizon_days": getattr(cand, "mu_horizon_days", None),
        }))
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
        sig = ctx.exits[-1][1]
        sig.source_job = "JointPortfolioQPJob"
        sig.source_task = "EmitOrdersFromQPSolutionTask"
        sig.decision_inputs = {
            "delta_w": float(dw),
            "target_w": float(target_w),
            "solver_status": getattr(sol, "status", None),
            "shares": float(qty),
            "held_shares": float(held),
            "expected_return_horizon_days": getattr(
                hs, "expected_return_horizon_days", None,
            ),
            "mu_horizon_days": getattr(hs, "mu_horizon_days", None),
        }
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
        sig = ctx.exits[-1][1]
        sig.source_job = "JointPortfolioQPJob"
        sig.source_task = "EmitOrdersFromQPSolutionTask"
        sig.decision_inputs = {
            "delta_w": float(dw),
            "target_w": float(target_w),
            "solver_status": getattr(sol, "status", None),
            "shares": float(long_close),
            "held_shares": float(held),
            "expected_return_horizon_days": getattr(
                hs, "expected_return_horizon_days", None,
            ),
            "mu_horizon_days": getattr(hs, "mu_horizon_days", None),
        }
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
        sig = ctx.exits[-1][1]
        sig.source_job = "JointPortfolioQPJob"
        sig.source_task = "EmitOrdersFromQPSolutionTask"
        sig.decision_inputs = {
            "delta_w": float(dw),
            "target_w": float(target_w),
            "solver_status": getattr(sol, "status", None),
            "shares": float(short_open),
            "held_shares": float(held),
        }
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
    "ApplySectorMetadataGuardTask",
    "ApplyConvictionCapTask",
    "BuildSectorConstraintMatrixTask",
    "BuildCorrelationGroupConstraintTask",
    "SolveMarkowitzQPTask",
    "EmitOrdersFromQPSolutionTask",
]
