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
            corr = json.loads(path.read_text())
        except Exception as exc:
            log.warning("ComputeFullSigmaTask: corr load failed (%s)", exc)
            ctx._qp_Sigma_full = None  # noqa: SLF001
            return
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
    (no correlation, equal variance). Typical operator setting 0.1–0.3.

    Reads:  ctx._qp_Sigma_full,
             ctx.config['rotation']['joint_actions']['qp_ledoit_wolf_lambda']
    Writes: ctx._qp_Sigma_full (in place; None if upstream produced None
             — diagonal-Σ fallback in solver is unaffected)
    """
    name = "ShrinkSigmaLedoitWolfTask"

    def run(self, ctx) -> bool | None:
        cfg = _qp_cfg(ctx)
        lam = float(cfg.get("qp_ledoit_wolf_lambda", 0.0))
        if lam <= 0.0:
            return                                      # default: off
        lam = min(lam, 1.0)
        S = _get_path(ctx, "_qp_Sigma_full")
        if S is None:
            return                                      # diagonal path
        n = S.shape[0]
        if n == 0:
            return
        avg_var = float(np.trace(S)) / max(n, 1)
        F = avg_var * np.eye(n)
        ctx._qp_Sigma_full = (1.0 - lam) * S + lam * F  # noqa: SLF001


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
    """Wash-sale mask: tickers sold within wash_sale_days have Δw_i ≤ 0.

    Implemented as a thin domain wrapper around BuildMaskFromConditionTask.
    Reads:  ctx._qp_tickers, ctx.last_sell_dates, ctx.config['wash_sale_days']
    Writes: ctx._qp_wash_mask (np.ndarray of bool)
    """
    name = "ComputeWashSaleMaskTask"

    def run(self, ctx) -> bool | None:
        wash_days = int((ctx.config or {}).get("wash_sale_days", 0))
        if wash_days <= 0:
            tickers = _get_path(ctx, "_qp_tickers") or []
            ctx._qp_wash_mask = np.zeros(len(tickers), dtype=bool)  # noqa: SLF001
            return
        from kernel.pipeline.atoms.vectors import BuildMaskFromConditionTask
        last_sells = _get_path(ctx, "last_sell_dates") or {}
        today = ctx.today

        def _is_recent_sell(c, t: str) -> bool:
            last = last_sells.get(t)
            if last is None:
                return False
            if isinstance(last, str):
                try:
                    last = _dt.date.fromisoformat(last[:10])
                except ValueError:
                    return False
            try:
                return (today - last).days < wash_days
            except Exception:
                return False

        BuildMaskFromConditionTask(
            "_qp_tickers", "_qp_wash_mask", _is_recent_sell,
        ).run(ctx)


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
        ctx._qp_w_lower = 0.0  # noqa: SLF001
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
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp
        cfg = _qp_cfg(ctx)
        sol = solve_portfolio_qp(
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
            # G10 — Rockafellar-Uryasev CVaR tail-risk multiplier (off
            # by default; cvar_lambda > 0 adds φ(z_α)/α · σ to gamma_eff)
            cvar_lambda=float(cfg.get("qp_cvar_lambda", 0.0)),
            cvar_alpha=float(cfg.get("qp_cvar_alpha", 0.05)),
            # G3 — Almgren-Chriss sqrt-impact (off by default; set
            # qp_impact_coef > 0 + ADV present to activate)
            impact_coef=float(cfg.get("qp_impact_coef", 0.0)),
            v_daily_dollar=_get_path(ctx, "_qp_v_daily_dollar"),
            nav_dollar=float(_get_path(ctx, "portfolio_value", 0.0) or 0.0),
            # G4 — Smoothed fixed cost (off by default)
            fixed_cost_per_trade=float(cfg.get("qp_fixed_cost_per_trade", 0.0)),
            fixed_cost_beta=float(cfg.get("qp_fixed_cost_beta", 200.0)),
            # 2026-05-05 cash-drag fix — budget mode + min_invested_pct.
            # equality forces Σw = 1 − cash_reserve (textbook Markowitz)
            # but breaks SLSQP feasibility on empty-portfolio start.
            # min_invested_pct > 0 imposes a SOFT floor (two-sided box,
            # stays feasible). Recommended starting point: 0.7
            # (require ≥70% deployed).
            budget_mode=str(cfg.get("qp_budget_mode", "inequality")),
            min_invested_pct=float(cfg.get("qp_min_invested_pct", 0.0)),
        )
        ctx._qp_solution = sol  # noqa: SLF001
        ctx._qp_n_buys = 0  # noqa: SLF001
        ctx._qp_n_sells = 0  # noqa: SLF001


# ── 7. Translate Δw → orders / exits ───────────────────────────────────────

class EmitOrdersFromQPSolutionTask(Task):
    """Translate Δw → ctx.orders (buys/top-ups) + ctx.exits (closes/trims).

    Reads:  ctx._qp_solution, ctx._qp_tickers, ctx.prices, ctx.holdings,
             ctx.portfolio_value, ctx.candidates,
             ctx.config['rotation']['joint_actions']['qp_min_dw_pct']
    Writes: ctx.orders (append), ctx.exits (append),
             ctx._qp_n_buys, ctx._qp_n_sells (counters for atom-side LogSummary)
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
        # 2026-05-05 cash-drag fix: per-asset no-trade band.
        # Davis-Norman (1990) / Constantinides (1979) closed form:
        #   ε_i ≈ (3κ/γ)^(1/3) × σ_i × √Δt
        # The classical answer to "when do I rebalance" — only trade
        # when |Δw_i| exceeds a volatility-scaled threshold. Smaller
        # ε for low-vol assets (rebalance often), larger ε for high-vol
        # (let positions drift). Pre-fix the only floor was the uniform
        # qp_min_dw_pct (0.02 NAV-fraction), which produces friction-
        # equivalent trades regardless of σ. Post-fix the threshold is
        # max(qp_min_dw_pct, qp_no_trade_band_factor × σ_i).
        # Default qp_no_trade_band_factor=0.0 → disabled (legacy parity).
        # Recommended starting point: 1.0 (one-σ band).
        no_trade_factor = float(cfg.get("qp_no_trade_band_factor", 0.0))
        sigma_vec = _get_path(ctx, "_qp_sigma")
        cands = {c.ticker: c for c in (ctx.candidates or [])}
        # 2026-05-05 wl183 incident bug 3: when ctx.buy_blocked OR
        # ctx.skip_buys is set, QP rebalance must not emit any buy.
        #   - buy_blocked: per-bar gate (DrawdownGate, VelocityCrash,
        #     EarningsBlackout regime). Pipeline gates Phase 2b but QP
        #     was free to emit top-ups → whiplash.
        #   - skip_buys: persistent drawdown-circuit halt (set by
        #     DrawdownCircuitTask, cleared on recovery). Same intent —
        #     no new exposure until circuit clears.
        # Pre-fix bar 03-19 (BULL_VOL conf=0.99 buys_blocked=True): QP
        # emitted +20% buys against the circuit; bar 03-20 (BULL_CALM
        # conf=0.50 transition=True): QP reversed at -24%. 10bps friction
        # round-trip × hundreds of regime flips = wl183 B2 Sharpe -0.07.
        # Fix: suppress dw>0 emissions on either flag. Sells still
        # allowed so the circuit can de-risk.
        buy_blocked = bool(getattr(ctx, "buy_blocked", False))
        skip_buys   = bool(getattr(ctx, "skip_buys",   False))
        buys_gated  = buy_blocked or skip_buys
        # 2026-05-05 wl183 incident bug 4: QP had no earnings awareness.
        # The buy-side EarningsFilterTask (in TickerCandidateJob) blocks
        # new entries within ±earnings_buffer_days of earnings, but QP
        # could still top-up an EXISTING holding right into earnings.
        # Same gap-risk rationale that justifies blocking new entries
        # applies to top-ups: an unexpected miss can move 5–15% on the
        # print, far larger than typical Δw the QP is optimizing over.
        # Fix: per-ticker check via the same is_earnings_blocked helper
        # the buy-side filter uses, gated on the same buffer config so
        # train/inference symmetry is preserved.
        from kernel.selection import is_earnings_blocked  # noqa: PLC0415
        earnings_cal = getattr(ctx, "earnings_calendar", None) or {}
        earn_buf = int((ctx.config.get("regime", {}) or {})
                          .get("earnings_buffer_days", 3))
        today = getattr(ctx, "today", None)
        n_blocked_buys = 0
        n_blocked_earnings = 0
        nb = ns = 0
        # Bug 9 fix (2026-05-05): defensive against non-finite Δw.
        # SolveMarkowitzQPTask returns sol.status="optimal" when SLSQP
        # converges, but the solver can occasionally produce NaN/inf
        # weights on numerically-degenerate inputs (e.g. zero-volatility
        # asset that wasn't pre-filtered, near-singular Σ). Pre-fix,
        # `int(abs(NaN) * ...)` raises ValueError mid-loop → uncaught →
        # the entire bar's Phase 3 unwinds, no Kelly/QP signal recorded.
        # Post-fix: skip non-finite Δw with a warning so the operator
        # can investigate without crashing the sim.
        import math as _math_dw  # noqa: PLC0415
        n_skipped_nonfinite = 0
        n_skipped_band = 0
        for i, t in enumerate(tickers):
            dw = float(sol.delta_w[i])
            if not _math_dw.isfinite(dw):
                n_skipped_nonfinite += 1
                continue
            # Per-asset no-trade band: max(min_dw_pct, factor × σ_i).
            # When sigma_vec missing or non-finite for asset i, fall
            # back to the uniform min_dw threshold (legacy parity).
            sig_i = 0.0
            if sigma_vec is not None and i < len(sigma_vec):
                sig_i_raw = float(sigma_vec[i])
                if _math_dw.isfinite(sig_i_raw) and sig_i_raw > 0:
                    sig_i = sig_i_raw
            threshold = max(min_dw, no_trade_factor * sig_i)
            if abs(dw) < threshold:
                if abs(dw) >= min_dw:
                    n_skipped_band += 1
                continue
            px = prices.get(t, 0.0)
            if not _math_dw.isfinite(px) or px <= 0:
                continue
            if not _math_dw.isfinite(nav) or nav <= 0:
                continue
            shares = int(abs(dw) * nav / px)
            if shares <= 0:
                continue
            if dw > 0:
                if buys_gated:
                    n_blocked_buys += 1
                    continue
                if today is not None and is_earnings_blocked(
                        t, today, earnings_cal, earn_buf):
                    n_blocked_earnings += 1
                    continue
                _emit_qp_buy(ctx, t, shares, px, sol, i, cands)
                nb += 1
            elif _emit_qp_sell(ctx, t, shares, dw, sol, i):
                ns += 1
        if n_blocked_buys:
            reason = ("buy_blocked=True" if buy_blocked
                      else "skip_buys=True (drawdown halt)")
            log.info(
                "EmitOrdersFromQPSolutionTask: %s — suppressed %d QP "
                "top-up BUY(s) (bar would have whiplashed against "
                "drawdown/velocity/blackout circuit)",
                reason, n_blocked_buys,
            )
        if n_blocked_earnings:
            log.info(
                "EmitOrdersFromQPSolutionTask: suppressed %d QP top-up "
                "BUY(s) within ±%d days of earnings (gap-risk parity "
                "with buy-side EarningsFilterTask)",
                n_blocked_earnings, earn_buf,
            )
        if n_skipped_nonfinite:
            log.warning(
                "EmitOrdersFromQPSolutionTask: skipped %d non-finite "
                "Δw entries (NaN/inf weights from solver — investigate "
                "Σ conditioning or μ/σ inputs)",
                n_skipped_nonfinite,
            )
        if n_skipped_band:
            log.info(
                "EmitOrdersFromQPSolutionTask: skipped %d trades by "
                "no-trade band (above %.2f%% min_dw but inside %.1fσ "
                "vol-scaled band) — Davis-Norman/Constantinides "
                "rebalance economy",
                n_skipped_band, min_dw * 100, no_trade_factor,
            )
        ctx._qp_n_buys = nb  # noqa: SLF001
        ctx._qp_n_sells = ns  # noqa: SLF001


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
    from kernel.exits import ExitSignal
    hs = (ctx.holdings or {}).get(ticker)
    if hs is None:
        return False
    held = int(getattr(hs, "shares", 0) or 0)
    qty = min(shares, held)
    if qty <= 0:
        return False
    exit_type = "qp_sell" if sol.target_w[i] > 1e-4 else "qp_close"
    ctx.exits.append((ticker, ExitSignal(
        should_exit=True, exit_type=exit_type,
        quantity=float(qty), reason=f"qp_dw={dw:+.4f}",
    )))
    log.info("QP_SELL %-6s  Δw=%+.4f  shares=%d  reason=%s",
             ticker, dw, qty, exit_type)
    return True


__all__ = [
    "BuildWeightVectorTask",
    "ComputeFullSigmaTask",
    "ShrinkSigmaLedoitWolfTask",
    "ComputeBrownSmithTaxCostTask",
    "ComputeWashSaleMaskTask",
    "BuildADVVectorTask",
    "ComputeQPConstraintsTask",
    "SolveMarkowitzQPTask",
    "EmitOrdersFromQPSolutionTask",
]
