"""JointPortfolioQPTask — single-shot Markowitz QP for buy/sell/rotate.

Stage 1 of the unified portfolio action redesign. Drop-in alternative
to JointActionTask; selected at runtime via:

    config["rotation"]["joint_actions"]["solver"] = "qp"   # or "greedy" (default)

Reads ctx state, builds a QP problem (w_current, μ, σ, wash-sale mask,
position caps), solves it, then translates Δw → ctx.orders + ctx.exits.

Sign of Δw IS the action:
  * Δw_i > 0   → BUY (or top-up an existing holding)
  * Δw_i < 0   → SELL (full close if w + Δw ≈ 0; else trim)
  * Δw_i = 0   → HOLD / NO-TRADE
  * Δw_A < 0  +  Δw_B > 0   → ROTATION (paired by sign convention)

Implemented constraints (Stage-1 subset):
  - cash reserve              (1' (w + Δw) ≤ 1 - cash_reserve)
  - per-position cap          (w + Δw ≤ max_position_pct)
  - no shorts                  (w + Δw ≥ 0)
  - wash sale                  (Δw_i ≤ 0 for recently-sold)
  - slippage band              (|Δw_i| ≤ dw_max)

NOT yet implemented (Stages 4-7):
  - DD scaler (Grossman-Zhou)            — Stage 4
  - Robust μ adjustment (Garlappi-UW)    — Stage 5
  - Signal combination (Treynor-Black)   — Stage 6
  - CVaR risk term (Rockafellar-Uryasev) — Stage 7

Reference: ``doc/components/portfolio-qp.md``.
"""
from __future__ import annotations

import datetime
import logging
from typing import Any

import numpy as np

from kernel.pipeline.context import InferenceContext
from kernel.pipeline.pipeline import Task
from kernel.portfolio_qp.qp_solver import solve_portfolio_qp

log = logging.getLogger("kernel.portfolio_qp.joint_qp")


class JointPortfolioQPTask(Task):
    """QP-based replacement for JointActionTask.

    Default OFF — opt in via `rotation.joint_actions.solver = "qp"`.
    Greedy path (`solver = "greedy"`) is the default until parity is
    fully validated on a 27-mo OOS sweep.
    """

    name = "JointPortfolioQPTask"

    def run(self, ctx: InferenceContext) -> bool | None:
        joint_cfg = (ctx.config.get("rotation", {})
                                .get("joint_actions", {}))
        solver = str(joint_cfg.get("solver", "greedy")).lower()
        if not joint_cfg.get("enabled", False):
            return False
        if solver != "qp":
            # Greedy path retains ownership of this bar
            return False

        if ctx.bear_only:
            log.info("JointPortfolioQPTask: BEAR — defer to legacy SelectionJob")
            return False

        # ── Build per-asset arrays in stable order ────────────────────────
        cands_by_t   = {c.ticker: c for c in (ctx.candidates or [])}
        held_tickers = list((ctx.holdings or {}).keys())
        cand_tickers = [t for t in cands_by_t.keys() if t not in held_tickers]
        tickers = held_tickers + cand_tickers
        if not tickers:
            log.info("JointPortfolioQPTask: no tickers to optimize")
            return True

        n = len(tickers)
        portfolio_value = float(ctx.portfolio_value or 0.0)
        if portfolio_value <= 0.0:
            log.warning("JointPortfolioQPTask: portfolio_value=0 — skipping")
            return True

        prices = {t: float((ctx.prices or {}).get(t, 0.0) or 0.0)
                  for t in tickers}

        w_current = np.zeros(n)
        for i, t in enumerate(tickers):
            hs = (ctx.holdings or {}).get(t)
            if hs is not None:
                shares = float(getattr(hs, "shares", 0.0) or 0.0)
                px     = prices.get(t, 0.0)
                if px > 0.0:
                    w_current[i] = (shares * px) / portfolio_value

        # μ: prefer NGBoost μ, fall back to panel_score, then 0
        mu = np.zeros(n)
        for i, t in enumerate(tickers):
            src = cands_by_t.get(t) or (ctx.holdings or {}).get(t)
            if src is None:
                continue
            mu_i = getattr(src, "mu", None)
            if mu_i is None:
                mu_i = getattr(src, "panel_score", None)
            if mu_i is None:
                mu_i = 0.0
            try:
                mu[i] = float(mu_i)
            except (TypeError, ValueError):
                mu[i] = 0.0

        # σ: NGBoost σ, fallback to a regime-typical default
        default_sigma = float(joint_cfg.get("default_sigma", 0.05))
        sigma = np.full(n, default_sigma)
        for i, t in enumerate(tickers):
            src = cands_by_t.get(t) or (ctx.holdings or {}).get(t)
            if src is None:
                continue
            s = getattr(src, "sigma", None)
            if s is not None:
                try:
                    val = float(s)
                    if val > 0.0:
                        sigma[i] = val
                except (TypeError, ValueError):
                    pass

        # Stage 8 (2026-04-29): build full Σ from rolling correlation matrix.
        # Pre-fix QP used diagonal Σ (independence assumption) — tech mega-caps
        # got over-allocated since the optimizer didn't see they share factor risk.
        # When `qp_use_full_sigma=true` (default true post-2026-04-29), load the
        # cached correlation artifact and compute Σ_ij = ρ_ij × σ_i × σ_j.
        Sigma_full = None
        use_full_sigma = bool(joint_cfg.get("qp_use_full_sigma", True))
        if use_full_sigma and ctx.strategy_dir is not None:
            corr_path = (ctx.strategy_dir / "artifacts" /
                         "watchlist-correlation.json")
            if corr_path.exists():
                try:
                    import json as _json  # noqa: PLC0415
                    corr_data = _json.loads(corr_path.read_text())
                    Sigma_full = np.zeros((n, n))
                    for i, ti in enumerate(tickers):
                        for j, tj in enumerate(tickers):
                            if i == j:
                                Sigma_full[i, j] = sigma[i] ** 2
                                continue
                            rho = corr_data.get(ti, {}).get(tj)
                            if rho is None:
                                rho = corr_data.get(tj, {}).get(ti, 0.0)
                            try:
                                rho_f = float(rho)
                            except (TypeError, ValueError):
                                rho_f = 0.0
                            # Defensive bounds: empirical correlations in [-1, 1].
                            rho_f = max(-0.99, min(0.99, rho_f))
                            Sigma_full[i, j] = rho_f * sigma[i] * sigma[j]
                    # Add small diagonal regularizer for PSD safety
                    Sigma_full += 1e-8 * np.eye(n)
                except Exception as exc:
                    log.warning(
                        "JointPortfolioQPTask: failed to load correlation matrix "
                        "(%s) — falling back to diagonal Σ", exc,
                    )
                    Sigma_full = None
            else:
                log.info(
                    "JointPortfolioQPTask: correlation artifact not found at %s — "
                    "using diagonal Σ", corr_path,
                )

        # Stage 8 (2026-04-29): per-position tax-cost vector.
        # For each held position with unrealized gain, compute the per-unit-sell
        # tax drag: gain_pct × tax_rate × position_share_of_NAV. New positions
        # (zero held) have zero tax cost. Cost is in NAV-fraction units, matching
        # the rest of the QP objective.
        tax_cost_vec = np.zeros(n)
        if joint_cfg.get("qp_tax_aware", True):
            st_rate = float(joint_cfg.get("qp_tax_rate_st", 0.30))   # short-term federal+state proxy
            lt_rate = float(joint_cfg.get("qp_tax_rate_lt", 0.15))   # long-term
            lt_threshold_days = int(joint_cfg.get("qp_lt_threshold_days", 365))
            today = ctx.today
            for i, t in enumerate(tickers):
                hs = (ctx.holdings or {}).get(t)
                if hs is None or w_current[i] <= 0:
                    continue
                entry_price = float(getattr(hs, "entry_price", 0.0) or 0.0)
                entry_date  = getattr(hs, "entry_date", None)
                if entry_price <= 0 or entry_date is None:
                    continue
                price = prices.get(t, 0.0)
                if price <= 0:
                    continue
                gain_pct = (price - entry_price) / entry_price
                if gain_pct <= 0:
                    continue   # no taxable gain → no drag
                try:
                    days_held = (today - entry_date).days
                except Exception:
                    days_held = 0
                rate = lt_rate if days_held >= lt_threshold_days else st_rate
                # Tax drag per unit of weight sold = gain_pct × rate.
                # When Δw_i = -1 (theoretical full liquidation), realized gain
                # in NAV units = w_i × gain_pct, taxed at `rate`. Expressed as
                # a coefficient on |Δw_i|, the per-unit drag is gain_pct × rate.
                tax_cost_vec[i] = gain_pct * rate

        # Wash-sale: recently sold tickers cannot be re-bought
        wash_days  = int(ctx.config.get("wash_sale_days", 0))
        last_sells = ctx.last_sell_dates or {}
        wash_mask = np.zeros(n, dtype=bool)
        if wash_days > 0:
            today = ctx.today
            for i, t in enumerate(tickers):
                last_sell = last_sells.get(t)
                if last_sell is None:
                    continue
                try:
                    if isinstance(last_sell, str):
                        last_sell = datetime.date.fromisoformat(last_sell)
                    if (today - last_sell).days < wash_days:
                        wash_mask[i] = True
                except Exception:
                    continue

        # Per-position cap from regime params
        regime_params = (ctx.config.get("regime_params", {})
                                  .get(ctx.regime, {}))
        max_pos_pct = float(regime_params.get(
            "max_position_pct",
            ctx.config.get("max_position_pct", 0.20),
        ))
        # Confidence scaling — call the canonical helper so the floor
        # logic, NaN handling, and behaviour match JointActionTask exactly.
        # (audit fix QP-CONF-CONSISTENCY 2026-04-26: pre-fix, this used
        # an open-coded floor at 0.5 which DID match by accident, but a
        # different `floor` config or NaN behavior would have diverged.)
        from kernel.regime import confidence_to_size_multiplier  # noqa: PLC0415
        conf_scale = confidence_to_size_multiplier(
            getattr(ctx, "confidence", None),
        )
        w_upper_arr = np.full(n, max_pos_pct * conf_scale)

        cash_reserve = float(regime_params.get(
            "cash_reserve_pct",
            ctx.config.get("cash_reserve_pct", 0.0),
        ))

        # ── Solve ─────────────────────────────────────────────────────────
        gamma = float(joint_cfg.get("qp_risk_aversion", 3.0))
        kappa = float(joint_cfg.get("qp_cost_kappa",
                                     joint_cfg.get("fee_pct", 0.0005)))
        dw_max_arr = np.full(n, float(joint_cfg.get("qp_dw_max", 0.50)))
        # Stage 2/4/5 advanced knobs — defaults preserve Stage-1 behavior
        signal_decay     = float(joint_cfg.get("qp_signal_decay", 0.0))
        robust_mu_kappa  = float(joint_cfg.get("qp_robust_mu_kappa", 0.0))
        # Drawdown — read from regime_state. May be either a RegimeState
        # dataclass (from kernel.regime) or a plain dict in some test paths.
        # Audit fix QP-REGIME-STATE-DUCK (2026-04-26): previously used
        # rs.get() unconditionally → AttributeError on RegimeState
        # instance, crashing every sim run that activated QP.
        rs = getattr(ctx, "regime_state", None)
        if rs is None:
            portfolio_dd = 0.0
        elif isinstance(rs, dict):
            portfolio_dd = float(rs.get("drawdown", 0.0) or 0.0)
        else:
            portfolio_dd = float(getattr(rs, "drawdown", 0.0) or 0.0)
        dd_limit = float(joint_cfg.get(
            "qp_drawdown_limit",
            ctx.config.get("regime", {}).get("drawdown_halt_pct", 0.20),
        ))

        # Stage 9 (2026-04-29): turnover hard cap. Default 0.30 = at most
        # 30% of NAV traded per bar — prevents the optimizer from churning
        # in response to small μ fluctuations.
        turnover_max = joint_cfg.get("qp_turnover_max", 0.30)
        if turnover_max is not None:
            try:
                turnover_max = float(turnover_max) if turnover_max else None
            except (TypeError, ValueError):
                turnover_max = None

        sol = solve_portfolio_qp(
            w_current      = w_current,
            mu             = mu,
            sigma          = sigma,
            Sigma          = Sigma_full,
            risk_aversion  = gamma,
            cost_kappa     = kappa,
            cash_reserve   = cash_reserve,
            w_upper        = w_upper_arr,
            w_lower        = 0.0,
            dw_max         = dw_max_arr,
            wash_sale_mask = wash_mask,
            signal_decay     = signal_decay,
            drawdown         = portfolio_dd,
            drawdown_limit   = dd_limit,
            robust_mu_kappa  = robust_mu_kappa,
            tax_cost_per_sell = tax_cost_vec,
            turnover_max     = turnover_max,
        )
        if sol.status != "optimal":
            log.warning(
                "JointPortfolioQPTask: solver returned status=%s — "
                "deferring to greedy", sol.status,
            )
            return False

        # ── Translate Δw → orders / exits ─────────────────────────────────
        from kernel.exits import ExitSignal  # noqa: PLC0415

        # Use a smaller threshold than the no-trade-band gate would imply,
        # since the QP itself produced these Δws (already past optimization).
        min_dw_pct = float(joint_cfg.get("qp_min_dw_pct", 0.005))   # 0.5% NAV
        n_buys = 0
        n_sells = 0
        for i, t in enumerate(tickers):
            dw = float(sol.delta_w[i])
            if abs(dw) < min_dw_pct:
                continue
            px = prices.get(t, 0.0)
            if px <= 0.0:
                continue
            invest_target = abs(dw) * portfolio_value
            shares = int(invest_target / px)
            if shares <= 0:
                continue
            if dw > 0:
                # BUY — top up or initiate
                ctx.orders.append({
                    "ticker":     t,
                    "shares":     shares,
                    "price":      px,
                    "invest":     shares * px,
                    "target_pct": float(sol.target_w[i]),
                    "rank_score": getattr(cands_by_t.get(t), "rank_score", None),
                    "mu":         float(mu[i]) if mu[i] != 0.0 else None,
                    "sigma":      float(sigma[i]) if sigma[i] > 0 else None,
                    "source":     "qp",
                })
                n_buys += 1
                log.info(
                    "QP_BUY  %-6s  Δw=%+.4f  shares=%d  px=%.2f  invest=$%.0f",
                    t, dw, shares, px, shares * px,
                )
            else:
                # SELL — full close if target weight ≈ 0, else trim
                hs = (ctx.holdings or {}).get(t)
                if hs is None:
                    continue
                held_shares = int(getattr(hs, "shares", 0) or 0)
                quantity = min(shares, held_shares)
                if quantity <= 0:
                    continue
                exit_type = ("qp_sell" if sol.target_w[i] > 1e-4
                              else "qp_close")
                sig = ExitSignal(
                    should_exit=True,
                    exit_type=exit_type,
                    quantity=float(quantity),
                    reason=f"qp_dw={dw:+.4f}",
                )
                ctx.exits.append((t, sig))
                n_sells += 1
                log.info(
                    "QP_SELL %-6s  Δw=%+.4f  shares=%d  reason=%s",
                    t, dw, quantity, exit_type,
                )

        ctx.counters["qp_buys"]  = ctx.counters.get("qp_buys", 0) + n_buys
        ctx.counters["qp_sells"] = ctx.counters.get("qp_sells", 0) + n_sells
        log.info(
            "JointPortfolioQPTask: solved n=%d  buys=%d  sells=%d  "
            "objective=%.6f  iter=%d",
            n, n_buys, n_sells, sol.objective, sol.n_iter,
        )
        return True
