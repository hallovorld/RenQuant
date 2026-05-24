"""BenchmarkSleeveTask — benchmark-aware core beta sleeve.

The panel model is an alpha sleeve: it decides which names deserve active
risk. It should not also be responsible for keeping the portfolio's market
beta near the benchmark. The 2026-05-23 WF audit found the opposite failure:
positive active-trade Sharpe, but ~94% average cash and beta ~0.06, so the
portfolio lost mainly by cash drag in risk-on regimes.

This task adds an opt-in, benchmark-aware core sleeve:

  target benchmark value = target_total_exposure(regime) - active_alpha_value

It runs after alpha selection/QP. It only uses residual cash for buys, never
forces alpha sells to fund beta, and it logs orders/exits with their own
attribution. Alpha admission remains separate from portfolio beta sizing.

References for the design:
  * Grinold & Kahn, Active Portfolio Management: separate benchmark exposure
    from active bets and evaluate active risk versus a benchmark.
  * Core-satellite portfolio construction: low-cost benchmark core plus
    active satellites, avoiding benchmark-relative cash drag when alpha
    capacity is intermittent.

Default is disabled; enabling requires an explicit config block.
"""
from __future__ import annotations

import logging
import math
from typing import Any

from .context import InferenceContext
from .order_attribution import stamp_order_attribution
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.benchmark_sleeve")


def _config(obj: Any) -> dict:
    if isinstance(obj, dict):
        return obj
    cfg = getattr(obj, "config", None)
    return cfg if isinstance(cfg, dict) else {}


def benchmark_sleeve_config(obj: Any) -> dict:
    cfg = _config(obj)
    sleeve = ((cfg.get("portfolio") or {}).get("benchmark_sleeve") or {})
    return sleeve if isinstance(sleeve, dict) else {}


def execution_allows_unsettled_buying_power(obj: Any) -> bool:
    """Whether executed sell proceeds may fund same-bar/new buys.

    Live Alpaca uses ``non_marginable_buying_power`` for RenQuant buys:
    executed sell proceeds can replenish buy budget without using 2x/4x
    margin buying power. Cash-account simulations can set
    ``execution.buying_power_mode="settled_cash"``; in that mode the
    benchmark sleeve must not present future sell proceeds as alpha funding.
    """
    cfg = _config(obj)
    mode = str(
        ((cfg.get("execution") or {}).get(
            "buying_power_mode", "non_marginable_buying_power",
        ))
    ).strip().lower()
    return mode in {
        "non_marginable_buying_power",
        "cash_plus_unsettled",
        "unsettled",
    }


def is_benchmark_sleeve_enabled(obj: Any) -> bool:
    return bool(benchmark_sleeve_config(obj).get("enabled", False))


def benchmark_sleeve_ticker(obj: Any, *, require_enabled: bool = True) -> str | None:
    cfg = _config(obj)
    sleeve = benchmark_sleeve_config(obj)
    if require_enabled and not sleeve.get("enabled", False):
        return None
    ticker = str(sleeve.get("ticker") or cfg.get("benchmark") or "SPY").strip().upper()
    return ticker or None


def exclude_benchmark_sleeve_from_alpha(obj: Any) -> bool:
    sleeve = benchmark_sleeve_config(obj)
    return bool(sleeve.get("enabled", False)) and bool(
        sleeve.get("exclude_from_alpha_pipeline", True)
    )


def decision_trace_tickers(config: dict) -> list[str]:
    """Watchlist plus the benchmark sleeve ticker when that sleeve is active."""
    out = list(config.get("watchlist", []) or [])
    ticker = benchmark_sleeve_ticker(config)
    if ticker and ticker not in out:
        out.append(ticker)
    return out


def benchmark_sleeve_alpha_funding_capacity(ctx: InferenceContext) -> float:
    """Return sleeve dollars QP may treat as alpha-funding liquidity.

    Core-satellite construction should let sufficiently qualified alpha
    satellite trades displace the passive benchmark core. Without this, a
    fully invested core sleeve starves the QP of cash and converts the active
    model into a no-op after day one. Disabled unless explicitly configured.
    """
    sleeve = benchmark_sleeve_config(ctx)
    if not (sleeve.get("enabled", False) and sleeve.get("fund_alpha_from_sleeve", False)):
        return 0.0
    if not execution_allows_unsettled_buying_power(ctx):
        return 0.0
    ticker = benchmark_sleeve_ticker(ctx)
    if not ticker:
        return 0.0
    nav = _finite_float(getattr(ctx, "portfolio_value", 0.0), 0.0)
    if nav <= 0:
        return 0.0
    budget = _finite_float(
        sleeve.get("alpha_funding_budget_pct", sleeve.get("alpha_budget_pct")),
        0.0,
    )
    if budget <= 0:
        return 0.0
    return min(_position_value(ctx, ticker), nav * min(max(budget, 0.0), 1.0))


def benchmark_sleeve_cash_reserve_credit(ctx: InferenceContext) -> float:
    """Return NAV fraction of cash reserve satisfied by the liquid sleeve."""
    sleeve = benchmark_sleeve_config(ctx)
    if not (sleeve.get("enabled", False) and sleeve.get("sleeve_counts_as_cash_reserve", False)):
        return 0.0
    ticker = benchmark_sleeve_ticker(ctx)
    if not ticker:
        return 0.0
    nav = _finite_float(getattr(ctx, "portfolio_value", 0.0), 0.0)
    if nav <= 0:
        return 0.0
    return min(max(_position_value(ctx, ticker) / nav, 0.0), 1.0)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _buy_cost_multiplier(config: dict) -> float:
    exec_cfg = (config or {}).get("execution", {}) or {}
    if bool(exec_cfg.get("legacy_no_fees", False)):
        return 1.0
    if not bool(exec_cfg.get("enabled", True)):
        return 1.0
    bps = (
        _finite_float(exec_cfg.get("half_spread_bps"), 2.0)
        + _finite_float(exec_cfg.get("commission_bps"), 0.0)
        + _finite_float(exec_cfg.get("qp_buy_cash_buffer_bps"), 1.0)
    )
    return 1.0 + max(0.0, bps) / 10000.0


def _sell_proceeds_multiplier(config: dict) -> float:
    exec_cfg = (config or {}).get("execution", {}) or {}
    if bool(exec_cfg.get("legacy_no_fees", False)):
        return 1.0
    if not bool(exec_cfg.get("enabled", True)):
        return 1.0
    bps = (
        _finite_float(exec_cfg.get("half_spread_bps"), 2.0)
        + _finite_float(exec_cfg.get("commission_bps"), 0.0)
        + _finite_float(exec_cfg.get("sec_fee_rate"), 27.0e-6) * 10000.0
        + _finite_float(exec_cfg.get("qp_buy_cash_buffer_bps"), 1.0)
    )
    return max(0.0, 1.0 - max(0.0, bps) / 10000.0)


def _position_value(ctx: InferenceContext, ticker: str) -> float:
    hs = (getattr(ctx, "holdings", None) or {}).get(ticker)
    if hs is None:
        return 0.0
    shares = _finite_float(getattr(hs, "shares", 0.0), 0.0)
    price = _finite_float((getattr(ctx, "prices", None) or {}).get(ticker), 0.0)
    if shares <= 0 or price <= 0:
        return 0.0
    return shares * price


def _pending_buy_invest(ctx: InferenceContext, *, exclude_ticker: str | None = None) -> float:
    total = 0.0
    for order in getattr(ctx, "orders", []) or []:
        if not isinstance(order, dict):
            continue
        ticker = str(order.get("ticker") or "")
        if exclude_ticker is not None and ticker == exclude_ticker:
            continue
        side = str(order.get("side") or order.get("action") or "BUY").upper()
        if side == "SELL":
            continue
        invest = _finite_float(order.get("invest"), float("nan"))
        if math.isfinite(invest) and invest > 0:
            total += invest
            continue
        shares = _finite_float(order.get("shares"), 0.0)
        price = _finite_float(order.get("price"), 0.0)
        if shares > 0 and price > 0:
            total += shares * price
    return total


def _exit_value(ctx: InferenceContext, ticker: str) -> float:
    hs = (getattr(ctx, "holdings", None) or {}).get(ticker)
    if hs is None:
        return 0.0
    current_shares = _finite_float(getattr(hs, "shares", 0.0), 0.0)
    price = _finite_float((getattr(ctx, "prices", None) or {}).get(ticker), 0.0)
    if current_shares <= 0 or price <= 0:
        return 0.0
    value = 0.0
    for ex_ticker, sig in getattr(ctx, "exits", []) or []:
        if ex_ticker != ticker:
            continue
        qty = getattr(sig, "quantity", None)
        if qty is not None:
            qty_f = _finite_float(qty, float("nan"))
            if math.isfinite(qty_f) and 0 < qty_f < current_shares:
                value += qty_f * price
                continue
        value += current_shares * price
    return min(value, current_shares * price)


def _active_alpha_exposure(ctx: InferenceContext, sleeve_ticker: str) -> float:
    exposure = 0.0
    for ticker in (getattr(ctx, "holdings", None) or {}):
        if ticker == sleeve_ticker:
            continue
        exposure += max(_position_value(ctx, ticker) - _exit_value(ctx, ticker), 0.0)
    exposure += _pending_buy_invest(ctx, exclude_ticker=sleeve_ticker)
    return exposure


def _existing_exit_tickers(ctx: InferenceContext) -> set[str]:
    out: set[str] = set()
    for item in getattr(ctx, "exits", []) or []:
        if isinstance(item, tuple) and item:
            out.add(str(item[0]))
            continue
        ticker = getattr(item, "ticker", None)
        if ticker:
            out.add(str(ticker))
    return out


def _existing_buy_tickers(ctx: InferenceContext) -> set[str]:
    return {
        str(o.get("ticker"))
        for o in (getattr(ctx, "orders", []) or [])
        if isinstance(o, dict) and o.get("ticker")
    }


def _target_exposure_for_regime(ctx: InferenceContext, sleeve: dict) -> float:
    target = sleeve.get("target_exposure")
    by_regime = sleeve.get("target_exposure_by_regime")
    if isinstance(by_regime, dict):
        for key in (getattr(ctx, "regime", None), getattr(ctx, "spy_regime", None)):
            if key in by_regime:
                target = by_regime[key]
                break
    target_f = _finite_float(target, 0.0)
    return min(max(target_f, 0.0), 1.0)


def solve_benchmark_sleeve_target(
    *,
    current_sleeve_weight: float,
    active_alpha_weight: float,
    target_total_exposure: float,
    max_sleeve_weight: float,
    available_cash_weight: float,
    turnover_penalty: float = 0.001,
) -> dict[str, Any]:
    """Solve the core-sleeve target with SciPy's HiGHS LP solver.

    Objective:
      minimize |active_alpha_weight + sleeve_weight - target_total_exposure|
             + λ |sleeve_weight - current_sleeve_weight|

    Subject to:
      0 <= sleeve_weight <= max_sleeve_weight
      sleeve_weight <= current_sleeve_weight + available_cash_weight

    This is a deliberately small mature-library LP, not a hand-rolled
    optimizer. If SciPy is unavailable or the solve fails while the feature is
    enabled, callers should fail closed instead of silently reverting to a
    weaker arithmetic rule.
    """
    cur = min(max(_finite_float(current_sleeve_weight), 0.0), 1.0)
    active = min(max(_finite_float(active_alpha_weight), 0.0), 1.0)
    target = min(max(_finite_float(target_total_exposure), 0.0), 1.0)
    max_w = min(max(_finite_float(max_sleeve_weight, 1.0), 0.0), 1.0)
    cash_w = min(max(_finite_float(available_cash_weight), 0.0), 1.0)
    upper = max(0.0, min(max_w, cur + cash_w))
    penalty = max(_finite_float(turnover_penalty, 0.001), 0.0)
    if upper <= 0:
        return {
            "target_weight": 0.0,
            "solver_status": "bounded_zero",
            "solver": "scipy_linprog_highs",
            "tracking_error_abs": abs(active - target),
        }

    try:
        from scipy.optimize import linprog  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - environment failure
        raise RuntimeError(
            "BenchmarkSleeveTask requires scipy.optimize.linprog when enabled"
        ) from exc

    # Vars: [x, e_plus, e_minus, t_plus, t_minus]
    # active + x - target = e_plus - e_minus
    # x - cur = t_plus - t_minus
    c = [0.0, 1.0, 1.0, penalty, penalty]
    a_eq = [
        [1.0, -1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, -1.0, 1.0],
    ]
    b_eq = [target - active, cur]
    bounds = [(0.0, upper), (0.0, None), (0.0, None), (0.0, None), (0.0, None)]
    res = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(
            "BenchmarkSleeveTask scipy linprog failed: "
            f"status={res.status} message={res.message}"
        )
    x = min(max(float(res.x[0]), 0.0), upper)
    return {
        "target_weight": x,
        "solver_status": str(res.message),
        "solver": "scipy_linprog_highs",
        "tracking_error_abs": abs(active + x - target),
        "turnover_abs": abs(x - cur),
        "upper_weight": upper,
    }


def _stamp_block(ctx: InferenceContext, ticker: str, reason: str) -> None:
    blocked = getattr(ctx, "_blocked_by_ticker", None)
    if blocked is None:
        blocked = {}
        ctx._blocked_by_ticker = blocked  # noqa: SLF001
    blocked.setdefault(ticker, reason)


class BenchmarkSleeveTask(Task):
    """Rebalance the benchmark sleeve after alpha orders have been decided."""

    name = "BenchmarkSleeveTask"

    def run(self, ctx: InferenceContext) -> bool | None:
        sleeve = benchmark_sleeve_config(ctx)
        if not sleeve.get("enabled", False):
            return None
        ticker = benchmark_sleeve_ticker(ctx)
        if not ticker:
            return None

        nav = _finite_float(getattr(ctx, "portfolio_value", 0.0), 0.0)
        cash = _finite_float(getattr(ctx, "cash", 0.0), 0.0)
        price = _finite_float((getattr(ctx, "prices", None) or {}).get(ticker), 0.0)
        if nav <= 0 or price <= 0:
            _stamp_block(ctx, ticker, "benchmark_sleeve_missing_price_or_nav")
            ctx.counters["benchmark_sleeve_missing_price"] = (
                ctx.counters.get("benchmark_sleeve_missing_price", 0) + 1
            )
            return None

        current_value = _position_value(ctx, ticker)
        alpha_value = _active_alpha_exposure(ctx, ticker)
        target_exposure = _target_exposure_for_regime(ctx, sleeve)
        max_sleeve_weight = min(
            max(_finite_float(sleeve.get("max_sleeve_pct"), 1.0), 0.0), 1.0,
        )
        pending = _pending_buy_invest(ctx, exclude_ticker=ticker)
        alpha_funding_gap = 0.0
        if (
            bool(sleeve.get("fund_alpha_from_sleeve", False))
            and execution_allows_unsettled_buying_power(ctx)
        ):
            alpha_funding_gap = max(pending - cash, 0.0)
        available_cash = max(cash - pending, 0.0)
        solve = solve_benchmark_sleeve_target(
            current_sleeve_weight=current_value / nav,
            active_alpha_weight=alpha_value / nav,
            target_total_exposure=target_exposure,
            max_sleeve_weight=max_sleeve_weight,
            available_cash_weight=available_cash / nav,
            turnover_penalty=_finite_float(sleeve.get("turnover_penalty"), 0.001),
        )
        target_value = float(solve["target_weight"]) * nav

        band = max(_finite_float(sleeve.get("rebalance_band_pct"), 0.05), 0.0) * nav
        min_trade = max(_finite_float(sleeve.get("min_trade_pct"), 0.02), 0.0) * nav
        threshold = max(band, min_trade)
        exiting = _existing_exit_tickers(ctx)
        buying = _existing_buy_tickers(ctx)
        if ticker in exiting or ticker in buying:
            _stamp_block(ctx, ticker, "benchmark_sleeve_already_touched")
            return None

        state = {
            "ticker": ticker,
            "current_value": current_value,
            "alpha_value": alpha_value,
            "target_total_exposure_value": target_exposure * nav,
            "target_sleeve_value": target_value,
            "cash": cash,
            "price": price,
            "threshold_value": threshold,
            "alpha_funding_gap_value": alpha_funding_gap,
            "optimizer": solve,
        }
        ctx._benchmark_sleeve_state = state  # noqa: SLF001

        delta = target_value - current_value
        if alpha_funding_gap > 0.0:
            reduction = max(alpha_funding_gap, -delta if delta < 0 else 0.0)
            return self._emit_sell(ctx, ticker, price, reduction, nav, current_value, state)
        if abs(delta) <= threshold:
            ctx.counters["benchmark_sleeve_noop"] = (
                ctx.counters.get("benchmark_sleeve_noop", 0) + 1
            )
            return None
        if delta > 0:
            return self._emit_buy(ctx, ticker, price, delta, cash, nav, current_value, sleeve, state)
        return self._emit_sell(ctx, ticker, price, -delta, nav, current_value, state)

    def _emit_buy(
        self,
        ctx: InferenceContext,
        ticker: str,
        price: float,
        desired_delta: float,
        cash: float,
        nav: float,
        current_value: float,
        sleeve: dict,
        state: dict,
    ) -> None:
        if bool(sleeve.get("respect_buy_gates", True)) and (
            bool(getattr(ctx, "buy_blocked", False))
            or bool(getattr(ctx, "skip_buys", False))
            or bool(getattr(ctx, "bear_only", False))
        ):
            _stamp_block(ctx, ticker, "benchmark_sleeve_buy_gate")
            ctx.counters["benchmark_sleeve_buy_gated"] = (
                ctx.counters.get("benchmark_sleeve_buy_gated", 0) + 1
            )
            return None

        pending = _pending_buy_invest(ctx, exclude_ticker=ticker)
        available = max(cash - pending, 0.0)
        buy_value = min(desired_delta, available)
        shares = int(buy_value // (price * _buy_cost_multiplier(ctx.config or {})))
        if shares < 1:
            _stamp_block(ctx, ticker, "benchmark_sleeve_insufficient_cash")
            ctx.counters["benchmark_sleeve_insufficient_cash"] = (
                ctx.counters.get("benchmark_sleeve_insufficient_cash", 0) + 1
            )
            return None

        invest = shares * price
        target_pct = (current_value + invest) / nav if nav > 0 else 0.0
        order = stamp_order_attribution({
            "ticker": ticker,
            "shares": float(shares),
            "price": price,
            "invest": invest,
            "target_pct": target_pct,
            "regime": getattr(ctx, "regime", None),
            "confidence": getattr(ctx, "confidence", None),
            "conviction": 1.0,
            "sigma_mult": 1.0,
            "rank_score": None,
            "rs_score": 0.0,
            "panel_score": None,
            "sigma": None,
            "mu": None,
            "kelly_target_pct": None,
            "detail": "benchmark_core_sleeve",
            "order_type": "BENCHMARK_SLEEVE_BUY",
        }, ctx=ctx, source_job="BenchmarkSleeveJob",
            source_task="BenchmarkSleeveTask",
            acceptance_reason="residual_cash_to_benchmark_core",
            decision_inputs={
                **state,
                "desired_delta_value": desired_delta,
                "pending_buy_cash": pending,
                "available_cash": available,
            })
        ctx.orders.append(order)
        ctx.counters["benchmark_sleeve_buys"] = (
            ctx.counters.get("benchmark_sleeve_buys", 0) + 1
        )
        log.info(
            "BenchmarkSleeveTask: BUY %s x%d target=%.1f%% "
            "(alpha=%.1f%% sleeve_before=%.1f%%)",
            ticker, shares, target_pct * 100,
            state["alpha_value"] / nav * 100,
            current_value / nav * 100,
        )

    def _emit_sell(
        self,
        ctx: InferenceContext,
        ticker: str,
        price: float,
        desired_reduction: float,
        nav: float,
        current_value: float,
        state: dict,
    ) -> None:
        hs = (getattr(ctx, "holdings", None) or {}).get(ticker)
        current_shares = _finite_float(getattr(hs, "shares", 0.0), 0.0) if hs else 0.0
        if current_shares <= 0:
            return None
        if _finite_float(state.get("alpha_funding_gap_value"), 0.0) > 0.0:
            unit_proceeds = price * _sell_proceeds_multiplier(ctx.config or {})
            if unit_proceeds <= 0:
                return None
            shares = min(int(math.ceil(desired_reduction / unit_proceeds)), int(current_shares))
        else:
            shares = min(int(desired_reduction // price), int(current_shares))
        if shares < 1:
            return None
        from kernel.exits import ExitSignal  # noqa: PLC0415
        sig = ExitSignal(
            should_exit=True,
            reason=(
                "benchmark sleeve rebalance "
                f"current={current_value / nav:.1%} "
                f"target={state['target_sleeve_value'] / nav:.1%}"
            ),
            exit_type="benchmark_sleeve_rebalance",
            quantity=float(shares) if shares < current_shares else None,
        )
        sig.source_job = "BenchmarkSleeveJob"
        sig.source_task = "BenchmarkSleeveTask"
        sig.order_source = "BenchmarkSleeveJob.BenchmarkSleeveTask"
        sig.source = sig.order_source
        ctx.exits.append((ticker, sig))
        ctx.counters["benchmark_sleeve_sells"] = (
            ctx.counters.get("benchmark_sleeve_sells", 0) + 1
        )
        log.info(
            "BenchmarkSleeveTask: SELL %s x%d "
            "(current=%.1f%% target=%.1f%%)",
            ticker, shares,
            current_value / nav * 100,
            state["target_sleeve_value"] / nav * 100,
        )
