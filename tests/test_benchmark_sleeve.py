"""BenchmarkSleeveTask tests.

The sleeve is a benchmark-aware beta overlay, not an alpha selector. These
tests pin three contracts:
  * target weight comes from a mature LP solver (SciPy/HiGHS), not a hidden
    heuristic;
  * buy/sell emissions are separately attributed from alpha/QP;
  * the benchmark sleeve is excluded from the alpha QP universe.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.exits import HoldingState  # noqa: E402
from kernel.pipeline.context import InferenceContext  # noqa: E402
from kernel.portfolio_qp.task_joint_qp import JointPortfolioQPTask  # noqa: E402
from kernel.pipeline.task_benchmark_sleeve import (  # noqa: E402
    BenchmarkSleeveTask,
    benchmark_sleeve_alpha_funding_capacity,
    benchmark_sleeve_cash_reserve_credit,
    decision_trace_tickers,
    solve_benchmark_sleeve_target,
)
from kernel.pipeline.pp_inference import _sell_universe  # noqa: E402
from kernel.portfolio_qp.job_qp import _BuildSourceMapTask  # noqa: E402


def _holding(shares: float, price: float = 100.0) -> HoldingState:
    return HoldingState(
        entry_price=price,
        entry_date=dt.date(2026, 1, 5),
        high_watermark=price,
        shares=shares,
    )


def _cfg(
    *,
    enabled: bool = True,
    target_by_regime: dict | None = None,
    fund_alpha: bool = False,
) -> dict:
    return {
        "benchmark": "SPY",
        "watchlist": ["AAA", "BBB"],
        "portfolio": {
            "benchmark_sleeve": {
                "enabled": enabled,
                "ticker": "SPY",
                "target_exposure_by_regime": target_by_regime or {
                    "BULL_CALM": 1.0,
                    "BULL_VOLATILE": 0.5,
                    "CHOPPY": 0.0,
                    "BEAR": 0.0,
                },
                "max_sleeve_pct": 1.0,
                "rebalance_band_pct": 0.0,
                "min_trade_pct": 0.0,
                "turnover_penalty": 0.001,
                "respect_buy_gates": True,
                "exclude_from_alpha_pipeline": True,
                "fund_alpha_from_sleeve": fund_alpha,
                "alpha_funding_budget_pct": 0.15,
                "sleeve_counts_as_cash_reserve": fund_alpha,
            }
        },
    }


def _ctx(
    *,
    cfg: dict | None = None,
    holdings: dict | None = None,
    orders: list | None = None,
    exits: list | None = None,
    prices: dict | None = None,
    portfolio_value: float = 100_000.0,
    cash: float = 100_000.0,
    buy_blocked: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        config=cfg or _cfg(),
        today=dt.date(2026, 5, 22),
        regime="BULL_CALM",
        confidence=1.0,
        portfolio_value=portfolio_value,
        cash=cash,
        prices=prices or {"SPY": 500.0},
        holdings=holdings or {},
        orders=list(orders or []),
        exits=list(exits or []),
        counters={},
        buy_blocked=buy_blocked,
        skip_buys=False,
        bear_only=False,
    )


def test_scipy_lp_solver_targets_residual_benchmark_exposure():
    out = solve_benchmark_sleeve_target(
        current_sleeve_weight=0.0,
        active_alpha_weight=0.25,
        target_total_exposure=1.0,
        max_sleeve_weight=1.0,
        available_cash_weight=1.0,
    )
    assert out["solver"] == "scipy_linprog_highs"
    assert out["target_weight"] == pytest.approx(0.75)
    assert out["tracking_error_abs"] == pytest.approx(0.0, abs=1e-9)


def test_disabled_is_noop():
    ctx = _ctx(cfg=_cfg(enabled=False))
    BenchmarkSleeveTask().run(ctx)
    assert ctx.orders == []
    assert ctx.exits == []


def test_emits_buy_for_unallocated_core_exposure():
    ctx = _ctx(prices={"SPY": 500.0}, cash=100_000.0)
    BenchmarkSleeveTask().run(ctx)
    assert len(ctx.orders) == 1
    order = ctx.orders[0]
    assert order["ticker"] == "SPY"
    assert order["shares"] == 200
    assert order["order_type"] == "BENCHMARK_SLEEVE_BUY"
    assert order["source_job"] == "BenchmarkSleeveJob"
    assert order["decision_inputs"]["optimizer"]["solver"] == "scipy_linprog_highs"


def test_pending_alpha_buy_reduces_sleeve_target():
    ctx = _ctx(
        prices={"SPY": 400.0},
        cash=100_000.0,
        orders=[{"ticker": "AAA", "shares": 200, "price": 100.0, "invest": 20_000.0}],
    )
    BenchmarkSleeveTask().run(ctx)
    spy_order = [o for o in ctx.orders if o.get("ticker") == "SPY"][0]
    assert spy_order["invest"] == pytest.approx(80_000.0)
    assert spy_order["shares"] == 200


def test_buy_gate_blocks_new_sleeve_buy():
    ctx = _ctx(buy_blocked=True)
    BenchmarkSleeveTask().run(ctx)
    assert ctx.orders == []
    assert ctx.counters["benchmark_sleeve_buy_gated"] == 1
    assert ctx._blocked_by_ticker["SPY"] == "benchmark_sleeve_buy_gate"


def test_emits_sell_when_regime_target_is_zero():
    cfg = _cfg(target_by_regime={"BULL_CALM": 0.0})
    ctx = _ctx(
        cfg=cfg,
        holdings={"SPY": _holding(100, 500.0)},
        prices={"SPY": 500.0},
        portfolio_value=100_000.0,
        cash=50_000.0,
    )
    BenchmarkSleeveTask().run(ctx)
    assert len(ctx.exits) == 1
    ticker, sig = ctx.exits[0]
    assert ticker == "SPY"
    assert sig.exit_type == "benchmark_sleeve_rebalance"
    assert sig.quantity is None
    assert sig.source_job == "BenchmarkSleeveJob"


def test_decision_trace_includes_sleeve_ticker_when_enabled():
    assert decision_trace_tickers(_cfg()) == ["AAA", "BBB", "SPY"]
    assert decision_trace_tickers(_cfg(enabled=False)) == ["AAA", "BBB"]


def test_qp_source_map_excludes_benchmark_sleeve_from_alpha_universe():
    ctx = _ctx(
        holdings={"SPY": _holding(10, 500.0)},
        prices={"SPY": 500.0, "AAA": 100.0},
        cash=95_000.0,
    )
    ctx.candidates = [
        SimpleNamespace(ticker="SPY", mu=0.9, sigma=0.1, panel_score=2.0),
        SimpleNamespace(ticker="AAA", mu=0.4, sigma=0.2, panel_score=1.0),
    ]
    ctx._qp_tickers = ["SPY", "AAA"]
    _BuildSourceMapTask().run(ctx)
    assert "SPY" not in ctx._qp_mu_source_map
    assert ctx._qp_tickers == ["AAA"]
    assert ctx._blocked_by_ticker["SPY"] == "benchmark_sleeve_excluded_from_alpha_qp"


def test_sell_universe_excludes_benchmark_sleeve_from_alpha_sell_chain():
    ctx = _ctx(
        holdings={"SPY": _holding(10, 500.0), "AAA": _holding(5, 100.0)},
        prices={"SPY": 500.0, "AAA": 100.0},
        cash=94_500.0,
    )

    assert _sell_universe(ctx) == ["AAA"]
    assert ctx._blocked_by_ticker["SPY"] == "benchmark_sleeve_alpha_sell_exempt"
    assert ctx.counters["benchmark_sleeve_alpha_sell_exempt"] == 1


def test_alpha_funding_capacity_uses_explicit_sleeve_budget():
    ctx = _ctx(
        cfg=_cfg(fund_alpha=True),
        holdings={"SPY": _holding(100, 500.0)},
        prices={"SPY": 500.0},
        portfolio_value=100_000.0,
        cash=0.0,
    )

    assert benchmark_sleeve_alpha_funding_capacity(ctx) == pytest.approx(15_000.0)
    assert benchmark_sleeve_cash_reserve_credit(ctx) == pytest.approx(0.50)


def test_qp_alpha_can_displace_benchmark_sleeve_when_enabled():
    cfg = {
        **_cfg(fund_alpha=True),
        "sector_map": {"AAA": "tech"},
        "max_positions_per_sector": 3,
        "regime_params": {
            "BULL_CALM": {
                "max_position_pct": 0.20,
                "cash_reserve_pct": 0.10,
                "max_concurrent_positions": 8,
            },
        },
        "rotation": {
            "joint_actions": {
                "enabled": True,
                "solver": "qp",
                "qp_risk_aversion": 3.0,
                "qp_cost_kappa": 0.0001,
                "qp_dw_max": 0.50,
                "qp_min_dw_pct": 0.005,
                "qp_signal_decay": 0.0,
                "qp_drawdown_limit": 0.20,
                "default_sigma": 0.05,
                "qp_sector_cap_enabled": False,
                "qp_correlation_cap_enabled": False,
            },
        },
    }
    ctx = InferenceContext(config=cfg, today=dt.date(2026, 5, 22))
    ctx.regime = "BULL_CALM"
    ctx.confidence = 1.0
    ctx.portfolio_value = 100_000.0
    ctx.cash = 0.0
    ctx.prices = {"SPY": 500.0, "AAA": 100.0}
    ctx.holdings = {"SPY": _holding(200, 500.0)}
    ctx.candidates = [
        SimpleNamespace(
            ticker="AAA",
            mu=0.06,
            sigma=0.12,
            panel_score=0.8,
            rank_score=0.75,
            rs_score=0.1,
            kelly_target_pct=0.15,
        ),
    ]
    ctx.corr_matrix = {"AAA": {}}

    JointPortfolioQPTask().run(ctx)

    assert ctx._qp_alpha_funding_cash == pytest.approx(15_000.0)
    assert ctx._qp_cash_reserve_effective == pytest.approx(0.0)
    assert ctx.orders, "QP should buy alpha by treating sleeve as funding liquidity"
    assert ctx.orders[0]["ticker"] == "AAA"

    BenchmarkSleeveTask().run(ctx)

    assert ctx.exits, "Benchmark sleeve should sell SPY to fund the alpha buy"
    ticker, sig = ctx.exits[0]
    assert ticker == "SPY"
    assert sig.exit_type == "benchmark_sleeve_rebalance"
    assert sig.source_job == "BenchmarkSleeveJob"


def test_alpha_funding_sell_bypasses_rebalance_band():
    cfg = _cfg(fund_alpha=True)
    cfg["portfolio"]["benchmark_sleeve"]["rebalance_band_pct"] = 0.05
    cfg["portfolio"]["benchmark_sleeve"]["min_trade_pct"] = 0.01
    ctx = _ctx(
        cfg=cfg,
        holdings={"SPY": _holding(200, 500.0)},
        orders=[{"ticker": "AAA", "side": "BUY", "invest": 1_500.0}],
        prices={"SPY": 500.0, "AAA": 100.0},
        portfolio_value=100_000.0,
        cash=0.0,
    )

    BenchmarkSleeveTask().run(ctx)

    assert ctx.exits, "Funding sells must not be suppressed by rebalance bands"
    ticker, sig = ctx.exits[0]
    assert ticker == "SPY"
    assert sig.exit_type == "benchmark_sleeve_rebalance"
    assert sig.quantity == pytest.approx(3.0)
    assert ctx._benchmark_sleeve_state["alpha_funding_gap_value"] == pytest.approx(1_500.0)


def test_alpha_funding_sell_happens_even_when_optimizer_cash_caps_target():
    cfg = _cfg(fund_alpha=True)
    cfg["portfolio"]["benchmark_sleeve"]["rebalance_band_pct"] = 0.0
    cfg["portfolio"]["benchmark_sleeve"]["min_trade_pct"] = 0.0
    ctx = _ctx(
        cfg=cfg,
        holdings={"SPY": _holding(160, 500.0)},
        orders=[{"ticker": "AAA", "side": "BUY", "invest": 5_000.0}],
        prices={"SPY": 500.0, "AAA": 100.0},
        portfolio_value=100_000.0,
        cash=0.0,
    )

    BenchmarkSleeveTask().run(ctx)

    assert ctx.orders == [{"ticker": "AAA", "side": "BUY", "invest": 5_000.0}]
    assert ctx.exits, "Sleeve funding must be real even when total exposure is below target"
    ticker, sig = ctx.exits[0]
    assert ticker == "SPY"
    assert sig.exit_type == "benchmark_sleeve_rebalance"
    assert sig.quantity == pytest.approx(10.0)
    assert ctx._benchmark_sleeve_state["target_sleeve_value"] == pytest.approx(80_000.0)
    assert ctx._benchmark_sleeve_state["alpha_funding_gap_value"] == pytest.approx(5_000.0)


def test_alpha_funding_sell_rounds_up_to_cover_buy_cash():
    ctx = _ctx(
        cfg=_cfg(fund_alpha=True),
        holdings={"SPY": _holding(200, 500.0)},
        orders=[{"ticker": "AAA", "side": "BUY", "invest": 1_499.0}],
        prices={"SPY": 500.0, "AAA": 100.0},
        portfolio_value=100_000.0,
        cash=0.0,
    )

    BenchmarkSleeveTask().run(ctx)

    assert ctx.exits
    _, sig = ctx.exits[0]
    assert sig.quantity == pytest.approx(3.0)
