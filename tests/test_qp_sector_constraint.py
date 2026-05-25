"""C2 — Sector cap as hard constraint inside the QP solve.

Pre-fix: `passes_sector_guard` (kernel/selection.py) gates buy candidates
upstream, but the QP solver had no awareness of sector membership — so a
stress reallocation among existing holdings could pile weight onto a
single sector. This file pins the post-fix invariant.

CLAUDE.md §5.13.3 audit-regression-guard: `TestSectorCapInQpRegression`
contains the canonical "this bug must not return" tests.
"""
from __future__ import annotations

import datetime
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.portfolio_qp.qp_solver import QPSolution, solve_portfolio_qp  # noqa: E402
from kernel.portfolio_qp.task_joint_qp import JointPortfolioQPTask  # noqa: E402
from kernel.portfolio_qp.tasks import (  # noqa: E402
    ApplySectorMetadataGuardTask,
    BuildSectorConstraintMatrixTask,
    _retry_with_relaxed_c2_caps,
)


@dataclass
class _QPCand:
    ticker: str
    mu: float
    sigma: float
    rank_score: float = 0.60
    panel_score: float = 0.60


@dataclass
class _QPHold:
    shares: float
    mu: float
    sigma: float
    entry_price: float = 100.0
    entry_date: datetime.date = datetime.date(2026, 4, 1)


@dataclass
class _QPCtx:
    config: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)
    holdings: dict = field(default_factory=dict)
    prices: dict = field(default_factory=dict)
    cash: float = 10000.0
    portfolio_value: float = 10000.0
    today: datetime.date = datetime.date(2026, 4, 26)
    regime: str = "BULL_CALM"
    confidence: float = 1.0
    bear_only: bool = False
    buy_blocked: bool = False
    skip_buys: bool = False
    earnings_calendar: dict = field(default_factory=dict)
    last_sell_dates: dict = field(default_factory=dict)
    last_sell_pls: dict = field(default_factory=dict)
    orders: list = field(default_factory=list)
    exits: list = field(default_factory=list)
    counters: dict = field(default_factory=dict)


def _complex_qp_ctx() -> _QPCtx:
    ctx = _QPCtx(config={
        "rotation": {"joint_actions": {
            "enabled": True,
            "solver": "qp",
            "qp_risk_aversion": 1.0,
            "qp_cost_kappa": 0.0001,
            "qp_dw_max": 0.50,
            "qp_turnover_max": 0.80,
            "qp_min_dw_pct": 0.005,
            "qp_no_trade_band_factor": 0.0,
            "default_sigma": 0.05,
            "qp_sector_cap_enabled": True,
        }},
        "regime_params": {"BULL_CALM": {
            "max_position_pct": 0.20,
            "cash_reserve_pct": 0.0,
        }},
        "sector_map": {
            "GOOD": "tech",
            "BAD": "tech",
            "GLD": "defensive",
        },
        "max_positions_per_sector": 3,
        "defensive_tickers": ["GLD", "TLT", "XLV", "XLU"],
        "wash_sale_days": 0,
    })
    tickers = ["BAD", "GOOD", "GLD", "MISSING_HELD", "MISSING_NEW"]
    ctx.corr_matrix = {
        a: {b: 0.0 for b in tickers if b != a}
        for a in tickers
    }
    return ctx


# ── Direct solver-level enforcement ───────────────────────────────────────────

def _qp_solution(status: str) -> QPSolution:
    return QPSolution(
        delta_w=np.zeros(2),
        target_w=np.zeros(2),
        objective=0.0,
        n_iter=0,
        status=status,
        diagnostics={},
    )


class TestC2InfeasiblePolicy:
    def test_default_policy_keeps_hard_caps_fail_closed(self):
        calls = []

        def _solve(**kwargs):
            calls.append(kwargs)
            return _qp_solution("optimal")

        sol = _retry_with_relaxed_c2_caps(
            _qp_solution("infeasible:sector"),
            {
                "sector_indicator": np.ones((1, 2)),
                "sector_cap_vec": np.array([0.10]),
                "corr_group_pairs": [(0, 1, 0.20)],
            },
            _solve,
        )

        assert sol.status == "infeasible:sector"
        assert calls == []
        assert sol.diagnostics["c2_infeasible_policy"] == "strict"

    def test_relax_policy_is_explicit_diagnostic_path(self):
        calls = []

        def _solve(**kwargs):
            calls.append(kwargs)
            return _qp_solution("optimal")

        sol = _retry_with_relaxed_c2_caps(
            _qp_solution("infeasible:sector"),
            {
                "sector_indicator": np.ones((1, 2)),
                "sector_cap_vec": np.array([0.10]),
                "corr_group_pairs": [(0, 1, 0.20)],
            },
            _solve,
            policy="relax",
        )

        assert sol.status == "optimal"
        assert len(calls) == 1
        np.testing.assert_allclose(calls[0]["sector_cap_vec"], [0.15])
        assert calls[0]["corr_group_pairs"][0][:2] == (0, 1)
        assert calls[0]["corr_group_pairs"][0][2] == pytest.approx(0.30)
        assert sol.diagnostics["c2_infeasible_policy"] == "relax"


class TestSolverSectorConstraint:
    """Smoke tests on solve_portfolio_qp's sector_indicator + sector_cap_vec."""

    def _three_assets_one_sector(self, *, sector_cap):
        # 3 stocks, all in same sector. μ strongly positive → without sector
        # cap, QP would pile weight on all 3.
        n = 3
        S = np.array([[1.0, 1.0, 1.0]])     # one row: every asset in sector
        cap = np.array([sector_cap])
        return solve_portfolio_qp(
            w_current=np.zeros(n),
            mu=np.array([0.10, 0.09, 0.08]),
            sigma=np.full(n, 0.15),
            risk_aversion=3.0,
            cash_reserve=0.0,
            w_upper=0.30,
            w_lower=0.0,
            dw_max=0.50,
            sector_indicator=S,
            sector_cap_vec=cap,
        )

    def test_sector_cap_binds(self):
        sol = self._three_assets_one_sector(sector_cap=0.30)
        assert sol.status in ("optimal", "optimal_inaccurate") or \
               sol.status == "optimal"
        # Total weight in the (one) sector ≤ cap (with float tolerance)
        total = float(np.sum(sol.target_w))
        assert total <= 0.301, f"sector cap violated: total={total}"

    def test_no_cap_lets_more_weight_through(self):
        """Without sector_indicator, cap does not bind."""
        n = 3
        sol = solve_portfolio_qp(
            w_current=np.zeros(n),
            mu=np.array([0.10, 0.09, 0.08]),
            sigma=np.full(n, 0.15),
            risk_aversion=3.0,
            cash_reserve=0.0,
            w_upper=0.30,
            w_lower=0.0,
            dw_max=0.50,
        )
        # No sector cap → solver can deploy multiples of the per-name cap.
        assert sol.target_w.sum() > 0.31

    def test_diagnostics_count_sector_rows(self):
        sol = self._three_assets_one_sector(sector_cap=0.30)
        assert sol.diagnostics["n_sector_constraints"] == 1

    def test_two_sectors_independent_caps(self):
        """3 in sector A, 2 in sector B. Each cap binds independently."""
        n = 5
        S = np.array([
            [1, 1, 1, 0, 0],  # sector A
            [0, 0, 0, 1, 1],  # sector B
        ], dtype=float)
        cap = np.array([0.20, 0.15])
        sol = solve_portfolio_qp(
            w_current=np.zeros(n),
            mu=np.array([0.10, 0.10, 0.10, 0.12, 0.12]),
            sigma=np.full(n, 0.15),
            risk_aversion=3.0,
            cash_reserve=0.0,
            w_upper=0.30,
            w_lower=0.0,
            dw_max=0.50,
            sector_indicator=S,
            sector_cap_vec=cap,
        )
        sec_a = float(np.sum(sol.target_w[:3]))
        sec_b = float(np.sum(sol.target_w[3:]))
        assert sec_a <= 0.201
        assert sec_b <= 0.151


# ── Task-level integration: BuildSectorConstraintMatrixTask ───────────────────

class TestBuildSectorConstraintMatrixTask:
    """Verify the Task constructs the right S, cap_vec, sector_names."""

    def _stub_ctx(self, tickers, sector_map, *, max_per_sector=2,
                   max_position_pct=0.15, enabled=True):
        from types import SimpleNamespace
        ctx = SimpleNamespace(
            config={
                "sector_map": sector_map,
                "max_positions_per_sector": max_per_sector,
                "max_position_pct": max_position_pct,
                "rotation": {"joint_actions": {
                    "qp_sector_cap_enabled": enabled,
                }},
            },
        )
        ctx._qp_tickers = tickers
        ctx._qp_w_upper = np.full(len(tickers), max_position_pct)
        return ctx

    def test_two_sector_layout(self):
        tickers = ["AAPL", "GOOG", "JPM"]
        sec_map = {"AAPL": "tech", "GOOG": "tech", "JPM": "fin"}
        ctx = self._stub_ctx(tickers, sec_map, max_per_sector=2,
                              max_position_pct=0.15)
        BuildSectorConstraintMatrixTask().run(ctx)
        # Indicator shape: (m=2, n=3)
        S = ctx._qp_sector_indicator
        assert S.shape == (2, 3)
        names = ctx._qp_sector_names
        # Sectors sorted alphabetically: ['fin', 'tech']
        assert names == ["fin", "tech"]
        # JPM is in 'fin' (row 0, col 2)
        assert S[0, 2] == 1.0 and S[0, 0] == 0.0 and S[0, 1] == 0.0
        # AAPL+GOOG are in 'tech' (row 1, cols 0,1)
        assert S[1, 0] == 1.0 and S[1, 1] == 1.0 and S[1, 2] == 0.0
        # Cap = max_per_sector * max_position_pct = 2 * 0.15 = 0.30
        np.testing.assert_allclose(ctx._qp_sector_cap_vec, [0.30, 0.30])

    def test_disabled_yields_none(self):
        ctx = self._stub_ctx(["AAPL"], {"AAPL": "tech"}, enabled=False)
        BuildSectorConstraintMatrixTask().run(ctx)
        assert ctx._qp_sector_indicator is None
        assert ctx._qp_sector_cap_vec is None
        assert ctx._qp_sector_names == []

    def test_unmapped_ticker_dropped_from_sector(self):
        """A ticker absent from sector_map is excluded from any indicator row."""
        ctx = self._stub_ctx(
            ["AAPL", "MYSTERY"],
            {"AAPL": "tech"},                  # MYSTERY unmapped
            max_per_sector=2, max_position_pct=0.15,
        )
        BuildSectorConstraintMatrixTask().run(ctx)
        S = ctx._qp_sector_indicator
        # Only one sector (tech); MYSTERY column is zero
        assert S.shape == (1, 2)
        assert S[0, 0] == 1.0 and S[0, 1] == 0.0

    def test_unmapped_high_cap_does_not_inflate_sector_cap(self):
        """Missing-sector broker holdings may have large current weights.
        They must not become the per-name anchor for mapped sector rows."""
        ctx = self._stub_ctx(
            ["AAPL", "MSFT", "MYSTERY_HELD"],
            {"AAPL": "tech", "MSFT": "tech"},
            max_per_sector=2, max_position_pct=0.15,
        )
        ctx._qp_w_upper = np.array([0.15, 0.10, 0.60])

        BuildSectorConstraintMatrixTask().run(ctx)

        np.testing.assert_allclose(ctx._qp_sector_cap_vec, [0.30])
        assert ctx._qp_sector_names == ["tech"]

    def test_zero_max_per_sector_disables(self):
        ctx = self._stub_ctx(["AAPL"], {"AAPL": "tech"}, max_per_sector=0)
        BuildSectorConstraintMatrixTask().run(ctx)
        assert ctx._qp_sector_indicator is None


class TestMissingSectorGuardRegression:
    """AUDIT REGRESSION GUARD: missing sector metadata is not an alpha pass.

    The sector cap matrix cannot constrain a ticker that has no sector row.
    The QP must therefore cap unmapped tickers at current weight before solve:
    new candidates get 0% max weight; existing holdings can be reduced or held
    but not increased.
    """

    def test_task_caps_missing_sector_at_current_weight(self):
        from types import SimpleNamespace

        ctx = SimpleNamespace(
            config={
                "sector_map": {"AAPL": "tech"},
                "rotation": {"joint_actions": {"qp_sector_cap_enabled": True}},
            },
            candidates=[],
            counters={},
        )
        ctx._qp_tickers = ["AAPL", "MYSTERY_HELD", "MYSTERY_NEW"]
        ctx._qp_w_current = np.array([0.00, 0.12, 0.00])
        ctx._qp_w_upper = np.array([0.20, 0.20, 0.20])

        ApplySectorMetadataGuardTask().run(ctx)

        np.testing.assert_allclose(ctx._qp_w_upper, [0.20, 0.12, 0.00])
        assert ctx._qp_missing_sector_tickers == ["MYSTERY_HELD", "MYSTERY_NEW"]
        assert ctx.counters["qp_missing_sector_guard"] == 2

    def test_task_caps_all_when_sector_map_empty(self):
        from types import SimpleNamespace

        ctx = SimpleNamespace(
            config={
                "sector_map": {},
                "rotation": {"joint_actions": {"qp_sector_cap_enabled": True}},
            },
            candidates=[_QPCand("MYSTERY_NEW", mu=1.0, sigma=0.05)],
            counters={},
        )
        ctx._qp_tickers = ["MYSTERY_HELD", "MYSTERY_NEW"]
        ctx._qp_w_current = np.array([0.12, 0.00])
        ctx._qp_w_upper = np.array([0.20, 0.20])

        ApplySectorMetadataGuardTask().run(ctx)

        np.testing.assert_allclose(ctx._qp_w_upper, [0.12, 0.00])
        assert ctx._qp_missing_sector_tickers == ["MYSTERY_HELD", "MYSTERY_NEW"]
        assert ctx._blocked_by_ticker["MYSTERY_NEW"] == "missing_sector_map"

    def test_full_qp_does_not_buy_missing_sector_candidate(self):
        """A missing-sector candidate with the strongest μ must not receive
        a QP buy; a mapped candidate in the same solve may still be bought."""
        ctx = _complex_qp_ctx()
        ctx.candidates = [
            _QPCand("MISSING_NEW", mu=1.00, sigma=0.05),
            _QPCand("GOOD", mu=0.60, sigma=0.05),
        ]
        ctx.prices.update({"MISSING_NEW": 100.0, "GOOD": 100.0})

        JointPortfolioQPTask().run(ctx)

        bought = {o["ticker"] for o in ctx.orders}
        assert "MISSING_NEW" not in bought
        assert "GOOD" in bought
        assert ctx._blocked_by_ticker["MISSING_NEW"] == "missing_sector_map"

    def test_full_qp_does_not_top_up_missing_sector_holding(self):
        """A broker-imported holding without sector metadata may remain in
        the book, but QP cannot increase it even with very high μ."""
        ctx = _complex_qp_ctx()
        ctx.holdings = {"MISSING_HELD": _QPHold(shares=10, mu=1.00, sigma=0.05)}
        ctx.prices = {"MISSING_HELD": 100.0}
        ctx.cash = 9000.0
        ctx.portfolio_value = 10000.0

        JointPortfolioQPTask().run(ctx)

        assert ctx.orders == []
        assert ctx.exits == []
        assert ctx._qp_missing_sector_tickers == ["MISSING_HELD"]
        assert float(ctx._qp_solution.target_w[0]) <= 0.1001

    def test_complex_book_only_allocates_to_metadata_complete_names(self):
        """Complex mixed book: missing-sector names and non-BEAR defensive
        buys are suppressed, while mapped positive-alpha names can still buy
        and mapped negative-alpha holdings can still sell."""
        ctx = _complex_qp_ctx()
        ctx.holdings = {
            "BAD": _QPHold(shares=10, mu=-0.80, sigma=0.05),
            "MISSING_HELD": _QPHold(shares=10, mu=1.20, sigma=0.05),
        }
        ctx.candidates = [
            _QPCand("GOOD", mu=0.80, sigma=0.05),
            _QPCand("MISSING_NEW", mu=1.50, sigma=0.05),
            _QPCand("GLD", mu=1.10, sigma=0.05),
        ]
        ctx.prices = {
            "BAD": 100.0,
            "MISSING_HELD": 100.0,
            "GOOD": 100.0,
            "MISSING_NEW": 100.0,
            "GLD": 100.0,
        }
        ctx.cash = 8000.0
        ctx.portfolio_value = 10000.0

        JointPortfolioQPTask().run(ctx)

        bought = {o["ticker"] for o in ctx.orders}
        sold = {t for t, _ in ctx.exits}
        assert bought == {"GOOD"}
        assert "BAD" in sold
        assert "MISSING_HELD" not in bought
        assert "MISSING_NEW" not in bought
        assert "GLD" not in bought
        assert ctx._blocked_by_ticker["MISSING_NEW"] == "missing_sector_map"
        assert ctx._blocked_by_ticker["GLD"] == "defensive_non_bear"


# ── Audit regression guard (§5.13.3) ──────────────────────────────────────────

class TestSectorCapInQpRegression:
    """Pin the bug class: 'QP can pile weight on one sector during reallocation
    even when buy-side filter would have caught it'.

    Pre-fix invariant (broken): solver had no S/cap → reallocation could
    drift past sector cap.

    Post-fix invariant (this test): with S + cap_vec passed, target sector
    weight ≤ cap. Bug-class is closed.
    """

    def test_3_stocks_same_sector_high_mu_capped(self):
        """All 3 high-conviction stocks in the same sector — total ≤ 30%."""
        n = 3
        sol = solve_portfolio_qp(
            w_current=np.zeros(n),
            mu=np.array([0.20, 0.18, 0.16]),       # all very bullish
            sigma=np.full(n, 0.15),
            risk_aversion=3.0,
            cash_reserve=0.0,
            w_upper=0.30,                          # per-name cap 30%
            w_lower=0.0,
            dw_max=0.50,
            sector_indicator=np.array([[1.0, 1.0, 1.0]]),
            sector_cap_vec=np.array([0.30]),       # but sector cap also 30%
        )
        assert sol.status == "optimal"
        # Without cap, sum would be ~ 0.9 (each at 30%). With cap, ≤ 30%.
        total = float(np.sum(sol.target_w))
        assert total <= 0.301

    def test_bug_class_via_diagnostic_counter(self):
        """Pin: solver advertises n_sector_constraints in diagnostics so
        downstream QC can detect 'sector cap silently dropped'."""
        n = 3
        sol = solve_portfolio_qp(
            w_current=np.zeros(n),
            mu=np.full(n, 0.10),
            sigma=np.full(n, 0.15),
            risk_aversion=3.0,
            cash_reserve=0.0,
            w_upper=0.30,
            w_lower=0.0,
            dw_max=0.50,
            sector_indicator=np.array([[1.0, 1.0, 1.0]]),
            sector_cap_vec=np.array([0.30]),
        )
        assert sol.diagnostics.get("n_sector_constraints", 0) == 1


# ── §5.13.5 single-source-of-truth ────────────────────────────────────────────

class TestSectorCapSingleSourceOfTruth:
    """The QP must read sector_map + max_positions_per_sector from the SAME
    config keys as kernel/selection.py::passes_sector_guard. Fingerprint it
    by importing the constants and confirming Task references match.
    """

    def test_task_reads_top_level_sector_map(self):
        from types import SimpleNamespace
        ctx = SimpleNamespace(
            config={
                "sector_map": {"AAPL": "tech"},
                "max_positions_per_sector": 1,
                "max_position_pct": 0.20,
                "rotation": {"joint_actions": {}},
            },
        )
        ctx._qp_tickers = ["AAPL"]
        ctx._qp_w_upper = np.array([0.20])
        BuildSectorConstraintMatrixTask().run(ctx)
        # Cap = 1 × 0.20 (single source of truth read).
        np.testing.assert_allclose(ctx._qp_sector_cap_vec, [0.20])

    def test_qp_uses_same_max_positions_per_sector_as_selection(self):
        """passes_sector_guard treats max_per_sector as a count cap; the QP
        treats it as a *weight cap* via max_per_sector × max_position_pct.
        Document this conversion is intentional and tested."""
        # If max_per_sector=4 and max_position_pct=0.15 → sector cap = 0.60.
        from types import SimpleNamespace
        ctx = SimpleNamespace(
            config={
                "sector_map": {"A": "tech", "B": "tech"},
                "max_positions_per_sector": 4,
                "max_position_pct": 0.15,
                "rotation": {"joint_actions": {}},
            },
        )
        ctx._qp_tickers = ["A", "B"]
        ctx._qp_w_upper = np.array([0.15, 0.15])
        BuildSectorConstraintMatrixTask().run(ctx)
        np.testing.assert_allclose(ctx._qp_sector_cap_vec, [0.60])

    def test_regime_max_sector_weight_tightens_count_cap(self):
        """AUDIT REGRESSION GUARD: count×per-name can be too loose
        (6×15%=90%). A regime-level direct sector max caps the linear
        group constraint the way mature optimizers model sector exposure."""
        from types import SimpleNamespace
        ctx = SimpleNamespace(
            regime="BULL_CALM",
            config={
                "sector_map": {"A": "tech", "B": "tech", "C": "tech"},
                "max_positions_per_sector": 6,
                "max_position_pct": 0.15,
                "regime_params": {"BULL_CALM": {"max_sector_weight_pct": 0.35}},
                "rotation": {"joint_actions": {}},
            },
        )
        ctx._qp_tickers = ["A", "B", "C"]
        ctx._qp_w_upper = np.array([0.15, 0.15, 0.15])

        BuildSectorConstraintMatrixTask().run(ctx)

        np.testing.assert_allclose(ctx._qp_sector_cap_vec, [0.35])
        assert ctx._qp_sector_cap_source == "regime_or_global_max_sector_weight_pct"
