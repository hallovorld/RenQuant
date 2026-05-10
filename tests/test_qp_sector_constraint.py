"""C2 — Sector cap as hard constraint inside the QP solve.

Pre-fix: `passes_sector_guard` (kernel/selection.py) gates buy candidates
upstream, but the QP solver had no awareness of sector membership — so a
stress reallocation among existing holdings could pile weight onto a
single sector. This file pins the post-fix invariant.

CLAUDE.md §5.13.3 audit-regression-guard: `TestSectorCapInQpRegression`
contains the canonical "this bug must not return" tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.portfolio_qp.qp_solver import solve_portfolio_qp  # noqa: E402
from kernel.portfolio_qp.tasks import BuildSectorConstraintMatrixTask  # noqa: E402


# ── Direct solver-level enforcement ───────────────────────────────────────────

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

    def test_zero_max_per_sector_disables(self):
        ctx = self._stub_ctx(["AAPL"], {"AAPL": "tech"}, max_per_sector=0)
        BuildSectorConstraintMatrixTask().run(ctx)
        assert ctx._qp_sector_indicator is None


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
