"""C2 — Correlation pair group cap inside the QP solve.

Pre-fix: `passes_correlation_guard` (kernel/selection.py) gates buy
candidates upstream, but the QP solver had no awareness of pair
correlation — so a stress reallocation could pile weight on two
highly-correlated holdings simultaneously.

Post-fix: for each (i, j) with |corr[i, j]| ≥ correlation_guard_threshold,
add a linear group cap `wp[i] + wp[j] ≤ 2 × per_name_cap`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.portfolio_qp.qp_solver import solve_portfolio_qp  # noqa: E402
from kernel.portfolio_qp.tasks import (  # noqa: E402
    BuildCorrelationGroupConstraintTask,
)


# ── Solver-level direct enforcement ───────────────────────────────────────────

class TestSolverCorrPairConstraint:

    def test_pair_cap_binds(self):
        """High-corr pair w/ both bullish μ: cap forces sum ≤ 0.20."""
        n = 3
        sol = solve_portfolio_qp(
            w_current=np.zeros(n),
            mu=np.array([0.10, 0.10, 0.10]),
            sigma=np.full(n, 0.15),
            risk_aversion=3.0,
            cash_reserve=0.0,
            w_upper=0.30,
            w_lower=0.0,
            dw_max=0.50,
            corr_group_pairs=[(0, 1, 0.20)],
        )
        # i=0 + i=1 ≤ 0.20 (pair cap)
        pair_sum = float(sol.target_w[0] + sol.target_w[1])
        assert pair_sum <= 0.201, f"pair cap violated: pair_sum={pair_sum}"

    def test_no_pairs_no_constraint(self):
        n = 2
        sol = solve_portfolio_qp(
            w_current=np.zeros(n),
            mu=np.array([0.10, 0.10]),
            sigma=np.full(n, 0.15),
            risk_aversion=3.0,
            cash_reserve=0.0,
            w_upper=0.20,
            w_lower=0.0,
            dw_max=0.50,
        )
        # Each at the per-name cap = 0.20 → sum = 0.40, no pair cap
        assert sol.target_w.sum() > 0.30

    def test_diagnostics_count_pairs(self):
        n = 4
        sol = solve_portfolio_qp(
            w_current=np.zeros(n),
            mu=np.array([0.10, 0.10, 0.10, 0.10]),
            sigma=np.full(n, 0.15),
            risk_aversion=3.0,
            cash_reserve=0.0,
            w_upper=0.20,
            w_lower=0.0,
            dw_max=0.50,
            corr_group_pairs=[(0, 1, 0.20), (2, 3, 0.20)],
        )
        assert sol.diagnostics["n_corr_pair_constraints"] == 2

    def test_invalid_pair_indices_skipped(self):
        """Bad indices (out of range / equal) → quietly dropped, no crash."""
        n = 2
        sol = solve_portfolio_qp(
            w_current=np.zeros(n),
            mu=np.array([0.10, 0.10]),
            sigma=np.full(n, 0.15),
            risk_aversion=3.0,
            cash_reserve=0.0,
            w_upper=0.20,
            w_lower=0.0,
            dw_max=0.50,
            corr_group_pairs=[
                (0, 0, 0.10),     # i==j (skipped)
                (0, 5, 0.10),     # j>=n (skipped)
                (-1, 0, 0.10),    # i<0 (skipped)
                (0, 1, 0.20),     # valid
            ],
        )
        assert sol.diagnostics["n_corr_pair_constraints"] == 1


# ── BuildCorrelationGroupConstraintTask integration ───────────────────────────

class TestBuildCorrelationGroupConstraintTask:

    def _stub_ctx(self, tickers, corr_matrix, *, threshold=0.7,
                   max_position_pct=0.15, enabled=True):
        from types import SimpleNamespace
        ctx = SimpleNamespace(
            config={
                "regime": {"correlation_guard_threshold": threshold},
                "max_position_pct": max_position_pct,
                "rotation": {"joint_actions": {
                    "qp_correlation_cap_enabled": enabled,
                }},
            },
        )
        ctx.corr_matrix = corr_matrix
        ctx._qp_tickers = tickers
        ctx._qp_w_upper = np.full(len(tickers), max_position_pct)
        ctx._qp_w_current = np.zeros(len(tickers))
        ctx.candidates = []
        ctx.counters = {}
        return ctx

    def test_high_corr_pair_added(self):
        """Pair w/ |ρ|=0.85 above 0.7 threshold → pair recorded."""
        ctx = self._stub_ctx(
            ["AAPL", "MSFT"],
            {"AAPL": {"MSFT": 0.85}, "MSFT": {"AAPL": 0.85}},
            threshold=0.7, max_position_pct=0.15,
        )
        BuildCorrelationGroupConstraintTask().run(ctx)
        pairs = ctx._qp_corr_group_pairs
        assert pairs is not None
        assert len(pairs) == 1
        i, j, gcap = pairs[0]
        assert (i, j) == (0, 1)
        # group cap = 2 × per_name_cap = 0.30
        assert abs(gcap - 0.30) < 1e-12

    def test_pair_cap_uses_pair_local_bounds_not_global_outlier(self):
        """A large unrelated holding cap must not loosen every high-corr pair."""
        ctx = self._stub_ctx(
            ["AAPL", "MSFT", "MYSTERY_HELD"],
            {"AAPL": {"MSFT": 0.85}},
            threshold=0.7, max_position_pct=0.15,
        )
        ctx._qp_w_upper = np.array([0.10, 0.12, 0.60])

        BuildCorrelationGroupConstraintTask().run(ctx)

        assert ctx._qp_corr_group_pairs == [(0, 1, 0.22)]

    def test_low_corr_pair_dropped(self):
        ctx = self._stub_ctx(
            ["AAPL", "JPM"],
            {"AAPL": {"JPM": 0.10}},
            threshold=0.7,
        )
        BuildCorrelationGroupConstraintTask().run(ctx)
        assert ctx._qp_corr_group_pairs is None

    def test_nan_correlation_treated_as_high(self):
        ctx = self._stub_ctx(
            ["A", "B"],
            {"A": {"B": float("nan")}},
            threshold=0.7,
        )
        BuildCorrelationGroupConstraintTask().run(ctx)
        # NaN → fail-conservative → counted as high-corr
        assert ctx._qp_corr_group_pairs is not None
        assert len(ctx._qp_corr_group_pairs) == 1

    def test_missing_matrix_caps_all_at_current_weight(self):
        from types import SimpleNamespace

        cand = SimpleNamespace(ticker="NEW")
        ctx = self._stub_ctx(["HELD", "NEW"], None)
        ctx._qp_w_current = np.array([0.11, 0.00])
        ctx._qp_w_upper = np.array([0.20, 0.20])
        ctx.candidates = [cand]

        BuildCorrelationGroupConstraintTask().run(ctx)

        np.testing.assert_allclose(ctx._qp_w_upper, [0.11, 0.00])
        assert ctx._qp_missing_correlation_tickers == ["HELD", "NEW"]
        assert ctx._blocked_by_ticker["NEW"] == "missing_correlation_matrix"

    def test_missing_pair_caps_incomplete_tickers(self):
        from types import SimpleNamespace

        cand = SimpleNamespace(ticker="C")
        ctx = self._stub_ctx(
            ["A", "B", "C"],
            {"A": {"B": 0.10}},  # A-C and B-C are missing
        )
        ctx._qp_w_current = np.array([0.05, 0.04, 0.00])
        ctx._qp_w_upper = np.array([0.20, 0.20, 0.20])
        ctx.candidates = [cand]

        BuildCorrelationGroupConstraintTask().run(ctx)

        np.testing.assert_allclose(ctx._qp_w_upper, [0.05, 0.04, 0.00])
        assert set(ctx._qp_missing_correlation_tickers) == {"A", "B", "C"}
        assert ctx._blocked_by_ticker["C"] == "missing_correlation_pair"

    def test_disabled_yields_none(self):
        ctx = self._stub_ctx(
            ["A", "B"],
            {"A": {"B": 0.95}},
            enabled=False,
        )
        BuildCorrelationGroupConstraintTask().run(ctx)
        assert ctx._qp_corr_group_pairs is None


# ── Audit regression guard (§5.13.3) ──────────────────────────────────────────

class TestCorrPairCapInQpRegression:
    """Pin: 'QP can over-allocate to a high-corr pair during reallocation'."""

    def test_high_corr_pair_capped_under_strong_signal(self):
        """Two highly-correlated stocks both with strong μ — sum ≤ group cap."""
        n = 2
        sol = solve_portfolio_qp(
            w_current=np.zeros(n),
            mu=np.array([0.20, 0.18]),
            sigma=np.full(n, 0.15),
            risk_aversion=3.0,
            cash_reserve=0.0,
            w_upper=0.30,
            w_lower=0.0,
            dw_max=0.50,
            corr_group_pairs=[(0, 1, 0.30)],
        )
        pair_sum = float(sol.target_w[0] + sol.target_w[1])
        assert pair_sum <= 0.301

    def test_violation_post_solve_resolution(self):
        """Pair cap is hard. With per-name cap 0.30 and pair cap 0.20, the
        binding sums to 0.20; adding a 3rd uncorrelated stock can still
        absorb leftover budget."""
        n = 3
        sol = solve_portfolio_qp(
            w_current=np.zeros(n),
            mu=np.array([0.10, 0.10, 0.10]),
            sigma=np.full(n, 0.15),
            risk_aversion=3.0,
            cash_reserve=0.0,
            w_upper=0.30,
            w_lower=0.0,
            dw_max=0.50,
            corr_group_pairs=[(0, 1, 0.20)],   # only stocks 0,1 correlated
        )
        assert sol.target_w[0] + sol.target_w[1] <= 0.201
        # Stock 2 not in any pair → free to reach per-name cap.
        # Solver will deploy it where pair budget is exhausted.
        assert sol.target_w[2] >= sol.target_w[0]   # uncorrelated wins
