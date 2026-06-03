"""Tests for ``solve_portfolio_qp_from_snapshot`` — Step 2 of §8 plan.

The wrapper is a strict delegate: every callable input that lands at
``solve_portfolio_qp`` via the snapshot path must be byte-identical to
what the existing kwargs path produces. This file pins that invariant
on a representative scenario set (basic feasible, infeasible cap
state, wash-sale block, sector cap, low-turnover squeeze).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.portfolio_qp.constraint_snapshot import ConstraintSnapshot  # noqa: E402
from kernel.portfolio_qp.qp_solver import (  # noqa: E402
    solve_portfolio_qp,
    solve_portfolio_qp_from_snapshot,
)


def _snap(
    *,
    w_current,
    w_upper_hard,
    w_upper,
    w_lower=0.0,
    dw_max=None,
    cash_reserve=0.0,
    turnover_max=0.30,
    drawdown=0.0,
    drawdown_limit=0.20,
    gross_max=None,
    wash_sale_mask=None,
    sector_indicator=None,
    sector_cap_vec=None,
    sector_names=None,
    corr_group_pairs=(),
) -> ConstraintSnapshot:
    n = len(w_current)
    return ConstraintSnapshot(
        n=n,
        tickers=tuple(f"T{i}" for i in range(n)),
        w_current=np.asarray(w_current, dtype=float),
        w_upper_hard=np.asarray(w_upper_hard, dtype=float),
        w_upper=np.asarray(w_upper, dtype=float),
        w_lower=w_lower,
        dw_max=np.asarray(dw_max if dw_max is not None else [0.5] * n, dtype=float),
        cash_reserve=cash_reserve,
        turnover_max=turnover_max,
        drawdown=drawdown,
        drawdown_limit=drawdown_limit,
        gross_max=gross_max,
        wash_sale_mask=np.asarray(
            wash_sale_mask if wash_sale_mask is not None else [False] * n,
            dtype=bool,
        ),
        sector_indicator=sector_indicator,
        sector_cap_vec=sector_cap_vec,
        sector_names=sector_names,
        corr_group_pairs=corr_group_pairs,
    )


def _assert_solutions_equal(via_kwargs, via_snapshot, *, atol=1e-12):
    """Two QPSolutions byte-equivalent within numerical noise."""
    assert via_kwargs.status == via_snapshot.status, (
        f"status differs: kwargs={via_kwargs.status!r} "
        f"snapshot={via_snapshot.status!r}"
    )
    np.testing.assert_allclose(
        via_kwargs.delta_w, via_snapshot.delta_w, atol=atol,
        err_msg="delta_w differs between kwargs and snapshot paths",
    )
    np.testing.assert_allclose(
        via_kwargs.target_w, via_snapshot.target_w, atol=atol,
        err_msg="target_w differs",
    )
    # Objective value may have solver-noise; broader atol is fine.
    assert abs(via_kwargs.objective - via_snapshot.objective) < 1e-8, (
        f"objective differs: {via_kwargs.objective} vs {via_snapshot.objective}"
    )


class TestSolverViaSnapshotByteEquivalence:
    """Snapshot path must produce identical QPSolution to kwargs path."""

    def test_basic_two_asset_feasible(self):
        kwargs = dict(
            w_current=np.zeros(2),
            mu=np.array([0.01, 0.02]),
            sigma=np.array([0.10, 0.15]),
            w_upper=np.full(2, 0.20),
            w_lower=0.0,
            dw_max=np.full(2, 0.50),
            cash_reserve=0.05,
            turnover_max=0.30,
            cost_kappa=0.002,
            risk_aversion=3.0,
        )
        via_kwargs = solve_portfolio_qp(**kwargs)

        snap = _snap(
            w_current=kwargs["w_current"],
            w_upper_hard=np.full(2, 0.20),
            w_upper=kwargs["w_upper"],
            cash_reserve=kwargs["cash_reserve"],
            turnover_max=kwargs["turnover_max"],
        )
        via_snapshot = solve_portfolio_qp_from_snapshot(
            snap,
            mu=kwargs["mu"],
            sigma=kwargs["sigma"],
            cost_kappa=kwargs["cost_kappa"],
            risk_aversion=kwargs["risk_aversion"],
        )
        _assert_solutions_equal(via_kwargs, via_snapshot)

    def test_over_cap_holding_infeasible_status_preserved(self):
        """v4 contract — over-cap holding produces infeasible status
        through both paths. The snapshot path must NOT silently widen
        the cap."""
        kwargs = dict(
            w_current=np.array([0.22]),
            mu=np.array([0.0]),
            sigma=np.array([0.10]),
            w_upper=np.array([0.15]),     # hard cap (v4 snapshot keeps this)
            w_lower=0.0,
            cash_reserve=0.0,
            cost_kappa=10.0,               # stiff so no-signal becomes infeasible
            turnover_max=0.01,
        )
        via_kwargs = solve_portfolio_qp(**kwargs)

        snap = _snap(
            w_current=kwargs["w_current"],
            w_upper_hard=np.array([0.15]),
            w_upper=kwargs["w_upper"],
            turnover_max=kwargs["turnover_max"],
        )
        via_snapshot = solve_portfolio_qp_from_snapshot(
            snap,
            mu=kwargs["mu"],
            sigma=kwargs["sigma"],
            cost_kappa=kwargs["cost_kappa"],
        )
        assert via_snapshot.status.startswith("infeasible")
        assert via_kwargs.status == via_snapshot.status

    def test_wash_sale_mask_round_trip(self):
        kwargs = dict(
            w_current=np.array([0.10, 0.05]),
            mu=np.array([0.02, 0.01]),
            sigma=np.array([0.10, 0.12]),
            w_upper=np.full(2, 0.20),
            w_lower=0.0,
            wash_sale_mask=np.array([False, True]),  # T1 cannot increase
            cash_reserve=0.0,
            turnover_max=0.30,
            cost_kappa=0.002,
        )
        via_kwargs = solve_portfolio_qp(**kwargs)

        snap = _snap(
            w_current=kwargs["w_current"],
            w_upper_hard=np.full(2, 0.20),
            w_upper=kwargs["w_upper"],
            wash_sale_mask=kwargs["wash_sale_mask"],
            turnover_max=kwargs["turnover_max"],
        )
        via_snapshot = solve_portfolio_qp_from_snapshot(
            snap, mu=kwargs["mu"], sigma=kwargs["sigma"],
            cost_kappa=kwargs["cost_kappa"],
        )
        _assert_solutions_equal(via_kwargs, via_snapshot)
        # T1 cannot go up via either path
        assert via_snapshot.delta_w[1] <= 1e-9

    def test_sector_cap_round_trip(self):
        """Two-sector universe with a tight cap on sector 0."""
        kwargs = dict(
            w_current=np.zeros(3),
            mu=np.array([0.03, 0.02, 0.01]),
            sigma=np.array([0.10, 0.12, 0.11]),
            w_upper=np.full(3, 0.20),
            w_lower=0.0,
            cash_reserve=0.0,
            turnover_max=0.30,
            cost_kappa=0.002,
            sector_indicator=np.array(
                [[1.0, 1.0, 0.0],   # sector 0: T0, T1
                 [0.0, 0.0, 1.0]],   # sector 1: T2
                dtype=float,
            ),
            sector_cap_vec=np.array([0.25, 1.0]),  # tight sector-0 cap
        )
        via_kwargs = solve_portfolio_qp(**kwargs)

        snap = _snap(
            w_current=kwargs["w_current"],
            w_upper_hard=np.full(3, 0.20),
            w_upper=kwargs["w_upper"],
            turnover_max=kwargs["turnover_max"],
            sector_indicator=kwargs["sector_indicator"],
            sector_cap_vec=kwargs["sector_cap_vec"],
            sector_names=("Tech", "Health"),
        )
        via_snapshot = solve_portfolio_qp_from_snapshot(
            snap, mu=kwargs["mu"], sigma=kwargs["sigma"],
            cost_kappa=kwargs["cost_kappa"],
        )
        _assert_solutions_equal(via_kwargs, via_snapshot)

    def test_full_Sigma_path_round_trip(self):
        """Pass Sigma rather than sigma — verifies the wrapper does
        not drop the Sigma kwarg en route."""
        Sigma = np.array([[0.01, 0.002], [0.002, 0.0144]])
        kwargs = dict(
            w_current=np.zeros(2),
            mu=np.array([0.02, 0.015]),
            Sigma=Sigma,
            w_upper=np.full(2, 0.20),
            w_lower=0.0,
            cash_reserve=0.05,
            turnover_max=0.30,
            cost_kappa=0.002,
        )
        via_kwargs = solve_portfolio_qp(**kwargs)

        snap = _snap(
            w_current=kwargs["w_current"],
            w_upper_hard=np.full(2, 0.20),
            w_upper=kwargs["w_upper"],
            cash_reserve=kwargs["cash_reserve"],
            turnover_max=kwargs["turnover_max"],
        )
        via_snapshot = solve_portfolio_qp_from_snapshot(
            snap, mu=kwargs["mu"], Sigma=Sigma,
            cost_kappa=kwargs["cost_kappa"],
        )
        _assert_solutions_equal(via_kwargs, via_snapshot)


class TestSolverViaSnapshotImmutabilityNotBroken:
    """The wrapper must not mutate the snapshot it received."""

    def test_solver_call_does_not_modify_snapshot_arrays(self):
        snap = _snap(
            w_current=np.zeros(2),
            w_upper_hard=np.full(2, 0.20),
            w_upper=np.full(2, 0.20),
        )
        before_w_upper = snap.w_upper.copy()
        before_w_current = snap.w_current.copy()

        _ = solve_portfolio_qp_from_snapshot(
            snap, mu=np.array([0.01, 0.02]),
            sigma=np.array([0.10, 0.15]),
            cost_kappa=0.002,
        )

        # Per-asset arrays in the snapshot are unmodified — they are
        # read-only (set by __post_init__) so a mutation would have
        # raised earlier; this is the belt-and-braces assertion.
        np.testing.assert_array_equal(snap.w_upper, before_w_upper)
        np.testing.assert_array_equal(snap.w_current, before_w_current)
