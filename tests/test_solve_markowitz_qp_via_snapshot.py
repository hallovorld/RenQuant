"""Tests for ``SolveMarkowitzQPTask`` snapshot fast-path — §8 Step 1e.

This file pins the byte-equivalence invariant for the migration:

    For the same ctx, the snapshot fast-path (``ctx._qp_constraint_snapshot``
    populated) and the legacy kwargs path (snapshot absent) must produce
    an identical QPSolution — same status, delta_w, target_w, objective.

The migration is strictly behaviour-preserving: the snapshot wrapper
:func:`solve_portfolio_qp_from_snapshot` is a strict delegate (pinned by
``test_solver_via_snapshot.py``), so end-to-end the Task must not move a
single bit. These tests run the Task against representative ctxs (basic
feasible, over-cap holding, wash-sale block, sector cap) with the
``ctx._qp_constraint_snapshot`` present and absent, and assert
equivalence.

If you ADD a new constraint kwarg to ``_build_solver_kwargs`` without
adding a snapshot field, ``TestSnapshotForecastKwargsCoverage`` will
fail and prompt you to extend ``_SNAPSHOT_FORECAST_KWARGS`` or the
snapshot contract.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.portfolio_qp.constraint_snapshot import (  # noqa: E402
    build_snapshot_from_ctx,
)
from kernel.portfolio_qp.tasks import SolveMarkowitzQPTask  # noqa: E402


# ── Shared ctx builder ──────────────────────────────────────────────────


def _base_ctx(
    *,
    tickers,
    w_current,
    mu,
    sigma,
    w_upper,
    w_upper_hard=None,
    w_lower=0.0,
    dw_max=None,
    cash_reserve=0.0,
    turnover_max=0.30,
    drawdown=0.0,
    drawdown_limit=0.20,
    wash_sale_mask=None,
    gross_max=None,
    sector_indicator=None,
    sector_cap_vec=None,
    corr_group_pairs=(),
    tax_cost=None,
    config=None,
    portfolio_value=100_000.0,
):
    n = len(tickers)
    if w_upper_hard is None:
        w_upper_hard = np.asarray(w_upper, dtype=float)
    if dw_max is None:
        dw_max = np.full(n, 0.5)
    if wash_sale_mask is None:
        wash_sale_mask = np.zeros(n, dtype=bool)
    if tax_cost is None:
        tax_cost = np.zeros(n)
    if config is None:
        config = {
            "rotation": {
                "joint_actions": {
                    "qp_risk_aversion": 3.0,
                    "qp_signal_decay": 0.0,
                    "qp_robust_mu_kappa": 0.0,
                    "qp_cvar_lambda": 0.0,
                    "qp_cvar_alpha": 0.05,
                    "qp_impact_coef": 0.0,
                    "qp_fixed_cost_per_trade": 0.0,
                    "qp_fixed_cost_beta": 200.0,
                    "qp_budget_mode": "inequality",
                    "qp_cash_drag_lambda": 0.0,
                    "qp_allow_optimal_inaccurate": False,
                    "qp_solver_backend": "cvxpy",
                    "qp_min_invested_pct": 0.0,
                    "qp_cost_kappa": 0.002,
                    "qp_c2_infeasible_policy": "strict",
                },
            },
        }
    ctx = SimpleNamespace(
        config=config,
        portfolio_value=portfolio_value,
        counters={},
        candidates=[],
        holdings={},
        # QP private fields populated by upstream Tasks:
        _qp_tickers=list(tickers),
        _qp_w_current=np.asarray(w_current, dtype=float),
        _qp_mu=np.asarray(mu, dtype=float),
        _qp_sigma=np.asarray(sigma, dtype=float),
        _qp_Sigma_full=None,
        _qp_w_upper=np.asarray(w_upper, dtype=float),
        _qp_w_upper_hard=np.asarray(w_upper_hard, dtype=float),
        _qp_w_lower=float(w_lower),
        _qp_dw_max=np.asarray(dw_max, dtype=float),
        _qp_cash_reserve=float(cash_reserve),
        _qp_turnover_max=turnover_max,
        _qp_drawdown=float(drawdown),
        _qp_drawdown_limit=float(drawdown_limit),
        _qp_wash_mask=np.asarray(wash_sale_mask, dtype=bool),
        _qp_tax_cost=np.asarray(tax_cost, dtype=float),
        _qp_v_daily_dollar=None,
        _qp_gross_max=gross_max,
        _qp_sector_indicator=sector_indicator,
        _qp_sector_cap_vec=sector_cap_vec,
        _qp_corr_group_pairs=corr_group_pairs,
    )
    return ctx


def _run_task_capture_solution(ctx):
    """Run SolveMarkowitzQPTask and return the QPSolution it stamped."""
    rv = SolveMarkowitzQPTask().run(ctx)
    sol = getattr(ctx, "_qp_solution", None)
    return rv, sol


def _stamp_snapshot(ctx):
    """Mirror BuildConstraintSnapshotTask: freeze ctx into ``_qp_constraint_snapshot``.

    Tests stamp the snapshot manually so each scenario can run BOTH paths
    against the same ctx state — one without the snapshot (legacy kwargs)
    and one with (Step 1e fast-path).
    """
    ctx._qp_constraint_snapshot = build_snapshot_from_ctx(ctx)  # noqa: SLF001


def _assert_solutions_equivalent(sol_a, sol_b, *, atol=1e-10):
    assert sol_a is not None and sol_b is not None
    assert sol_a.status == sol_b.status, (
        f"status differs: legacy={sol_a.status!r} snapshot={sol_b.status!r}"
    )
    np.testing.assert_allclose(
        sol_a.delta_w, sol_b.delta_w, atol=atol,
        err_msg="delta_w differs between legacy kwargs and snapshot fast-path",
    )
    np.testing.assert_allclose(
        sol_a.target_w, sol_b.target_w, atol=atol,
        err_msg="target_w differs between legacy kwargs and snapshot fast-path",
    )
    assert abs(sol_a.objective - sol_b.objective) < 1e-7, (
        f"objective differs: legacy={sol_a.objective} snapshot={sol_b.objective}"
    )


# ── Byte-equivalence scenarios ─────────────────────────────────────────


class TestSolveMarkowitzQPSnapshotByteEquivalence:
    """The runtime fast-path must produce identical QPSolution to the legacy path.

    Each scenario builds a ctx, runs SolveMarkowitzQPTask twice — once with
    ``_qp_constraint_snapshot`` absent (legacy kwargs path), once with the
    snapshot stamped (Step 1e fast-path) — and asserts every field of the
    stamped QPSolution matches.
    """

    def test_basic_feasible_two_asset_equivalence(self):
        common = dict(
            tickers=["AAPL", "MSFT"],
            w_current=np.zeros(2),
            mu=np.array([0.01, 0.02]),
            sigma=np.array([0.10, 0.15]),
            w_upper=np.full(2, 0.20),
            cash_reserve=0.05,
            turnover_max=0.30,
        )
        ctx_legacy = _base_ctx(**common)
        _, sol_legacy = _run_task_capture_solution(ctx_legacy)

        ctx_snap = _base_ctx(**common)
        _stamp_snapshot(ctx_snap)
        _, sol_snap = _run_task_capture_solution(ctx_snap)

        _assert_solutions_equivalent(sol_legacy, sol_snap)
        # And the snapshot path actually used the contract — the snapshot
        # is still present (and read-only) post-call.
        assert ctx_snap._qp_constraint_snapshot is not None
        # Legacy ctx never had a snapshot stamped.
        assert getattr(ctx_legacy, "_qp_constraint_snapshot", None) is None

    def test_over_cap_holding_infeasible_status_preserved(self):
        """Audit #2 / issue #70 regression — over-cap holding produces
        infeasible status through BOTH paths. The snapshot path must not
        silently widen the hard cap."""
        common = dict(
            tickers=["ORCL"],
            w_current=np.array([0.22]),
            mu=np.array([0.0]),
            sigma=np.array([0.10]),
            w_upper=np.array([0.15]),
            w_upper_hard=np.array([0.15]),
            cash_reserve=0.0,
            turnover_max=0.01,    # tight, forces infeasibility for over-cap
        )
        # Stiff cost_kappa so no-signal stays in the infeasibility regime
        cfg_stiff = {
            "rotation": {
                "joint_actions": {
                    "qp_solver_backend": "cvxpy",
                    "qp_cost_kappa": 10.0,
                    "qp_risk_aversion": 3.0,
                    "qp_c2_infeasible_policy": "strict",
                },
            },
        }
        ctx_legacy = _base_ctx(config=cfg_stiff, **common)
        _, sol_legacy = _run_task_capture_solution(ctx_legacy)

        ctx_snap = _base_ctx(config=cfg_stiff, **common)
        _stamp_snapshot(ctx_snap)
        _, sol_snap = _run_task_capture_solution(ctx_snap)

        assert sol_snap.status.startswith("infeasible")
        assert sol_legacy.status == sol_snap.status

    def test_wash_sale_mask_equivalence(self):
        """Wash-sale mask on T1 prevents Δw>0 via both paths."""
        common = dict(
            tickers=["AAPL", "MSFT"],
            w_current=np.array([0.10, 0.05]),
            mu=np.array([0.02, 0.01]),
            sigma=np.array([0.10, 0.12]),
            w_upper=np.full(2, 0.20),
            wash_sale_mask=np.array([False, True]),
            turnover_max=0.30,
        )
        ctx_legacy = _base_ctx(**common)
        _, sol_legacy = _run_task_capture_solution(ctx_legacy)

        ctx_snap = _base_ctx(**common)
        _stamp_snapshot(ctx_snap)
        _, sol_snap = _run_task_capture_solution(ctx_snap)

        _assert_solutions_equivalent(sol_legacy, sol_snap)
        # T1 cannot grow on either path
        assert sol_snap.delta_w[1] <= 1e-9
        assert sol_legacy.delta_w[1] <= 1e-9

    def test_sector_cap_equivalence(self):
        """Two-sector universe with a tight cap on sector 0 — equivalence on the
        sector-cap surface that the snapshot owns."""
        sector_indicator = np.array(
            [[1.0, 1.0, 0.0],   # sector 0: T0, T1
             [0.0, 0.0, 1.0]],   # sector 1: T2
            dtype=float,
        )
        sector_cap_vec = np.array([0.25, 1.0])  # tight sector-0 cap
        common = dict(
            tickers=["A", "B", "C"],
            w_current=np.zeros(3),
            mu=np.array([0.03, 0.02, 0.01]),
            sigma=np.array([0.10, 0.12, 0.11]),
            w_upper=np.full(3, 0.20),
            cash_reserve=0.0,
            turnover_max=0.30,
            sector_indicator=sector_indicator,
            sector_cap_vec=sector_cap_vec,
        )
        ctx_legacy = _base_ctx(**common)
        _, sol_legacy = _run_task_capture_solution(ctx_legacy)

        ctx_snap = _base_ctx(**common)
        _stamp_snapshot(ctx_snap)
        _, sol_snap = _run_task_capture_solution(ctx_snap)

        _assert_solutions_equivalent(sol_legacy, sol_snap)
        # Sanity: sum of T0 + T1 weights respects the 0.25 cap (within solver tol)
        assert sol_snap.target_w[0] + sol_snap.target_w[1] <= 0.25 + 1e-6


# ── Coverage + performance guards ──────────────────────────────────────


class TestSnapshotForecastKwargsCoverage:
    """Static guard: every kwarg ``_build_solver_kwargs`` produces is
    either owned by the snapshot OR listed in ``_SNAPSHOT_FORECAST_KWARGS``.

    If this fails, someone added a knob without deciding which surface
    owns it — making the snapshot path silently drop the new kwarg.
    """

    # Kwargs the snapshot already owns (must NOT be in _SNAPSHOT_FORECAST_KWARGS).
    SNAPSHOT_OWNED = frozenset({
        "w_current", "w_upper", "w_lower", "dw_max",
        "cash_reserve", "wash_sale_mask",
        "drawdown", "drawdown_limit",
        "turnover_max", "gross_max",
        "sector_indicator", "sector_cap_vec",
        "corr_group_pairs",
    })

    def test_every_built_kwarg_has_an_owner(self):
        # Construct a minimal-but-real ctx so _build_solver_kwargs runs.
        ctx = _base_ctx(
            tickers=["A"],
            w_current=np.zeros(1),
            mu=np.array([0.01]),
            sigma=np.array([0.10]),
            w_upper=np.full(1, 0.20),
        )
        cfg = ctx.config["rotation"]["joint_actions"]
        kwargs = SolveMarkowitzQPTask._build_solver_kwargs(ctx, cfg)
        forecast_set = set(SolveMarkowitzQPTask._SNAPSHOT_FORECAST_KWARGS)

        missing = []
        for k in kwargs:
            if k in self.SNAPSHOT_OWNED:
                continue
            if k in forecast_set:
                continue
            missing.append(k)
        assert not missing, (
            f"kwargs without a snapshot/forecast owner: {missing} — extend "
            "either the ConstraintSnapshot or _SNAPSHOT_FORECAST_KWARGS"
        )

    def test_forecast_kwargs_and_snapshot_owned_disjoint(self):
        forecast_set = set(SolveMarkowitzQPTask._SNAPSHOT_FORECAST_KWARGS)
        overlap = forecast_set & self.SNAPSHOT_OWNED
        assert not overlap, (
            f"snapshot-owned kwargs duplicated in _SNAPSHOT_FORECAST_KWARGS: "
            f"{overlap} — this would double-pass and break the wrapper"
        )


class TestSnapshotFastPathOverhead:
    """The fast-path must add no measurable overhead beyond a log line."""

    def test_runtime_overhead_within_solver_noise(self):
        common = dict(
            tickers=["AAPL", "MSFT", "GOOG"],
            w_current=np.zeros(3),
            mu=np.array([0.01, 0.02, 0.015]),
            sigma=np.array([0.10, 0.15, 0.12]),
            w_upper=np.full(3, 0.20),
            cash_reserve=0.05,
            turnover_max=0.30,
        )
        # Warmup so cvxpy compilation cost doesn't dominate the first measurement.
        ctx_warm = _base_ctx(**common)
        SolveMarkowitzQPTask().run(ctx_warm)

        # 5 runs each side, take min wall time (least noisy).
        ts_legacy = []
        ts_snap = []
        for _ in range(5):
            ctx_l = _base_ctx(**common)
            t0 = time.perf_counter()
            SolveMarkowitzQPTask().run(ctx_l)
            ts_legacy.append(time.perf_counter() - t0)

            ctx_s = _base_ctx(**common)
            _stamp_snapshot(ctx_s)
            t0 = time.perf_counter()
            SolveMarkowitzQPTask().run(ctx_s)
            ts_snap.append(time.perf_counter() - t0)

        # The snapshot path should not be more than 3× the legacy path
        # — generous because cvxpy solve times have high jitter. The
        # actual added work is one dict comprehension over ~19 keys plus
        # one debug log; well under a millisecond.
        assert min(ts_snap) < 3.0 * min(ts_legacy), (
            f"snapshot fast-path overhead too large: "
            f"snap_min={min(ts_snap)*1000:.2f}ms vs "
            f"legacy_min={min(ts_legacy)*1000:.2f}ms"
        )


# ── Fallback: snapshot missing → legacy path still works ───────────────


class TestSnapshotMissingFallback:
    """Snapshot absent → the Task continues via the legacy kwargs path
    with no observable change. Pins the ``snap is None`` early-return in
    ``_initial_solve``."""

    def test_snapshot_absent_runs_legacy_path(self):
        ctx = _base_ctx(
            tickers=["AAPL"],
            w_current=np.zeros(1),
            mu=np.array([0.01]),
            sigma=np.array([0.10]),
            w_upper=np.full(1, 0.20),
        )
        # Important: do NOT stamp a snapshot.
        assert getattr(ctx, "_qp_constraint_snapshot", None) is None

        rv = SolveMarkowitzQPTask().run(ctx)
        # SolveMarkowitzQPTask returns False only on cvxportfolio
        # unsupported-constraints path. cvxpy path returns None.
        assert rv is None
        assert ctx._qp_solution is not None
        assert ctx._qp_status  # any non-empty status string is fine
