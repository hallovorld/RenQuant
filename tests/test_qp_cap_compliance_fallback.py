"""Regression coverage for _retry_for_per_asset_cap_compliance.

Audit #2 / issue #70: when prod QP is infeasible AND at least one holding
is over its per-asset cap, allow a deterministic force-sell-to-cap fallback
(opt-in via ``cfg.portfolio_qp.allow_cap_compliance_sells_on_infeasible``).

Mirror of ``renquant-pipeline/tests/test_qp_cap_compliance_fallback.py``
because ``kernel/*`` lives in both repos (Phase 1 byte-equivalent rollback
under ``RQ_DAILY_RUNNER=umbrella``).

These tests pin the invariants:

1. No-op when QP is feasible (sol.status="optimal").
2. No-op when no holding is over cap, even if QP is infeasible.
3. Generates sells to bring each over-cap holding to exactly cap.
4. Other holdings get Δw = 0.
5. Synthetic solution has status="cap_compliance_fallback".
6. Diagnostics record n_sold + total_sold for audit visibility.
7. Returns the original solution if w_current or w_upper is missing.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

# Ensure umbrella kernel/* is importable when run from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY) not in sys.path:
    sys.path.insert(0, str(_STRATEGY))

from kernel.portfolio_qp.tasks import (  # noqa: E402
    _retry_for_per_asset_cap_compliance,
)


@dataclass
class _FakeQPSolution:
    """Minimal stand-in for `qp_solver.QPSolution` — captures the surface
    the fallback function touches without pulling in the real cvxpy build."""
    status: str
    delta_w: np.ndarray
    target_w: np.ndarray
    objective: float = 0.0
    n_iter: int = -1
    diagnostics: dict = field(default_factory=dict)


def _kwargs(w_current, w_upper):
    return {
        "w_current": np.asarray(w_current, dtype=float),
        "w_upper": np.asarray(w_upper, dtype=float),
    }


def _solve_noop(**_kwargs):  # solve_fn — not actually called by the fallback
    raise AssertionError(
        "solve_fn must not be called; cap-compliance fallback is deterministic"
    )


def test_noop_when_status_is_optimal():
    n = 3
    sol = _FakeQPSolution(
        status="optimal",
        delta_w=np.zeros(n),
        target_w=np.zeros(n),
    )
    result = _retry_for_per_asset_cap_compliance(
        sol, _kwargs([0.20, 0.10, 0.05], [0.15, 0.15, 0.15]), _solve_noop,
    )
    assert result is sol  # unchanged identity
    assert result.status == "optimal"


def test_noop_when_no_holding_over_cap():
    """Infeasible status but every holding ≤ cap — return original."""
    n = 3
    sol = _FakeQPSolution(
        status="infeasible:primal_infeasible",
        delta_w=np.zeros(n),
        target_w=np.array([0.10, 0.10, 0.10]),
    )
    result = _retry_for_per_asset_cap_compliance(
        sol, _kwargs([0.10, 0.10, 0.10], [0.15, 0.15, 0.15]), _solve_noop,
    )
    assert result is sol
    assert result.status.startswith("infeasible")


def test_force_sell_brings_over_cap_holding_to_cap():
    """One holding at 0.20, cap=0.15 → fallback emits Δw=-0.05 to bring to cap."""
    n = 3
    sol = _FakeQPSolution(
        status="infeasible:primal_infeasible",
        delta_w=np.zeros(n),
        target_w=np.array([0.20, 0.10, 0.05]),
    )
    result = _retry_for_per_asset_cap_compliance(
        sol, _kwargs([0.20, 0.10, 0.05], [0.15, 0.15, 0.15]), _solve_noop,
    )
    assert result.status == "cap_compliance_fallback"
    np.testing.assert_allclose(result.delta_w, [-0.05, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(result.target_w, [0.15, 0.10, 0.05], atol=1e-9)


def test_only_over_cap_assets_get_sold_others_held():
    """Mixed: 2 over cap, 2 under cap — Δw nonzero only for over-cap pair."""
    sol = _FakeQPSolution(
        status="infeasible:something",
        delta_w=np.zeros(4),
        target_w=np.array([0.18, 0.05, 0.16, 0.08]),
    )
    result = _retry_for_per_asset_cap_compliance(
        sol, _kwargs([0.18, 0.05, 0.16, 0.08], [0.15, 0.15, 0.15, 0.15]),
        _solve_noop,
    )
    assert result.status == "cap_compliance_fallback"
    np.testing.assert_allclose(
        result.delta_w, [-0.03, 0.0, -0.01, 0.0], atol=1e-9,
    )
    np.testing.assert_allclose(
        result.target_w, [0.15, 0.05, 0.15, 0.08], atol=1e-9,
    )


def test_per_asset_caps_can_differ():
    """w_upper per-asset (not scalar) — fallback honors each cap."""
    sol = _FakeQPSolution(
        status="infeasible:primal_infeasible",
        delta_w=np.zeros(3),
        target_w=np.array([0.12, 0.20, 0.10]),
    )
    result = _retry_for_per_asset_cap_compliance(
        sol,
        _kwargs([0.12, 0.20, 0.10], [0.10, 0.15, 0.15]),  # caps differ!
        _solve_noop,
    )
    assert result.status == "cap_compliance_fallback"
    np.testing.assert_allclose(
        result.delta_w, [-0.02, -0.05, 0.0], atol=1e-9,
    )


def test_diagnostics_record_n_sold_and_total():
    """Diagnostics must carry n_sold + total_sold for audit visibility."""
    sol = _FakeQPSolution(
        status="infeasible:foo",
        delta_w=np.zeros(3),
        target_w=np.array([0.20, 0.18, 0.05]),
        diagnostics={"prior": "value"},
    )
    result = _retry_for_per_asset_cap_compliance(
        sol, _kwargs([0.20, 0.18, 0.05], [0.15, 0.15, 0.15]), _solve_noop,
    )
    d = result.diagnostics
    assert d["c2_infeasible_policy"] == "cap_compliance_fallback"
    assert d["cap_compliance_n_sold"] == 2
    assert d["cap_compliance_total_sold"] == pytest.approx(0.05 + 0.03, abs=1e-9)
    # Prior diagnostics preserved (we merge into the existing dict).
    assert d["prior"] == "value"


def test_noop_when_w_current_missing():
    """Defensive: missing kwarg → return original sol unchanged."""
    sol = _FakeQPSolution(
        status="infeasible:foo",
        delta_w=np.zeros(2),
        target_w=np.zeros(2),
    )
    result = _retry_for_per_asset_cap_compliance(
        sol, {"w_upper": np.array([0.15, 0.15])}, _solve_noop,
    )
    assert result is sol


def test_noop_when_n_zero():
    """Empty portfolio: nothing to do."""
    sol = _FakeQPSolution(
        status="infeasible:foo",
        delta_w=np.zeros(0),
        target_w=np.zeros(0),
    )
    result = _retry_for_per_asset_cap_compliance(
        sol, _kwargs([], [0.15]), _solve_noop,
    )
    assert result is sol


def test_tiny_overcap_within_numerical_tolerance_is_noop():
    """w_current = w_upper + 1e-10 should NOT trigger force-sell. Per-asset
    cap check uses a 1e-9 tolerance to avoid floating-point noise."""
    sol = _FakeQPSolution(
        status="infeasible:foo",
        delta_w=np.zeros(2),
        target_w=np.array([0.15 + 1e-10, 0.10]),
    )
    result = _retry_for_per_asset_cap_compliance(
        sol, _kwargs([0.15 + 1e-10, 0.10], [0.15, 0.15]), _solve_noop,
    )
    assert result is sol
    assert result.status.startswith("infeasible")


# ── codex #75 blocker: cap_compliance_fallback ≠ QP failure ────────────────

def test_solve_task_does_not_stamp_failure_for_cap_compliance_fallback():
    """codex #75 blocker fix: when SolveMarkowitzQPTask returns the fallback,
    upstream stamping must NOT mark it as a QP failure. Replays the
    failure-stamping branch with a synthetic cap_compliance_fallback
    solution and asserts (a) no _qp_failure_reason, (b) no counters
    incremented, (c) ctx._qp_status + diagnostics still recorded."""
    import types

    from kernel.portfolio_qp import tasks as qp_tasks  # noqa: E402

    sol = _FakeQPSolution(
        status="cap_compliance_fallback",
        delta_w=np.array([-0.05, 0.0, 0.0]),
        target_w=np.array([0.15, 0.10, 0.05]),
        diagnostics={"c2_infeasible_policy": "cap_compliance_fallback",
                     "cap_compliance_n_sold": 1,
                     "cap_compliance_total_sold": 0.05},
    )
    ctx = types.SimpleNamespace(counters={})
    ctx._qp_tickers = ["A", "B", "C"]                               # noqa: SLF001

    ctx._qp_solution = sol                                          # noqa: SLF001
    ctx._qp_status = str(sol.status)                                # noqa: SLF001
    ctx._qp_diagnostics = dict(sol.diagnostics or {})               # noqa: SLF001
    if sol.status not in qp_tasks.QP_EMITTABLE_STATUSES:
        reason = (
            "qp_no_signal" if sol.status == "optimal_no_signal"
            else f"qp_global:{sol.status}"
        )
        ctx._qp_failure_reason = reason                             # noqa: SLF001
        qp_tasks._stamp_all_qp_blocks(ctx, reason)
        qp_tasks._stamp_qp_failure_counter(ctx, ctx._qp_status)     # noqa: SLF001

    assert not hasattr(ctx, "_qp_failure_reason"), (
        f"cap_compliance_fallback must NOT set _qp_failure_reason; "
        f"got {getattr(ctx, '_qp_failure_reason', None)!r}"
    )
    for key in ("qp_infeasible", "qp_missing_solution", "qp_optimal_no_signal",
                "qp_other_nonoptimal"):
        assert ctx.counters.get(key, 0) == 0, (
            f"cap_compliance_fallback bumped counters.{key} = {ctx.counters[key]}"
        )
    assert ctx._qp_status == "cap_compliance_fallback"
    assert ctx._qp_diagnostics["c2_infeasible_policy"] == "cap_compliance_fallback"


def test_emit_and_solve_use_same_emittable_status_set():
    """Codex #75 invariant: the emittable set must be a single source of
    truth. Pin that the class-level alias is identical to the module-level
    constant."""
    from kernel.portfolio_qp.tasks import (  # noqa: E402
        EmitOrdersFromQPSolutionTask,
        QP_EMITTABLE_STATUSES,
    )

    assert EmitOrdersFromQPSolutionTask._EMITTABLE_STATUSES is QP_EMITTABLE_STATUSES
    assert "optimal" in QP_EMITTABLE_STATUSES
    assert "cap_compliance_fallback" in QP_EMITTABLE_STATUSES


def test_optimal_no_signal_still_treated_as_failure():
    """Sanity guard: ``optimal_no_signal`` is NOT in QP_EMITTABLE_STATUSES, so
    a true no-signal solve must still set _qp_failure_reason=qp_no_signal."""
    import types

    from kernel.portfolio_qp import tasks as qp_tasks  # noqa: E402

    sol = _FakeQPSolution(
        status="optimal_no_signal",
        delta_w=np.zeros(2),
        target_w=np.array([0.1, 0.1]),
    )
    ctx = types.SimpleNamespace(counters={})
    ctx._qp_tickers = ["A", "B"]                                    # noqa: SLF001

    ctx._qp_solution = sol                                          # noqa: SLF001
    ctx._qp_status = str(sol.status)                                # noqa: SLF001
    if sol.status not in qp_tasks.QP_EMITTABLE_STATUSES:
        reason = (
            "qp_no_signal" if sol.status == "optimal_no_signal"
            else f"qp_global:{sol.status}"
        )
        ctx._qp_failure_reason = reason                             # noqa: SLF001
        qp_tasks._stamp_qp_failure_counter(ctx, ctx._qp_status)     # noqa: SLF001

    assert getattr(ctx, "_qp_failure_reason", None) == "qp_no_signal"
    assert ctx.counters.get("qp_optimal_no_signal", 0) >= 1
