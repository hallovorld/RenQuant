"""Regression tests for QP failure counter stamping (codex PR #48 review #1).

Mirror of renquant-pipeline/tests/test_qp_failure_counters.py. Umbrella
copy because the kernel/* lives in both repos for byte-equivalent
rollback (RQ_DAILY_RUNNER=umbrella).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

# Ensure umbrella kernel/* is importable when run from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY) not in sys.path:
    sys.path.insert(0, str(_STRATEGY))

import pytest

from kernel.portfolio_qp.tasks import _stamp_qp_failure_counter


def _ctx() -> types.SimpleNamespace:
    return types.SimpleNamespace(counters={})


def test_infeasible_stamps_qp_infeasible():
    ctx = _ctx()
    _stamp_qp_failure_counter(ctx, "infeasible")
    assert ctx.counters == {"qp_infeasible": 1}


def test_infeasible_with_reason_suffix_stamps_qp_infeasible():
    ctx = _ctx()
    _stamp_qp_failure_counter(ctx, "infeasible:cov_nan_pair")
    assert ctx.counters == {"qp_infeasible": 1}


def test_missing_solution_stamps_qp_missing_solution():
    ctx = _ctx()
    _stamp_qp_failure_counter(ctx, "missing_solution")
    assert ctx.counters == {"qp_missing_solution": 1}


def test_optimal_no_signal_stamps_qp_optimal_no_signal():
    ctx = _ctx()
    _stamp_qp_failure_counter(ctx, "optimal_no_signal")
    assert ctx.counters == {"qp_optimal_no_signal": 1}


def test_plain_optimal_stamps_nothing():
    ctx = _ctx()
    _stamp_qp_failure_counter(ctx, "optimal")
    assert ctx.counters == {}


def test_other_nonoptimal_stamps_qp_other_nonoptimal():
    ctx = _ctx()
    _stamp_qp_failure_counter(ctx, "qp_global:unbounded")
    assert ctx.counters == {"qp_other_nonoptimal": 1}


def test_empty_status_is_noop():
    ctx = _ctx()
    _stamp_qp_failure_counter(ctx, "")
    _stamp_qp_failure_counter(ctx, None)  # type: ignore[arg-type]
    assert ctx.counters == {}


def test_missing_counters_dict_is_noop():
    ctx = types.SimpleNamespace()
    _stamp_qp_failure_counter(ctx, "infeasible")


def test_repeated_calls_accumulate():
    ctx = _ctx()
    _stamp_qp_failure_counter(ctx, "infeasible")
    _stamp_qp_failure_counter(ctx, "infeasible:cov")
    _stamp_qp_failure_counter(ctx, "missing_solution")
    assert ctx.counters == {"qp_infeasible": 2, "qp_missing_solution": 1}


def test_compute_full_sigma_fail_stamps_counter():
    from kernel.portfolio_qp.tasks import ComputeFullSigmaTask
    task = ComputeFullSigmaTask()
    ctx = types.SimpleNamespace(counters={})
    task._fail_full_sigma(ctx, "cov_nan_pair")
    assert ctx._qp_status.startswith("infeasible:")
    assert ctx.counters.get("qp_infeasible") == 1
