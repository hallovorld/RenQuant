"""Tests for baseline allocators (§8 Step 4 scaffolding).

The three baselines (equal_weight_top_k / inverse_vol_top_k /
fractional_kelly_top_k) are the simplest possible "competitive" rules.
Their job in the offline A/B replay is to bound the QP from below —
if the QP's optimization gain doesn't beat 1/N within top-K, we are
paying the complexity tax for noise (parent memo §2 + §4).

These tests pin:
1. Each baseline returns a valid AllocatorResult on a healthy snapshot.
2. Per-asset hard cap is respected (no Δw above ``w_upper_hard``).
3. Wash-sale-masked names cannot increase.
4. Cash-budget constraint Σw ≤ 1 - cash_reserve is respected.
5. ``no_candidates`` status when no μ̂ is positive.
6. Kelly guardrails (μ-shrinkage + edge floor) actually shrink positions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.portfolio_qp.baseline_allocators import (  # noqa: E402
    AllocatorResult,
    equal_weight_top_k,
    fractional_kelly_top_k,
    inverse_vol_top_k,
)
from kernel.portfolio_qp.constraint_snapshot import ConstraintSnapshot  # noqa: E402


def _snap(
    n: int,
    *,
    w_current=None,
    w_upper_hard=None,
    w_upper=None,
    cash_reserve: float = 0.0,
    wash_sale_mask=None,
) -> ConstraintSnapshot:
    return ConstraintSnapshot(
        n=n,
        tickers=tuple(f"T{i}" for i in range(n)),
        w_current=np.asarray(
            w_current if w_current is not None else np.zeros(n), dtype=float,
        ),
        w_upper_hard=np.asarray(
            w_upper_hard if w_upper_hard is not None else np.full(n, 0.20),
            dtype=float,
        ),
        w_upper=np.asarray(
            w_upper if w_upper is not None else np.full(n, 0.20),
            dtype=float,
        ),
        w_lower=0.0,
        dw_max=np.full(n, 0.5),
        cash_reserve=cash_reserve,
        turnover_max=0.30,
        drawdown=0.0,
        drawdown_limit=0.20,
        gross_max=None,
        wash_sale_mask=np.asarray(
            wash_sale_mask if wash_sale_mask is not None else np.zeros(n, dtype=bool),
            dtype=bool,
        ),
    )


class TestEqualWeightTopK:
    def test_basic_5_assets_K3(self):
        # Raise the hard cap so the cap doesn't bind; we want to see
        # the equal-weight assignment cleanly.
        snap = _snap(5, w_upper_hard=np.full(5, 0.40))
        mu = np.array([0.05, 0.03, 0.04, 0.01, 0.02])
        res = equal_weight_top_k(snap, mu=mu, K=3)
        assert isinstance(res, AllocatorResult)
        assert res.status == "optimal"
        # Top-3 by μ̂ = indices 0, 2, 1 (in descending μ̂ order)
        assert set(res.selected_indices) == {0, 1, 2}
        # Each top-K name gets 1/K of the budget (= 1.0 since cash_reserve=0)
        for i in res.selected_indices:
            assert abs(res.target_w[i] - 1.0 / 3.0) < 1e-9
        # Untouched names: target_w = 0
        for i in (3, 4):
            assert res.target_w[i] == 0.0
        # Σ target_w = budget
        assert abs(res.target_w.sum() - 1.0) < 1e-9

    def test_cash_reserve_respected(self):
        # Raise the hard cap so the cap doesn't bind first
        snap = _snap(3, w_upper_hard=np.full(3, 0.50), cash_reserve=0.10)
        mu = np.array([0.05, 0.04, 0.03])
        res = equal_weight_top_k(snap, mu=mu, K=3)
        assert abs(res.target_w.sum() - 0.90) < 1e-9

    def test_hard_cap_clips_oversize(self):
        """K=2 with budget 1.0 → 0.5 each, but cap is 0.20."""
        snap = _snap(4, w_upper_hard=np.full(4, 0.20))
        mu = np.array([0.05, 0.04, 0.03, 0.02])
        res = equal_weight_top_k(snap, mu=mu, K=2)
        # Each top-2 name capped at 0.20
        for i in res.selected_indices:
            assert res.target_w[i] <= 0.20 + 1e-9

    def test_no_positive_mu_returns_no_candidates(self):
        snap = _snap(3)
        mu = np.array([-0.01, -0.02, 0.0])
        res = equal_weight_top_k(snap, mu=mu, K=2)
        assert res.status == "no_candidates"
        np.testing.assert_array_equal(res.target_w, np.zeros(3))

    def test_wash_sale_masked_cannot_increase(self):
        snap = _snap(
            3,
            w_current=np.array([0.0, 0.10, 0.0]),
            wash_sale_mask=np.array([False, True, False]),
        )
        mu = np.array([0.05, 0.04, 0.03])  # T1 is top-3 but wash-sale-masked
        res = equal_weight_top_k(snap, mu=mu, K=3)
        # T1 cannot increase from its 0.10 current
        assert res.target_w[1] <= 0.10 + 1e-9


class TestInverseVolTopK:
    def test_lower_sigma_higher_weight(self):
        # Raise hard cap so cap doesn't equalise the three names
        snap = _snap(3, w_upper_hard=np.full(3, 0.60))
        mu = np.array([0.05, 0.04, 0.03])  # all positive
        sigma = np.array([0.10, 0.20, 0.30])  # T0 lowest vol
        res = inverse_vol_top_k(snap, mu=mu, sigma=sigma, K=3)
        assert res.status == "optimal"
        # T0 gets the largest weight (lowest σ)
        assert res.target_w[0] > res.target_w[1] > res.target_w[2]

    def test_hard_cap_respected(self):
        snap = _snap(2, w_upper_hard=np.full(2, 0.30))
        mu = np.array([0.05, 0.04])
        sigma = np.array([0.05, 0.50])  # huge inv-vol ratio
        res = inverse_vol_top_k(snap, mu=mu, sigma=sigma, K=2)
        assert res.target_w[0] <= 0.30 + 1e-9
        assert res.target_w[1] <= 0.30 + 1e-9

    def test_no_positive_mu(self):
        snap = _snap(3)
        mu = np.array([-0.01, 0.0, -0.02])
        sigma = np.array([0.10, 0.10, 0.10])
        res = inverse_vol_top_k(snap, mu=mu, sigma=sigma, K=2)
        assert res.status == "no_candidates"


class TestFractionalKellyTopK:
    def test_basic_kelly_sizing(self):
        snap = _snap(3)
        mu = np.array([0.05, 0.04, 0.03])
        sigma = np.array([0.10, 0.10, 0.10])
        # Full Kelly would give 0.05 / 0.10² = 5.0 (capped by hard cap)
        # 25% fractional Kelly → 1.25 (still capped)
        # σ² = 0.01, mu=0.05 → f* = 0.25 * 0.05 / 0.01 = 1.25 → capped at 0.20
        res = fractional_kelly_top_k(
            snap, mu=mu, sigma=sigma, K=3, kelly_fraction=0.25,
        )
        assert res.status == "optimal"
        # All three sized; the cap binds
        for i in (0, 1, 2):
            assert res.target_w[i] > 0.0

    def test_mu_shrinkage_reduces_position(self):
        """Higher μ-shrinkage → smaller positions (codex MED-7 guard)."""
        # Raise hard cap so cap doesn't bind in either case
        snap = _snap(2, w_upper_hard=np.full(2, 1.00))
        mu = np.array([0.05, 0.04])
        sigma = np.array([0.20, 0.20])  # bigger σ so Kelly target is small enough to not hit cap
        res_no_shrink = fractional_kelly_top_k(
            snap, mu=mu, sigma=sigma, K=2,
            kelly_fraction=0.10, mu_shrinkage=0.0,
        )
        res_shrunk = fractional_kelly_top_k(
            snap, mu=mu, sigma=sigma, K=2,
            kelly_fraction=0.10, mu_shrinkage=0.3,
        )
        assert res_shrunk.target_w[0] < res_no_shrink.target_w[0]
        assert res_shrunk.target_w[1] < res_no_shrink.target_w[1]

    def test_edge_floor_drops_low_mu(self):
        """Edge floor drops names below the uncertainty threshold."""
        snap = _snap(3)
        mu = np.array([0.05, 0.005, 0.04])  # T1 has tiny μ̂
        sigma = np.array([0.10, 0.10, 0.10])
        res = fractional_kelly_top_k(
            snap, mu=mu, sigma=sigma, K=3,
            kelly_fraction=0.25, edge_floor=0.01,
        )
        # T1's μ̂=0.005 is below the floor → dropped to zero
        assert res.target_w[1] == 0.0
        # T0 and T2 still sized
        assert res.target_w[0] > 0.0
        assert res.target_w[2] > 0.0

    def test_no_positive_mu_after_shrinkage(self):
        snap = _snap(3)
        mu = np.array([0.005, 0.005, 0.005])
        sigma = np.array([0.10, 0.10, 0.10])
        # μ̂ - 0.5·σ = 0.005 - 0.05 < 0 → all dropped
        res = fractional_kelly_top_k(
            snap, mu=mu, sigma=sigma, K=3,
            kelly_fraction=0.25, mu_shrinkage=0.5,
        )
        assert res.status == "no_candidates"


class TestAllAllocatorsSatisfyContract:
    """Cross-allocator invariants — these MUST hold for every baseline."""

    @pytest.fixture
    def snap_and_signals(self):
        snap = _snap(5, cash_reserve=0.05)
        mu = np.array([0.05, 0.04, 0.03, 0.02, 0.01])
        sigma = np.array([0.10, 0.12, 0.15, 0.18, 0.20])
        return snap, mu, sigma

    def test_delta_w_plus_w_current_equals_target_w(self, snap_and_signals):
        snap, mu, sigma = snap_and_signals
        for res in (
            equal_weight_top_k(snap, mu=mu, K=3),
            inverse_vol_top_k(snap, mu=mu, sigma=sigma, K=3),
            fractional_kelly_top_k(snap, mu=mu, sigma=sigma, K=3),
        ):
            np.testing.assert_allclose(
                res.target_w, snap.w_current + res.delta_w, atol=1e-12,
            )

    def test_target_w_within_hard_cap(self, snap_and_signals):
        snap, mu, sigma = snap_and_signals
        for res in (
            equal_weight_top_k(snap, mu=mu, K=3),
            inverse_vol_top_k(snap, mu=mu, sigma=sigma, K=3),
            fractional_kelly_top_k(snap, mu=mu, sigma=sigma, K=3),
        ):
            assert (res.target_w <= snap.w_upper_hard + 1e-9).all()
            assert (res.target_w >= -1e-9).all()  # long-only

    def test_cash_budget_respected(self, snap_and_signals):
        snap, mu, sigma = snap_and_signals
        budget = 1.0 - snap.cash_reserve
        for res in (
            equal_weight_top_k(snap, mu=mu, K=3),
            inverse_vol_top_k(snap, mu=mu, sigma=sigma, K=3),
            fractional_kelly_top_k(snap, mu=mu, sigma=sigma, K=3),
        ):
            assert res.target_w.sum() <= budget + 1e-9
