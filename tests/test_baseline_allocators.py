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
    hard_only_qp_allocator,
    hybrid_option_f_allocator,
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
    dw_max=None,
    turnover_max=0.30,
    gross_max=None,
    sector_indicator=None,
    sector_cap_vec=None,
    sector_names=None,
    corr_group_pairs=(),
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
        dw_max=np.asarray(
            dw_max if dw_max is not None else np.full(n, 0.5),
            dtype=float,
        ),
        cash_reserve=cash_reserve,
        turnover_max=turnover_max,
        drawdown=0.0,
        drawdown_limit=0.20,
        gross_max=gross_max,
        wash_sale_mask=np.asarray(
            wash_sale_mask if wash_sale_mask is not None else np.zeros(n, dtype=bool),
            dtype=bool,
        ),
        sector_indicator=sector_indicator,
        sector_cap_vec=sector_cap_vec,
        sector_names=sector_names,
        corr_group_pairs=corr_group_pairs,
    )


class TestEqualWeightTopK:
    def test_basic_5_assets_K3(self):
        # Raise the hard cap so the cap doesn't bind; we want to see
        # the equal-weight assignment cleanly. Also disable turnover
        # cap so the from-cash ‖Δw‖₁=1.0 is allowed.
        snap = _snap(5, w_upper_hard=np.full(5, 0.40), turnover_max=None)
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
        # Raise hard cap + disable turnover cap so cash budget binds first
        snap = _snap(
            3, w_upper_hard=np.full(3, 0.50),
            cash_reserve=0.10, turnover_max=None,
        )
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


class TestFullHardConstraintEnforcement:
    """**Codex #130 review HIGH regression guard.** The baseline allocators
    must respect every ``ConstraintSnapshot`` hard constraint family,
    not just ``w_upper_hard`` + wash-sale + cash budget. Codex's exact
    repro: snap.dw_max=[0.05]·2, turnover_max=0.05, sector cap 0.20,
    corr-pair cap 0.20 — equal-weight returned target_w=[0.5,0.5]
    violating all four.
    """

    def test_dw_max_respected(self):
        # dw_max = 0.05 per asset. Without it equal-weight top-2 would
        # be 0.50 each (Δw=0.50); with dw_max it must clip to ≤ 0.05.
        snap = _snap(
            2,
            w_upper_hard=np.full(2, 1.0),
            dw_max=np.array([0.05, 0.05]),
            turnover_max=None,
        )
        mu = np.array([0.05, 0.04])
        res = equal_weight_top_k(snap, mu=mu, K=2)
        assert np.all(np.abs(res.delta_w) <= 0.05 + 1e-9), (
            f"dw_max violated: |Δw|={np.abs(res.delta_w)}"
        )

    def test_turnover_max_respected(self):
        # turnover cap 0.05 — equal-weight top-2 wants ‖Δw‖₁=1.0
        snap = _snap(
            2,
            w_upper_hard=np.full(2, 1.0),
            dw_max=np.full(2, 1.0),  # disable dw_max
            turnover_max=0.05,
        )
        mu = np.array([0.05, 0.04])
        res = equal_weight_top_k(snap, mu=mu, K=2)
        l1 = float(np.sum(np.abs(res.delta_w)))
        assert l1 <= 0.05 + 1e-9, f"turnover cap violated: ‖Δw‖₁={l1}"

    def test_sector_cap_respected(self):
        # 2 names in sector 0 with cap 0.20 — equal-weight top-2 wants
        # 0.50 each = 1.00 sector load.
        snap = _snap(
            2,
            w_upper_hard=np.full(2, 1.0),
            dw_max=np.full(2, 1.0),
            turnover_max=None,
            sector_indicator=np.array([[1.0, 1.0]]),
            sector_cap_vec=np.array([0.20]),
            sector_names=("Tech",),
        )
        mu = np.array([0.05, 0.04])
        res = equal_weight_top_k(snap, mu=mu, K=2)
        sector_load = float(res.target_w[0] + res.target_w[1])
        assert sector_load <= 0.20 + 1e-9, (
            f"sector cap violated: load={sector_load}"
        )

    def test_correlation_pair_cap_respected(self):
        # Pair (0, 1) capped at 0.20 — equal-weight top-2 wants 1.00
        snap = _snap(
            2,
            w_upper_hard=np.full(2, 1.0),
            dw_max=np.full(2, 1.0),
            turnover_max=None,
            corr_group_pairs=((0, 1, 0.20),),
        )
        mu = np.array([0.05, 0.04])
        res = equal_weight_top_k(snap, mu=mu, K=2)
        pair_sum = float(res.target_w[0] + res.target_w[1])
        assert pair_sum <= 0.20 + 1e-9, (
            f"corr-pair cap violated: sum={pair_sum}"
        )

    def test_gross_max_respected(self):
        # gross cap 0.30 — equal-weight top-2 wants 1.00
        snap = _snap(
            2,
            w_upper_hard=np.full(2, 1.0),
            dw_max=np.full(2, 1.0),
            turnover_max=None,
            gross_max=0.30,
        )
        mu = np.array([0.05, 0.04])
        res = equal_weight_top_k(snap, mu=mu, K=2)
        gross = float(np.sum(np.abs(res.target_w)))
        assert gross <= 0.30 + 1e-9, f"gross cap violated: ‖w‖₁={gross}"

    def test_codex_exact_repro_all_four_constraints(self):
        """Codex's #130 repro values — multiple constraints simultaneously."""
        snap = _snap(
            2,
            w_upper_hard=np.full(2, 1.0),
            dw_max=np.array([0.05, 0.05]),
            turnover_max=0.05,
            sector_indicator=np.array([[1.0, 1.0]]),
            sector_cap_vec=np.array([0.20]),
            sector_names=("Tech",),
            corr_group_pairs=((0, 1, 0.20),),
        )
        mu = np.array([0.05, 0.04])
        # All three baselines must respect every constraint
        for name, res in [
            ("equal_weight", equal_weight_top_k(snap, mu=mu, K=2)),
            ("inverse_vol", inverse_vol_top_k(snap, mu=mu, sigma=np.full(2, 0.1), K=2)),
            ("fractional_kelly", fractional_kelly_top_k(snap, mu=mu, sigma=np.full(2, 0.1), K=2)),
        ]:
            max_dw = float(np.max(np.abs(res.delta_w)))
            l1 = float(np.sum(np.abs(res.delta_w)))
            sector_load = float(res.target_w[0] + res.target_w[1])
            pair_sum = sector_load  # same names in pair
            assert max_dw <= 0.05 + 1e-9, f"{name}: dw_max violated ({max_dw})"
            assert l1 <= 0.05 + 1e-9, f"{name}: turnover_max violated ({l1})"
            assert sector_load <= 0.20 + 1e-9, f"{name}: sector cap violated ({sector_load})"
            assert pair_sum <= 0.20 + 1e-9, f"{name}: corr-pair cap violated ({pair_sum})"

    def test_current_holding_above_cap_reports_infeasible_not_optimal(self):
        """If current book cannot be repaired within movement limits, fail loud."""
        snap = _snap(
            1,
            w_current=np.array([1.0]),
            w_upper_hard=np.array([0.20]),
            dw_max=np.array([0.05]),
            turnover_max=0.05,
        )
        res = equal_weight_top_k(snap, mu=np.array([0.05]), K=1)
        assert res.status.startswith("infeasible:"), res
        assert res.status in {"infeasible:dw_max", "infeasible:turnover_max"}
        assert np.abs(res.delta_w[0]) > snap.dw_max[0] + 1e-9

    def test_current_sector_over_cap_reports_infeasible_not_optimal(self):
        """Turnover scaling must not hide a sector cap violation."""
        snap = _snap(
            2,
            w_current=np.array([0.60, 0.60]),
            w_upper_hard=np.full(2, 1.0),
            dw_max=np.full(2, 1.0),
            turnover_max=0.05,
            sector_indicator=np.array([[1.0, 1.0]]),
            sector_cap_vec=np.array([0.20]),
            sector_names=("Tech",),
        )
        res = equal_weight_top_k(snap, mu=np.array([0.05, 0.04]), K=2)
        assert res.status.startswith("infeasible:"), res
        assert float(res.target_w.sum()) > 0.20 + 1e-9


class TestHardOnlyQPAllocator:
    """§8 Step 4f — 5th baseline. The QP solver with EVERY soft-penalty
    objective term zeroed (cvar=0, robust=0, cash_drag=0, signal_decay=0,
    tax=0, impact=0). Isolates the mean-variance core + hard constraints
    so the offline A/B can attribute lift between the soft-penalty stack
    and the optimisation gain.
    """

    def test_basic_feasible_solve(self):
        """Healthy 4-asset snapshot → optimal status, valid allocator output.

        μ̂ all positive, σ̂ uniform, no binding caps → solver should
        invest non-trivially in the highest-μ̂ names, hard caps respected.
        """
        snap = _snap(4, w_upper_hard=np.full(4, 0.40), turnover_max=None)
        mu = np.array([0.05, 0.04, 0.03, 0.02])
        sigma = np.array([0.10, 0.10, 0.10, 0.10])
        res = hard_only_qp_allocator(snap, mu=mu, sigma=sigma)
        assert isinstance(res, AllocatorResult)
        assert res.status == "optimal"
        # Hard cap respected per-asset
        assert (res.target_w <= snap.w_upper_hard + 1e-6).all()
        assert (res.target_w >= -1e-6).all()
        # Cash budget respected
        assert res.target_w.sum() <= 1.0 - snap.cash_reserve + 1e-6
        # delta_w + w_current == target_w (math sanity)
        np.testing.assert_allclose(
            res.target_w, snap.w_current + res.delta_w, atol=1e-9,
        )
        # selected_indices match |Δw| > 1e-9
        expected_sel = tuple(
            int(i) for i in np.where(np.abs(res.delta_w) > 1e-9)[0]
        )
        assert res.selected_indices == expected_sel
        # At least one name was actually sized (μ̂ > 0 everywhere)
        assert len(res.selected_indices) > 0

    def test_over_cap_holding_infeasible(self):
        """Contradictory hard constraints → ``infeasible:hard_only_qp:...``.

        High cash_reserve (0.95) + dw_max=0 (can't sell) + existing
        holdings totalling 0.50 → solver cannot satisfy
        Σwp ≤ 0.05 because wp is locked to w_current. The allocator
        must surface this as ``infeasible:hard_only_qp:<solver_status>``
        per the §8 Step 4f spec.
        """
        snap = _snap(
            3,
            w_current=np.array([0.20, 0.20, 0.10]),
            w_upper_hard=np.full(3, 0.30),
            w_upper=np.full(3, 0.30),
            cash_reserve=0.95,
            dw_max=np.zeros(3),  # locked — no trade possible
            turnover_max=None,
        )
        mu = np.array([0.05, 0.04, 0.03])
        sigma = np.array([0.10, 0.10, 0.10])
        res = hard_only_qp_allocator(snap, mu=mu, sigma=sigma)
        assert isinstance(res, AllocatorResult)
        assert res.status.startswith("infeasible:hard_only_qp:"), (
            f"expected infeasible:hard_only_qp:* prefix, got {res.status!r}"
        )
        # Infeasible fallback per solver convention: Δw=0, target=w_current
        np.testing.assert_allclose(res.delta_w, np.zeros(3), atol=1e-9)
        np.testing.assert_allclose(res.target_w, snap.w_current, atol=1e-9)

    def test_sector_cap_respected(self):
        """Hard sector cap binds — solver output must respect it.

        3 assets all in sector 0 with cap 0.15; the unconstrained
        QP would push allocation toward the 0.40 per-asset cap. After
        the sector cap the sector load must be ≤ 0.15.
        """
        snap = _snap(
            3,
            w_upper_hard=np.full(3, 0.40),
            w_upper=np.full(3, 0.40),
            turnover_max=None,
            sector_indicator=np.array([[1.0, 1.0, 1.0]]),
            sector_cap_vec=np.array([0.15]),
            sector_names=("Tech",),
        )
        mu = np.array([0.05, 0.04, 0.03])
        sigma = np.array([0.10, 0.10, 0.10])
        res = hard_only_qp_allocator(snap, mu=mu, sigma=sigma)
        assert res.status == "optimal"
        sector_load = float(res.target_w.sum())
        assert sector_load <= 0.15 + 1e-6, (
            f"sector cap violated: load={sector_load}"
        )

    def test_replay_harness_shape_contract(self):
        """Integration with the §8 Step 4b replay harness contract.

        The harness expects ``AllocatorResult`` with:
          - ``delta_w``: np.ndarray, shape (snap.n,), float
          - ``target_w``: np.ndarray, shape (snap.n,), float
          - ``status``: str
          - ``selected_indices``: tuple of int
        AND ``allocator(snap, mu=..., sigma=...)`` must be callable
        with that exact kwargs shape (the
        ``AllocatorFn = Callable[..., AllocatorResult]`` contract
        documented in ``allocator_replay.py``).

        We replay one synthetic bar end-to-end against the harness
        contract — fwd_return → daily P&L → turnover cost — without
        importing the (not-yet-merged) replay harness module. Shape +
        dtype + cost-math sanity is what the harness keys off.
        """
        n = 4
        snap = _snap(n, w_upper_hard=np.full(n, 0.40), turnover_max=None)
        mu = np.array([0.05, 0.04, 0.03, 0.02])
        sigma = np.array([0.10, 0.10, 0.10, 0.10])
        fwd_return = np.array([0.01, -0.005, 0.003, 0.002])
        cost_per_trade_bps = 5.0

        # Call signature must accept (snap, mu=, sigma=) per AllocatorFn
        res = hard_only_qp_allocator(snap, mu=mu, sigma=sigma)

        # Replay-harness shape contract — these are the EXACT
        # access patterns in allocator_replay.replay_one_allocator.
        assert isinstance(res, AllocatorResult)
        assert isinstance(res.delta_w, np.ndarray)
        assert isinstance(res.target_w, np.ndarray)
        assert res.delta_w.shape == (n,)
        assert res.target_w.shape == (n,)
        assert res.delta_w.dtype == np.float64
        assert res.target_w.dtype == np.float64
        assert isinstance(res.status, str)
        assert isinstance(res.selected_indices, tuple)
        for i in res.selected_indices:
            assert isinstance(i, int)

        # Reproduce the replay-harness daily-return math on this bar
        # (gross P&L − turnover cost) — proves the harness can compute
        # well-defined floats from the allocator's outputs.
        gross = float(np.sum(res.target_w * fwd_return))
        turn = float(np.sum(np.abs(res.delta_w)))
        cost = turn * cost_per_trade_bps * 1e-4
        daily = gross - cost
        assert np.isfinite(daily), "daily P&L must be finite"
        assert turn >= 0.0


class TestHybridOptionFAllocator:
    """§8 Step 4d — Hybrid Option F (parent memo §5)."""

    def test_basic_feasible_path_optimal_no_qp_fallback(self):
        """Stages 1-3 succeed; status='optimal', QP not invoked."""
        snap = _snap(
            5,
            w_upper_hard=np.full(5, 1.0),
            w_upper=np.full(5, 1.0),
            dw_max=np.full(5, 1.0),
            turnover_max=None,
            cash_reserve=0.05,
        )
        mu = np.array([0.05, 0.04, 0.03, 0.02, 0.01])
        sigma = np.full(5, 0.30)
        res = hybrid_option_f_allocator(
            snap, mu=mu, sigma=sigma, K=3,
            kelly_fraction=0.10, mu_shrinkage=0.0,
            edge_floor=0.0, min_dw=0.01,
        )
        assert isinstance(res, AllocatorResult)
        assert res.status == "optimal", (
            f"expected 'optimal' (no fallback), got {res.status!r}"
        )
        assert set(res.selected_indices) == {0, 1, 2}
        for i in res.selected_indices:
            assert res.target_w[i] > 0.0
        for i in (3, 4):
            assert res.target_w[i] == 0.0
        np.testing.assert_allclose(
            res.target_w, snap.w_current + res.delta_w, atol=1e-12,
        )

    def test_low_mu_falls_back_to_no_candidates(self):
        """Stage 1 returns no_candidates below the edge floor."""
        snap = _snap(3, w_upper_hard=np.full(3, 1.0), turnover_max=None)
        mu = np.array([0.005, 0.005, 0.005])
        sigma = np.full(3, 0.10)
        res = hybrid_option_f_allocator(
            snap, mu=mu, sigma=sigma, K=3,
            kelly_fraction=0.25, mu_shrinkage=0.5,
            edge_floor=0.001, min_dw=0.01,
        )
        assert res.status == "no_candidates"
        np.testing.assert_array_equal(res.target_w, snap.w_current)
        np.testing.assert_array_equal(res.delta_w, np.zeros(3))

    def test_over_cap_holding_triggers_qp_fallback_and_respects_hard_cap(self):
        """Over-cap current holding routes to QP fallback."""
        snap = _snap(
            3,
            w_current=np.array([0.50, 0.0, 0.0]),
            w_upper_hard=np.array([0.20, 0.20, 0.20]),
            w_upper=np.array([0.20, 0.20, 0.20]),
            dw_max=np.array([0.10, 0.10, 0.10]),
            turnover_max=None,
            cash_reserve=0.0,
        )
        mu = np.array([0.05, 0.04, 0.03])
        sigma = np.full(3, 0.10)
        res = hybrid_option_f_allocator(
            snap, mu=mu, sigma=sigma, K=3,
            kelly_fraction=0.25, mu_shrinkage=0.0,
            edge_floor=0.0, min_dw=0.01,
        )
        assert res.status in {
            "optimal:qp_fallback",
            "infeasible:hybrid_qp_fallback",
        }, f"expected fallback, got {res.status!r}"
        if res.status == "optimal:qp_fallback":
            assert res.target_w[0] <= snap.w_upper_hard[0] + 1e-9, (
                f"hard cap violated post-QP: target[0]={res.target_w[0]}"
            )
            assert (res.target_w <= snap.w_upper_hard + 1e-9).all()

    def test_sector_cap_violation_triggers_qp_fallback(self):
        """Sector-cap-violating raw target routes to QP fallback."""
        snap = _snap(
            3,
            w_upper_hard=np.full(3, 0.20),
            w_upper=np.full(3, 0.20),
            dw_max=np.full(3, 0.50),
            turnover_max=None,
            sector_indicator=np.array([[1.0, 1.0, 1.0]]),
            sector_cap_vec=np.array([0.15]),
            sector_names=("Tech",),
        )
        mu = np.array([0.05, 0.04, 0.03])
        sigma = np.full(3, 0.20)
        res = hybrid_option_f_allocator(
            snap, mu=mu, sigma=sigma, K=3,
            kelly_fraction=0.25, mu_shrinkage=0.0,
            edge_floor=0.0, min_dw=0.005,
        )
        assert res.status.startswith(("optimal:qp_fallback", "infeasible:")), (
            f"expected fallback or infeasible, got {res.status!r}"
        )
        if res.status == "optimal:qp_fallback":
            sector_load = float(res.target_w.sum())
            assert sector_load <= 0.15 + 1e-6, (
                f"sector cap violated post-QP: load={sector_load}"
            )

    def test_tight_turnover_cap_triggers_qp_fallback(self):
        """Stage 4 handles turnover caps that the min_dw band cannot."""
        snap = _snap(
            3,
            w_upper_hard=np.full(3, 1.0),
            w_upper=np.full(3, 1.0),
            dw_max=np.full(3, 1.0),
            turnover_max=0.05,
            cash_reserve=0.0,
        )
        mu = np.array([0.05, 0.04, 0.03])
        sigma = np.full(3, 0.30)
        res = hybrid_option_f_allocator(
            snap, mu=mu, sigma=sigma, K=3,
            kelly_fraction=0.10, mu_shrinkage=0.0,
            edge_floor=0.0, min_dw=0.001,
        )
        assert res.status in {
            "optimal:qp_fallback",
            "infeasible:hybrid_qp_fallback",
        }, f"expected fallback, got {res.status!r}"
        if res.status == "optimal:qp_fallback":
            l1 = float(np.sum(np.abs(res.delta_w)))
            assert l1 <= 0.05 + 1e-6, (
                f"turnover cap violated post-QP: ‖Δw‖₁={l1}"
            )

    def test_min_dw_band_zeros_tiny_trades(self):
        """Stage 3 snaps |Δw| < min_dw back to w_current."""
        snap = _snap(
            3,
            w_current=np.array([0.10, 0.10, 0.10]),
            w_upper_hard=np.full(3, 1.0),
            w_upper=np.full(3, 1.0),
            dw_max=np.full(3, 1.0),
            turnover_max=None,
        )
        mu = np.array([0.001, 0.001, 0.001])
        sigma = np.full(3, 0.05)
        res = hybrid_option_f_allocator(
            snap, mu=mu, sigma=sigma, K=3,
            kelly_fraction=0.25, mu_shrinkage=0.0,
            edge_floor=0.0, min_dw=0.02,
        )
        assert res.status == "optimal", (
            f"expected 'optimal' (band snap to w_current), got {res.status!r}"
        )
        np.testing.assert_allclose(
            res.target_w, snap.w_current, atol=0.02,
        )

    def test_replay_harness_integration_on_10_bar_sequence(self):
        """End-to-end 10-bar sequence keeps allocator contract intact."""
        rng = np.random.default_rng(2026)
        n_bars = 10
        n = 4
        w = np.zeros(n)
        statuses: list[str] = []
        for bar in range(n_bars):
            mu = rng.uniform(0.005, 0.05, size=n)
            sigma = rng.uniform(0.10, 0.30, size=n)
            snap = ConstraintSnapshot(
                n=n,
                tickers=tuple(f"T{i}" for i in range(n)),
                w_current=w.copy(),
                w_upper_hard=np.full(n, 0.30),
                w_upper=np.full(n, 0.30),
                w_lower=0.0,
                dw_max=np.full(n, 0.20),
                cash_reserve=0.05,
                turnover_max=0.15,
                drawdown=0.0,
                drawdown_limit=0.20,
                gross_max=None,
                wash_sale_mask=np.zeros(n, dtype=bool),
            )
            res = hybrid_option_f_allocator(
                snap, mu=mu, sigma=sigma, K=3,
                kelly_fraction=0.25, mu_shrinkage=0.1,
                edge_floor=0.001, min_dw=0.02,
            )
            assert isinstance(res, AllocatorResult)
            assert (res.target_w <= snap.w_upper_hard + 1e-6).all(), (
                f"bar {bar}: hard cap violated"
            )
            assert (res.target_w >= -1e-9).all(), (
                f"bar {bar}: long-only violated"
            )
            np.testing.assert_allclose(
                res.target_w, snap.w_current + res.delta_w, atol=1e-9,
            )
            statuses.append(res.status)
            w = res.target_w
        assert any(s == "optimal" or s == "optimal:qp_fallback" for s in statuses), (
            f"expected at least one feasible bar, got statuses={statuses}"
        )
