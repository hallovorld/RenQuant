"""Tests for the ConstraintSnapshot contract (Step 1 of §8 plan, PR #125).

The snapshot is the single hard-constraint contract every candidate
allocator (current QP, simplified-QP, Hybrid, Level-2 MPO, …) will
consume. These tests pin:

1. Constructor validation fails loud on each bug class the #123
   review caught (soft cap > hard cap; shape mismatch; non-finite
   entries; sector matrix inconsistency).
2. The snapshot is immutable: every per-asset array is read-only and
   the dataclass is frozen.
3. ``build_snapshot_from_ctx`` produces a snapshot whose fields are
   byte-equivalent to what ``solve_portfolio_qp`` consumes today via
   kwargs — so the follow-up PR that migrates the solver to read
   from the snapshot is a no-op behaviour change.

Reference: doc/research/2026-06-02-qp-architecture-review-and-alternatives.md §8.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.portfolio_qp.constraint_snapshot import (  # noqa: E402
    ConstraintSnapshot,
    build_snapshot_from_ctx,
)


def _valid_kwargs(n: int = 2) -> dict:
    """Minimal valid constructor kwargs for a 2-asset snapshot."""
    return dict(
        n=n,
        tickers=tuple(f"T{i}" for i in range(n)),
        w_current=np.zeros(n),
        w_upper_hard=np.full(n, 0.20),
        w_upper=np.full(n, 0.20),
        w_lower=0.0,
        dw_max=np.full(n, 0.50),
        cash_reserve=0.0,
        turnover_max=0.30,
        drawdown=0.0,
        drawdown_limit=0.20,
        gross_max=None,
        wash_sale_mask=np.zeros(n, dtype=bool),
    )


class TestSnapshotConstructorValidation:
    """The constructor must fail loud on every bug class #123 caught."""

    def test_valid_snapshot_constructs(self):
        snap = ConstraintSnapshot(**_valid_kwargs())
        assert snap.n == 2
        assert snap.tickers == ("T0", "T1")
        assert snap.contract_version == "v1-2026-06-03"

    def test_n_mismatches_tickers_fails(self):
        kw = _valid_kwargs()
        kw["tickers"] = ("only_one",)
        with pytest.raises(ValueError, match="n=2 != len.tickers.=1"):
            ConstraintSnapshot(**kw)

    def test_w_current_wrong_shape_fails(self):
        kw = _valid_kwargs()
        kw["w_current"] = np.zeros(3)
        with pytest.raises(ValueError, match="w_current shape .* != .2,."):
            ConstraintSnapshot(**kw)

    def test_w_upper_hard_must_be_ndarray(self):
        kw = _valid_kwargs()
        kw["w_upper_hard"] = [0.20, 0.20]
        with pytest.raises(TypeError, match="w_upper_hard must be np.ndarray"):
            ConstraintSnapshot(**kw)

    def test_non_finite_w_current_fails(self):
        kw = _valid_kwargs()
        kw["w_current"] = np.array([0.10, np.nan])
        with pytest.raises(ValueError, match="w_current has non-finite entries"):
            ConstraintSnapshot(**kw)

    def test_soft_cap_above_hard_cap_fails(self):
        """**The #123 v1/v2/v3 bug class, asserted at construction.**

        Three rejected revisions of PR #123 produced exactly this state
        (soft cap > hard cap, masking cap-compliance fallback). The
        snapshot fails loud here so a future Builder cannot ship the
        same bug.
        """
        kw = _valid_kwargs()
        kw["w_upper_hard"] = np.array([0.15, 0.15])
        kw["w_upper"] = np.array([0.22, 0.15])  # row 0 above hard
        with pytest.raises(
            ValueError, match="soft cap exceeds hard cap at indices"
        ):
            ConstraintSnapshot(**kw)

    def test_cash_reserve_outside_unit_interval_fails(self):
        kw = _valid_kwargs()
        kw["cash_reserve"] = 1.5
        with pytest.raises(ValueError, match="cash_reserve 1.5 must be in"):
            ConstraintSnapshot(**kw)

    def test_negative_turnover_max_fails(self):
        kw = _valid_kwargs()
        kw["turnover_max"] = -0.01
        with pytest.raises(ValueError, match="turnover_max .* must be >= 0"):
            ConstraintSnapshot(**kw)

    def test_sector_matrix_without_cap_vec_fails(self):
        kw = _valid_kwargs()
        kw["sector_indicator"] = np.ones((2, 2))
        with pytest.raises(
            ValueError, match="sector_indicator and sector_cap_vec must both"
        ):
            ConstraintSnapshot(**kw)

    def test_sector_matrix_wrong_n_fails(self):
        kw = _valid_kwargs()
        kw["sector_indicator"] = np.ones((1, 3))  # 3 != n=2
        kw["sector_cap_vec"] = np.array([0.5])
        with pytest.raises(
            ValueError, match=r"sector_indicator shape .* != \(S, 2\)"
        ):
            ConstraintSnapshot(**kw)


class TestSnapshotImmutability:
    """Frozen dataclass + read-only arrays — consumers cannot mutate."""

    def test_dataclass_is_frozen(self):
        snap = ConstraintSnapshot(**_valid_kwargs())
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            snap.w_upper = np.zeros(2)  # type: ignore[misc]

    def test_per_asset_arrays_are_read_only(self):
        snap = ConstraintSnapshot(**_valid_kwargs())
        for attr in ("w_current", "w_upper_hard", "w_upper",
                     "dw_max", "wash_sale_mask"):
            arr = getattr(snap, attr)
            with pytest.raises(ValueError, match="read-only|assignment"):
                arr[0] = 99.0


class TestSnapshotDoesNotMutateSourceArrays:
    """**Codex #126 review regression guard.** The snapshot must own its
    own arrays; constructing one must NOT freeze (or otherwise mutate)
    the caller's arrays.

    Bug codex caught: ``__post_init__`` was calling
    ``arr.flags.writeable = False`` on the *passed-in* arrays, so
    after a snapshot was built the caller's ``ctx._qp_w_*`` arrays
    became read-only. That violated the "does not mutate ctx"
    contract and broke the additive premise of this PR.
    """

    def test_direct_constructor_leaves_caller_arrays_writable(self):
        kw = _valid_kwargs()
        # Hold references to the caller-owned arrays
        caller_w_current = kw["w_current"]
        caller_w_upper = kw["w_upper"]
        caller_w_upper_hard = kw["w_upper_hard"]
        caller_dw_max = kw["dw_max"]
        caller_wash = kw["wash_sale_mask"]

        snap = ConstraintSnapshot(**kw)

        # The caller's arrays must remain writable AFTER construction
        for name, arr in [
            ("w_current", caller_w_current),
            ("w_upper", caller_w_upper),
            ("w_upper_hard", caller_w_upper_hard),
            ("dw_max", caller_dw_max),
            ("wash_sale_mask", caller_wash),
        ]:
            assert arr.flags.writeable, (
                f"caller's {name} got frozen by snapshot construction "
                f"(codex #126 bug)"
            )

        # The snapshot's arrays must be DISTINCT objects (defensive copy)
        for attr in ("w_current", "w_upper", "w_upper_hard",
                     "dw_max", "wash_sale_mask"):
            snap_arr = getattr(snap, attr)
            caller_arr = kw[attr]
            assert snap_arr is not caller_arr, (
                f"snapshot {attr} is the same object as the caller's — "
                "no defensive copy was made"
            )

    def test_build_from_ctx_leaves_ctx_arrays_writable(self):
        """The codex repro, exactly as posted on #126."""
        from types import SimpleNamespace
        ctx = SimpleNamespace()
        ctx._qp_tickers = ["A", "B"]
        ctx._qp_w_current = np.array([0.10, 0.05])
        ctx._qp_w_upper_hard = np.array([0.20, 0.20])
        ctx._qp_w_upper = np.array([0.20, 0.20])
        ctx._qp_dw_max = np.array([0.50, 0.50])
        ctx._qp_wash_mask = np.array([False, False])

        _ = build_snapshot_from_ctx(ctx)

        # The codex repro: this assignment must succeed
        ctx._qp_w_upper[0] = 0.1
        assert ctx._qp_w_upper[0] == 0.1, (
            "ctx._qp_w_upper was frozen by build_snapshot_from_ctx"
        )
        # All other ctx arrays remain writable
        ctx._qp_w_current[1] = 0.99
        ctx._qp_w_upper_hard[0] = 0.30
        ctx._qp_dw_max[0] = 0.40
        ctx._qp_wash_mask[1] = True

    def test_caller_mutation_does_not_leak_into_snapshot(self):
        """The dual direction: the snapshot's contents are independent
        of the caller's subsequent mutations."""
        from types import SimpleNamespace
        ctx = SimpleNamespace()
        ctx._qp_tickers = ["A"]
        ctx._qp_w_current = np.array([0.10])
        ctx._qp_w_upper_hard = np.array([0.20])
        ctx._qp_w_upper = np.array([0.15])

        snap = build_snapshot_from_ctx(ctx)

        # Caller mutates their own array
        ctx._qp_w_upper[0] = 0.05

        # The snapshot's value is unchanged (defensive copy worked)
        assert float(snap.w_upper[0]) == 0.15, (
            f"snapshot w_upper={snap.w_upper[0]} leaked from caller's "
            f"mutation (expected 0.15)"
        )


class TestSnapshotMatchesKwargs:
    """``build_snapshot_from_ctx`` must produce a snapshot whose fields
    match what ``solve_portfolio_qp`` consumes via kwargs today.

    This pins the byte-equivalence invariant the follow-up PR (which
    migrates the solver call sites) relies on.
    """

    def test_basic_ctx_roundtrip(self):
        ctx = SimpleNamespace()
        ctx._qp_tickers = ["AAPL", "MSFT", "GOOG"]
        ctx._qp_w_current = np.array([0.10, 0.05, 0.0])
        ctx._qp_w_upper_hard = np.array([0.20, 0.20, 0.20])
        ctx._qp_w_upper = np.array([0.15, 0.20, 0.20])  # AAPL soft-scaled
        ctx._qp_w_lower = 0.0
        ctx._qp_dw_max = np.full(3, 0.5)
        ctx._qp_cash_reserve = 0.05
        ctx._qp_turnover_max = 0.30
        ctx._qp_drawdown = 0.02
        ctx._qp_drawdown_limit = 0.20
        ctx._qp_gross_max = None
        ctx._qp_wash_mask = np.array([False, True, False])
        ctx.regime = "BULL_CALM"
        ctx.confidence = 0.7

        snap = build_snapshot_from_ctx(ctx)

        # Identity-equivalent fields the solver kwargs consume:
        assert snap.tickers == ("AAPL", "MSFT", "GOOG")
        np.testing.assert_array_equal(snap.w_current, ctx._qp_w_current)
        np.testing.assert_array_equal(snap.w_upper_hard, ctx._qp_w_upper_hard)
        np.testing.assert_array_equal(snap.w_upper, ctx._qp_w_upper)
        assert snap.w_lower == 0.0
        np.testing.assert_array_equal(snap.dw_max, ctx._qp_dw_max)
        assert snap.cash_reserve == 0.05
        assert snap.turnover_max == 0.30
        assert snap.drawdown_limit == 0.20
        np.testing.assert_array_equal(snap.wash_sale_mask, ctx._qp_wash_mask)
        # Provenance fields preserved
        assert snap.regime == "BULL_CALM"
        assert snap.confidence == 0.7

    def test_over_cap_holding_post_v4_snapshot_is_valid(self):
        """v4 contract: hard-cap-aware soft scaler keeps w_upper at the
        hard cap on over-cap rows. The snapshot of such a state must
        construct without raising — the validation only forbids soft >
        hard, not w_current > w_upper.
        """
        ctx = SimpleNamespace()
        ctx._qp_tickers = ["ORCL"]
        ctx._qp_w_current = np.array([0.22])      # over-cap holding
        ctx._qp_w_upper_hard = np.array([0.15])   # hard cap
        ctx._qp_w_upper = np.array([0.15])        # v4 keeps hard for over-cap rows
        ctx._qp_w_lower = 0.0
        ctx._qp_dw_max = np.array([0.5])
        ctx._qp_cash_reserve = 0.0
        ctx._qp_turnover_max = 0.30
        ctx._qp_drawdown = 0.0
        ctx._qp_drawdown_limit = 0.20
        ctx._qp_wash_mask = np.array([False])

        snap = build_snapshot_from_ctx(ctx)

        # Snapshot constructed — cap-compliance fallback path stays viable
        # because soft (0.15) == hard (0.15); solver sees infeasible for
        # the over-cap row exactly as the v4 test asserts.
        assert float(snap.w_upper[0]) == 0.15
        assert float(snap.w_upper_hard[0]) == 0.15
        assert float(snap.w_current[0]) == 0.22

    def test_missing_optional_fields_defaults(self):
        """ctx fields the user did not populate fall through to sane
        defaults; build does not raise on a minimally-populated ctx.
        """
        ctx = SimpleNamespace()
        ctx._qp_tickers = ["T0"]
        ctx._qp_w_current = np.zeros(1)
        ctx._qp_w_upper_hard = np.array([0.20])
        ctx._qp_w_upper = np.array([0.20])

        snap = build_snapshot_from_ctx(ctx)

        assert snap.w_lower == 0.0
        assert snap.cash_reserve == 0.0
        assert snap.turnover_max is None
        assert snap.gross_max is None
        np.testing.assert_array_equal(snap.wash_sale_mask, np.array([False]))
        assert snap.sector_indicator is None
        assert snap.regime is None
        assert snap.contract_version == "v1-2026-06-03"

    def test_sector_cap_fields_round_trip(self):
        ctx = SimpleNamespace()
        ctx._qp_tickers = ["A", "B", "C"]
        ctx._qp_w_current = np.zeros(3)
        ctx._qp_w_upper_hard = np.full(3, 0.20)
        ctx._qp_w_upper = np.full(3, 0.20)
        ctx._qp_sector_indicator = np.array(
            [[1.0, 0.0, 1.0],   # sector 0: A, C
             [0.0, 1.0, 0.0]]   # sector 1: B
        )
        ctx._qp_sector_cap_vec = np.array([0.30, 0.20])
        ctx._qp_sector_names = ("Tech", "Health")
        ctx._qp_missing_sector_tickers = ()
        ctx._qp_sector_cap_source = "config.sector_map"

        snap = build_snapshot_from_ctx(ctx)

        assert snap.sector_indicator is not None
        assert snap.sector_indicator.shape == (2, 3)
        np.testing.assert_array_equal(
            snap.sector_cap_vec, np.array([0.30, 0.20])
        )
        assert snap.sector_names == ("Tech", "Health")
        assert snap.sector_cap_source == "config.sector_map"
