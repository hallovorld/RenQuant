"""ConstraintSnapshot — single, immutable hard-constraint contract.

Step 1 of the §8 measurement-and-contract plan (codex + gemini
convergent recommendation on PR #125, 2026-06-03).

**Why this module exists.** Before this PR the per-asset box bound was
*built* by four composed Tasks (``ComputeQPConstraintsTask →
ApplyExposureScalingTask → ApplyConvictionCapTask → sector/corr``)
each mutating ``ctx._qp_*`` fields in place. Three rejected revisions
of PR #123 (v1 / v2 / v3) all surfaced as constraint-composition
bugs: one Task widened the hard cap, another shrunk the soft cap
below the held weight, the cap-compliance retry consumed a value that
neither Task realised they were authoring jointly. The *class* — many
Tasks producing pieces of one constraint vector with no shared
contract — was open after PR #123 v4.

The :class:`ConstraintSnapshot` is the contract every candidate
allocator (current QP, simplified-QP, Hybrid, Level-2 MPO, inverse-
vol top-K, …) consumes. It carries the full hard-constraint set as
typed, frozen arrays with provenance for debugging. Once constructed
it cannot be mutated; the only way to change a constraint is to
construct a new snapshot upstream.

**This PR is strictly additive.** The existing Tasks still run and
still stamp the same ``ctx._qp_*`` fields; :func:`build_snapshot`
just *also* assembles a frozen view over them. Follow-up PRs migrate
the solver to read from the snapshot directly and (separately)
collapse the four composed Tasks into a single
``BuildQPConstraintsTask``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ConstraintSnapshot:
    """Immutable per-bar hard-constraint contract for the QP / Hybrid / MPO.

    Field naming mirrors the ``solve_portfolio_qp`` kwargs so the call
    sites are mechanical to migrate. Fields are validated at
    construction (:func:`_validate`) — a malformed snapshot fails loud
    at build time rather than silently degrading a downstream
    allocator.

    All per-asset arrays are length ``n`` and indexed identically to
    :attr:`tickers`. NumPy arrays are marked read-only so callers
    cannot mutate the snapshot via the array handles.
    """

    # ── Universe ──────────────────────────────────────────────────
    n: int
    tickers: tuple[str, ...]

    # ── Per-asset caps (the central object) ───────────────────────
    w_current: np.ndarray            # current weights, shape (n,)
    w_upper_hard: np.ndarray         # immutable hard risk cap, shape (n,)
    w_upper: np.ndarray              # soft target cap after scaling, shape (n,)
    w_lower: float                   # short-side cap (0 for long-only)
    dw_max: np.ndarray               # per-bar slippage cap, shape (n,)

    # ── Scalar limits ─────────────────────────────────────────────
    cash_reserve: float              # fraction of NAV held in cash
    turnover_max: Optional[float]    # L1 ‖Δw‖₁ cap; None disables
    drawdown: float                  # current drawdown (informational)
    drawdown_limit: float            # halts buys above this
    gross_max: Optional[float]       # ‖w‖₁ cap; None for long-only

    # ── Masks (per-asset, dtype bool) ─────────────────────────────
    wash_sale_mask: np.ndarray       # True ⇒ Δwᵢ ≤ 0 hard
    # Future masks (no_rebuy / forced_sells / soft_sell_thesis_age)
    # surface here when their consumer Tasks land in this contract.

    # ── Sector cap (optional) ─────────────────────────────────────
    sector_indicator: Optional[np.ndarray] = None    # shape (S, n)
    sector_cap_vec: Optional[np.ndarray] = None      # shape (S,)
    sector_names: Optional[tuple[str, ...]] = None
    missing_sector_tickers: tuple[str, ...] = ()

    # ── Correlation-group cap (optional) ──────────────────────────
    corr_group_pairs: tuple = ()                     # list of (i, j, cap)
    missing_correlation_tickers: tuple[str, ...] = ()

    # ── Provenance (for debugging + tests) ────────────────────────
    regime: Optional[str] = None
    confidence: Optional[float] = None
    conviction_caps: Optional[dict] = None
    sector_cap_source: Optional[str] = None
    contract_version: str = "v1-2026-06-03"

    def __post_init__(self) -> None:
        # Validate first (shape / dtype / finiteness / soft<=hard).
        _validate(self)
        # **Defensive copy** every ndarray BEFORE freezing it. Codex
        # #126 review caught the bug where constructing a snapshot
        # marked the caller's array read-only via ``flags.writeable``,
        # mutating their ctx and breaking the "snapshot does not
        # mutate ctx" contract. The snapshot owns its own arrays; the
        # caller keeps theirs writable.
        #
        # Frozen dataclasses require ``object.__setattr__`` for the
        # in-place attribute replacement.
        for attr in (
            "w_current", "w_upper_hard", "w_upper",
            "dw_max", "wash_sale_mask",
            "sector_indicator", "sector_cap_vec",
        ):
            arr = getattr(self, attr)
            if isinstance(arr, np.ndarray):
                owned = arr.copy()
                owned.flags.writeable = False
                object.__setattr__(self, attr, owned)


def _validate(snap: "ConstraintSnapshot") -> None:
    """Loud-failure validation of a snapshot.

    Anything that would silently produce an infeasible / incorrect
    QP fails here instead, with a message that names the violated
    invariant. Mirrors the bug class codex caught on #123 v3 — the
    silent soft-vs-hard cap conflation — by asserting at construction.
    """
    n = snap.n
    if n != len(snap.tickers):
        raise ValueError(
            f"ConstraintSnapshot: n={n} != len(tickers)={len(snap.tickers)}"
        )

    per_asset = {
        "w_current": snap.w_current,
        "w_upper_hard": snap.w_upper_hard,
        "w_upper": snap.w_upper,
        "dw_max": snap.dw_max,
        "wash_sale_mask": snap.wash_sale_mask,
    }
    for name, arr in per_asset.items():
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be np.ndarray, got {type(arr)}")
        if arr.shape != (n,):
            raise ValueError(
                f"{name} shape {arr.shape} != ({n},)"
            )

    # Finiteness on float arrays
    for name in ("w_current", "w_upper_hard", "w_upper", "dw_max"):
        arr = per_asset[name]
        if not np.isfinite(arr).all():
            bad = np.where(~np.isfinite(arr))[0].tolist()
            raise ValueError(
                f"{name} has non-finite entries at indices {bad}"
            )

    # The contract that codex's #123 review pinned in code:
    # the SOFT cap (w_upper) must never exceed the HARD cap (w_upper_hard).
    # PR #123 v1/v2/v3 violated this in three different ways.
    if (snap.w_upper > snap.w_upper_hard + 1e-9).any():
        bad = np.where(snap.w_upper > snap.w_upper_hard + 1e-9)[0]
        raise ValueError(
            "ConstraintSnapshot: soft cap exceeds hard cap at indices "
            f"{bad.tolist()} — w_upper={snap.w_upper[bad].tolist()} "
            f"vs w_upper_hard={snap.w_upper_hard[bad].tolist()}. "
            "This is the #123 v1/v2/v3 bug class; building the "
            "snapshot must not produce a soft cap above the hard cap."
        )

    # Cash reserve and drawdown bounds
    if not (0.0 <= snap.cash_reserve <= 1.0):
        raise ValueError(
            f"cash_reserve {snap.cash_reserve} must be in [0, 1]"
        )
    if snap.turnover_max is not None and snap.turnover_max < 0:
        raise ValueError(f"turnover_max {snap.turnover_max} must be >= 0")
    if snap.gross_max is not None and snap.gross_max <= 0:
        raise ValueError(f"gross_max {snap.gross_max} must be > 0")

    # Sector cap consistency
    if snap.sector_indicator is not None or snap.sector_cap_vec is not None:
        if snap.sector_indicator is None or snap.sector_cap_vec is None:
            raise ValueError(
                "sector_indicator and sector_cap_vec must both be set or both None"
            )
        if snap.sector_indicator.ndim != 2 or snap.sector_indicator.shape[1] != n:
            raise ValueError(
                f"sector_indicator shape {snap.sector_indicator.shape} "
                f"!= (S, {n})"
            )
        if snap.sector_indicator.shape[0] != snap.sector_cap_vec.shape[0]:
            raise ValueError(
                f"sector indicator rows {snap.sector_indicator.shape[0]} "
                f"!= sector_cap_vec length {snap.sector_cap_vec.shape[0]}"
            )


def build_snapshot_from_ctx(ctx) -> ConstraintSnapshot:
    """Read the existing ``ctx._qp_*`` constraint fields and freeze them.

    **Strictly additive.** Does not mutate ctx. Existing Tasks
    (``ComputeQPConstraintsTask`` / ``ApplyExposureScalingTask`` /
    ``ApplyConvictionCapTask`` / sector + corr Tasks) must have run
    before this is called. The returned snapshot is the contract
    follow-up PRs will route the solver through.

    A snapshot built from this function is byte-equivalent to the
    fields ``solve_portfolio_qp`` consumes via kwargs today — pinned
    by ``tests/test_constraint_snapshot.py::TestSnapshotMatchesKwargs``.
    """
    tickers = tuple(_get(ctx, "_qp_tickers") or ())
    n = len(tickers)

    w_current = _asarray_or_default(_get(ctx, "_qp_w_current"), np.zeros(n))
    w_upper_hard = np.asarray(_get(ctx, "_qp_w_upper_hard"), dtype=float)
    w_upper = np.asarray(_get(ctx, "_qp_w_upper"), dtype=float)
    dw_max = _asarray_or_default(_get(ctx, "_qp_dw_max"), np.full(n, 0.5))
    wash = _get(ctx, "_qp_wash_mask")
    wash_mask = (
        np.asarray(wash, dtype=bool) if wash is not None else np.zeros(n, dtype=bool)
    )

    return ConstraintSnapshot(
        n=n,
        tickers=tickers,
        w_current=w_current,
        w_upper_hard=w_upper_hard,
        w_upper=w_upper,
        w_lower=float(_get(ctx, "_qp_w_lower") or 0.0),
        dw_max=dw_max,
        cash_reserve=float(_get(ctx, "_qp_cash_reserve") or 0.0),
        turnover_max=_get(ctx, "_qp_turnover_max"),
        drawdown=float(_get(ctx, "_qp_drawdown") or 0.0),
        drawdown_limit=float(_get(ctx, "_qp_drawdown_limit") or 1.0),
        gross_max=_get(ctx, "_qp_gross_max"),
        wash_sale_mask=wash_mask,
        sector_indicator=_to_ndarray_or_none(_get(ctx, "_qp_sector_indicator")),
        sector_cap_vec=_to_ndarray_or_none(_get(ctx, "_qp_sector_cap_vec")),
        sector_names=_to_tuple_or_none(_get(ctx, "_qp_sector_names")),
        missing_sector_tickers=tuple(_get(ctx, "_qp_missing_sector_tickers") or ()),
        corr_group_pairs=tuple(_get(ctx, "_qp_corr_group_pairs") or ()),
        missing_correlation_tickers=tuple(
            _get(ctx, "_qp_missing_correlation_tickers") or ()
        ),
        regime=getattr(ctx, "regime", None),
        confidence=getattr(ctx, "confidence", None),
        conviction_caps=_get(ctx, "_qp_conviction_caps"),
        sector_cap_source=_get(ctx, "_qp_sector_cap_source"),
    )


def _get(ctx, name: str):
    """Read a private QP field via getattr — robust to non-dict ctx types."""
    return getattr(ctx, name, None)


def _asarray_or_default(value, default: np.ndarray) -> np.ndarray:
    """``np.asarray(value or default)`` is ambiguous when value is an
    ndarray; this helper resolves the truthiness explicitly.
    """
    if value is None:
        return default
    arr = np.asarray(value, dtype=float)
    return default if arr.size == 0 else arr


def _to_ndarray_or_none(value):
    if value is None:
        return None
    return np.asarray(value, dtype=float)


def _to_tuple_or_none(value):
    if value is None:
        return None
    return tuple(value)
