"""Baseline allocators for the §8 Step 4 offline WF A/B replay.

Three closed-form, no-solver baselines used to bound the QP and Hybrid
candidates from below in the offline A/B replay (PR #125, §8 Step 4).
Each baseline consumes the same immutable :class:`ConstraintSnapshot`
contract introduced in #126 so the replay harness has a single input
interface to test against.

The baselines are deliberately simple — they are NOT recommendations
for production sizing. Their purpose is to answer: "given μ̂ at the
current IC and Σ̂ noise level, does the QP's optimization gain
actually beat the simplest possible rule?". If a closed-form baseline
matches the QP's Sharpe within DSR/PBO tolerance, the complexity tax
(§2 of the parent memo) is paying for noise.

References
----------
- DeMiguel, Garlappi & Uppal (2009) *RFS* 22(5) — naive 1/N benchmark
- Kelly (1956) / Thorp (1969) — per-name fractional Kelly
- López de Prado (2018) ch.3 — inverse-vol weighting as risk parity floor
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from kernel.portfolio_qp.constraint_snapshot import ConstraintSnapshot


@dataclass(frozen=True)
class AllocatorResult:
    """Output of a baseline allocator.

    Shape-compatible with ``QPSolution`` so the A/B harness can compute
    metrics against either via a single code path. The status strings
    mirror the QP convention: ``"optimal"`` on a clean allocation,
    ``"all_cash"`` when no candidate has positive expected return.
    """

    delta_w: np.ndarray              # per-asset Δw (n,)
    target_w: np.ndarray             # post-trade weights w_current + Δw (n,)
    status: str                       # "optimal" / "all_cash" / "no_candidates"
    selected_indices: tuple[int, ...]  # indices that were sized (post top-K)


def _select_top_k(
    mu: np.ndarray,
    snap: ConstraintSnapshot,
    K: int,
) -> tuple[int, ...]:
    """Top-K names by μ̂ that are NOT wash-sale-blocked.

    Wash-sale-masked names can be HELD but not increased; we still
    allow them in the selection so the baseline can choose to hold
    them (Δw=0) but not buy more. The Δw assignment downstream
    enforces the no-buy bound.
    """
    candidate_idx = [
        i for i in range(snap.n)
        if np.isfinite(mu[i]) and mu[i] > 0.0
    ]
    if not candidate_idx:
        return ()
    candidate_idx.sort(key=lambda i: -float(mu[i]))
    return tuple(candidate_idx[:K])


def _build_result(
    snap: ConstraintSnapshot,
    target_pct: np.ndarray,
    selected: tuple[int, ...],
    status: str,
) -> AllocatorResult:
    """Project the proposed target into the full snapshot feasible set.

    Codex #130 review HIGH: an earlier version applied only
    ``w_upper_hard`` clip + wash-sale + cash budget, ignoring
    ``dw_max``, ``turnover_max``, sector caps, correlation-group
    caps, and ``gross_max``. The replay harness then compared QP
    against infeasible closed-form portfolios. This routine now
    sequentially projects to satisfy every hard constraint family
    advertised by :class:`ConstraintSnapshot`.

    Order is intentional — earlier projections feed later ones, so
    later projections never re-introduce a violation in the earlier
    family. When a family cannot be satisfied without exiting all
    positions, the returned ``status`` is
    ``"infeasible:<family>"`` and the replay harness counts the
    violation.
    """
    n = snap.n
    target = np.clip(np.asarray(target_pct, dtype=float), 0.0, snap.w_upper_hard)

    # ── 1. Wash-sale: Δw ≤ 0 for masked names ─────────────────
    if snap.wash_sale_mask.any():
        target = np.where(
            snap.wash_sale_mask,
            np.minimum(target, snap.w_current),
            target,
        )

    # ── 2. dw_max: |Δw| ≤ dw_max per asset ───────────────────
    if snap.dw_max is not None:
        target = np.clip(
            target,
            snap.w_current - snap.dw_max,
            snap.w_current + snap.dw_max,
        )
        target = np.clip(target, 0.0, snap.w_upper_hard)

    # ── 3. Cash budget Σw ≤ 1 - cash_reserve ──────────────────
    budget = max(0.0, 1.0 - float(snap.cash_reserve))
    total = float(target.sum())
    if total > budget and total > 0.0:
        target = target * (budget / total)
        target = np.clip(target, 0.0, snap.w_upper_hard)

    # ── 4. Sector cap S @ w ≤ sector_cap_vec ──────────────────
    if snap.sector_indicator is not None and snap.sector_cap_vec is not None:
        S = snap.sector_indicator
        cap = snap.sector_cap_vec
        # Per-sector scaling: if S_s @ w > cap_s, scale all names in
        # sector s by cap_s / (S_s @ w). Iterate (rarely > 1 round)
        # until satisfied or a sector becomes infeasible.
        for _ in range(5):  # bounded iterations
            sector_loads = S @ target
            over = sector_loads > cap + 1e-9
            if not over.any():
                break
            for s in np.where(over)[0]:
                load = float(sector_loads[s])
                if load <= 0:
                    continue
                scale = float(cap[s]) / load
                in_sector = S[s].astype(bool)
                target[in_sector] *= scale
            target = np.clip(target, 0.0, snap.w_upper_hard)
        else:
            return AllocatorResult(
                delta_w=np.zeros(n), target_w=snap.w_current,
                status="infeasible:sector_cap",
                selected_indices=selected,
            )

    # ── 5. Correlation-group cap w_i + w_j ≤ corr_cap ─────────
    for trip in snap.corr_group_pairs or ():
        # trip is (i, j, cap) per ConstraintSnapshot docstring
        try:
            i, j, cap = int(trip[0]), int(trip[1]), float(trip[2])
        except (TypeError, IndexError, ValueError):
            continue
        if i >= n or j >= n:
            continue
        pair_sum = float(target[i] + target[j])
        if pair_sum > cap + 1e-9 and pair_sum > 0:
            scale = cap / pair_sum
            target[i] *= scale
            target[j] *= scale

    # ── 6. Turnover cap ‖Δw‖₁ ≤ turnover_max ──────────────────
    if snap.turnover_max is not None:
        delta = target - snap.w_current
        l1 = float(np.sum(np.abs(delta)))
        if l1 > float(snap.turnover_max) + 1e-9 and l1 > 0:
            scale = float(snap.turnover_max) / l1
            target = snap.w_current + delta * scale

    # ── 7. Gross cap ‖w‖₁ ≤ gross_max ─────────────────────────
    if snap.gross_max is not None:
        gross = float(np.sum(np.abs(target)))
        if gross > float(snap.gross_max) + 1e-9 and gross > 0:
            scale = float(snap.gross_max) / gross
            target = target * scale

    target = np.clip(target, 0.0, snap.w_upper_hard)

    return AllocatorResult(
        delta_w=target - snap.w_current,
        target_w=target,
        status=status,
        selected_indices=selected,
    )


def equal_weight_top_k(
    snap: ConstraintSnapshot,
    *,
    mu: Sequence[float],
    K: int = 5,
) -> AllocatorResult:
    """1/N within top-K candidates (DeMiguel 2009 benchmark).

    Selects the K names with the largest positive μ̂ and assigns each
    ``min(1/K · budget, w_upper_hard[i])``. Held names that don't make
    top-K are dropped to zero (target_w → 0 → Δw < 0 sell). This is
    the simplest possible "competitive" baseline.

    Returns ``status="no_candidates"`` when no μ̂ is positive.
    """
    mu_arr = np.asarray(mu, dtype=float)
    if mu_arr.shape != (snap.n,):
        raise ValueError(f"mu shape {mu_arr.shape} != ({snap.n},)")
    selected = _select_top_k(mu_arr, snap, K)
    if not selected:
        return _build_result(snap, np.zeros(snap.n), (), "no_candidates")
    budget = max(0.0, 1.0 - float(snap.cash_reserve))
    per_name = budget / len(selected)
    target = np.zeros(snap.n)
    for i in selected:
        target[i] = per_name
    return _build_result(snap, target, selected, "optimal")


def inverse_vol_top_k(
    snap: ConstraintSnapshot,
    *,
    mu: Sequence[float],
    sigma: Sequence[float],
    K: int = 5,
) -> AllocatorResult:
    """Inverse-volatility weight within top-K candidates (risk-parity floor).

    Selects the K names with the largest positive μ̂; assigns
    ``w_i ∝ 1/σ_i`` then renormalises to the budget. Per-asset hard
    cap applied after. Closed-form, no solver, ignores correlation
    (this is the López de Prado risk-parity *floor* — full HRP would
    cluster first).
    """
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    if mu_arr.shape != (snap.n,) or sigma_arr.shape != (snap.n,):
        raise ValueError(
            f"mu/sigma shape mismatch with snap.n={snap.n}: "
            f"mu={mu_arr.shape} sigma={sigma_arr.shape}"
        )
    selected = _select_top_k(mu_arr, snap, K)
    if not selected:
        return _build_result(snap, np.zeros(snap.n), (), "no_candidates")
    inv_sig = np.array([
        1.0 / max(float(sigma_arr[i]), 1e-6) for i in selected
    ])
    inv_sig_sum = float(inv_sig.sum())
    if inv_sig_sum <= 0.0:
        return _build_result(snap, np.zeros(snap.n), selected, "no_candidates")
    weights = inv_sig / inv_sig_sum
    budget = max(0.0, 1.0 - float(snap.cash_reserve))
    target = np.zeros(snap.n)
    for w, i in zip(weights, selected, strict=True):
        target[i] = float(w) * budget
    return _build_result(snap, target, selected, "optimal")


def fractional_kelly_top_k(
    snap: ConstraintSnapshot,
    *,
    mu: Sequence[float],
    sigma: Sequence[float],
    K: int = 5,
    kelly_fraction: float = 0.25,
    mu_shrinkage: float = 0.0,
    edge_floor: Optional[float] = None,
) -> AllocatorResult:
    """Per-name fractional Kelly with μ̂ shrinkage and edge floor.

    f*_i = kelly_fraction · max(μ̂_i - shrinkage·σ_i, 0) / σ_i²

    Required guardrails (codex MED-7 on parent memo §3 / 3-questions
    addendum §3):

    * ``kelly_fraction`` — fractional Kelly. Full Kelly with point
      estimates blows up under μ̂ noise (Thorp 1969 caveat); 25% is
      the conservative-bank-rate default.
    * ``mu_shrinkage`` — subtract a multiple of σ from μ̂ before sizing
      (Garlappi-Uppal-Wang 2007 robust shrinkage; conservative under
      μ-noise).
    * ``edge_floor`` — drop names whose shrunk μ̂ is below this
      threshold (uncertainty-aware edge floor).

    Selection is top-K by post-shrinkage μ̂. Per-asset hard cap and
    budget normalisation applied after sizing.
    """
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    if mu_arr.shape != (snap.n,) or sigma_arr.shape != (snap.n,):
        raise ValueError(
            f"mu/sigma shape mismatch with snap.n={snap.n}: "
            f"mu={mu_arr.shape} sigma={sigma_arr.shape}"
        )
    # μ̂ shrinkage (Garlappi-Uppal-Wang 2007 style: subtract κ·σ)
    shrunk_mu = mu_arr - float(mu_shrinkage) * sigma_arr
    # Edge floor: drop names below the uncertainty floor
    if edge_floor is not None:
        shrunk_mu = np.where(shrunk_mu >= float(edge_floor), shrunk_mu, 0.0)
    selected = _select_top_k(shrunk_mu, snap, K)
    if not selected:
        return _build_result(snap, np.zeros(snap.n), (), "no_candidates")
    target = np.zeros(snap.n)
    for i in selected:
        s = max(float(sigma_arr[i]), 1e-6)
        f_star = float(kelly_fraction) * max(float(shrunk_mu[i]), 0.0) / (s * s)
        target[i] = max(f_star, 0.0)
    return _build_result(snap, target, selected, "optimal")
