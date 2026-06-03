"""Allocator A/B replay harness — §8 Step 4b (PR #125).

Runs N allocators on a shared sequence of per-bar inputs
(:class:`AllocatorReplayBar`) and produces per-allocator paired-daily
returns + Sharpe / MDD / turnover + per-regime stratified attribution.

This module is **the math**, not the data loader. A separate
follow-up PR wires the WF cut loader (training artifact + holdout
dates + per-cut Σ̂) to this module's input shape. Tests in this PR
use synthetic bars so the harness math can be pinned independently
of the production artifact storage.

Output is :class:`ReplayResult` per allocator — JSON-serialisable so
the decision-grade A/B replay artifact can be committed under
``doc/research/evidence/``.

The harness deliberately keeps DSR / PBO out of this scaffolding
module; those are added in Step 4c via ``kernel.metrics`` (lifted to
renquant-common) so the same multiple-comparison correction applies
to both the QP and Hybrid candidate evaluations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from kernel.portfolio_qp.baseline_allocators import AllocatorResult
from kernel.portfolio_qp.constraint_snapshot import ConstraintSnapshot


# An allocator callable: takes (snapshot, mu, sigma) and returns AllocatorResult.
# (The richer QP / Hybrid signatures absorb extra kwargs via partial.)
AllocatorFn = Callable[..., AllocatorResult]


@dataclass(frozen=True)
class AllocatorReplayBar:
    """One bar of input to the A/B replay.

    The same bar is fed to every allocator under test — they see the
    same snapshot, μ̂, σ̂, and realised forward return. Paired-daily-
    returns + DSR/PBO comparisons all key off this shared input.
    """

    bar_date: str                          # ISO date (informational)
    snap: ConstraintSnapshot
    mu: np.ndarray                          # shape (n,)
    sigma: np.ndarray                       # shape (n,)
    fwd_return: np.ndarray                  # shape (n,) — realised per-asset return
    regime: Optional[str] = None            # for per-regime stratification
    cost_per_trade_bps: float = 5.0         # 5 bp round-trip transaction cost


@dataclass
class ReplayResult:
    """Per-allocator output of the A/B replay.

    Attributes are JSON-serialisable (numpy → list inside
    :meth:`to_dict`). Sharpe is the annualised mean / std of daily
    net-of-cost returns; MDD is the maximum drawdown of the cumulative
    return series; turnover is the mean per-bar L1 |Δw|.

    **Constraint violation tracking** (codex #131 review HIGH): the
    Step 4 A/B gate requires *zero hard-constraint regressions vs the
    ConstraintSnapshot*. The harness validates every allocator output
    against the full hard-constraint set advertised by the snapshot
    and tallies per-family violations.
    """

    name: str
    bars: int
    daily_returns_net: list[float] = field(default_factory=list)
    turnover: list[float] = field(default_factory=list)
    cap_violations: int = 0  # any-family violation count (legacy alias)
    fallback_to_no_candidates: int = 0
    per_regime: dict[str, list[float]] = field(default_factory=dict)
    # Per-family violation counters (codex #131 review HIGH)
    violations_per_family: dict[str, int] = field(default_factory=dict)

    @property
    def sharpe_annual(self) -> Optional[float]:
        r = np.asarray(self.daily_returns_net, dtype=float)
        if len(r) < 2:
            return None
        sd = float(np.std(r, ddof=1))
        if sd < 1e-12:
            return None
        return float(np.mean(r) / sd * np.sqrt(252.0))

    @property
    def mean_daily_return(self) -> float:
        r = self.daily_returns_net
        return float(np.mean(r)) if r else 0.0

    @property
    def cumulative_return(self) -> float:
        if not self.daily_returns_net:
            return 0.0
        return float(np.prod(1.0 + np.asarray(self.daily_returns_net)) - 1.0)

    @property
    def max_drawdown(self) -> float:
        if not self.daily_returns_net:
            return 0.0
        equity = np.cumprod(1.0 + np.asarray(self.daily_returns_net))
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        return float(dd.min())

    @property
    def mean_turnover(self) -> float:
        return float(np.mean(self.turnover)) if self.turnover else 0.0

    def per_regime_sharpe(self) -> dict[str, Optional[float]]:
        out: dict[str, Optional[float]] = {}
        for regime, returns in self.per_regime.items():
            arr = np.asarray(returns, dtype=float)
            if len(arr) < 2:
                out[regime] = None
                continue
            sd = float(np.std(arr, ddof=1))
            out[regime] = (
                None if sd < 1e-12
                else float(np.mean(arr) / sd * np.sqrt(252.0))
            )
        return out

    def total_violations(self) -> int:
        return int(sum(self.violations_per_family.values()))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "bars": self.bars,
            "sharpe_annual": self.sharpe_annual,
            "mean_daily_return": self.mean_daily_return,
            "cumulative_return": self.cumulative_return,
            "max_drawdown": self.max_drawdown,
            "mean_turnover": self.mean_turnover,
            "cap_violations": self.cap_violations,
            "violations_per_family": dict(self.violations_per_family),
            "total_violations": self.total_violations(),
            "fallback_to_no_candidates": self.fallback_to_no_candidates,
            "per_regime_sharpe": self.per_regime_sharpe(),
            "per_regime_n_bars": {
                r: len(v) for r, v in self.per_regime.items()
            },
        }


# Hard-constraint families surfaced by ConstraintSnapshot.
_VIOLATION_FAMILIES = (
    "w_upper_hard",
    "w_lower",
    "wash_sale",
    "dw_max",
    "cash_budget",
    "turnover_max",
    "sector_cap",
    "corr_group_cap",
    "gross_max",
)


def check_snapshot_feasibility(
    snap: "ConstraintSnapshot",
    target_w: np.ndarray,
    delta_w: np.ndarray,
    *,
    tol: float = 1e-9,
) -> dict[str, int]:
    """Validate ``target_w`` / ``delta_w`` against the full snapshot
    hard-constraint set and return per-family violation counts (0 or 1
    each — one bar can contribute at most one violation per family).

    Codex #131 review HIGH fix: replay was previously only counting
    ``target_w > w_upper_hard`` and missing every other family. Step
    4's gate of *zero hard-constraint regressions vs ConstraintSnapshot*
    requires the full check.
    """
    fam: dict[str, int] = {name: 0 for name in _VIOLATION_FAMILIES}
    n = snap.n

    if (target_w > snap.w_upper_hard + tol).any():
        fam["w_upper_hard"] = 1
    if (target_w < snap.w_lower - tol).any():
        fam["w_lower"] = 1
    if snap.wash_sale_mask.any():
        # Δw_i must be ≤ 0 for masked names
        if (delta_w[snap.wash_sale_mask.astype(bool)] > tol).any():
            fam["wash_sale"] = 1
    if snap.dw_max is not None:
        if (np.abs(delta_w) > snap.dw_max + tol).any():
            fam["dw_max"] = 1
    budget = max(0.0, 1.0 - float(snap.cash_reserve))
    if float(target_w.sum()) > budget + tol:
        fam["cash_budget"] = 1
    if snap.turnover_max is not None:
        if float(np.sum(np.abs(delta_w))) > float(snap.turnover_max) + tol:
            fam["turnover_max"] = 1
    if snap.sector_indicator is not None and snap.sector_cap_vec is not None:
        loads = snap.sector_indicator @ target_w
        if (loads > snap.sector_cap_vec + tol).any():
            fam["sector_cap"] = 1
    for trip in snap.corr_group_pairs or ():
        try:
            i, j, cap = int(trip[0]), int(trip[1]), float(trip[2])
        except (TypeError, IndexError, ValueError):
            continue
        if 0 <= i < n and 0 <= j < n:
            if float(target_w[i] + target_w[j]) > cap + tol:
                fam["corr_group_cap"] = 1
    if snap.gross_max is not None:
        if float(np.sum(np.abs(target_w))) > float(snap.gross_max) + tol:
            fam["gross_max"] = 1
    return fam


def replay_one_allocator(
    name: str,
    allocator: AllocatorFn,
    bars: Sequence[AllocatorReplayBar],
) -> ReplayResult:
    """Run a single allocator over the bar sequence and collect metrics.

    **no_candidates accounting** (codex #131 review HIGH-2): the
    allocator's returned ``target_w`` (typically zeros = liquidate to
    cash) and ``delta_w`` (= ``-w_current``) ARE the action the
    allocator chose; the harness must honour them. Previously
    ``no_candidates`` bars were short-circuited to zero return / zero
    turnover, which silently discarded the liquidation cost and
    over-stated baselines vs QP.
    """
    res = ReplayResult(name=name, bars=len(bars))
    for bar in bars:
        try:
            alloc = allocator(bar.snap, mu=bar.mu, sigma=bar.sigma)
        except TypeError:
            # Allocator may not accept sigma (e.g. equal-weight).
            alloc = allocator(bar.snap, mu=bar.mu)
        if alloc.status == "no_candidates":
            res.fallback_to_no_candidates += 1
        # ALWAYS compute gross + cost from the allocator's own
        # target_w / delta_w — no_candidates means "go to cash" which
        # has a real liquidation cost.
        gross = float(np.sum(alloc.target_w * bar.fwd_return))
        turn = float(np.sum(np.abs(alloc.delta_w)))
        cost = turn * bar.cost_per_trade_bps * 1e-4
        daily = gross - cost
        # Per-family feasibility check (codex #131 review HIGH-1)
        family_viol = check_snapshot_feasibility(
            bar.snap, alloc.target_w, alloc.delta_w,
        )
        for fam_name, count in family_viol.items():
            if count > 0:
                res.violations_per_family[fam_name] = (
                    res.violations_per_family.get(fam_name, 0) + count
                )
        if any(v > 0 for v in family_viol.values()):
            res.cap_violations += 1  # legacy any-family counter
        res.daily_returns_net.append(daily)
        res.turnover.append(turn)
        if bar.regime is not None:
            res.per_regime.setdefault(bar.regime, []).append(daily)
    return res


def replay_all(
    allocators: dict[str, AllocatorFn],
    bars: Sequence[AllocatorReplayBar],
) -> dict[str, ReplayResult]:
    """Run every allocator over the same bar sequence.

    Returns ``{name: ReplayResult}``. The bar sequence is shared so
    downstream paired-daily-returns + DSR / PBO comparisons key off
    a consistent input.
    """
    out: dict[str, ReplayResult] = {}
    for name, fn in allocators.items():
        out[name] = replay_one_allocator(name, fn, bars)
    return out


def paired_daily_returns(
    results: dict[str, ReplayResult],
) -> dict[str, np.ndarray]:
    """Return ``{name: np.ndarray}`` aligned by bar index.

    All result objects must have the same ``bars`` count (i.e. they
    were produced by ``replay_all`` over the same bar sequence) —
    otherwise paired daily returns are not well-defined.
    """
    if not results:
        return {}
    bar_counts = {name: r.bars for name, r in results.items()}
    if len(set(bar_counts.values())) != 1:
        raise ValueError(
            f"Allocators produced different bar counts: {bar_counts}. "
            "Paired daily returns are only valid for results from the "
            "same bar sequence."
        )
    return {
        name: np.asarray(r.daily_returns_net, dtype=float)
        for name, r in results.items()
    }
