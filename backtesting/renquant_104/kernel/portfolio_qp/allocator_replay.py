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
    """

    name: str
    bars: int
    daily_returns_net: list[float] = field(default_factory=list)
    turnover: list[float] = field(default_factory=list)
    cap_violations: int = 0
    fallback_to_no_candidates: int = 0
    per_regime: dict[str, list[float]] = field(default_factory=dict)

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
            "fallback_to_no_candidates": self.fallback_to_no_candidates,
            "per_regime_sharpe": self.per_regime_sharpe(),
            "per_regime_n_bars": {
                r: len(v) for r, v in self.per_regime.items()
            },
        }


def replay_one_allocator(
    name: str,
    allocator: AllocatorFn,
    bars: Sequence[AllocatorReplayBar],
) -> ReplayResult:
    """Run a single allocator over the bar sequence and collect metrics."""
    res = ReplayResult(name=name, bars=len(bars))
    for bar in bars:
        try:
            alloc = allocator(bar.snap, mu=bar.mu, sigma=bar.sigma)
        except TypeError:
            # Allocator may not accept sigma (e.g. equal-weight).
            alloc = allocator(bar.snap, mu=bar.mu)
        if alloc.status == "no_candidates":
            res.fallback_to_no_candidates += 1
            daily = 0.0
            turn = 0.0
        else:
            # Gross daily return: Σ target_w_i · fwd_return_i
            gross = float(np.sum(alloc.target_w * bar.fwd_return))
            # Transaction cost: |Δw|₁ · cost_bps × 1e-4
            turn = float(np.sum(np.abs(alloc.delta_w)))
            cost = turn * bar.cost_per_trade_bps * 1e-4
            daily = gross - cost
            # Cap-violation accounting (target above hard cap by > 1e-9)
            if (alloc.target_w > bar.snap.w_upper_hard + 1e-9).any():
                res.cap_violations += 1
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
