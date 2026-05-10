"""Probability of Backtest Overfitting via Combinatorially Symmetric
Cross-Validation (CSCV).

Reference: Bailey, Borwein, López de Prado, Zhu (2017),
"The Probability of Backtest Overfitting", J. Computational Finance.

Algorithm (Section 4):
  1. Build (T × N) returns matrix M (rows=periods, cols=strategies).
  2. Partition T rows into S even slices (S even, default 16).
  3. For each combination C of S/2 IS slices (complement = OOS):
       a. i* = argmax IS Sharpe across N strategies.
       b. r* = OOS rank of i* (1=worst, N=best); ω̄ = r*/(N+1).
       c. λ_C = log(ω̄/(1−ω̄)); λ ≤ 0 ⇔ chosen strategy ≤ OOS median.
  4. PBO = fraction of partitions with λ_C ≤ 0.

Skilled strategy → PBO ≪ 0.5; overfit → PBO → 0.5.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Optional

import numpy as np


def _slice_sharpe(slab: np.ndarray) -> np.ndarray:
    """Per-strategy Sharpe over rows of slab; NaN where std==0."""
    if slab.size == 0:
        return np.full(slab.shape[1] if slab.ndim == 2 else 0, np.nan)
    mu, sd = slab.mean(axis=0), slab.std(axis=0, ddof=1)
    return np.where(sd > 0, mu / np.where(sd > 0, sd, 1.0), np.nan)


def _split_indices(n_rows: int, n_slices: int) -> list[np.ndarray]:
    """n_slices index arrays partitioning [0, n_rows); sizes differ ≤ 1."""
    base, extra = n_rows // n_slices, n_rows % n_slices
    out, cursor = [], 0
    for s in range(n_slices):
        size = base + (1 if s < extra else 0)
        out.append(np.arange(cursor, cursor + size))
        cursor += size
    return out


def _logit_omega(rank_one_indexed: int, n_strategies: int) -> float:
    """ω̄ = r/(N+1); λ = log(ω̄/(1−ω̄)). The (N+1) keeps λ finite."""
    omega = rank_one_indexed / (n_strategies + 1.0)
    return math.log(omega / (1.0 - omega))


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_slices: int = 16,
    max_combinations: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """CSCV PBO estimate ∈ [0, 1].

    Parameters
    ----------
    returns_matrix : (T, N) array
        Each column is a candidate strategy's return series.
    n_slices : int, must be even, ≥ 4
        S in the paper. 16 is the canonical default.
    max_combinations : int or None
        If C(n_slices, n_slices/2) is too large, randomly sample this
        many partitions. None = enumerate all.
    rng : np.random.Generator
        Used only when max_combinations is set.
    """
    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (T × N)")
    T, N = M.shape
    if N < 2:
        raise ValueError("PBO requires at least 2 candidate strategies")
    if n_slices % 2 != 0 or n_slices < 4:
        raise ValueError("n_slices must be even and >= 4")
    if T < n_slices:
        raise ValueError(f"need T >= n_slices ({n_slices}); got T={T}")

    slices = _split_indices(T, n_slices)
    half = n_slices // 2
    all_combos = list(combinations(range(n_slices), half))
    if max_combinations is not None and max_combinations < len(all_combos):
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(all_combos), size=max_combinations, replace=False)
        combos = [all_combos[i] for i in idx]
    else:
        combos = all_combos

    overfit_count, valid = 0, 0
    for combo in combos:
        is_idx = np.concatenate([slices[s] for s in combo])
        oos_idx = np.concatenate([slices[s] for s in range(n_slices) if s not in combo])
        is_sr = _slice_sharpe(M[is_idx])
        oos_sr = _slice_sharpe(M[oos_idx])
        if not np.isfinite(is_sr).any() or not np.isfinite(oos_sr).any():
            continue
        best_i = int(np.nanargmax(is_sr))
        finite_mask = np.isfinite(oos_sr)
        if not finite_mask[best_i]:
            continue
        # Rank: NaN OOS values placed at the lowest end via -inf.
        order = np.argsort(np.where(finite_mask, oos_sr, -np.inf))
        ranks = np.empty(N, dtype=int)
        ranks[order] = np.arange(1, N + 1)
        if _logit_omega(ranks[best_i], N) <= 0.0:
            overfit_count += 1
        valid += 1
    return overfit_count / valid if valid else float("nan")
