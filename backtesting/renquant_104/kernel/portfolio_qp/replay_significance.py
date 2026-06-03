"""DSR / PBO wired into the A/B replay harness — §8 Step 4c (PR #125).

The §7.3 multi-measurement requirement and the §8 plan both call for
Deflated Sharpe Ratio (Bailey-López de Prado 2014) and Probability of
Backtest Overfitting (Bailey-Borwein-López de Prado-Zhu 2015) to be
applied to the A/B replay's paired-daily-returns output. Without this,
the Sharpe rankings the replay produces are vulnerable to
selection-bias inflation across the 5 candidate allocators.

This module is the thin adapter that takes the
:class:`AllocatorReplay` output and runs DSR / PBO on it via the
existing :mod:`kernel.metrics` implementations (lifted to renquant-
common per the merged refactor PRs #51 / #52).

Output is the canonical evidence-artifact dict the offline A/B
replay's verdict JSON will commit under ``doc/research/evidence/``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from kernel.metrics.deflated_sharpe import deflated_sharpe_ratio
from kernel.metrics.pbo import probability_of_backtest_overfitting
from kernel.portfolio_qp.allocator_replay import ReplayResult


@dataclass(frozen=True)
class SignificanceVerdict:
    """Per-allocator DSR / PBO output of the A/B replay's significance pass.

    The fields are JSON-serialisable; the verdict block is what the
    Step 4g evidence-artifact JSON will commit.

    DSR ≥ 0.95 = selection-bias-corrected significance at 5%.
    PBO < 0.5 = better-than-coin-flip out-of-sample.

    The §8 Step 4 A/B gate is deliberately stricter than CLAUDE.md
    §7.4's literal disjunction because it selects among multiple
    allocator candidates.
    """

    name: str
    sharpe_raw_annual: Optional[float]      # bare annualised SR (252-day)
    dsr: Optional[float]                     # ∈ [0, 1], higher = more robust
    pbo: Optional[float] = None              # ∈ [0, 1], lower = more robust (shared)
    pbo_se: Optional[float] = None           # bootstrap SE when available
    n_returns: int = 0
    n_trials: int = 1                         # number of allocator candidates


def compute_significance_verdicts(
    results: dict[str, ReplayResult],
    *,
    pbo_n_slices: int = 16,
    pbo_max_combinations: Optional[int] = None,
    pbo_rng_seed: int = 0,
) -> dict[str, SignificanceVerdict]:
    """Run DSR + PBO across the A/B replay's allocators.

    DSR is per-allocator (uses each allocator's own returns; deflates
    against the count of candidates as the trial count).

    PBO is a single number across the candidate matrix (T × N) —
    Bailey-Borwein-López de Prado-Zhu 2015's CSCV procedure — so each
    allocator's ``pbo`` field carries the same shared value.

    Returns ``{name: SignificanceVerdict}``.
    """
    if not results:
        return {}

    # Align: every allocator must have the same bar count for the
    # paired comparison + the PBO matrix to be well-defined.
    bar_counts = {n: r.bars for n, r in results.items()}
    if len(set(bar_counts.values())) != 1:
        raise ValueError(
            "Allocators produced different bar counts: %s. "
            "Significance comparison requires a shared bar sequence." % bar_counts
        )
    n_returns = next(iter(bar_counts.values()))
    n_trials = len(results)

    # Build the (T, N) returns matrix in a stable allocator order.
    names = list(results.keys())
    returns_matrix = np.column_stack([
        np.asarray(results[name].daily_returns_net, dtype=float)
        for name in names
    ])

    # PBO needs ≥ 2 candidates AND T ≥ n_slices.
    pbo_value: Optional[float]
    if n_trials >= 2 and n_returns >= pbo_n_slices:
        pbo_value = float(probability_of_backtest_overfitting(
            returns_matrix,
            n_slices=pbo_n_slices,
            max_combinations=pbo_max_combinations,
            rng=np.random.default_rng(pbo_rng_seed),
        ))
    else:
        pbo_value = None

    out: dict[str, SignificanceVerdict] = {}
    for name in names:
        r = results[name]
        sr = r.sharpe_annual
        # DSR requires ≥ 30 observations for the higher-moment
        # correction to be meaningful.
        if sr is None or n_returns < 30:
            dsr_value: Optional[float] = None
        else:
            arr = np.asarray(r.daily_returns_net, dtype=float)
            skew = _safe_skew(arr)
            kurt = _safe_excess_kurtosis(arr)
            dsr_value = float(deflated_sharpe_ratio(
                sr_observed=float(sr),
                n_returns=int(n_returns),
                n_trials=int(n_trials),
                skew=float(skew),
                excess_kurtosis=float(kurt),
            ))
        out[name] = SignificanceVerdict(
            name=name,
            sharpe_raw_annual=sr,
            dsr=dsr_value,
            pbo=pbo_value,
            pbo_se=None,
            n_returns=int(n_returns),
            n_trials=int(n_trials),
        )
    return out


def verdicts_to_dict(
    verdicts: dict[str, SignificanceVerdict],
) -> dict:
    """JSON-serialisable view of the significance block.

    Shape matches what the §8 Step 4g evidence artifact will commit
    under ``doc/research/evidence/``.
    """
    return {
        name: {
            "name": v.name,
            "sharpe_raw_annual": v.sharpe_raw_annual,
            "dsr": v.dsr,
            "pbo": v.pbo,
            "pbo_se": v.pbo_se,
            "n_returns": v.n_returns,
            "n_trials": v.n_trials,
            "live_promotable_per_section_8": _passes_section_8_gate(v),
            # Back-compat alias for callers/tests that pre-date the PR #134
            # schema correction. The semantics are the stricter §8 gate.
            "live_promotable_per_clause_7_4": _passes_section_8_gate(v),
        }
        for name, v in verdicts.items()
    }


def _passes_section_8_gate(v: SignificanceVerdict) -> bool:
    """Strict §8 Step 4 significance gate for multi-candidate selection."""
    if v.dsr is None or v.dsr < 0.95:
        return False
    if v.pbo is None:
        return True
    if v.pbo >= 0.5:
        return False
    return v.pbo_se is None or (v.pbo + v.pbo_se) < 0.55


def _safe_skew(arr: np.ndarray) -> float:
    if len(arr) < 3:
        return 0.0
    sd = float(np.std(arr))
    if sd < 1e-12:
        return 0.0
    centered = arr - float(np.mean(arr))
    return float(np.mean(centered ** 3) / sd ** 3)


def _safe_excess_kurtosis(arr: np.ndarray) -> float:
    if len(arr) < 4:
        return 0.0
    sd = float(np.std(arr))
    if sd < 1e-12:
        return 0.0
    centered = arr - float(np.mean(arr))
    return float(np.mean(centered ** 4) / sd ** 4 - 3.0)
