"""Stage 6 — Treynor-Black 1973 inverse-variance signal combination.

Takes K signal sources (per-ticker μ̂_per_ticker, panel μ̂_panel,
NGBoost μ̂_ngb, RS, …) plus their estimated IC variances, and produces
the optimal weighted-mean μ_combined per asset.

Reference:
- Treynor, J.L. & Black, F. (1973). "How to Use Security Analysis to
  Improve Portfolio Selection." Journal of Business 46(1), 66–86.
- Grinold, R.C. & Kahn, R.N. (1999). Active Portfolio Management. 2nd ed.

Theoretical optimum (independent signals):
    μ_combined_i  =  Σ_k (w_k · μ̂_k,i)
    w_k          =  IC_k² / σ²(IC_k)        (normalised across k)

When signals correlate, full Treynor-Black solves:
    μ_combined  =  μ̄ + Σ_signals · IC · σ_residual_inv

Stage 6 ships the diagonal-IC case (signals assumed uncorrelated). Full
covariance treatment is a future stage if signal correlation grows.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

log = logging.getLogger("kernel.portfolio_qp.signal_combiner")


def combine_signals(
    signals: dict[str, np.ndarray],
    ic_means: dict[str, float] | None = None,
    ic_stds:  dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Inverse-variance weighted mean across signal sources.

    Args:
        signals: dict {source_name: μ-vector of length n}. All vectors
            must share the same length and order. Sources with non-
            positive weight (insufficient IC) are silently dropped.
        ic_means: per-source IC estimate (e.g. CPCV mean). If None,
            uses uniform 1.0.
        ic_stds: per-source IC std (e.g. CPCV std). If None, uses
            uniform 1.0.

    Returns (combined_mu_vector, weights_dict). weights_dict maps
    source_name → normalised weight (sums to 1.0 across active sources).

    Edge cases:
        - Empty signals dict → returns zero vector, empty weights.
        - All signals are NaN-only → returns zero, empty weights.
        - Single signal → returns it unchanged with weight {name: 1.0}.
    """
    if not signals:
        return np.zeros(0), {}

    # Coerce to ndarray + validate shape consistency
    n = None
    arrs: dict[str, np.ndarray] = {}
    for k, v in signals.items():
        a = np.asarray(v, dtype=float)
        if n is None:
            n = a.shape[0]
        elif a.shape[0] != n:
            raise ValueError(
                f"signals[{k!r}] length {a.shape[0]} != expected {n}",
            )
        arrs[k] = a

    if n is None or n == 0:
        return np.zeros(0), {}

    # Drop sources with no IC info → uniform priors
    ic_means = ic_means or {}
    ic_stds  = ic_stds  or {}

    # IR² = (IC / std(IC))² → inverse-variance proxy
    # Audit fix SC-NEG-IC (2026-04-26): negative-IC sources are
    # BIASED (point in the wrong direction). Squaring IC drops the
    # sign so a -0.05 IC source would get the same weight as +0.05 —
    # propagating the wrong direction into the combined signal. Fix:
    # track sign separately and FLIP the source vector when summing.
    raw_weights: dict[str, float] = {}
    sign_per_src: dict[str, float] = {}
    for k in arrs:
        ic_m = float(ic_means.get(k, 1.0))
        ic_s = float(ic_stds.get(k, 1.0))
        if ic_s <= 1e-12 or not np.isfinite(ic_s):
            ic_s = 1.0
        if not np.isfinite(ic_m):
            ic_m = 0.0
        ir_sq = (ic_m / ic_s) ** 2
        raw_weights[k] = ir_sq
        sign_per_src[k] = 1.0 if ic_m >= 0.0 else -1.0

    total = float(sum(raw_weights.values()))
    if total <= 0.0:
        # No source has any informational ratio — fall back to equal
        # weights so caller gets the simple mean.
        n_src = len(arrs)
        weights = {k: 1.0 / n_src for k in arrs}
    else:
        weights = {k: w / total for k, w in raw_weights.items()}

    combined = np.zeros(n)
    for k, a in arrs.items():
        a_clean = np.where(np.isfinite(a), a, 0.0)
        combined = combined + weights[k] * sign_per_src[k] * a_clean

    log.debug(
        "combine_signals: %d sources, weights=%s",
        len(arrs),
        {k: round(v, 3) for k, v in weights.items()},
    )
    return combined, weights
