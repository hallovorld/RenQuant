"""§7.2 placebo loaders for the QP Step-4 A/B replay (#212 / Step-4h
follow-up). Per §7.2.1 R2, any Sharpe/return quoted from the replay needs
a companion placebo verdict; these loaders provide it via the
`--loader-module` injection hook on `run_ab_replay`.

Mechanism: load the real bars, then break the decision<->outcome
alignment. If an allocator's measured edge is REAL (depends on the
mu/sigma signal predicting fwd_return), the paired ΔSharpe collapses
toward 0 under the placebo. If the edge SURVIVES the placebo, it is a
structural artifact of the allocator, not signal — and must not be
promoted.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from kernel.portfolio_qp.wf_replay_loader import (
    load_replay_bars_from_sim_db,
)

_SEED = 42


def load_shuffle_placebo(root, start, end, *, fwd_horizon_days):
    """Shuffle-label placebo (§7.2): per-bar permute fwd_return so each
    asset's realised return is reassigned to a random asset in the same
    bar. This severs the mu<->fwd asset alignment the allocators size on
    while preserving the bar's marginal return distribution. A real edge
    -> paired ΔSharpe ~ 0 here."""
    bars = load_replay_bars_from_sim_db(
        root, start, end, fwd_horizon_days=fwd_horizon_days,
    )
    rng = np.random.default_rng(_SEED)
    out = []
    for b in bars:
        fwd = np.array(b.fwd_return, dtype=float, copy=True)
        rng.shuffle(fwd)
        out.append(replace(b, fwd_return=fwd))
    return out


def load_timeshift_placebo(root, start, end, *, fwd_horizon_days):
    """Time-shift placebo (§7.2): each bar realises the NEXT bar's
    per-asset returns, re-aligned by ticker (intersection only). Breaks
    the time alignment between the decision date's mu and that date's
    realised fwd. Bars whose ticker set does not overlap the next bar
    fall back to a within-bar shuffle so no bar leaks true alignment."""
    bars = load_replay_bars_from_sim_db(
        root, start, end, fwd_horizon_days=fwd_horizon_days,
    )
    if len(bars) < 2:
        return bars
    rng = np.random.default_rng(_SEED)
    out = []
    for i, b in enumerate(bars):
        nxt = bars[(i + 1) % len(bars)]
        nxt_map = {t: nxt.fwd_return[j] for j, t in enumerate(nxt.snap.tickers)}
        fwd = np.array(b.fwd_return, dtype=float, copy=True)
        matched = 0
        for j, t in enumerate(b.snap.tickers):
            if t in nxt_map:
                fwd[j] = nxt_map[t]
                matched += 1
        if matched == 0:  # no ticker overlap -> shuffle so no true leak
            rng.shuffle(fwd)
        out.append(replace(b, fwd_return=fwd))
    return out
