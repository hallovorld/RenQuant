"""Sim NAV mark-to-market — sim.py decomposition (S2 item 5).

EXTRACTED 2026-06-15 from adapters/sim.py _portfolio_value. The portfolio
net-asset-value computation, carrying three hard-won correctness invariants
(see inline): lookahead-safe price fallback, non-finite-price poisoning guard,
and T+N pending-settlement inclusion. SimAdapter keeps a thin method delegate;
behavior is unchanged. No SimAdapter state — self-deps are passed in.
"""
from __future__ import annotations

import math
from typing import Any


def portfolio_value(
    prices: dict,
    today_ts=None,
    *,
    cash: float,
    t2_queue: Any,
    pos_shares: dict,
    ohlcv: dict,
) -> float:
    """Mark-to-market the held positions.

    Bug 25 fix (2026-04-24): when a holding has no price in the per-bar
    ``prices`` dict (delisted / suspended / new IPO not yet trading), we fall
    back to the last AVAILABLE close ON OR BEFORE ``today_ts`` — NOT
    ``df.iloc[-1]`` of the full ohlcv (which is the LAST historical bar =
    future data in a sim).

    Audit fix SA-1 (Round 9, 2026-04-25): pre-fix, NaN/inf in either
    ``prices.get(t)`` or the fallback close silently propagated into
    ``total += shares * NaN = NaN``. Once corrupted, every subsequent call
    returned NaN — equity curve filled with NaN, total_ret/APY came out NaN.
    Now: skip non-finite prices (treat as zero contribution) so a single bad
    bar doesn't poison the rest of the simulation.

    Bug #C fix (2026-05-11): include the T+N pending-settlement balance.
    Pre-fix, NAV returned cash + position MTM but ignored sell proceeds
    sitting in ``t2_queue``. On sell day shares drop but cash is unchanged
    (proceeds queued) ⇒ phantom NAV drop = sale amount, recovered two bars
    later when the queue drains — inflating measured ann_vol. Invariant
    (CLAUDE.md §5.3): NAV ≡ free_cash + pending_settle + Σ(shares × price).
    """
    total = cash
    if t2_queue is not None:
        pending = t2_queue.pending_total()
        if math.isfinite(pending):
            total += pending
    for t, shares in pos_shares.items():
        p = prices.get(t)
        if p is None or not math.isfinite(p):
            df = ohlcv.get(t)
            if df is not None and not df.empty:
                if today_ts is not None:
                    truncated = df.loc[:today_ts]
                    if not truncated.empty:
                        cand = float(truncated["close"].iloc[-1])
                        p = cand if math.isfinite(cand) else None
                    else:
                        p = None
                else:
                    # No truncation hint — caller is responsible for not
                    # introducing lookahead.
                    cand = float(df["close"].iloc[-1])
                    p = cand if math.isfinite(cand) else None
        if p is not None and math.isfinite(p):
            total += shares * p
    return total
