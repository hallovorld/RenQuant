"""Triple-Barrier labeling — faithful port of López de Prado AFML 2018.

This module re-implements the algorithm from:

    Marcos López de Prado, *Advances in Financial Machine Learning*,
    Wiley 2018, ISBN 978-1-119-48208-6.
        Chapter 3.4 — "The Triple-Barrier Method"
        Snippet 3.4 — "applyPtSlOnT1" (pp. 47-49)

The mlfinlab wrapper (Hudson & Thames) is the canonical OSS implementation
but was moved behind their commercial Foundation tier in 2023 — no longer
on PyPI. This file is a direct port of the textbook pseudocode and pins
the same constants (pt_mult / sl_mult / vertical horizon naming).

For the meta-labeling-on-exit adaptation (AFML ch.20 pp. 295-305), see
``meta_label_for_exit_event`` at the bottom of this module.

References
----------
* López de Prado 2018 AFML ch.3 Snippet 3.4 ("applyPtSlOnT1", pp. 47-49)
* López de Prado 2018 AFML ch.20 "Meta-Labeling" (pp. 295-305)
* `doc/research/meta-labeling-exit-policy.md` — RenQuant-specific design
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def apply_triple_barrier(
    close:              pd.Series,
    *,
    entry_idx:          pd.Timestamp,
    entry_price:        float,
    pt_mult:            float,
    sl_mult:            float,
    sigma_daily:        float,
    max_horizon_days:   int,
    return_terminal_sign: bool = False,
) -> Optional[Tuple[int, pd.Timestamp, float]]:
    """Apply triple-barrier labeling per AFML Snippet 3.4.

    Parameters
    ----------
    close : pd.Series
        Indexed by date, values are close prices for the instrument.
    entry_idx : pd.Timestamp
        Time index for the entry event (the bar AT which we want to label
        an outcome). The walk-forward window starts at the NEXT bar.
    entry_price : float
        Reference price for barrier computation. Usually ``close.loc[entry_idx]``
        but exposed as a separate arg so meta-label callers can use the
        decision-time price (e.g. an intraday trigger price) rather than
        the bar close.
    pt_mult, sl_mult : float
        Profit-taking and stop-loss multipliers on ``sigma_daily``. Per
        AFML Snippet 3.4, the upper barrier is ``entry × (1 + pt_mult ×
        sigma_daily)`` and the lower is ``entry × (1 - sl_mult ×
        sigma_daily)``. Setting either to 0 disables that barrier.
    sigma_daily : float
        Daily-σ estimate (typically a 20-day rolling realized vol). The
        barrier width scales with this so calm names get tighter bands
        and volatile names get wider — same property as σ-aware stops
        in kernel/exits.py.
    max_horizon_days : int
        Vertical barrier — drop the event if neither side barrier fires
        within this many BUSINESS DAYS of ``entry_idx``.
    return_terminal_sign : bool, default False
        If True and vertical hit, return ``sign(terminal_price - entry_price)``
        instead of 0. AFML uses 0 by default; mlfinlab's ``get_bins`` has
        an option to use terminal sign.

    Returns
    -------
    (label, exit_date, exit_price) | None
        label ∈ {+1, -1, 0}: +1 upper hit, -1 lower hit, 0 vertical hit
        (optionally sign-of-terminal when ``return_terminal_sign``).
        Returns None when ``entry_idx`` is past the series end or has
        no forward data.
    """
    # Slice the forward window: STRICTLY after entry_idx, up to
    # ``max_horizon_days`` business days. The "+ 1" + ``iloc[1:]`` skips
    # the entry bar itself so we never "hit" on the same bar (mirrors
    # AFML's t0 < t indexing).
    if entry_idx not in close.index:
        return None
    pos = close.index.get_loc(entry_idx)
    # Take the next max_horizon_days bars (or fewer if we're near the end)
    end_pos = min(pos + 1 + max_horizon_days, len(close))
    window = close.iloc[pos + 1 : end_pos]
    if len(window) == 0:
        return None

    upper = entry_price * (1.0 + pt_mult * sigma_daily) if pt_mult > 0 else float("inf")
    lower = entry_price * (1.0 - sl_mult * sigma_daily) if sl_mult > 0 else float("-inf")

    # Walk forward bar-by-bar. AFML's Snippet 3.4 uses pandas where()
    # masks for parallel speed, but the loop is identical in semantics
    # and clearer for a 30-feature classifier feeding ~5000 events.
    for date, price in window.items():
        if not np.isfinite(price):
            continue   # data gap — skip but keep scanning
        if price >= upper:
            return (+1, date, float(price))
        if price <= lower:
            return (-1, date, float(price))

    # Vertical barrier hit — neither side fired within the window.
    terminal_date = window.index[-1]
    terminal_price = float(window.iloc[-1])
    if return_terminal_sign:
        delta = terminal_price - entry_price
        if delta > 0:
            return (+1, terminal_date, terminal_price)
        if delta < 0:
            return (-1, terminal_date, terminal_price)
    return (0, terminal_date, terminal_price)


def meta_label_for_exit_event(
    close:        pd.Series,
    *,
    event_idx:    pd.Timestamp,
    event_price:  float,
    sigma_daily:  float,
    fwd_window:   int = 20,
    pt_mult:      float = 10.0,
    sl_mult:      float = 10.0,
) -> Optional[int]:
    """Binary meta-label for an exit decision per AFML ch.20.

    "Did the path-rule exit at ``event_idx`` correctly anticipate
    further loss, or was it a false positive (the position recovered)?"

    Encoding:
      * label = 1 ⇒ EXIT WAS CORRECT — price hit lower barrier first
                    (continued falling)
      * label = 0 ⇒ EXIT WAS WRONG    — price hit upper barrier first
                    OR vertical with terminal ≥ entry (would have
                    recovered if we'd held)

    The barrier widths use the same pt_mult / sl_mult / sigma_daily
    formulation as the triple-barrier source. Defaults pt=sl=10 with
    σ_daily=0.01 produce ±10% bands — close to the empirical p10 of
    stop_loss exits (-20%) and p75 of profitable model_sell exits
    (+9.5%) per the baseline distribution analysis (P4.1 data).

    Returns None when no forward data is available.
    """
    res = apply_triple_barrier(
        close,
        entry_idx=event_idx,
        entry_price=event_price,
        pt_mult=pt_mult,
        sl_mult=sl_mult,
        sigma_daily=sigma_daily,
        max_horizon_days=fwd_window,
        return_terminal_sign=True,   # vertical hit → sign of terminal
    )
    if res is None:
        return None
    label, _, _ = res
    # Map AFML label {-1, 0, +1} → meta-label {1, 0}
    # AFML label = -1 (lower hit / continued fall)  → meta = 1 (correct exit)
    # AFML label = +1 (upper hit / recovery)        → meta = 0 (false positive)
    # AFML label =  0 (vertical, flat-ish)          → meta = 0 (default to "would have held")
    return 1 if label == -1 else 0
