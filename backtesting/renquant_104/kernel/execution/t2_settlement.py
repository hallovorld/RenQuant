"""T+N cash-settlement queue for sell proceeds.

US equity settlement is T+1 for most broker-dealer transactions since
May 28, 2024 under SEC Rule 15c6-1 amendments. The queue remains named
``T2CashQueue`` for compatibility with older imports, but the default
settlement lag is now one NYSE session. Tests may still pass
``settlement_days=2`` when they intentionally exercise the legacy
conservative convention.

Per CLAUDE.md §5.13.5: this is the only T+N settlement implementation;
SimAdapter (and any future runner reconciliation) must route through
:class:`T2CashQueue` rather than rolling their own.

We use the NYSE trading calendar from ``pandas_market_calendars`` so
holidays / half-days skip correctly. Per the spec: sell Tue 12/24
(Christmas Eve) → settle Mon 12/30 (skipping Wed 12/25 + the
intervening weekend).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import pandas as pd

try:
    import pandas_market_calendars as mcal
    _NYSE = mcal.get_calendar("NYSE")
except Exception:  # pragma: no cover — defensive, mcal is in .venv
    _NYSE = None


def _settle_date(sale_date: pd.Timestamp, n_days: int) -> pd.Timestamp:
    """Trading-day arithmetic: ``sale_date + n_days`` NYSE sessions.

    Falls back to calendar-day arithmetic (skipping weekends only) if
    pandas_market_calendars is unavailable. Holidays would then NOT
    skip — caller should treat that as a degraded path.
    """
    sale = pd.Timestamp(sale_date).normalize()
    if _NYSE is None:
        # Skip weekends only. Caller should warn.
        out = sale
        added = 0
        while added < n_days:
            out = out + pd.Timedelta(days=1)
            if out.weekday() < 5:
                added += 1
        return out
    # Get sufficient trading sessions starting strictly AFTER sale_date.
    end = sale + pd.Timedelta(days=n_days * 2 + 10)
    sessions = _NYSE.valid_days(start_date=sale + pd.Timedelta(days=1),
                                end_date=end)
    if len(sessions) < n_days:
        # Extend window — only matters around very long holiday clusters.
        end = sale + pd.Timedelta(days=n_days * 5 + 30)
        sessions = _NYSE.valid_days(start_date=sale + pd.Timedelta(days=1),
                                    end_date=end)
    # tz-strip to align with naive timestamps used elsewhere in sim.
    sess_naive = [pd.Timestamp(s).tz_localize(None).normalize() for s in sessions]
    return sess_naive[n_days - 1]


@dataclass
class PendingCashEntry:
    """One sell's proceeds, waiting to settle.

    ``settle_date`` is normalized (midnight). ``amount`` is post-fee
    proceeds (i.e. what the broker will credit on settlement — net of
    SEC + TAF + custom commission; tax is paid separately and stays
    immediate).
    """

    settle_date: pd.Timestamp
    amount: float


@dataclass
class T2CashQueue:
    """FIFO-by-settle-date queue of pending sell proceeds.

    Invariants:
    * ``add_pending`` is the only mutator that grows the queue.
    * ``drain(today)`` removes every entry with ``settle_date <= today``
      and returns the summed amount. Idempotent on dates with no
      pending entries.
    * ``pending_total()`` reports the sum of NOT-YET-settled amounts
      (useful for sanity-checking `live - settled = pending`).
    """

    settlement_days: int = 1
    _pending: List[PendingCashEntry] = field(default_factory=list)

    def add_pending(self, sale_date: pd.Timestamp, amount: float) -> None:
        """Queue proceeds for T+N settlement.

        Per §5.13.11: non-finite or non-positive amount is silently
        dropped (defensive — caller should already have rejected the
        bad fill upstream). Logging is the caller's responsibility.
        """
        if not math.isfinite(amount) or amount <= 0:
            return
        sd = _settle_date(sale_date, self.settlement_days)
        self._pending.append(PendingCashEntry(settle_date=sd, amount=float(amount)))

    def drain(self, today: pd.Timestamp) -> float:
        """Settle every entry with ``settle_date <= today``. Returns sum.

        Mutates ``self._pending`` in place. Safe to call once per bar
        at the top of the loop.
        """
        today_norm = pd.Timestamp(today).normalize()
        settled_total = 0.0
        kept: List[PendingCashEntry] = []
        for entry in self._pending:
            if entry.settle_date <= today_norm:
                settled_total += entry.amount
            else:
                kept.append(entry)
        self._pending = kept
        return settled_total

    def pending_total(self) -> float:
        """Sum of amounts not yet settled. Read-only."""
        return float(sum(e.amount for e in self._pending))

    def __len__(self) -> int:
        return len(self._pending)
