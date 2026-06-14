"""Alert escalation lifecycle — detection without lifecycle = noise (L6).

UMBRELLA MIRROR of renquant_pipeline.kernel.alert_lifecycle — the strategy-104
kernel is self-contained (stdlib only, no pipeline imports), so the L6 audit
stack is carried here verbatim rather than cross-imported. Keep the two copies
in sync (the parallel-copy convention the kernel already follows for
persistence / decision_trace).

Design: renquant-orchestrator
doc/research/2026-06-12-engineering-architecture-deep-plan.md §12.3 / L6
audit sidecar. Graduates scripts/engineering/alert_lifecycle_prototype.py.

Motivating incident: the fundamentals-stale warning fired DAILY for ~4
months and was ignored — 121 identical ntfys train the operator to mute
the channel. The fix is a lifecycle, not louder alerts:

  NEW ──(first ntfy)──> WARN ──(unacked ≥ escalate_after_days)──> CRITICAL
                          │                                          │
                          └──────────(acked: never escalates)        │
  any state ──(condition absent on a later day)──> RESOLVED          │
  CRITICAL scopes are returned by blocking_scopes() for the L6 barrier.

Dedup key = (audit, scope, cause_hash): a 121-day-old condition is ONE
escalating incident with exactly TWO notifications (NEW + escalation),
not 121 identical warnings. Pure: no DB, no ntfy — the caller persists
the book and fires notifications off the returned deltas.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

WARN = "WARN"
CRITICAL = "CRITICAL"
RESOLVED = "RESOLVED"


@dataclass
class Alert:
    audit: str
    scope: str
    cause_hash: str
    first_seen: dt.date
    last_seen: dt.date
    state: str = WARN
    acked: bool = False
    notifications: int = 0   # count of ntfys this incident has earned

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.audit, self.scope, self.cause_hash)


@dataclass
class AlertBook:
    """In-memory incident book. Caller owns persistence + notification I/O."""
    escalate_after_days: int = 5
    alerts: dict = field(default_factory=dict)

    def observe(self, audit: str, scope: str, cause_hash: str,
                today: dt.date) -> Alert:
        """Record that (audit, scope, cause_hash) is true today. Returns the
        (possibly newly-created or newly-escalated) Alert. A rising
        ``notifications`` count vs the prior call is the caller's signal to
        send an ntfy."""
        k = (audit, scope, cause_hash)
        a = self.alerts.get(k)
        if a is None or a.state == RESOLVED:
            a = Alert(audit, scope, cause_hash, today, today,
                      state=WARN, notifications=1)   # one ntfy on NEW
            self.alerts[k] = a
            return a
        a.last_seen = today                          # dedup: no new ntfy
        if (not a.acked and a.state == WARN
                and (today - a.first_seen).days >= self.escalate_after_days):
            a.state = CRITICAL                       # blocks scope at L6 barrier
            a.notifications += 1                     # one escalation ntfy
        return a

    def ack(self, audit: str, scope: str, cause_hash: str) -> None:
        """Operator acknowledgement: the incident stays open and tracked but
        never escalates to CRITICAL (it is a known/accepted condition)."""
        a = self.alerts.get((audit, scope, cause_hash))
        if a is not None:
            a.acked = True

    def resolve_if_absent(self, seen_today: set, today: dt.date) -> list[Alert]:
        """Mark any open incident NOT observed today as RESOLVED. Returns the
        list of newly-resolved alerts (the caller may send a resolution
        ntfy).

        Book-wide: every caller that uses this MUST pass a ``seen_today``
        covering ALL audits/scopes it observed this run, or it will resolve
        incidents owned by other audits. An audit that only knows about its
        own ``(audit, scope)`` slice should call ``resolve_audit_scope``
        instead — it cannot construct a complete book-wide seen set."""
        resolved = []
        for k, a in self.alerts.items():
            if a.state != RESOLVED and k not in seen_today and a.last_seen < today:
                a.state = RESOLVED
                resolved.append(a)
        return resolved

    def resolve_audit_scope(self, audit: str, scope: str, today: dt.date,
                            *, seen: set | frozenset = frozenset()) -> list[Alert]:
        """Resolve open incidents for a SINGLE ``(audit, scope)`` whose cause
        was not observed today — the lifecycle-isolated counterpart to
        ``resolve_if_absent``. A clean audit run uses this to close only ITS
        OWN stale incidents, leaving every other audit/scope untouched (the
        isolation implied by the ``(audit, scope, cause_hash)`` dedup key).

        ``seen`` is the set of full keys this run DID observe and must keep
        open (empty for a fully-clean run). Returns the newly-resolved
        alerts."""
        resolved = []
        for k, a in self.alerts.items():
            if (a.audit == audit and a.scope == scope
                    and a.state != RESOLVED and k not in seen
                    and a.last_seen < today):
                a.state = RESOLVED
                resolved.append(a)
        return resolved

    def blocking_scopes(self) -> set:
        """Scopes with a live CRITICAL incident — the L6 barrier blocks these."""
        return {a.scope for a in self.alerts.values() if a.state == CRITICAL}

    def open_incidents(self) -> list[Alert]:
        return [a for a in self.alerts.values() if a.state != RESOLVED]
