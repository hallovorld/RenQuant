"""Post-scoring drift audit — the L6 sidecar integration point.

UMBRELLA MIRROR of renquant_pipeline.kernel.score_audit — keep in sync.

Design: renquant-orchestrator
doc/research/2026-06-12-engineering-architecture-deep-plan.md §L6 audit
sidecar. Ties together the three primitives shipped separately:

  * kernel.score_drift.load_score_drift_from_db   — MEASURE
  * persistence.record_score_drift_audit          — PERSIST (queryable history)
  * kernel.alert_lifecycle.AlertBook              — ESCALATE (no daily noise)

One call, ``run_score_drift_audit``, does all three: compute the latest
full-run PSI vs the trailing baseline, append it to score_drift_audits,
and (when an AlertBook is supplied) fold the verdict into the escalation
lifecycle so a CRITICAL that fires every run until retrain is ONE
escalating incident, not N daily alarms. Pure orchestration — no ntfy,
no broker; the caller fires notifications off the returned alert's rising
notification count and persists the book.

Designed to run concurrently with / just after scoring, per the operator
mandate "pipeline 中应该有自行审计 task … early detect data abnormal".
Read-mostly: it reads candidate_scores and appends only to the audit
log; it never touches a decision.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from kernel.score_drift import DriftReport, load_score_drift_from_db

AUDIT_NAME = "score_drift"


@dataclass(frozen=True)
class ScoreAuditResult:
    report: DriftReport | None       # None when too few full runs to baseline
    persisted: bool
    alert_state: str | None          # WARN/CRITICAL/RESOLVED when a book was used
    notifications: int | None        # the book's count for this incident


def _cause_hash(report: DriftReport) -> str:
    """Stable incident cause for the lifecycle dedup. Bucketing PSI to one
    decimal keeps a slowly-worsening CRITICAL as ONE incident rather than
    a new one each run; the severity band is the real signal."""
    return f"{report.severity}:psi~{report.psi:.1f}"


def run_score_drift_audit(
    conn,
    *,
    run_id: str | None,
    run_date: dt.date,
    book=None,
    scope: str = "panel",
    trailing: int = 20,
) -> ScoreAuditResult:
    """Measure → persist → (optionally) escalate the score-distribution drift.

    ``conn``  : a runs DB connection (candidate_scores read; audit append).
    ``book``  : an optional kernel.alert_lifecycle.AlertBook. When given, a
                non-INFO verdict is observed into the lifecycle; INFO
                resolves any open incident for this scope.
    Returns a ScoreAuditResult. No-op-safe: a None conn yields an empty
    result.
    """
    if conn is None:
        return ScoreAuditResult(None, False, None, None)

    # deferred import: persistence imports many heavy deps; keep this light
    from kernel.persistence import (  # noqa: PLC0415
        record_score_drift_audit,
    )

    report = load_score_drift_from_db(conn, trailing=trailing)
    if report is None:
        return ScoreAuditResult(None, False, None, None)

    persisted = record_score_drift_audit(
        conn, run_id=run_id, run_date=run_date, report=report) == 1

    alert_state = None
    notifications = None
    if book is not None:
        if report.ok:   # INFO — the drift cleared; close only THIS audit+scope
            # Lifecycle isolation: resolve only the score_drift incident(s)
            # for the target scope. A book-wide resolve_if_absent here would
            # silently RESOLVE unrelated open incidents (other audits, other
            # scopes) whose last_seen predates this run — see PR #137 review.
            book.resolve_audit_scope(AUDIT_NAME, scope, run_date)
            alert_state = "RESOLVED" if not _has_open(book, scope) else None
        else:           # WARN/CRITICAL — fold into the escalation lifecycle
            alert = book.observe(AUDIT_NAME, scope, _cause_hash(report), run_date)
            alert_state = alert.state
            notifications = alert.notifications
    return ScoreAuditResult(report, persisted, alert_state, notifications)


def _has_open(book, scope: str) -> bool:
    """Whether a score_drift incident for ``scope`` is still open. Scoped to
    AUDIT_NAME so the INFO verdict reflects THIS audit's incident, not an
    unrelated audit that happens to share the scope."""
    return any(a.audit == AUDIT_NAME and a.scope == scope
               and a.state != "RESOLVED"
               for a in book.alerts.values())
