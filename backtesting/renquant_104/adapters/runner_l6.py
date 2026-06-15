"""L6 score-drift audit sidecar — umbrella runner commit() entry point.

Degrade-safe wrapper that runs the L6 score-distribution drift audit after the
runner has persisted this bar's candidate_scores: load the alert book,
measure + persist the PSI drift vs the trailing baseline, fold the verdict into
the escalation lifecycle, save the book back.

AUDIT-ONLY — it reads candidate_scores and appends to the score_drift_audits /
alert_incidents tables; it NEVER touches a trade decision, cash, or position.
On ANY failure (missing L6 stack behind a lagging pin, missing tables, a bad
row) it logs once and returns None rather than raise, so it can never perturb
a live commit().
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("adapters.runner")  # same logger — audit is a runner concern


def run_l6_score_audit_sidecar(
    conn: Any,
    *,
    run_id: Any,
    run_date: Any,
    scope: str = "panel",
    escalate_after_days: int = 5,
):
    """Run the L6 score-drift audit against ``conn``. Returns the
    ScoreAuditResult, or None on a no-op / degraded run. Never raises."""
    if conn is None:
        return None
    try:
        from kernel.persistence import load_alert_book, save_alert_book
        from kernel.score_audit import run_score_drift_audit
    except Exception:  # L6 stack not present (e.g. lagging runtime pin)
        return None
    try:
        book = load_alert_book(conn, escalate_after_days=escalate_after_days)
        result = run_score_drift_audit(
            conn, run_id=run_id, run_date=run_date, book=book, scope=scope)
        save_alert_book(conn, book)
        return result
    except Exception as exc:  # never block the bar on an audit write
        log.warning("L6-SCORE-AUDIT sidecar degraded: %s", exc)
        return None
