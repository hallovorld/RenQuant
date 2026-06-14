"""Score-distribution drift audit — PSI of rank_score vs a trailing baseline.

UMBRELLA MIRROR of renquant_pipeline.kernel.score_drift — carried verbatim in
the self-contained strategy-104 kernel (stdlib + numpy only). Keep in sync.

Design: renquant-orchestrator
doc/research/2026-06-12-engineering-architecture-deep-plan.md §L6 audit
sidecar (catalog item 3) + the operator's "pipeline 中应该有自行审计 task …
early detect data abnormal" mandate. Graduates
scripts/engineering/score_drift_audit_prototype.py.

Population Stability Index on today's calibrated rank_score distribution
vs a trailing-N-run baseline. PSI bands are the industry standard:
  < 0.10  INFO     (stable)
  < 0.25  WARN     (moderate shift — investigate)
  >= 0.25 CRITICAL (population changed — calibrator collapse / feature
                    drift / scorer swap)

Pure core (psi / severity / score_drift_report) — no DB, no I/O — so it
unit-tests without fixtures; the DB-query helper is a thin separable
adapter. Read-only by construction: this module never writes a decision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INFO_BAND = 0.10
WARN_BAND = 0.25
MIN_SCORES_PER_RUN = 30   # a "full scoring run" floor; below = sell-only/partial


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. Quantile bins from ``expected``; ±inf
    edges so out-of-range ``actual`` lands in the tail bins; 1e-6 floor so
    an empty bin never produces a div-by-zero or log(0)."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    e, _ = np.histogram(expected, qs)
    a, _ = np.histogram(actual, qs)
    e = np.clip(e / e.sum(), 1e-6, None)
    a = np.clip(a / a.sum(), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def severity(value: float) -> str:
    return ("INFO" if value < INFO_BAND
            else "WARN" if value < WARN_BAND
            else "CRITICAL")


@dataclass(frozen=True)
class DriftReport:
    psi: float
    severity: str
    n_baseline: int
    n_current: int
    ok: bool          # True for INFO; WARN/CRITICAL are findings


def score_drift_report(baseline: np.ndarray, current: np.ndarray,
                       bins: int = 10) -> DriftReport:
    """PSI + banded verdict for two score arrays.

    Degenerate inputs (either side too small to bin) return a WARN
    finding rather than a number — "we could not measure stability" is a
    signal, not a pass (no-silent-continue)."""
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)
    if baseline.size < bins or current.size == 0:
        return DriftReport(psi=float("nan"), severity="WARN",
                           n_baseline=int(baseline.size),
                           n_current=int(current.size), ok=False)
    v = psi(baseline, current, bins=bins)
    sev = severity(v)
    return DriftReport(psi=v, severity=sev, n_baseline=int(baseline.size),
                       n_current=int(current.size), ok=(sev == "INFO"))


def load_score_drift_from_db(conn, *, trailing: int = 20,
                             bins: int = 10) -> DriftReport | None:
    """Build a DriftReport from a runs DB's candidate_scores table:
    latest full scoring run vs the prior ``trailing`` full runs. Returns
    None when there are too few full runs to baseline. Read-only."""
    rows = conn.execute(
        "SELECT run_id, rank_score FROM candidate_scores "
        "WHERE rank_score IS NOT NULL").fetchall()
    by_run: dict[str, list[float]] = {}
    for run_id, score in rows:
        by_run.setdefault(str(run_id), []).append(float(score))
    full = sorted(rid for rid, vals in by_run.items()
                  if len(vals) >= MIN_SCORES_PER_RUN)  # run_id is date-prefixed
    if len(full) < 3:
        return None
    latest = full[-1]
    baseline_ids = full[-(trailing + 1):-1]
    baseline = np.array([s for rid in baseline_ids for s in by_run[rid]])
    current = np.array(by_run[latest])
    return score_drift_report(baseline, current, bins=bins)
