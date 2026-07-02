"""Shadow model scoring — record alt-model decisions via MLflow tracking.

Per 2026-05-18 user request: inference accepts shadow models that run
the full pipeline but DON'T submit orders; all data recorded for review.

Per 2026-05-18 second update: use 3rd-party library (MLflow) instead of
custom SQLite — battle-tested experiment tracking, built-in UI for
comparison, standard schema.

MLflow tracking layout:
  experiment_name = ranking.panel_scoring.shadow_experiment
                    (default: "renquant_104_shadow")
  per inference: ONE MLflow Run per (date, shadow_model)
    tags:    as_of_date / shadow_name / shadow_kind / primary_kind
    metrics: mean_primary_score / mean_shadow_score / mean_diff
             corr_primary_shadow / rank_agreement_top5 / top5_overlap
    artifact: comparison.csv (per-ticker primary vs shadow scores)

Query later (UI or programmatic):
  $ mlflow ui --backend-store-uri file:./mlruns
  → http://127.0.0.1:5000 → experiment → compare runs

  # Or programmatic:
  import mlflow
  exp = mlflow.get_experiment_by_name("renquant_104_shadow")
  runs = mlflow.search_runs(exp.experiment_id, filter_string="tags.shadow_name='patchtst_v1'")
  print(runs[["start_time", "metrics.mean_diff", "metrics.corr_primary_shadow"]])

Why MLflow over custom SQLite:
  - Standard schema, well-documented
  - Built-in comparison UI
  - Run-level filtering/aggregation
  - Artifact storage (per-bar comparison tables)
  - 3rd-party maintained, battle-tested in production
  - No new dependency (mlflow 3.12.0 already installed)

Tests in tests/test_shadow_scoring.py.
"""
from __future__ import annotations
import datetime
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from kernel.pipeline.context import InferenceContext
from kernel.pipeline.pipeline import Job, Task

log = logging.getLogger("kernel.panel_pipeline.shadow_scoring")

# OMP fix per [[concurrency_resource_budget]]: ensure single-thread BEFORE
# any torch model construction in shadow scorers (PatchTST etc).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Default MLflow tracking URI: file-based local store at repo/mlruns
_DEFAULT_TRACKING_URI = "file:" + str(Path(__file__).resolve().parents[4] / "mlruns")
_DEFAULT_EXPERIMENT = "renquant_104_shadow"
_SCORER_CACHE: dict[tuple[str, str], object] = {}

# 2026-07-01 (operator incident: "[SHADOW]...BUY OXY" ntfy misread as a
# PatchTST recommendation — see doc/progress/2026-07-01-shadow-ntfy-top-picks.md):
# size of the per-shadow-model top-N recommendation list surfaced in
# ctx._shadow_summary[i]["top_picks"] and rendered as the ntfy
# "SHADOW-PICKS[name]" line. Configurable via
# ranking.panel_scoring.shadow_top_n_picks; default kept small for ntfy
# length budget.
_DEFAULT_TOP_N_PICKS = 5

# 2026-07-01 ROUND 2 (Codex CHANGES_REQUESTED on umbrella PR #426 — see
# doc/progress/2026-07-01-shadow-ntfy-top-picks.md addendum): the top-N list
# above is a RAW diagnostic rank, not a validated recommendation. Before it
# is ever surfaced as "actionable" this module binds it to an admission
# verdict — is the shadow ARTIFACT fresh enough, and is the scored universe
# complete enough, for rank-1-of-N to mean anything comparable day to day.
#
# Tier NAMING mirrors (not imports — model_freshness_monitor.py lives in the
# renquant-orchestrator repo, a separate Python package/deploy unit)
# renquant-orchestrator/src/renquant_orchestrator/model_freshness_monitor.py
# TIER_HEALTHY/WARN/ESCALATE/BREACH/UNKNOWN + that module's SHADOW_POLICY
# cadence (warn=28d/escalate=33d/breach=35d), so an operator who already
# reads that monitor's alerts sees the SAME vocabulary here. A real shadow
# PatchTST artifact has been observed ~140 DAYS stale in this codebase —
# 4x past breach — which is exactly the silent-overclaim case this gates.
_FRESHNESS_WARN_DAYS = 28
_FRESHNESS_ESCALATE_DAYS = 33
_FRESHNESS_BREACH_DAYS = 35
_FRESHNESS_TIER_HEALTHY = "healthy"
_FRESHNESS_TIER_WARN = "warn"
_FRESHNESS_TIER_ESCALATE = "escalate"
_FRESHNESS_TIER_BREACH = "breach"
_FRESHNESS_TIER_UNKNOWN = "unknown"

# Coverage: fraction of the CONFIGURED watchlist the shadow model actually
# scored today (n_candidates / len(config["watchlist"])). Field naming
# mirrors the coverage / n_have / n_expected convention already used by this
# repo's own preflight (kernel/preflight.py _check_feature_coverage,
# _check_sector_map_coverage) and renquant-pipeline's
# task_data_verification.py (`coverage`, `n_have`/`n_expected`). The default
# floor sits in the same 0.70-0.90 band those checks already use; an 83/292
# (~28%) censored subset — the real incident this responds to — fails
# clearly under any threshold in that band.
_DEFAULT_MIN_COVERAGE = 0.80

# 2026-07-01 ROUND 3 (Codex #426 review point 1 named BOTH "trained cutoff"
# AND "feature-data cutoff" as separate provenance to bind): binding
# DATA-cutoff field priority, most-binding first — mirrors orchestrator's
# model_freshness_monitor.py DATA_CUTOFF_FIELDS. Preferred over
# ``trained_date`` whenever present: ``trained_date`` is run time, not a
# data-freshness axis, and a fresh ``trained_date`` over stale/absent DATA
# must never certify freshness — this codebase hit exactly that bug class
# before (2026-06-15 "model stale-by-split-recipe" incident: a live model
# failed WF sanity because a fresh-looking retrain was still keyed to a
# stale val-tail cutoff). ``hf_patchtst_scorer.py`` already stamps
# ``effective_train_cutoff_date`` into ``scorer.metadata`` at load time
# (from ckpt/contract/sidecar, ``_coalesce``) for exactly this reason — the
# real PatchTST-shadow incident this round responds to has this field
# available; round-2 computed age from ``trained_date`` alone and left it
# unused.
_DATA_CUTOFF_FIELDS = (
    "label_observation_cutoff",
    "effective_selection_cutoff_date",
    "effective_train_cutoff_date",
    "data_cutoff_date",
    "live_train_end",
    "cutoff_date",
)

# 2026-07-01 ROUND 4 (Codex CHANGES_REQUESTED on umbrella PR #426 round 4 —
# see doc/progress/2026-07-01-shadow-ntfy-top-picks.md addendum #4): FOUR
# remaining fail-closed gaps, plus a SCOPE NARROWING that is the main ask of
# this round.
#
# GAP 1 — no ``trained_date`` fallback. A missing/unparseable binding DATA
# cutoff is UNKNOWN, full stop; ``trained_date`` is DISPLAYED as
# process-liveness context only and is NEVER used to compute an age or
# certify actionability (round-3 already preferred a binding cutoff when
# BOTH were present, but still fell back to ``trained_date`` when only it
# was present — that fallback reopened the exact stale-data-spoof round-3
# closed for the "both present" case: a fresh ``trained_date`` over
# genuinely stale/absent DATA).
#
# GAP 2 — ``n_expected<=0`` BLOCKS. An unknown/unresolvable universe
# denominator (watchlist not configured/available) previously degraded
# coverage to "unknown, does not block" — this let picks pass with a
# denominator no one can verify. Now ``n_expected<=0`` fails closed like any
# other missing-provenance case.
#
# GAP 3 — a missing artifact fingerprint BLOCKS. Immutable artifact
# identity is mandatory for an actionable verdict: a missing fingerprint
# previously still produced a ``run_id`` keyed on the ``nofingerprint``
# sentinel and proceeded — now it fails closed instead.
#
# GAP 4 — horizon-aware age compensation for ``label_observation_cutoff``.
# For a fwd-N-session-label model, a causally valid label-observation cutoff
# is intentionally horizon-lagged (the label needs N sessions forward to be
# observed, so even a same-day retrain's cutoff sits ~N business days behind
# the raw data frontier by construction) — comparing that RAW age directly
# against a short-window freshness threshold marks genuinely fresh artifacts
# stale. This reuses the horizon-aware age-compensation PATTERN already
# merged in renquant-orchestrator's model_freshness_monitor.py
# (``_subtract_business_days`` / ``_expected_lag_calendar_days`` — ported
# here, not imported: separate Python package/deploy unit): the RAW age is
# NEVER mutated, only the axis's own EXPECTED lag is computed and used to
# derive a distinct, separately-persisted "horizon-compensated" age that the
# freshness tier is judged against. The lag is keyed on the model's OWN
# stamped horizon (``artifact_meta["lookahead_days"]``, already stamped by
# ``hf_patchtst_scorer.py``) when present, falling back to the documented
# PatchTST fwd_60d convention otherwise. A cutoff LATER than ``as_of_date``
# (look-ahead) is checked against the RAW age BEFORE any compensation and
# fails closed regardless — compensation only ever excuses an EXPECTED lag,
# never a genuine future cutoff.
#
# SCOPE NARROWING (the main ask of this round): the freshness/coverage
# thresholds throughout this module are UNVALIDATED OPERATIONAL GUESSES, not
# empirically-grounded bands — no preregistered shadow evaluation has
# established minimum coverage/freshness bands on fixed sessions with
# costs/top-N stability. Per the review: "keep all picks explicitly NOT
# ACTIONABLE by default... A config flag may enable experimental display,
# but the safe default cannot authorize discretionary capital from
# warn-tier/raw-rank output." This converges the feature to the same
# "Stage-1 operations-only, no execution-quality claim until preregistered
# validation" discipline already established this session for the
# renquant105 architecture (observe/collect first, claim actionability only
# after validated evidence) — the design rationale is the SAME principle,
# not an arbitrary restriction.
#
# So ``actionable`` is now a two-part AND, both computed by
# ``_compute_admission``:
#   gates_passed = every fail-closed gate below passes (freshness tier incl.
#                  GAP 1/4, coverage incl. GAP 2, fingerprint incl. GAP 3) —
#                  "would be actionable if experimental display were on".
#   actionable   = gates_passed AND the explicit opt-in
#                  ``shadow_experimental_actionable_display`` config flag
#                  (``ranking.panel_scoring.shadow_experimental_actionable_display``
#                  in strategy_config*.json — mirrors the flat
#                  ``shadow_*`` key convention this section already uses:
#                  ``shadow_top_n_picks``, ``shadow_min_coverage``, etc.).
# The flag can only ever RAISE an already-gates_passed verdict to
# actionable — ``gates_passed`` is computed byte-for-byte identically
# whether the flag is set or not, so the flag NEVER bypasses a failed gate.
_DEFAULT_EXPERIMENTAL_ACTIONABLE_DISPLAY = False

# GAP 4: horizon compensation applies ONLY to this axis (mirrors
# orchestrator's ``_AXIS_EXPECTED_LAG_BDAYS``, keyed only on
# ``label_observation_cutoff`` — every other binding-cutoff field has no
# inherent fwd-label horizon lag).
_LABEL_OBSERVATION_FIELD = "label_observation_cutoff"
# Fallback business-day horizon used ONLY when the axis needs compensation
# (binding field == ``_LABEL_OBSERVATION_FIELD``) but the artifact did not
# stamp its own ``lookahead_days`` — the documented PatchTST fwd_60d
# convention (matches orchestrator's ``_LABEL_OBSERVATION_LOOKAHEAD_BDAYS``).
_DEFAULT_LABEL_OBSERVATION_LOOKAHEAD_BDAYS = 60


def _subtract_business_days(base: datetime.date, n: int) -> datetime.date:
    """Subtract ``n`` Mon-Fri business days from ``base`` (no holiday
    calendar — sufficient for a fixed, documented label horizon). Ports
    orchestrator's ``model_freshness_monitor._subtract_business_days``
    weekday semantics (not imported: separate Python package)."""
    current = base
    remaining = n
    while remaining > 0:
        current -= datetime.timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def _expected_lag_calendar_days(
    binding_field: str | None,
    lookahead_bdays: object,
    today: datetime.date,
) -> float:
    """Calendar-day width of ``binding_field``'s EXPECTED fwd-label-horizon
    lag as of ``today`` — 0 for every axis except ``label_observation_cutoff``
    (GAP 4). ``lookahead_bdays`` is the model's OWN stamped horizon
    (``artifact_meta["lookahead_days"]``); a missing/unparseable/non-positive
    value falls back to ``_DEFAULT_LABEL_OBSERVATION_LOOKAHEAD_BDAYS``. Ports
    orchestrator's ``_expected_lag_calendar_days`` (not imported)."""
    if binding_field != _LABEL_OBSERVATION_FIELD:
        return 0.0
    try:
        bdays = int(lookahead_bdays) if lookahead_bdays else 0
    except (TypeError, ValueError):
        bdays = 0
    if bdays <= 0:
        bdays = _DEFAULT_LABEL_OBSERVATION_LOOKAHEAD_BDAYS
    expected_frontier = _subtract_business_days(today, bdays)
    return float((today - expected_frontier).days)


def _parse_iso_date(value: object) -> datetime.date | None:
    if not value:
        return None
    text = str(value).strip()
    if len(text) < 10:
        return None
    try:
        return datetime.date.fromisoformat(text[:10])
    except ValueError:
        return None


def _binding_cutoff(artifact_meta: dict) -> tuple[datetime.date | None, Optional[str]]:
    """First present, parseable field in ``_DATA_CUTOFF_FIELDS`` (most
    binding first). ``(None, None)`` when no binding DATA cutoff is present
    — this is UNKNOWN provenance, not automatically "old"; the caller falls
    back to ``trained_date`` (a strictly weaker signal — see module note
    above) only in that case."""
    for field_name in _DATA_CUTOFF_FIELDS:
        parsed = _parse_iso_date(artifact_meta.get(field_name))
        if parsed is not None:
            return parsed, field_name
    return None, None


def _ensure_mlflow_setup(tracking_uri: Optional[str] = None,
                         experiment_name: Optional[str] = None) -> str:
    """Set MLflow tracking URI + experiment. Returns experiment_id."""
    import mlflow  # noqa: PLC0415
    mlflow.set_tracking_uri(tracking_uri or _DEFAULT_TRACKING_URI)
    name = experiment_name or _DEFAULT_EXPERIMENT
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
    else:
        exp_id = exp.experiment_id
    return exp_id


def _log_shadow_run(experiment_id: str, as_of_date, shadow_name: str,
                    shadow_kind: str, primary_kind: str,
                    primary_scores: dict[str, float],
                    shadow_scores: dict[str, float],
                    primary_ranks: dict[str, int],
                    shadow_ranks: dict[str, int]) -> None:
    """Persist one shadow run's comparison to MLflow."""
    import mlflow  # noqa: PLC0415

    # Aggregate metrics
    tickers = sorted(set(primary_scores) & set(shadow_scores))
    if not tickers:
        return
    ps = np.array([primary_scores[t] for t in tickers])
    ss = np.array([shadow_scores[t] for t in tickers])
    diffs = ss - ps
    # Rank agreement: how many of top-5 primary are in top-5 shadow
    n_top = min(5, len(tickers))
    top_primary = sorted(primary_ranks.items(), key=lambda x: x[1])[:n_top]
    top_shadow = sorted(shadow_ranks.items(), key=lambda x: x[1])[:n_top]
    top_primary_set = {t for t, _ in top_primary}
    top_shadow_set = {t for t, _ in top_shadow}
    overlap = len(top_primary_set & top_shadow_set)
    # Pearson correlation
    if np.std(ps) > 1e-9 and np.std(ss) > 1e-9:
        corr = float(np.corrcoef(ps, ss)[0, 1])
    else:
        corr = float("nan")

    run_name = f"{shadow_name}_{as_of_date}"
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        mlflow.set_tags({
            "as_of_date": str(as_of_date),
            "shadow_name": shadow_name,
            "shadow_kind": shadow_kind,
            "primary_kind": primary_kind,
            "n_candidates": str(len(tickers)),
        })
        mlflow.log_metrics({
            "mean_primary_score": float(np.mean(ps)),
            "mean_shadow_score": float(np.mean(ss)),
            "mean_diff": float(np.mean(diffs)),
            "std_diff": float(np.std(diffs)),
            "corr_primary_shadow": corr,
            f"top{n_top}_overlap": float(overlap),
            f"top{n_top}_overlap_pct": float(overlap / n_top) if n_top else 0.0,
        })
        # Per-ticker comparison table as artifact
        comparison = pd.DataFrame({
            "ticker": tickers,
            "primary_score": ps,
            "shadow_score": ss,
            "diff": diffs,
            "primary_rank": [primary_ranks.get(t, -1) for t in tickers],
            "shadow_rank": [shadow_ranks.get(t, -1) for t in tickers],
            "rank_diff": [shadow_ranks.get(t, 0) - primary_ranks.get(t, 0)
                          for t in tickers],
        })
        # MLflow log_table writes to artifacts/<artifact_file>
        mlflow.log_table(comparison, "comparison.json")


def _freshness_tier(age_days: float | None) -> str:
    """Bucket artifact age into healthy/warn/escalate/breach/unknown.

    ``None``/non-finite age (missing or unparseable cutoff) is UNKNOWN,
    never a silent pass — mirrors this repo's own P-MODEL-STALENESS
    convention (renquant-pipeline
    kernel/preflight_pipeline/tasks/staleness.py: missing provenance is a
    fail, not a skip).

    2026-07-01 ROUND 3: a NEGATIVE age (a cutoff LATER than ``as_of_date`` —
    a look-ahead) fails closed to ``breach`` rather than reading as healthy
    (mirrors orchestrator's model_freshness_monitor.py look-ahead guard —
    PR #211 lesson there: a window bounded on only one side lets a negative
    age silently read healthy).
    """
    if age_days is None:
        return _FRESHNESS_TIER_UNKNOWN
    try:
        age = float(age_days)
    except (TypeError, ValueError):
        return _FRESHNESS_TIER_UNKNOWN
    if age != age or age in (float("inf"), float("-inf")):  # NaN/inf check
        return _FRESHNESS_TIER_UNKNOWN
    if age < 0:
        return _FRESHNESS_TIER_BREACH  # look-ahead cutoff, never healthy
    if age >= _FRESHNESS_BREACH_DAYS:
        return _FRESHNESS_TIER_BREACH
    if age >= _FRESHNESS_ESCALATE_DAYS:
        return _FRESHNESS_TIER_ESCALATE
    if age >= _FRESHNESS_WARN_DAYS:
        return _FRESHNESS_TIER_WARN
    return _FRESHNESS_TIER_HEALTHY


def _compute_admission(
    *,
    name: str,
    as_of_date,
    artifact_meta: dict,
    n_scored: int,
    n_expected: int,
    min_coverage: float,
    experimental_actionable_display: bool = _DEFAULT_EXPERIMENTAL_ACTIONABLE_DISPLAY,
) -> dict:
    """Bind today's shadow top-picks to provenance + an actionability
    verdict. See doc/progress/2026-07-01-shadow-ntfy-top-picks.md addendum
    (Codex CHANGES_REQUESTED, rounds 2-4 on umbrella PR #426).

    This deliberately does NOT re-check feature-DATA freshness: primary and
    shadow score the SAME ``ctx._panel_matrix`` / ``panel_history``, already
    gated upstream by ``DataFreshnessGateTask``
    (kernel/pipeline/task_data_freshness.py) before Phase 3
    (``PanelScoringJob``, which runs ``ApplyShadowScoringTask``) executes.
    What IS shadow-specific and NOT covered by that upstream gate:

      1. the shadow ARTIFACT's own training-cutoff staleness — primary has
         a retrain-cadence gate; a shadow artifact can silently rot behind
         it with nothing re-checking it (real observed case: ~140d stale).
      2. the shadow-SCORED universe's coverage of the configured watchlist —
         shadow scorers often use different feature_cols/seq_len than
         primary, censoring a different (often much smaller) subset, so
         "rank 1" here is not comparable to primary's rank 1, or to a
         different day's shadow run, without knowing n_scored/n_expected.

    ROUND 4 fail-closed gates (see the module-level comment block above
    ``_DEFAULT_EXPERIMENTAL_ACTIONABLE_DISPLAY`` for the full rationale):

      GAP 1 — ``trained_date`` is NEVER used to compute an age or certify
      freshness, even when no binding DATA cutoff is present. It is
      returned in ``trained_date`` for DISPLAY (process-liveness) only. A
      missing/unparseable binding cutoff is UNKNOWN, full stop — no
      fallback (round-3 still fell back to it when it was the ONLY field
      present, reopening the stale-data spoof for that case).
      GAP 2 — ``n_expected<=0`` (unknown/unresolvable universe size) BLOCKS
      (``coverage_ok=False``), rather than degrading to "unknown, does not
      block" as round-3 did.
      GAP 3 — a missing/placeholder artifact fingerprint BLOCKS
      (``fingerprint_ok=False``) rather than proceeding with the
      ``"nofingerprint"`` sentinel baked into ``run_id``.
      GAP 4 — ``label_observation_cutoff`` gets a horizon-aware age
      compensation (``horizon_compensated_age_days``, distinct from the
      literal, unadjusted ``age_days``) before it is tiered, so a genuinely
      fresh fwd-N-session-label retrain does not read born-stale. A
      look-ahead cutoff (later than ``as_of_date``) is checked against the
      RAW age BEFORE compensation and fails closed regardless.

    Fail-closed gate outcome is ``gates_passed`` — every gate above passing.
    The final ``actionable`` is the SCOPE-NARROWED verdict:
    ``gates_passed AND experimental_actionable_display``. The freshness/
    coverage thresholds here are unvalidated operational guesses (no
    preregistered shadow evaluation has run), so the safe DEFAULT
    (``experimental_actionable_display=False``) is always NOT ACTIONABLE
    even when every gate passes — the flag is an explicit opt-in that can
    only ever RAISE an already-``gates_passed`` verdict, never bypass a
    failed gate.
    """
    today = (as_of_date if isinstance(as_of_date, datetime.date)
              else datetime.date.today())
    trained_date = artifact_meta.get("trained_date")
    binding_cutoff, binding_cutoff_field = _binding_cutoff(artifact_meta)

    # GAP 1: age is derived ONLY from a binding DATA cutoff. No
    # ``trained_date`` fallback — see docstring.
    age_days: float | None = None
    horizon_compensated_age_days: float | None = None
    horizon_lag_days = 0.0
    if binding_cutoff is not None:
        age_days = float((today - binding_cutoff).days)

    if binding_cutoff is None:
        tier = _FRESHNESS_TIER_UNKNOWN
    elif age_days < 0:
        # Look-ahead: judged on the RAW age, BEFORE any horizon
        # compensation — GAP 4 never excuses a genuine future cutoff.
        tier = _FRESHNESS_TIER_BREACH
    else:
        # GAP 4: widen the effective age for a fwd-N-session-label axis by
        # its EXPECTED lag (never mutating ``age_days`` itself — persisted
        # separately below).
        horizon_lag_days = _expected_lag_calendar_days(
            binding_cutoff_field, artifact_meta.get("lookahead_days"), today)
        horizon_compensated_age_days = max(0.0, age_days - horizon_lag_days)
        tier = _freshness_tier(horizon_compensated_age_days)

    # GAP 2: n_expected<=0 (unknown/unresolvable universe size) BLOCKS —
    # coverage cannot be verified, so it must not silently pass.
    if n_expected is None or n_expected <= 0:
        coverage = None
        coverage_ok = False
    else:
        coverage = float(n_scored) / n_expected
        coverage_ok = coverage >= min_coverage

    # GAP 3: a missing/placeholder fingerprint BLOCKS — immutable artifact
    # identity is mandatory for an actionable verdict.
    fingerprint = (
        artifact_meta.get("artifact_fingerprint")
        or artifact_meta.get("artifact_sha256")
        or artifact_meta.get("model_content_fingerprint")
    )
    fingerprint_ok = bool(fingerprint) and bool(str(fingerprint).strip())
    fingerprint_short = str(fingerprint)[:12] if fingerprint_ok else "nofingerprint"
    run_id = f"{as_of_date}:{name}:{fingerprint_short}"

    reasons: list[str] = []
    cutoff_desc = (
        f"{binding_cutoff_field}={binding_cutoff.isoformat()}" if binding_cutoff is not None
        else (f"trained_date={trained_date} is informational only, not a freshness axis"
              if trained_date else "no binding data cutoff or trained_date")
    )
    if tier == _FRESHNESS_TIER_UNKNOWN:
        reasons.append(f"freshness=unknown ({cutoff_desc}); binding DATA cutoff "
                        "missing/unparseable (fail-closed, GAP 1)")
    elif age_days is not None and age_days < 0:
        reasons.append(f"look-ahead cutoff {cutoff_desc} is later than as_of_date={today}")
    elif tier == _FRESHNESS_TIER_BREACH:
        reasons.append(
            f"artifact {horizon_compensated_age_days:.0f}d stale (raw {age_days:.0f}d, "
            f"{cutoff_desc}, breach>={_FRESHNESS_BREACH_DAYS}d)")
    elif tier == _FRESHNESS_TIER_ESCALATE:
        reasons.append(
            f"artifact {horizon_compensated_age_days:.0f}d stale (raw {age_days:.0f}d, "
            f"{cutoff_desc}, escalate>={_FRESHNESS_ESCALATE_DAYS}d)")
    if not coverage_ok:
        if coverage is None:
            reasons.append(
                f"n_expected={n_expected} unknown/unresolvable universe size — "
                "coverage cannot be verified (fail-closed, GAP 2)")
        else:
            reasons.append(
                f"coverage {n_scored}/{n_expected} ({coverage:.0%}) < {min_coverage:.0%}")
    if not fingerprint_ok:
        reasons.append(
            "missing artifact fingerprint — immutable artifact identity required "
            "(fail-closed, GAP 3)")

    # Only healthy/warn tiers are actionable — escalate is one step short of
    # breach and, per the incident this responds to, presenting it as a
    # plain pick is exactly the overclaim risk being closed here.
    fresh_ok = tier in (_FRESHNESS_TIER_HEALTHY, _FRESHNESS_TIER_WARN)
    gates_passed = fresh_ok and coverage_ok and fingerprint_ok

    # SCOPE NARROWING (round 4 main ask): default NOT ACTIONABLE regardless
    # of gates_passed, unless the explicit opt-in flag is set. The flag
    # never bypasses a gate — it only lets an already-gates_passed verdict
    # render as actionable.
    actionable = gates_passed and experimental_actionable_display
    if gates_passed and not experimental_actionable_display:
        reasons.append(
            "NOT ACTIONABLE by default pending a preregistered shadow "
            "evaluation of these freshness/coverage thresholds (Stage-1 "
            "observability-only, matching the renquant105 discipline); set "
            "ranking.panel_scoring.shadow_experimental_actionable_display="
            "true to opt in once gates pass")

    return {
        "verdict": tier,
        "gates_passed": gates_passed,
        "actionable": actionable,
        "experimental_actionable_display": experimental_actionable_display,
        "trained_date": trained_date,
        "binding_cutoff": binding_cutoff.isoformat() if binding_cutoff is not None else None,
        "binding_cutoff_field": binding_cutoff_field,
        "age_days": age_days,
        "horizon_compensated_age_days": horizon_compensated_age_days,
        "horizon_lag_days": horizon_lag_days,
        "artifact_fingerprint": fingerprint if fingerprint_ok else None,
        "fingerprint_ok": fingerprint_ok,
        "n_scored": n_scored,
        "n_expected": n_expected if (n_expected is not None and n_expected > 0) else None,
        "coverage": round(coverage, 4) if coverage is not None else None,
        "coverage_ok": coverage_ok,
        "min_coverage": min_coverage,
        "reasons": reasons,
        "run_id": run_id,
    }


def _compute_shadow_summary(
    name: str,
    kind: str,
    primary_scores: dict[str, float],
    sorted_primary: list[tuple[str, float]],
    primary_ranks: dict[str, int],
    shadow_dict: dict[str, float],
    sorted_shadow: list[tuple[str, float]],
    shadow_ranks: dict[str, int],
    top_n_picks: int,
    *,
    as_of_date=None,
    artifact_meta: dict | None = None,
    n_expected_universe: int = 0,
    min_coverage: float = _DEFAULT_MIN_COVERAGE,
    experimental_actionable_display: bool = _DEFAULT_EXPERIMENTAL_ACTIONABLE_DISPLAY,
) -> dict:
    """Build the compact per-shadow-model ntfy/audit summary dict.

    2026-05-19 (user mandate "want to know what shadow will do in ntfy"):
    single-line-of-ntfy-friendly rollup — shadow top-3 picks, top-10 overlap
    with primary, Spearman rank correlation.

    2026-07-01 EXTENDED (operator incident: a "[SHADOW]...BUY OXY" ntfy was
    misread as "the shadow PatchTST model recommends OXY" — see
    doc/progress/2026-07-01-shadow-ntfy-top-picks.md; operator mandate
    "shadow的message应该给出带有信心指数的推荐"): adds a top-N recommendation
    list (``top_picks``) with an HONEST, relative-only confidence indicator
    (rank / percentile / z-score within today's scored universe). Shadow
    scorers have NO fitted probability calibrator — only
    ApplyGlobalCalibrationTask calibrates the PRIMARY score — so this
    deliberately never emits a fabricated "% confidence" number.

    Pulled out of ``ApplyShadowScoringTask.run`` as a pure function (no I/O,
    no ctx access) so it is unit-testable against a small hand-computed
    fixture without mocking MLflow / the model registry / scorer loading.
    Callers MUST pass in the SAME score/rank arrays already used elsewhere
    for this shadow model (``sorted_primary``/``primary_ranks`` from the
    primary panel score, ``shadow_dict``/``sorted_shadow``/``shadow_ranks``
    from this shadow model's own score) — this function never re-scores or
    recomputes those arrays from scratch.

    2026-07-01 ROUND 2 (Codex CHANGES_REQUESTED — see
    doc/progress/2026-07-01-shadow-ntfy-top-picks.md addendum): a raw rank
    is not itself an actionable recommendation. Every call now computes an
    ``admission`` verdict (see ``_compute_admission``) binding the picks to
    artifact freshness + scored-universe coverage, and the returned
    ``actionable``/``run_id`` top-level keys let ``live/runner.py`` decide
    whether to render the ranked picks at all or label them NOT ACTIONABLE.
    ``as_of_date``/``artifact_meta``/``n_expected_universe`` are keyword-only
    with permissive defaults so existing positional callers keep working —
    but omitting them means ``artifact_meta`` is empty (no ``trained_date``)
    and the verdict is fail-closed UNKNOWN/NOT-actionable, by design.

    2026-07-01 ROUND 4: ``experimental_actionable_display`` is threaded
    straight through to ``_compute_admission`` (see its docstring for the
    scope-narrowing default) — omitting it keeps the safe default (always
    NOT ACTIONABLE, even when every other gate passes).
    """
    import numpy as _np  # noqa: PLC0415

    top10_primary = set(t for t, _ in sorted_primary[:10])
    top10_shadow = set(t for t, _ in sorted_shadow[:10])
    overlap = len(top10_primary & top10_shadow)
    common = sorted(set(primary_scores) & set(shadow_dict))
    if len(common) >= 5:
        pr = _np.array([primary_ranks[t] for t in common])
        sr = _np.array([shadow_ranks[t] for t in common])
        from scipy.stats import spearmanr as _sp  # noqa: PLC0415
        rho, _ = _sp(pr, sr)
        rho = float(rho) if _np.isfinite(rho) else float("nan")
    else:
        rho = float("nan")
    top3 = [t for t, _ in sorted_shadow[:3]]

    primary_top_n_set = {t for t, _ in sorted_primary[:top_n_picks]}
    shadow_vals = _np.array(list(shadow_dict.values()), dtype=float)
    shadow_mean = float(_np.mean(shadow_vals)) if shadow_vals.size else float("nan")
    shadow_std = float(_np.std(shadow_vals)) if shadow_vals.size else float("nan")
    n_universe = len(shadow_dict)
    top_picks = []
    for t, sc in sorted_shadow[:top_n_picks]:
        rank = shadow_ranks[t]
        # shadow_percentile: 100.0 = best (rank 1), approaches 0 as rank
        # worsens — the conventional "higher percentile = better" reading.
        # FIXED 2026-07-01 round 2 (Codex CHANGES_REQUESTED): was
        # `rank / n * 100`, which gave the BEST-ranked name the LOWEST
        # percentile (rank 1 of 83 -> 1.2), exactly backwards from how
        # "percentile" is normally read and flagged as misleading.
        pct = ((n_universe - rank + 1) / n_universe * 100.0) if n_universe else float("nan")
        if shadow_std and shadow_std > 1e-12:
            z = (float(sc) - shadow_mean) / shadow_std
        else:
            z = float("nan")
        top_picks.append({
            "ticker": t,
            "shadow_score": float(sc),
            "shadow_rank": rank,
            "shadow_percentile": round(pct, 1) if pct == pct else float("nan"),
            "shadow_zscore": round(z, 2) if z == z else float("nan"),
            # NOT determinable here: ApplyShadowScoringTask runs inside
            # PanelScoringJob (Phase 3), which executes BEFORE
            # RankingJob/SelectionJob populate ctx.ranked / ctx.orders.
            # Whether primary actually SELECTED/BOUGHT this ticker today is
            # unknown at this point in the pipeline — left None rather than
            # guessed. live/runner.py overlays the real value from
            # ctx.orders_placed at ntfy-render time, once the full pipeline
            # + adapter.commit has run.
            "in_primary_admitted": None,
            "in_primary_topN": t in primary_top_n_set,
        })

    admission = _compute_admission(
        name=name,
        as_of_date=as_of_date if as_of_date is not None else datetime.date.today(),
        artifact_meta=artifact_meta or {},
        n_scored=n_universe,
        n_expected=n_expected_universe,
        min_coverage=min_coverage,
        experimental_actionable_display=experimental_actionable_display,
    )

    return {
        "name": name, "kind": kind,
        "top3": top3,
        "top10_overlap": overlap,
        "n_candidates": len(shadow_dict),
        "spearman_vs_primary": rho,
        "top_picks": top_picks,
        "top_picks_n": top_n_picks,
        "admission": admission,
        "actionable": admission["actionable"],
        "gates_passed": admission["gates_passed"],
        "run_id": admission["run_id"],
    }


class ApplyShadowScoringTask(Task):
    """Run each configured shadow model on the SAME candidates as primary,
    record scores via MLflow tracking. Read-only — no order submission.

    Reads:
      - ctx.candidates (with .panel_score set by main)
      - ctx.config["ranking"]["panel_scoring"]["shadow_models"]
      - ctx.config["ranking"]["panel_scoring"]["shadow_tracking_uri"]
        (default: file:<repo>/mlruns)
      - ctx.config["ranking"]["panel_scoring"]["shadow_experiment"]
        (default: "renquant_104_shadow")

    Writes:
      - MLflow run per shadow model per inference bar

    Soft-fail: shadow errors logged, primary pipeline unaffected.
    """

    name = "ApplyShadowScoringTask"

    def run(self, ctx: InferenceContext) -> bool | None:
        panel_cfg = ctx.config.get("ranking", {}).get("panel_scoring", {})
        if panel_cfg.get("shadow_enabled", True) is False:
            return None
        shadow_models = panel_cfg.get("shadow_models", []) or []
        if not shadow_models:
            return None

        # Primary scores (must be set by main ApplyScoresTask)
        cands = list(ctx.candidates) if ctx.candidates else []
        if not cands:
            log.info("ApplyShadowScoringTask: 0 candidates — skip")
            return None
        primary_scores = {c.ticker: float(c.panel_score)
                          for c in cands if c.panel_score is not None}
        if not primary_scores:
            return None
        sorted_primary = sorted(primary_scores.items(), key=lambda x: -x[1])
        primary_ranks = {t: i + 1 for i, (t, _) in enumerate(sorted_primary)}
        primary_kind = panel_cfg.get("kind", "xgb")
        top_n_picks = int(panel_cfg.get("shadow_top_n_picks", _DEFAULT_TOP_N_PICKS) or
                          _DEFAULT_TOP_N_PICKS)
        # 2026-07-01 round 2: admission-verdict inputs shared by every
        # shadow model this cycle — see _compute_admission docstring.
        as_of_date = getattr(ctx, "today", datetime.date.today())
        n_expected_universe = len(ctx.config.get("watchlist", []) or [])
        min_coverage = float(
            panel_cfg.get("shadow_min_coverage", _DEFAULT_MIN_COVERAGE)
            or _DEFAULT_MIN_COVERAGE)
        # 2026-07-01 ROUND 4: explicit opt-in to render gates_passed picks
        # as actionable — see the module-level comment block above
        # _DEFAULT_EXPERIMENTAL_ACTIONABLE_DISPLAY. Default False: picks
        # always render NOT ACTIONABLE until a preregistered shadow
        # evaluation validates these thresholds.
        experimental_actionable_display = bool(
            panel_cfg.get("shadow_experimental_actionable_display",
                          _DEFAULT_EXPERIMENTAL_ACTIONABLE_DISPLAY))

        shadow_log_mlflow = bool(panel_cfg.get("shadow_log_mlflow", True))
        exp_id = None
        if shadow_log_mlflow:
            try:
                exp_id = _ensure_mlflow_setup(
                    panel_cfg.get("shadow_tracking_uri"),
                    panel_cfg.get("shadow_experiment"))
            except Exception as exc:
                log.warning("ApplyShadowScoringTask: MLflow setup failed: %s — skip",
                             exc)
                return None

        from kernel.panel_pipeline.model_registry import registry  # noqa: PLC0415
        repo = Path(__file__).resolve().parents[4]

        for sm in shadow_models:
            name = sm.get("name", "unnamed_shadow")
            kind = sm.get("kind")
            artifact_path = sm.get("artifact_path")
            if not kind or not artifact_path:
                log.warning("ApplyShadowScoringTask: shadow %s missing "
                             "kind/artifact_path", name)
                continue
            p = Path(artifact_path)
            if not p.is_absolute():
                p = repo / p
            try:
                handler = registry.get(kind)
            except ValueError as exc:
                log.warning("ApplyShadowScoringTask: %s", exc)
                continue

            # Inject shadow's feature_cols + seq_len + regime_router into config copy
            shadow_panel_cfg = dict(panel_cfg)
            if "feature_cols" in sm:
                shadow_panel_cfg["feature_cols"] = sm["feature_cols"]
            if "seq_len" in sm:
                shadow_panel_cfg["seq_len"] = sm["seq_len"]
            if "regime_router" in sm:  # composite scorer sub-config
                shadow_panel_cfg["regime_router"] = sm["regime_router"]
            shadow_cfg = dict(ctx.config)
            shadow_cfg.setdefault("ranking", {})["panel_scoring"] = shadow_panel_cfg

            cache_key = (kind, str(p))
            scorer = _SCORER_CACHE.get(cache_key)
            if scorer is None:
                try:
                    scorer = handler.scorer_loader(p, shadow_cfg)
                except Exception as exc:
                    log.warning("ApplyShadowScoringTask: shadow %s (%s) load failed: %s",
                                 name, kind, exc)
                    continue
                _SCORER_CACHE[cache_key] = scorer

            # 2026-07-01 round 2: scorer.metadata carries trained_date /
            # effective_train_cutoff_date / artifact_fingerprint, stamped by
            # panel_scorer.stamp_artifact_metadata at load time for every
            # registered scorer kind — feeds _compute_admission below.
            artifact_meta = getattr(scorer, "metadata", {}) or {}

            target_tickers = list(primary_scores.keys())
            try:
                if getattr(scorer, "requires_history", False):
                    panel_history = getattr(ctx, "_panel_history", None)
                    if panel_history is None:
                        panel_parquet = (repo / "data"
                                          / "alpha158_291_fundamental_dataset.parquet")
                        full = pd.read_parquet(panel_parquet)
                        full["date"] = pd.to_datetime(full["date"])
                        today_ts = pd.Timestamp(getattr(ctx, "today",
                                                          datetime.date.today()))
                        past = full[full["date"] < today_ts]
                        dates = sorted(past["date"].unique())[-scorer.seq_len:]
                        panel_history = past[past["date"].isin(dates)]
                    # If scorer accepts current_regime (RegimeRouterScorer), pass it
                    import inspect as _inspect  # noqa: PLC0415
                    sig = _inspect.signature(scorer.score_with_history)
                    if "current_regime" in sig.parameters:
                        series = scorer.score_with_history(
                            panel_history, target_tickers,
                            current_regime=getattr(ctx, "regime", "BULL_CALM"))
                    else:
                        series = scorer.score_with_history(
                            panel_history, target_tickers)
                else:
                    X = getattr(ctx, "_panel_matrix", None)
                    if X is None or X.empty:
                        log.warning("ApplyShadowScoringTask: shadow %s needs "
                                     "matrix but ctx._panel_matrix empty", name)
                        continue
                    fc = scorer.feature_cols
                    missing = [c for c in fc if c not in X.columns]
                    if missing:
                        log.warning("ApplyShadowScoringTask: shadow %s missing "
                                     "cols: %s", name, missing[:5])
                        continue
                    series = scorer.score(X[fc].fillna(0))
            except Exception as exc:
                log.warning("ApplyShadowScoringTask: shadow %s score failed: %s",
                             name, exc)
                continue

            shadow_dict = series.to_dict()
            sorted_shadow = sorted(shadow_dict.items(), key=lambda x: -x[1])
            shadow_ranks = {t: i + 1 for i, (t, _) in enumerate(sorted_shadow)}

            # 2026-05-19 (user mandate "want to know what shadow will do in
            # ntfy"): stash a compact summary on ctx so live.runner can
            # surface it — see _compute_shadow_summary docstring for the
            # 2026-07-01 top-N recommendation extension. REUSES the exact
            # same primary_scores/sorted_primary/primary_ranks and
            # shadow_dict/sorted_shadow/shadow_ranks arrays already built
            # above — no re-scoring, no second pass.
            try:
                summary = _compute_shadow_summary(
                    name, kind,
                    primary_scores, sorted_primary, primary_ranks,
                    shadow_dict, sorted_shadow, shadow_ranks,
                    top_n_picks,
                    as_of_date=as_of_date,
                    artifact_meta=artifact_meta,
                    n_expected_universe=n_expected_universe,
                    min_coverage=min_coverage,
                    experimental_actionable_display=experimental_actionable_display,
                )
                if not summary["actionable"]:
                    log.warning(
                        "ApplyShadowScoringTask: shadow %s top_picks NOT "
                        "ACTIONABLE (%s)", name,
                        "; ".join(summary["admission"]["reasons"]) or "n/a")
                if not hasattr(ctx, "_shadow_summary"):
                    ctx._shadow_summary = []  # noqa: SLF001
                ctx._shadow_summary.append(summary)  # noqa: SLF001
            except Exception as exc:
                log.warning("ApplyShadowScoringTask: ctx summary failed for %s: %s",
                             name, exc)

            if not shadow_log_mlflow or exp_id is None:
                log.info("ApplyShadowScoringTask: shadow %s (%s) scored %d "
                         "candidates (MLflow disabled)",
                         name, kind, len(shadow_dict))
                continue

            try:
                _log_shadow_run(
                    exp_id, getattr(ctx, "today", datetime.date.today()),
                    name, kind, primary_kind,
                    primary_scores, shadow_dict,
                    primary_ranks, shadow_ranks,
                )
                log.info("ApplyShadowScoringTask: shadow %s (%s) logged %d "
                         "candidates via MLflow", name, kind, len(shadow_dict))
            except Exception as exc:
                log.warning("ApplyShadowScoringTask: MLflow log failed for %s: %s",
                             name, exc)

        return None


__all__ = ["ApplyShadowScoringTask"]
