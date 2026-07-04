"""Fail-closed acceptance for per-ticker tournament model writes (campaign A2).

Context — finding F-17 (umbrella compliance audit, RQ PR #444; campaign plan
orchestrator PR #297, fix A2): the Sunday weekly tournament
(``scripts/weekly_tournament_retrain.sh`` → ``train_104.py --skip-panel
--force``) trains the per-ticker buy-admission models (RF / Q-learning /
per-ticker XGB / Manual) and wrote them STRAIGHT to production
``models/<TICKER>/``. ``train_104.py`` auto-disables ``ModelAcceptanceGate``
under ``--skip-panel`` (commit ba44e58, 2026-05-22 "Harden alpha158 staged
retraining gates") because that gate evaluates a *panel* candidate artifact
(staging panel-ltr.json vs active) — with ``--skip-panel`` no panel candidate
exists, so the panel gate is structurally inapplicable to per-ticker
tournament output. Consequence: a broken retrain (degenerate scores,
frozen/regressed training data, metric collapse) silently replaced every
good per-ticker model, and a mid-run crash could leave a ticker dir
partially written.

This module is the tournament-shaped acceptance the panel gate could not
provide: a minimal, fail-closed, per-ticker verdict evaluated BEFORE anything
touches ``models/<TICKER>/``, plus a staging-then-swap write pattern (see
``kernel.pipeline.pp_training._run_gated_export``) so a failing candidate or
a mid-write crash can never partially clobber the incumbent model.

PROTECTION CONTRACT (operator-pinned, P0): a HEALTHY tournament run must
produce byte-identical ``models/<TICKER>/`` contents vs the pre-gate code.
Only the FAILURE path changes: bad candidates are rejected, the incumbent
model is kept, and the rejection is loud (log + aggregated ntfy WARN from
train_104.py + per-ticker verdict archive). Enforced by the A/B test in
``tests/test_tournament_acceptance.py``.

Gates — all hard, all read from what the tournament already computes
(``run_tournament`` result dict + the ticker feature frame + the incumbent's
on-disk policy metadata); no new model evaluation is introduced:

  T1_model_present    a trained model object exists (belt-and-suspenders —
                      the chain already skips export when it is None)
  T2_sample_size      train_rows / oos_rows floors (mirrors the tournament's
                      own insufficient-data guard)
  T3_nondegenerate    OOS raw scores: enough finite values and not constant
                      (constant/NaN score output = broken model, the exact
                      failure the load-time sharpe floor cannot see)
  T4_data_cutoff      candidate feature-data end must not be in the future,
                      not be absurdly stale, and must NEVER regress vs the
                      incumbent's ``live_train_end`` (same non-regression
                      policy as scripts/tournament_retrain_marker.py)
  T5_metric_collapse  candidate OOS Sharpe must not mechanically collapse vs
                      the incumbent: reject only when it is BOTH below an
                      absolute floor AND far below the incumbent. Honest
                      degradation still ships — a worse fresh Sharpe is
                      admission INFORMATION consumed by LoadUniverseJob's
                      ``ranking.universe_floor``; rejecting it would freeze
                      stale good-looking metadata into the admission path.

Comparison gates (T4 regression leg, T5) skip-pass when there is no readable
incumbent: with nothing to protect, rejection would only starve coverage.
Absolute legs (future date, staleness, degeneracy) always apply.

Config — ``strategy_config.json``::

    "acceptance": {
        "enabled": true,              # existing panel-gate master switch
        "tournament": {
            "enabled": true,          # default: follows acceptance.enabled,
                                      # else true (fail-closed by default)
            "min_train_rows": 60,
            "min_oos_rows": 30,
            "min_oos_scores": 10,
            "max_data_staleness_days": 45,
            "collapse_sharpe_floor": -1.0,
            "collapse_min_drop": 2.0
        }
    }

``train_104.py --skip-acceptance`` (operator override, DANGEROUS) disables
this gate for one run, matching its existing panel semantics.

Reuses GateResult / AcceptanceVerdict from ``kernel.model_acceptance`` —
single-impl-imports-only, no hand-copied result types.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from kernel.model_acceptance import AcceptanceVerdict, GateResult

log = logging.getLogger("kernel.tournament_acceptance")


# ── Config ────────────────────────────────────────────────────────────────────

_DEFAULTS: dict[str, float | int] = {
    "min_train_rows": 60,
    "min_oos_rows": 30,
    "min_oos_scores": 10,
    # Feature-data end lags sample_end by ~lookahead trading days (label
    # dropna) and sample_end is bumped manually (last: 2026-06-09 →
    # "2026-06-30"), so healthy runs sit ~7-30 days behind the wall clock.
    # 45d matches the P-FUND-FRESHNESS convention and still catches the
    # 61-day silent-ageing incident class (2026-06-30) with margin.
    "max_data_staleness_days": 45,
    # Weekly 90d-OOS per-ticker Sharpe is noisy (healthy refreshes swing by
    # whole units) — reject only the mechanical-collapse pattern: below the
    # absolute floor AND far below the incumbent, both at once.
    "collapse_sharpe_floor": -1.0,
    "collapse_min_drop": 2.0,
}


def tournament_acceptance_config(config: dict[str, Any] | None) -> dict[str, Any] | None:
    """Resolve the tournament-acceptance config block; None when disabled.

    Default is ENABLED (fail-closed): ``acceptance.tournament.enabled``
    falls back to ``acceptance.enabled`` (the existing master switch),
    which itself defaults to True.
    """
    acc = (config or {}).get("acceptance") or {}
    tourn = acc.get("tournament") or {}
    enabled_default = bool(acc.get("enabled", True))
    if not bool(tourn.get("enabled", enabled_default)):
        return None
    resolved: dict[str, Any] = dict(_DEFAULTS)
    for key, default in _DEFAULTS.items():
        if key in tourn:
            try:
                resolved[key] = type(default)(tourn[key])
            except (TypeError, ValueError):
                log.warning(
                    "tournament_acceptance: malformed config %s=%r — using default %r",
                    key, tourn[key], default,
                )
    return resolved


def load_incumbent_metadata(models_dir: Path, ticker: str) -> dict[str, Any] | None:
    """Read the incumbent's policy metadata; None when absent/unreadable.

    Unreadable incumbents degrade the COMPARISON gates to their no-prior
    (skip-pass) semantics — the absolute gates still apply.
    """
    meta_path = Path(models_dir) / ticker / f"{ticker}-policy-metadata.json"
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("tournament_acceptance: cannot read incumbent metadata %s: %s",
                    meta_path, exc)
        return None
    return payload if isinstance(payload, dict) else None


# ── Gates ─────────────────────────────────────────────────────────────────────

def _gate_t1_model_present(result: dict[str, Any]) -> GateResult:
    present = result.get("model") is not None
    return GateResult(
        "T1_model_present", "hard", present, None, None,
        "trained model present" if present else "tournament produced no model object",
    )


def _gate_t2_sample_size(result: dict[str, Any], cfg: dict[str, Any]) -> GateResult:
    train_rows = int(result.get("train_rows") or 0)
    oos_rows = int(result.get("oos_rows") or 0)
    min_train = int(cfg["min_train_rows"])
    min_oos = int(cfg["min_oos_rows"])
    passed = train_rows >= min_train and oos_rows >= min_oos
    return GateResult(
        "T2_sample_size", "hard", passed, float(min(train_rows, oos_rows)), None,
        f"train_rows={train_rows} (min {min_train}), oos_rows={oos_rows} (min {min_oos})",
    )


def _gate_t3_nondegenerate_scores(result: dict[str, Any], cfg: dict[str, Any]) -> GateResult:
    import numpy as np  # noqa: PLC0415

    scores = result.get("oos_raw_scores")
    min_scores = int(cfg["min_oos_scores"])
    if scores is None:
        return GateResult("T3_nondegenerate", "hard", False, None, float(min_scores),
                          "no oos_raw_scores on tournament result — cannot verify "
                          "candidate output (fail closed)")
    try:
        values = np.asarray(scores, dtype=float)
    except (TypeError, ValueError):
        return GateResult("T3_nondegenerate", "hard", False, None, float(min_scores),
                          f"oos_raw_scores not numeric ({type(scores).__name__})")
    finite = values[np.isfinite(values)]
    n_finite = int(finite.size)
    n_unique = int(np.unique(finite).size)
    if n_finite < min_scores:
        return GateResult("T3_nondegenerate", "hard", False, float(n_finite),
                          float(min_scores),
                          f"only {n_finite} finite OOS scores (min {min_scores})")
    if n_unique < 2:
        return GateResult("T3_nondegenerate", "hard", False, float(n_unique), 2.0,
                          f"OOS scores are constant/degenerate "
                          f"(n_finite={n_finite}, n_unique={n_unique})")
    return GateResult("T3_nondegenerate", "hard", True, float(n_unique), 2.0,
                      f"n_finite={n_finite}, n_unique={n_unique}")


def _gate_t4_data_cutoff(
    feature_frame: Any,
    incumbent_meta: dict[str, Any] | None,
    cfg: dict[str, Any],
    today: Any,
) -> GateResult:
    import pandas as pd  # noqa: PLC0415

    max_stale = int(cfg["max_data_staleness_days"])
    try:
        candidate_end = pd.Timestamp(feature_frame.index.max()).normalize()
    except Exception as exc:  # noqa: BLE001 — unverifiable data end = fail closed
        return GateResult("T4_data_cutoff", "hard", False, None, None,
                          f"cannot derive candidate data end from feature frame: {exc}")
    if pd.isna(candidate_end):
        return GateResult("T4_data_cutoff", "hard", False, None, None,
                          "candidate feature frame has no valid dates")
    today_ts = pd.Timestamp(today).normalize()
    if candidate_end > today_ts + pd.Timedelta(days=1):
        return GateResult("T4_data_cutoff", "hard", False, None, None,
                          f"candidate data end {candidate_end.date()} is in the "
                          f"future (today {today_ts.date()})")
    age_days = int((today_ts - candidate_end).days)
    if age_days > max_stale:
        return GateResult("T4_data_cutoff", "hard", False, float(age_days),
                          float(max_stale),
                          f"candidate data end {candidate_end.date()} is {age_days}d "
                          f"stale (max {max_stale}d) — frozen data feed?")
    incumbent_end = None
    if incumbent_meta:
        raw = incumbent_meta.get("live_train_end")
        if raw:
            try:
                incumbent_end = pd.Timestamp(str(raw)).normalize()
            except (TypeError, ValueError):
                incumbent_end = None  # unreadable prior → skip regression leg
    if incumbent_end is not None and candidate_end < incumbent_end:
        return GateResult("T4_data_cutoff", "hard", False, float(age_days),
                          float(max_stale),
                          f"candidate data end {candidate_end.date()} REGRESSES vs "
                          f"incumbent live_train_end {incumbent_end.date()}")
    detail = f"candidate data end {candidate_end.date()} (age {age_days}d)"
    if incumbent_end is not None:
        detail += f", incumbent {incumbent_end.date()} — non-regressing"
    else:
        detail += ", no incumbent cutoff to compare"
    return GateResult("T4_data_cutoff", "hard", True, float(age_days),
                      float(max_stale), detail)


def _gate_t5_metric_collapse(
    result: dict[str, Any],
    incumbent_meta: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> GateResult:
    floor = float(cfg["collapse_sharpe_floor"])
    min_drop = float(cfg["collapse_min_drop"])
    candidate = result.get("sharpe")
    if candidate is None:
        return GateResult("T5_metric_collapse", "hard", False, None, floor,
                          "tournament result has no sharpe — cannot verify (fail closed)")
    candidate = float(candidate)
    incumbent = None
    if incumbent_meta:
        raw = incumbent_meta.get("sharpe")
        if isinstance(raw, (int, float)):
            incumbent = float(raw)
    if incumbent is None:
        return GateResult("T5_metric_collapse", "hard", True, candidate, floor,
                          f"no incumbent sharpe to compare (candidate={candidate:+.3f}) — skip")
    drop = incumbent - candidate
    collapsed = candidate < floor and drop >= min_drop
    return GateResult(
        "T5_metric_collapse", "hard", not collapsed, candidate, floor,
        f"candidate={candidate:+.3f} incumbent={incumbent:+.3f} drop={drop:+.3f} "
        f"(floor={floor:+.2f}, min_drop={min_drop:.2f})",
    )


def evaluate_tournament_candidate(
    ticker: str,
    result: dict[str, Any],
    feature_frame: Any,
    incumbent_meta: dict[str, Any] | None,
    cfg: dict[str, Any],
    today: Any = None,
) -> AcceptanceVerdict:
    """Run all tournament gates for one ticker's candidate; pure (no writes)."""
    import pandas as pd  # noqa: PLC0415

    if today is None:
        today = pd.Timestamp.today().normalize()
    results: list[GateResult] = [
        _gate_t1_model_present(result),
        _gate_t2_sample_size(result, cfg),
        _gate_t3_nondegenerate_scores(result, cfg),
        _gate_t4_data_cutoff(feature_frame, incumbent_meta, cfg, today),
        _gate_t5_metric_collapse(result, incumbent_meta, cfg),
    ]
    all_hard_passed = all(r.passed for r in results if r.severity == "hard")
    verdict = AcceptanceVerdict(all_hard_passed=all_hard_passed, results=results)
    if not all_hard_passed:
        log.warning("tournament_acceptance[%s]: REJECT\n%s", ticker, verdict.summary())
    return verdict


# ── Staging promote / rejection archive ──────────────────────────────────────

def promote_staged_ticker_dir(staged_dir: Path, live_dir: Path, ticker: str) -> list[str]:
    """Atomically (per file) move a fully-staged ticker bundle into production.

    * Non-metadata artifacts first, metadata LAST — the metadata file is the
      commit point every consumer (LoadUniverseJob, model TTL check,
      tournament_retrain_marker) keys on.
    * Each move is a same-filesystem ``os.replace`` (staging lives under
      ``models/.staging/``), so a reader sees the prior or the new file,
      never a partial one.
    * RF / Q-learning / Manual ``save()`` embed ``str(directory / artifact)``
      — an ABSOLUTE path — in ``metadata["artifacts"]``. Those values are
      rewritten from the staging prefix to the live prefix so the promoted
      bytes are identical to what the pre-gate in-place write produced
      (loaders resolve artifacts relative to the model dir, but the stamped
      path is part of the byte-invariance contract).
    * Files already in ``live_dir`` that the staged bundle does not contain
      (e.g. a previous winner's artifacts of another model type) are left
      untouched — exactly as the pre-gate in-place write left them.
    """
    staged_dir = Path(staged_dir)
    live_dir = Path(live_dir)
    live_dir.mkdir(parents=True, exist_ok=True)
    meta_name = f"{ticker}-policy-metadata.json"
    promoted: list[str] = []
    for path in sorted(p for p in staged_dir.iterdir() if p.is_file()):
        if path.name == meta_name:
            continue
        os.replace(str(path), str(live_dir / path.name))
        promoted.append(path.name)
    meta_path = staged_dir / meta_name
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        artifacts = meta.get("artifacts")
        if isinstance(artifacts, dict):
            staged_prefix = str(staged_dir)
            live_prefix = str(live_dir)
            for key, value in artifacts.items():
                if isinstance(value, str) and value.startswith(staged_prefix):
                    artifacts[key] = live_prefix + value[len(staged_prefix):]
        rewritten = staged_dir / (meta_name + ".promote")
        rewritten.write_text(json.dumps(meta, indent=2))
        os.replace(str(rewritten), str(live_dir / meta_name))
        meta_path.unlink(missing_ok=True)
        promoted.append(meta_name)
    return promoted


def archive_rejection(strategy_dir: Path, ticker: str, verdict: AcceptanceVerdict) -> Path:
    """Persist a per-ticker rejection verdict for forensics (text, not bytes)."""
    log_dir = Path(strategy_dir) / "artifacts" / "_tournament_acceptance_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = verdict.timestamp.strftime("%Y-%m-%dT%H%M%S")
    out = log_dir / f"{ts}_REJECTED_{ticker}.txt"
    out.write_text(verdict.summary())
    return out


__all__ = [
    "tournament_acceptance_config",
    "load_incumbent_metadata",
    "evaluate_tournament_candidate",
    "promote_staged_ticker_dir",
    "archive_rejection",
]
