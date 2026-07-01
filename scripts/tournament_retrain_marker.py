#!/usr/bin/env python
"""Artifact-derived completion evidence for the weekly per-ticker tournament retrain.

Why this module exists (Codex review on PR #420, 2026-06-30)
------------------------------------------------------------
The first cut of ``scripts/weekly_tournament_retrain.sh`` stamped a
"last successful tournament retrain" marker by:

  * counting *pre-existing* ``models/*`` directories (``ls -d models/*/``),
  * stamping ``trained_date`` with **today's wall clock**.

Both signals are process-derived, not artifact-derived, so the marker could
certify a **partial or no-op** retrain as a globally fresh population refresh:

  * ``ls`` counts stale orphan directories from prior watchlists (on the live
    tree, 97 of 230 ``models/*`` dirs were NOT in the current 142-name
    watchlist), so the count is satisfied even if nothing was retrained.
  * ``trained_date = today`` advances regardless of whether any per-ticker
    artifact actually changed or its effective data cutoff moved.

This module makes completion evidence **artifact-derived and per ticker**:

  1. The caller FREEZES the expected watchlist *before* launching training and
     captures a ``launch_epoch`` (seconds).
  2. For each expected ticker we open its ``<T>-policy-metadata.json`` and
     require the file was **REWRITTEN this invocation** — its mtime must be
     ``>= launch_epoch``. A pre-existing dir that training did not touch keeps
     an older mtime and is classified ``stale`` (never counted as fresh).
  3. For each rewritten ticker we record its **effective data cutoff**
     (``live_train_end`` — the last training bar — falling back to
     ``trained_date`` with the source recorded) and an **artifact digest**
     (sha256 of the metadata bytes).
  4. Certification requires a **pre-registered coverage policy**: the fraction
     of expected tickers freshly rewritten must meet ``min_coverage`` AND there
     must be **zero** ``stale`` (pre-existing dir masquerading as fresh) AND
     zero rewritten-but-unparseable artifacts. Otherwise the job is NOT
     certified and no fresh marker is stamped.
  5. The marker reports **min/max data cutoff** and an **explicit partial
     status** — freshness is the artifact data cutoff, never the wall clock.

Scope: this certifies **cadence coverage only** — that the scheduled retrain
actually rewrote the population. It says NOTHING about model quality or
promotion; OOS/shadow/WF evaluation against the pinned incumbent stays owned by
``weekly_wf_promote.sh``. Keeping cadence-completion separate from
model-quality is deliberate (Codex review, same thread).

The heavy lifting lives in pure, side-effect-light functions
(:func:`evaluate_ticker`, :func:`build_marker_evidence`) so the marker logic is
unit-testable without executing the bash wrapper — see
``tests/test_tournament_retrain_marker.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

META_SUFFIX = "-policy-metadata.json"
SCHEMA = "tournament_retrain_marker/v1"

# States a per-ticker artifact can be in relative to this invocation.
STATE_SUCCEEDED = "succeeded"      # rewritten this run, parseable, has a data cutoff
STATE_STALE = "stale"              # artifact exists but was NOT rewritten this run
STATE_MISSING = "missing"          # no artifact on disk at all (never trained / new name)
STATE_UNPARSEABLE = "unparseable"  # rewritten this run but no readable cutoff


@dataclass
class TickerEvidence:
    ticker: str
    state: str
    metadata_path: str | None = None
    mtime: float | None = None
    rewritten: bool = False
    data_cutoff: str | None = None
    cutoff_source: str | None = None  # "live_train_end" | "trained_date" | None
    digest: str | None = None         # sha256 of the metadata bytes
    reason: str | None = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate_ticker(models_dir: Path, ticker: str, launch_epoch: float) -> TickerEvidence:
    """Classify one expected ticker's per-ticker artifact against ``launch_epoch``.

    ``launch_epoch`` is the wall-clock second captured *immediately before*
    training was launched. Any artifact training rewrote will have an
    ``mtime >= launch_epoch``; anything older was not touched this run.
    """
    meta = models_dir / ticker / f"{ticker}{META_SUFFIX}"
    if not meta.exists():
        return TickerEvidence(
            ticker,
            STATE_MISSING,
            reason="no policy-metadata artifact on disk",
        )

    mtime = meta.stat().st_mtime
    if mtime < launch_epoch:
        return TickerEvidence(
            ticker,
            STATE_STALE,
            metadata_path=str(meta),
            mtime=mtime,
            rewritten=False,
            reason=(
                f"metadata mtime {mtime:.3f} predates launch {launch_epoch:.3f} "
                "— pre-existing dir, NOT rewritten this invocation"
            ),
        )

    try:
        payload = json.loads(meta.read_text())
    except (ValueError, OSError) as exc:
        return TickerEvidence(
            ticker,
            STATE_UNPARSEABLE,
            metadata_path=str(meta),
            mtime=mtime,
            rewritten=True,
            reason=f"metadata rewritten but unreadable: {exc}",
        )

    cutoff = payload.get("live_train_end")
    source: str | None = "live_train_end"
    if not cutoff:
        cutoff = payload.get("trained_date")
        source = "trained_date" if cutoff else None
    if not cutoff:
        return TickerEvidence(
            ticker,
            STATE_UNPARSEABLE,
            metadata_path=str(meta),
            mtime=mtime,
            rewritten=True,
            reason="metadata rewritten but has no live_train_end / trained_date",
        )

    return TickerEvidence(
        ticker,
        STATE_SUCCEEDED,
        metadata_path=str(meta),
        mtime=mtime,
        rewritten=True,
        data_cutoff=str(cutoff),
        cutoff_source=source,
        digest=_sha256(meta),
    )


def build_marker_evidence(
    models_dir: str | Path,
    expected_tickers: Iterable[str],
    launch_epoch: float,
    *,
    min_coverage: float = 1.0,
) -> dict:
    """Build artifact-derived completion evidence for the frozen expected set.

    Certification (``certified=True``) requires ALL of:

      * coverage (fresh-succeeded / expected) ``>= min_coverage``;
      * **zero** ``stale`` artifacts — a pre-existing dir that was not rewritten
        this invocation must never be certified as fresh (the exact bug Codex
        flagged);
      * **zero** rewritten-but-unparseable artifacts.

    ``missing`` (no artifact on disk — an ETF/benchmark the tournament does not
    train, or a newly-added watchlist name not yet trained) is tolerated only
    insofar as the coverage floor is still met; it can never masquerade as
    fresh because it contributes no cutoff and no digest.
    """
    models_dir = Path(models_dir)
    expected = sorted({str(t).strip() for t in expected_tickers if str(t).strip()})
    if not expected:
        raise ValueError(
            "expected_tickers is empty — refuse to certify against an empty universe"
        )
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError(f"min_coverage must be in (0, 1]; got {min_coverage!r}")

    per = [evaluate_ticker(models_dir, t, launch_epoch) for t in expected]

    succeeded = sorted(e.ticker for e in per if e.state == STATE_SUCCEEDED)
    stale = sorted(e.ticker for e in per if e.state == STATE_STALE)
    missing = sorted(e.ticker for e in per if e.state == STATE_MISSING)
    unparseable = sorted(e.ticker for e in per if e.state == STATE_UNPARSEABLE)
    failed = sorted(stale + unparseable)  # artifacts present but not fresh-valid

    coverage = len(succeeded) / len(expected)
    cutoffs = sorted(e.data_cutoff for e in per if e.state == STATE_SUCCEEDED and e.data_cutoff)
    min_cutoff = cutoffs[0] if cutoffs else None
    max_cutoff = cutoffs[-1] if cutoffs else None

    policy_met = coverage >= min_coverage
    no_stale_masquerade = not stale
    no_unparseable = not unparseable
    certified = policy_met and no_stale_masquerade and no_unparseable

    if not certified:
        status = "failed"
    elif coverage >= 1.0:
        status = "success"
    else:
        status = "partial"

    return {
        "schema": SCHEMA,
        "scope": "cadence_completion_only",  # NOT model quality / promotion
        "certified": certified,
        "status": status,
        "coverage_policy": {"min_coverage": min_coverage},
        "coverage": round(coverage, 6),
        "policy_met": policy_met,
        "no_stale_masquerade": no_stale_masquerade,
        "no_unparseable": no_unparseable,
        "launch_epoch": launch_epoch,
        "expected_count": len(expected),
        "succeeded_count": len(succeeded),
        "stale_count": len(stale),
        "missing_count": len(missing),
        "unparseable_count": len(unparseable),
        "min_data_cutoff": min_cutoff,
        "max_data_cutoff": max_cutoff,
        "sets": {
            "attempted": expected,
            "succeeded": succeeded,
            "failed": failed,
            "stale": stale,
            "missing": missing,
            "unparseable": unparseable,
        },
        "per_ticker": {
            e.ticker: {k: v for k, v in asdict(e).items() if k != "ticker"}
            for e in per
        },
    }


def _load_watchlist(path: Path) -> list[str]:
    """Load the frozen expected watchlist. Accepts a JSON list or ``{"watchlist": [...]}``."""
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("watchlist", [])
    if not isinstance(data, list):
        raise ValueError(f"watchlist file {path} did not contain a JSON list")
    return [str(t) for t in data]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stamp artifact-derived per-ticker tournament retrain completion marker."
    )
    p.add_argument("--models-dir", required=True,
                   help="backtesting/renquant_104/models directory")
    p.add_argument("--watchlist", required=True,
                   help="path to the frozen expected-watchlist JSON (list or {'watchlist': [...]})")
    p.add_argument("--launch-epoch", required=True, type=float,
                   help="epoch seconds captured immediately BEFORE training launched")
    p.add_argument("--run-id", required=True, help="unique id for this invocation")
    p.add_argument("--marker", required=True, help="output marker path (written only when certified)")
    p.add_argument("--min-coverage", type=float, default=1.0,
                   help="pre-registered coverage floor in (0, 1] (default 1.0)")
    p.add_argument("--exit-code", type=int, default=0, help="train_104.py exit code")
    p.add_argument("--command", default="scripts/train_104.py --skip-panel --force")
    p.add_argument("--host", default="unknown")
    p.add_argument("--log", default="")
    p.add_argument("--completed-at", default=None, help="UTC ISO-8601; default now")
    p.add_argument("--date", default=None, help="wall date YYYY-MM-DD; default today (informational)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    completed_at = args.completed_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    wall_date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    expected = _load_watchlist(Path(args.watchlist))
    try:
        evidence = build_marker_evidence(
            args.models_dir, expected, args.launch_epoch, min_coverage=args.min_coverage
        )
    except ValueError as exc:
        print(f"tournament_retrain_marker: REFUSING to certify — {exc}", file=sys.stderr)
        return 4

    payload = {
        "job": "weekly_tournament_retrain",
        **evidence,
        "run_id": args.run_id,
        "command": args.command,
        "exit_code": args.exit_code,
        "completed_at": completed_at,   # UTC ISO — when the job finished (informational)
        "host": args.host,
        "log": args.log,
        # Freshness the monitor consumes is ARTIFACT-DERIVED: the min effective
        # data cutoff (the tournament is only as fresh as its stalest ticker).
        # Keep the `trained_date` field name for monitor compatibility but bind
        # it to the cutoff, NOT the wall clock.
        "trained_date": evidence["min_data_cutoff"],
        "trained_date_source": "min_data_cutoff",
        "wall_clock_date": wall_date,   # informational only — NOT the freshness signal
    }

    # Emit the full evidence to stdout (captured in the job log) either way.
    print(json.dumps({k: payload[k] for k in (
        "status", "certified", "coverage", "coverage_policy",
        "expected_count", "succeeded_count", "stale_count", "missing_count",
        "unparseable_count", "min_data_cutoff", "max_data_cutoff",
    )}, indent=2))

    if not payload["certified"]:
        print(
            "tournament_retrain_marker: NOT CERTIFIED "
            f"(status={payload['status']} coverage={payload['coverage']:.3f} "
            f"min={args.min_coverage} stale={payload['stale_count']} "
            f"unparseable={payload['unparseable_count']} missing={payload['missing_count']}). "
            "Leaving prior marker untouched so its data cutoff keeps ageing.",
            file=sys.stderr,
        )
        if payload["stale_count"]:
            print(f"  stale (pre-existing, not rewritten): {payload['sets']['stale']}", file=sys.stderr)
        if payload["unparseable_count"]:
            print(f"  unparseable: {payload['sets']['unparseable']}", file=sys.stderr)
        return 3

    marker = Path(args.marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(
        f"tournament_retrain_marker: CERTIFIED status={payload['status']} "
        f"coverage={payload['coverage']:.3f} "
        f"cutoff=[{payload['min_data_cutoff']}..{payload['max_data_cutoff']}] "
        f"→ stamped {marker}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
