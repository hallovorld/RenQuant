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

Round 1 fix (this module's first cut) made completion evidence artifact- and
per-ticker-derived (mtime >= launch_epoch, per-ticker digest, coverage floor)
but Codex flagged four residual gaps in the SAME review thread (2026-06-30,
second pass) that round 2 (this version) closes:

  1. **No pre-run baseline.** ``mtime >= launch_epoch`` only proves a file was
     *touched*, not that training *completed*, changed model bytes, or moved
     the data cutoff — a no-op writer / ``cp -p`` restamp / failed run that
     rewrites identical bytes with a fresh mtime passed. Fix: the caller now
     captures a **pre-run baseline** (:func:`capture_baseline`, CLI
     ``--emit-baseline``) of each expected ticker's digest + data cutoff
     *before* launching training. Post-run, :func:`evaluate_ticker` requires
     the digest to have **changed** from that baseline, or an **explicit**
     ``no_change_reason`` in the rewritten payload — an unexplained
     byte-identical rewrite is ``unverified_no_change`` and blocks
     certification.
  2. **``trained_date`` fallback reintroduced the wall-clock spoof.** Removed.
     ``live_train_end`` is the only accepted data-cutoff field; a rewritten
     artifact without it is ``unparseable``, never certified.
  3. **``exit_code`` was recorded but not enforced.** Certification now
     **requires** ``exit_code == 0`` as an independent, always-checked gate —
     artifact freshness can never override a failed training process, even if
     the marker script is invoked directly (not just via the shell wrapper's
     control flow).
  4. **The 0.90 coverage floor was an unregistered magic number.** Replaced
     with an explicit ``non_trainable`` map (``ticker -> justification``) of
     intentionally non-trained names (benchmark / sector / defensive ETFs —
     see ``weekly_tournament_retrain.sh``, derived live from
     ``strategy_config.json`` so it can never silently drift out of sync).
     Certification now requires **100% coverage of the trainable set**
     (``expected - non_trainable``); every exclusion must carry a non-empty
     reason and must itself be a member of the frozen watchlist.

Scope: this certifies **cadence coverage only** — that the scheduled retrain
actually rewrote the trainable population with evidence the bytes moved (or
an explicitly justified no-op) and the data window did not regress. It says
NOTHING about model quality or promotion; OOS/shadow/WF evaluation against the
pinned incumbent stays owned by ``weekly_wf_promote.sh``. Keeping
cadence-completion separate from model-quality is deliberate (Codex review,
same thread).

The heavy lifting lives in pure, side-effect-light functions
(:func:`evaluate_ticker`, :func:`build_marker_evidence`,
:func:`capture_baseline`) so the marker logic is unit-testable without
executing the bash wrapper — see ``tests/test_tournament_retrain_marker.py``.
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
SCHEMA = "tournament_retrain_marker/v2"

# States a per-ticker artifact can be in relative to this invocation.
STATE_SUCCEEDED = "succeeded"                  # rewritten, parseable, cutoff present, identity proven
STATE_STALE = "stale"                          # artifact exists but was NOT rewritten this run
STATE_MISSING = "missing"                      # no artifact on disk at all (never trained / new name)
STATE_UNPARSEABLE = "unparseable"              # rewritten this run but no readable cutoff
STATE_CUTOFF_REGRESSED = "cutoff_regressed"    # rewritten, but data cutoff moved BACKWARD vs baseline
STATE_UNVERIFIED_NO_CHANGE = "unverified_no_change"  # rewritten, byte-identical to baseline, unexplained

# Terminal states that mean "artifact present but not certifiable fresh".
_BLOCKING_STATES = (
    STATE_STALE,
    STATE_MISSING,
    STATE_UNPARSEABLE,
    STATE_CUTOFF_REGRESSED,
    STATE_UNVERIFIED_NO_CHANGE,
)


@dataclass
class TickerEvidence:
    ticker: str
    state: str
    metadata_path: str | None = None
    mtime: float | None = None
    rewritten: bool = False
    data_cutoff: str | None = None
    cutoff_source: str | None = None  # "live_train_end" | None (no other source is accepted)
    digest: str | None = None         # sha256 of the metadata bytes, this invocation
    baseline_status: str | None = None  # "new" | "changed" | "unchanged_explicit"
    baseline_digest: str | None = None
    baseline_cutoff: str | None = None
    reason: str | None = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_cutoff(payload: dict) -> str | None:
    """Effective data cutoff. ``live_train_end`` ONLY — no ``trained_date``
    fallback (Codex review, PR #420 round 2): ``trained_date`` is a wall-clock
    stamp, not proof the training window advanced, and falling back to it
    reintroduces the exact freshness spoof this module exists to remove.
    """
    cutoff = payload.get("live_train_end")
    return str(cutoff) if cutoff else None


def capture_baseline(models_dir: str | Path, expected_tickers: Iterable[str]) -> dict[str, dict]:
    """Snapshot PRE-RUN per-ticker artifact identity for the frozen expected set.

    Must be called BEFORE training launches. Records each ticker's current
    on-disk digest + data cutoff (when a readable, cutoff-bearing artifact
    already exists) so :func:`evaluate_ticker` can prove — after training —
    that a rewritten artifact's bytes actually changed, rather than being a
    no-op copy/restamp with only a touched mtime.

    A ticker absent from the returned mapping (no artifact, unparseable, or no
    ``live_train_end``) has no baseline to compare against; ``evaluate_ticker``
    then treats any post-run rewrite as a first-ever ("new") training and does
    not require a digest change.
    """
    models_dir = Path(models_dir)
    snapshot: dict[str, dict] = {}
    for ticker in sorted({str(t).strip() for t in expected_tickers if str(t).strip()}):
        meta = models_dir / ticker / f"{ticker}{META_SUFFIX}"
        if not meta.exists():
            continue
        try:
            payload = json.loads(meta.read_text())
        except (ValueError, OSError):
            continue
        cutoff = _read_cutoff(payload)
        if not cutoff:
            continue
        snapshot[ticker] = {
            "digest": _sha256(meta),
            "data_cutoff": cutoff,
            "mtime": meta.stat().st_mtime,
        }
    return snapshot


def evaluate_ticker(
    models_dir: Path,
    ticker: str,
    launch_epoch: float,
    baseline_entry: dict | None = None,
) -> TickerEvidence:
    """Classify one expected ticker's per-ticker artifact against ``launch_epoch``
    and, when available, a pre-run ``baseline_entry`` (see
    :func:`capture_baseline`).

    ``launch_epoch`` is the wall-clock second captured *immediately before*
    training was launched. Any artifact training rewrote will have an
    ``mtime >= launch_epoch``; anything older was not touched this run.

    When ``baseline_entry`` is provided (pre-run digest + data cutoff for this
    ticker), a rewritten artifact must additionally prove **identity change**:

      * its data cutoff must not have moved BACKWARD vs the baseline
        (``cutoff_regressed`` otherwise);
      * its digest must differ from the baseline digest, OR the rewritten
        payload must carry an explicit, non-empty ``no_change_reason`` string
        justifying a byte-identical re-write (``unverified_no_change``
        otherwise — an unexplained no-op cannot certify).

    ``baseline_entry is None`` means no pre-run artifact existed to compare
    against (first-ever training for this ticker, or the caller did not
    capture a baseline) — the rewrite is accepted without an identity check.
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

    cutoff = _read_cutoff(payload)
    if not cutoff:
        return TickerEvidence(
            ticker,
            STATE_UNPARSEABLE,
            metadata_path=str(meta),
            mtime=mtime,
            rewritten=True,
            reason=(
                "metadata rewritten but has no live_train_end "
                "(trained_date fallback removed — Codex review #420)"
            ),
        )

    digest = _sha256(meta)
    baseline_digest = baseline_entry.get("digest") if baseline_entry else None
    baseline_cutoff = baseline_entry.get("data_cutoff") if baseline_entry else None

    if baseline_entry is None:
        return TickerEvidence(
            ticker,
            STATE_SUCCEEDED,
            metadata_path=str(meta),
            mtime=mtime,
            rewritten=True,
            data_cutoff=cutoff,
            cutoff_source="live_train_end",
            digest=digest,
            baseline_status="new",
        )

    if baseline_cutoff and cutoff < baseline_cutoff:
        return TickerEvidence(
            ticker,
            STATE_CUTOFF_REGRESSED,
            metadata_path=str(meta),
            mtime=mtime,
            rewritten=True,
            data_cutoff=cutoff,
            cutoff_source="live_train_end",
            digest=digest,
            baseline_status="regressed",
            baseline_digest=baseline_digest,
            baseline_cutoff=baseline_cutoff,
            reason=f"data cutoff regressed: post-run {cutoff} < pre-run baseline {baseline_cutoff}",
        )

    if digest == baseline_digest:
        no_change_reason = payload.get("no_change_reason")
        if no_change_reason:
            return TickerEvidence(
                ticker,
                STATE_SUCCEEDED,
                metadata_path=str(meta),
                mtime=mtime,
                rewritten=True,
                data_cutoff=cutoff,
                cutoff_source="live_train_end",
                digest=digest,
                baseline_status="unchanged_explicit",
                baseline_digest=baseline_digest,
                baseline_cutoff=baseline_cutoff,
                reason=f"digest unchanged from baseline; explicit no_change_reason={no_change_reason!r}",
            )
        return TickerEvidence(
            ticker,
            STATE_UNVERIFIED_NO_CHANGE,
            metadata_path=str(meta),
            mtime=mtime,
            rewritten=True,
            data_cutoff=cutoff,
            cutoff_source="live_train_end",
            digest=digest,
            baseline_status="unchanged_unverified",
            baseline_digest=baseline_digest,
            baseline_cutoff=baseline_cutoff,
            reason=(
                "post-run artifact is byte-identical to the pre-run baseline with no "
                "explicit no_change_reason — cannot prove training actually ran "
                "(Codex review #420)"
            ),
        )

    return TickerEvidence(
        ticker,
        STATE_SUCCEEDED,
        metadata_path=str(meta),
        mtime=mtime,
        rewritten=True,
        data_cutoff=cutoff,
        cutoff_source="live_train_end",
        digest=digest,
        baseline_status="changed",
        baseline_digest=baseline_digest,
        baseline_cutoff=baseline_cutoff,
    )


def build_marker_evidence(
    models_dir: str | Path,
    expected_tickers: Iterable[str],
    launch_epoch: float,
    *,
    exit_code: int,
    baseline: dict[str, dict] | None = None,
    non_trainable: dict[str, str] | None = None,
) -> dict:
    """Build artifact-derived completion evidence for the frozen expected set.

    Certification (``certified=True``) requires ALL of:

      * ``exit_code == 0`` — the actual ``train_104.py`` exit status, checked
        HERE (not just by shell control flow) so artifact freshness can never
        override a failed training process (Codex review, PR #420 round 2);
      * **100% of the trainable set** (``expected - non_trainable``) is
        ``succeeded`` — rewritten this invocation, parseable, has a
        ``live_train_end`` cutoff, and (when a baseline entry exists) proves
        digest identity change or an explicit no-change justification, with a
        non-regressing cutoff. Zero tolerance for stale / missing /
        unparseable / cutoff-regressed / unverified-no-change among trainable
        tickers — this is the exact bug Codex flagged, now backed by evidence
        beyond mtime alone.

    ``non_trainable`` enumerates tickers the tournament is not expected to
    train (e.g. benchmark / sector / defensive ETFs) mapped to a
    justification string. Every key MUST be a member of ``expected_tickers``
    (no silent scope expansion) and MUST carry a non-empty reason. Excluded
    tickers are informational only — being ``missing`` never blocks
    certification, but being freshly ``succeeded`` is recorded too.
    """
    models_dir = Path(models_dir)
    expected = sorted({str(t).strip() for t in expected_tickers if str(t).strip()})
    if not expected:
        raise ValueError(
            "expected_tickers is empty — refuse to certify against an empty universe"
        )

    non_trainable = dict(non_trainable or {})
    unknown_exclusions = sorted(set(non_trainable) - set(expected))
    if unknown_exclusions:
        raise ValueError(
            "non_trainable exclusions not present in expected_tickers "
            f"(remove them or add them to the watchlist): {unknown_exclusions}"
        )
    unjustified = sorted(t for t, reason in non_trainable.items() if not str(reason or "").strip())
    if unjustified:
        raise ValueError(
            f"non_trainable exclusions require a non-empty justification reason: {unjustified}"
        )

    trainable = sorted(set(expected) - set(non_trainable))
    excluded = sorted(non_trainable)
    if not trainable:
        raise ValueError(
            "non_trainable excludes the entire expected watchlist — nothing left to certify"
        )

    baseline = baseline or {}
    per = [
        evaluate_ticker(models_dir, t, launch_epoch, baseline.get(t))
        for t in expected
    ]
    by_ticker = {e.ticker: e for e in per}

    def _bucket(tickers: list[str], state: str) -> list[str]:
        return sorted(t for t in tickers if by_ticker[t].state == state)

    trainable_succeeded = _bucket(trainable, STATE_SUCCEEDED)
    trainable_stale = _bucket(trainable, STATE_STALE)
    trainable_missing = _bucket(trainable, STATE_MISSING)
    trainable_unparseable = _bucket(trainable, STATE_UNPARSEABLE)
    trainable_cutoff_regressed = _bucket(trainable, STATE_CUTOFF_REGRESSED)
    trainable_unverified_no_change = _bucket(trainable, STATE_UNVERIFIED_NO_CHANGE)
    trainable_blocking = sorted(
        t for t in trainable if by_ticker[t].state in _BLOCKING_STATES
    )

    excluded_succeeded = _bucket(excluded, STATE_SUCCEEDED)
    excluded_missing = _bucket(excluded, STATE_MISSING)
    excluded_other = sorted(set(excluded) - set(excluded_succeeded) - set(excluded_missing))

    trainable_coverage = len(trainable_succeeded) / len(trainable)

    exit_code_ok = int(exit_code) == 0
    trainable_fully_fresh = trainable_coverage >= 1.0 and not trainable_blocking
    certified = exit_code_ok and trainable_fully_fresh
    status = "success" if certified else "failed"

    cutoffs = sorted(e.data_cutoff for e in per if e.state == STATE_SUCCEEDED and e.data_cutoff)
    min_cutoff = cutoffs[0] if cutoffs else None
    max_cutoff = cutoffs[-1] if cutoffs else None

    return {
        "schema": SCHEMA,
        "scope": "cadence_completion_only",  # NOT model quality / promotion
        "certified": certified,
        "status": status,
        "exit_code": int(exit_code),
        "exit_code_ok": exit_code_ok,
        "coverage_policy": {
            "trainable_required_coverage": 1.0,
            "excluded_tickers": {t: non_trainable[t] for t in excluded},
        },
        "trainable_coverage": round(trainable_coverage, 6),
        "trainable_fully_fresh": trainable_fully_fresh,
        "expected_count": len(expected),
        "trainable_count": len(trainable),
        "excluded_count": len(excluded),
        "trainable_succeeded_count": len(trainable_succeeded),
        "trainable_blocking_count": len(trainable_blocking),
        "min_data_cutoff": min_cutoff,
        "max_data_cutoff": max_cutoff,
        "sets": {
            "attempted": expected,
            "trainable": trainable,
            "excluded": excluded,
            "trainable_succeeded": trainable_succeeded,
            "trainable_stale": trainable_stale,
            "trainable_missing": trainable_missing,
            "trainable_unparseable": trainable_unparseable,
            "trainable_cutoff_regressed": trainable_cutoff_regressed,
            "trainable_unverified_no_change": trainable_unverified_no_change,
            "trainable_blocking": trainable_blocking,
            "excluded_succeeded": excluded_succeeded,
            "excluded_missing": excluded_missing,
            "excluded_other": excluded_other,
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


def _load_non_trainable(path: Path | None) -> dict[str, str]:
    """Load ``{ticker: justification}`` for intentionally non-trained tickers.

    Accepts a JSON object (``{"SPY": "benchmark, not admitted"}``) or a list
    of ``{"ticker": ..., "reason": ...}`` objects. Missing path -> ``{}`` (no
    exclusions; every expected ticker is then required at 100%).
    """
    if path is None:
        return {}
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    if isinstance(data, list):
        out: dict[str, str] = {}
        for entry in data:
            out[str(entry["ticker"])] = str(entry.get("reason", ""))
        return out
    raise ValueError(f"non-trainable file {path} must be a JSON object or list")


def _load_baseline(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    return json.loads(path.read_text())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stamp artifact-derived per-ticker tournament retrain completion marker."
    )
    p.add_argument("--models-dir", required=True,
                   help="backtesting/renquant_104/models directory")
    p.add_argument("--watchlist", required=True,
                   help="path to the frozen expected-watchlist JSON (list or {'watchlist': [...]})")

    p.add_argument("--emit-baseline", default=None,
                   help="If set, write a PRE-RUN baseline snapshot (per-ticker digest + data "
                        "cutoff for --watchlist tickers currently on disk) to this path and exit "
                        "0. Must be called BEFORE launching training. Does not certify anything.")

    p.add_argument("--launch-epoch", type=float, default=None,
                   help="epoch seconds captured immediately BEFORE training launched "
                        "(required unless --emit-baseline is given)")
    p.add_argument("--run-id", default=None, help="unique id for this invocation "
                   "(required unless --emit-baseline is given)")
    p.add_argument("--marker", default=None,
                   help="output marker path, written only when certified "
                        "(required unless --emit-baseline is given)")
    p.add_argument("--exit-code", type=int, default=0,
                   help="train_104.py exit code — certification REQUIRES this to be 0")
    p.add_argument("--baseline", default=None,
                   help="path to a pre-run baseline snapshot (from --emit-baseline) used to "
                        "prove post-run artifact identity change / non-regression")
    p.add_argument("--non-trainable", default=None,
                   help="path to a JSON map/list of {ticker: justification} for intentionally "
                        "non-trained expected tickers (e.g. benchmark/sector ETFs). Every other "
                        "expected ticker is required at 100%% coverage.")

    p.add_argument("--command", default="scripts/train_104.py --skip-panel --force")
    p.add_argument("--host", default="unknown")
    p.add_argument("--log", default="")
    p.add_argument("--completed-at", default=None, help="UTC ISO-8601; default now")
    p.add_argument("--date", default=None, help="wall date YYYY-MM-DD; default today (informational)")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    expected = _load_watchlist(Path(args.watchlist))

    if args.emit_baseline:
        snapshot = capture_baseline(Path(args.models_dir), expected)
        out = Path(args.emit_baseline)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(snapshot, f, indent=2)
            f.write("\n")
        print(
            f"tournament_retrain_marker: wrote PRE-RUN baseline for {len(snapshot)}/{len(expected)} "
            f"tickers → {out}"
        )
        return 0

    missing_required = [
        flag for flag, val in (
            ("--launch-epoch", args.launch_epoch),
            ("--run-id", args.run_id),
            ("--marker", args.marker),
        ) if val is None
    ]
    if missing_required:
        parser.error(
            f"{', '.join(missing_required)} required unless --emit-baseline is given"
        )

    completed_at = args.completed_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    wall_date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    non_trainable = _load_non_trainable(Path(args.non_trainable) if args.non_trainable else None)
    baseline = _load_baseline(Path(args.baseline) if args.baseline else None)

    try:
        evidence = build_marker_evidence(
            args.models_dir,
            expected,
            args.launch_epoch,
            exit_code=args.exit_code,
            baseline=baseline,
            non_trainable=non_trainable,
        )
    except ValueError as exc:
        print(f"tournament_retrain_marker: REFUSING to certify — {exc}", file=sys.stderr)
        return 4

    payload = {
        "job": "weekly_tournament_retrain",
        **evidence,
        "run_id": args.run_id,
        "command": args.command,
        "completed_at": completed_at,   # UTC ISO — when the job finished (informational)
        "host": args.host,
        "log": args.log,
        # Freshness the monitor consumes is ARTIFACT-DERIVED: the min effective
        # data cutoff (the tournament is only as fresh as its stalest trainable
        # ticker). Keep the `trained_date` field name for monitor compatibility
        # but bind it to the cutoff, NOT the wall clock.
        "trained_date": evidence["min_data_cutoff"],
        "trained_date_source": "min_data_cutoff",
        "wall_clock_date": wall_date,   # informational only — NOT the freshness signal
    }

    # Emit the full evidence to stdout (captured in the job log) either way.
    print(json.dumps({k: payload[k] for k in (
        "status", "certified", "exit_code", "exit_code_ok",
        "trainable_coverage", "expected_count", "trainable_count", "excluded_count",
        "trainable_succeeded_count", "trainable_blocking_count",
        "min_data_cutoff", "max_data_cutoff",
    )}, indent=2))

    if not payload["certified"]:
        print(
            "tournament_retrain_marker: NOT CERTIFIED "
            f"(status={payload['status']} exit_code={payload['exit_code']} "
            f"trainable_coverage={payload['trainable_coverage']:.3f} "
            f"blocking={payload['trainable_blocking_count']}). "
            "Leaving prior marker untouched so its data cutoff keeps ageing.",
            file=sys.stderr,
        )
        if not payload["exit_code_ok"]:
            print(f"  train_104.py exited non-zero: {payload['exit_code']}", file=sys.stderr)
        if payload["sets"]["trainable_blocking"]:
            print(f"  trainable tickers not fresh-certified: {payload['sets']['trainable_blocking']}", file=sys.stderr)
        return 3

    marker = Path(args.marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(
        f"tournament_retrain_marker: CERTIFIED status={payload['status']} "
        f"trainable_coverage={payload['trainable_coverage']:.3f} "
        f"cutoff=[{payload['min_data_cutoff']}..{payload['max_data_cutoff']}] "
        f"→ stamped {marker}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
