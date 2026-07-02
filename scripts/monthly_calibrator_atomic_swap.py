#!/usr/bin/env python3
"""Atomic staging -> production publish for the monthly calibrator refresh.

2026-07-01 REVIEW FIX ROUND 2 (PR #425 CHANGES_REQUESTED, Codex): the fit
step in `monthly_calibrator_refresh.sh` wrote DIRECTLY to `PROD_CAL` (the
live production calibrator path) BEFORE Step 3/3b's validation (the
pool_ic/non-collapse quality gate + the scorer/calibrator binding gate)
ran. During the fit-to-validation window, the LIVE RUNTIME could read the
new, unvalidated/mismatched calibrator — a real exposure window, not
merely a "roll back after the fact" bug. Separately: if NO prior
calibrator existed (first-ever fit) and validation failed, the failure
branches alerted + exited but left the REJECTED artifact sitting at
`PROD_CAL` — so a rejected artifact remained published in the no-baseline
case. Rollback-after-exposure is not the same as atomic admission.

This module is the extracted, independently-testable staging/publish logic
`monthly_calibrator_refresh.sh` now delegates to — same convention as
`scripts/verify_calibrator_scorer_binding.py` (see its docstring): pull
shell-embedded logic into an importable module so it can be unit +
integration tested without a strategy venv, rather than leaving it
un-unit-testable inline.

Building blocks:
  * `sha256_file`        — content digest of a calibrator artifact.
  * `atomic_publish`      — re-verify the staging artifact's digest still
                            matches what was gate-checked (TOCTOU guard),
                            THEN `os.replace()` staging -> prod. Both
                            checks happen BEFORE any filesystem mutation,
                            so a raise here always leaves `prod_path`
                            untouched. `os.replace` is a single rename
                            syscall on the SAME filesystem (staging lives
                            at `PROD_CAL` + a unique suffix, in the same
                            directory) — POSIX-atomic, so a concurrent
                            reader of `PROD_CAL` always observes either the
                            fully-old or fully-new file, never a partial
                            write and never the path transiently missing.
  * `quarantine_staging`  — on ANY gate failure, move the staging artifact
                            out of the way. This function never touches
                            `prod_path` — it does not even accept one as an
                            argument — so by construction a failed fit/gate
                            leaves production byte-identical to how the run
                            started, including the no-baseline (first-ever
                            fit) case, where production simply continues to
                            not exist.
  * `build_receipt` / `write_receipt` — bind the CHECKED scorer identity/
                            fingerprints + the candidate calibrator's exact
                            digest into a receipt written alongside the
                            publish/quarantine decision, so what actually
                            gets swapped into `PROD_CAL` is provably the
                            same artifact `verify_calibrator_scorer_binding.py`
                            evaluated — closing the TOCTOU gap where
                            something could swap the staging file's
                            contents between the gate check and the
                            publish.

Usage (see `monthly_calibrator_refresh.sh` Step 3c)::

    scripts/monthly_calibrator_atomic_swap.py sha256 --path "$STAGING_CAL"

    scripts/monthly_calibrator_atomic_swap.py publish \\
        --staging "$STAGING_CAL" --prod "$PROD_CAL" \\
        --expected-sha256 "$CANDIDATE_SHA256" \\
        --receipt-out "$RECEIPT" \\
        --scorer-path "$PROD_SCORER" \\
        --scorer-fingerprints-json "$SCORER_FPS_JSON" \\
        --calibrator-fingerprints-json "$CAL_FPS_JSON" \\
        --pool-ic "$NEW_POOL_IC" --n-unique "$NEW_N_UNIQUE"

    scripts/monthly_calibrator_atomic_swap.py quarantine \\
        --staging "$STAGING_CAL" --reason "..." \\
        --receipt-out "$RECEIPT" ...(same metadata flags)...

Exit codes: 0 = success, 1 = failure (digest mismatch / missing staging /
IO error). Every failure path leaves `PROD_CAL` untouched.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


class DigestMismatchError(RuntimeError):
    """Staging artifact's bytes changed between gate-check and publish."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _parse_json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        val = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(v) for v in val] if isinstance(val, list) else []


def _parse_float(raw: str | None) -> float | None:
    if raw is None or raw in ("None", ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def build_receipt(
    *,
    status: str,
    candidate_path: Path,
    candidate_sha256: str,
    scorer_path: Path | None = None,
    scorer_fingerprints: list[str] | None = None,
    calibrator_fingerprints: list[str] | None = None,
    pool_ic: float | None = None,
    n_unique_prob_y: float | None = None,
    reason: str | None = None,
    prod_path: Path | None = None,
) -> dict[str, Any]:
    """Build the acceptance receipt binding scorer identity + candidate digest.

    This is what makes the publish provable: the receipt records the exact
    sha256 of the calibrator bytes that were run through Step 3 (quality
    gate) and Step 3b (binding gate), and the scorer path/fingerprint(s) it
    was checked against. `atomic_publish` re-verifies the digest
    immediately before the filesystem swap, so a receipt can never describe
    an artifact other than the one actually published.
    """
    return {
        "status": status,
        "timestamp": _utc_now_iso(),
        "candidate_calibrator_path": str(candidate_path),
        "candidate_sha256": candidate_sha256,
        "scorer_path": str(scorer_path) if scorer_path is not None else None,
        "scorer_fingerprints": scorer_fingerprints or [],
        "calibrator_fingerprints": calibrator_fingerprints or [],
        "pool_ic": pool_ic,
        "n_unique_prob_y": n_unique_prob_y,
        "prod_path": str(prod_path) if prod_path is not None else None,
        "reason": reason,
    }


def write_receipt(receipt_path: Path, receipt: dict[str, Any]) -> None:
    receipt_path = Path(receipt_path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def atomic_publish(
    staging_path: Path,
    prod_path: Path,
    *,
    expected_sha256: str,
) -> str:
    """Atomically swap `staging_path` into `prod_path`.

    Raises `FileNotFoundError` if staging is missing, `DigestMismatchError`
    if the staging bytes no longer match `expected_sha256` (the artifact
    that was actually gate-checked) — both raised BEFORE any filesystem
    mutation, so `prod_path` is guaranteed untouched whenever this raises.

    Returns the sha256 that was published (always == expected_sha256).
    """
    staging_path = Path(staging_path)
    prod_path = Path(prod_path)
    if not staging_path.exists():
        raise FileNotFoundError(
            f"staging calibrator missing at publish time: {staging_path}"
        )

    actual_sha256 = sha256_file(staging_path)
    if actual_sha256 != expected_sha256:
        raise DigestMismatchError(
            "staging calibrator bytes changed between gate-check and publish "
            f"(TOCTOU) — expected sha256={expected_sha256} got={actual_sha256}. "
            "Refusing to publish; production calibrator is untouched."
        )

    # Same-directory rename (staging lives at f"{prod}.staging-<run-id>.json")
    # is a single POSIX rename syscall — atomic. A concurrent reader of
    # prod_path always observes either the fully-old or fully-new file.
    os.replace(staging_path, prod_path)
    return actual_sha256


def quarantine_staging(
    staging_path: Path,
    *,
    reason: str,
    quarantine_dir: Path | None = None,
) -> Path | None:
    """Move a rejected/failed staging artifact out of the way.

    `prod_path` is never referenced here — by construction this function
    can never touch production; it only ever moves the staging file (or
    does nothing). Returns the quarantine destination, or `None` if there
    was no staging file to begin with (e.g. `fit_calibrator` crashed before
    writing anything) — the correct, no-op outcome for the first-install
    (no-baseline) failure case: no production artifact, and nothing to
    quarantine either.
    """
    staging_path = Path(staging_path)
    if not staging_path.exists():
        return None
    if quarantine_dir is None:
        quarantine_dir = staging_path.parent / "_rejected_calibrators"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    ts = _utc_now_iso().replace(":", "").replace("+00:00", "Z")
    dest = quarantine_dir / f"{ts}_REJECTED_{staging_path.name}"
    os.replace(staging_path, dest)
    reason_path = dest.with_name(dest.name + ".reason.txt")
    reason_path.write_text(reason + "\n")
    return dest


def _receipt_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "scorer_path": Path(args.scorer_path) if args.scorer_path else None,
        "scorer_fingerprints": _parse_json_list(args.scorer_fingerprints_json),
        "calibrator_fingerprints": _parse_json_list(args.calibrator_fingerprints_json),
        "pool_ic": _parse_float(args.pool_ic),
        "n_unique_prob_y": _parse_float(args.n_unique),
    }


def _add_receipt_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--receipt-out", default=None)
    p.add_argument("--scorer-path", default=None)
    p.add_argument("--scorer-fingerprints-json", default=None)
    p.add_argument("--calibrator-fingerprints-json", default=None)
    p.add_argument("--pool-ic", default=None)
    p.add_argument("--n-unique", default=None)
    p.add_argument("--reason", default=None)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_sha = sub.add_parser(
        "sha256", help="print the sha256 of a file (candidate digest capture point)",
    )
    p_sha.add_argument("--path", required=True)

    p_pub = sub.add_parser(
        "publish", help="atomically swap staging into prod after all gates pass",
    )
    p_pub.add_argument("--staging", required=True)
    p_pub.add_argument("--prod", required=True)
    p_pub.add_argument("--expected-sha256", required=True)
    _add_receipt_args(p_pub)

    p_q = sub.add_parser(
        "quarantine",
        help="quarantine a rejected/failed staging artifact; prod is untouched",
    )
    p_q.add_argument("--staging", required=True)
    _add_receipt_args(p_q)

    args = ap.parse_args()

    if args.command == "sha256":
        print(sha256_file(Path(args.path)))
        return 0

    if args.command == "publish":
        staging = Path(args.staging)
        prod = Path(args.prod)
        try:
            actual = atomic_publish(staging, prod, expected_sha256=args.expected_sha256)
        except (FileNotFoundError, DigestMismatchError) as exc:
            print(f"PUBLISH FAILED: {exc}", file=sys.stderr)
            if args.receipt_out:
                receipt = build_receipt(
                    status="error",
                    candidate_path=staging,
                    candidate_sha256=args.expected_sha256,
                    reason=str(exc),
                    prod_path=prod,
                    **_receipt_kwargs_from_args(args),
                )
                write_receipt(Path(args.receipt_out), receipt)
            return 1
        print(f"PUBLISHED: {staging.name} -> {prod.name} (sha256={actual})")
        if args.receipt_out:
            receipt = build_receipt(
                status="published",
                candidate_path=prod,   # post-swap, the artifact now lives at prod
                candidate_sha256=actual,
                reason=args.reason or "all gates passed",
                prod_path=prod,
                **_receipt_kwargs_from_args(args),
            )
            write_receipt(Path(args.receipt_out), receipt)
        return 0

    if args.command == "quarantine":
        staging = Path(args.staging)
        reason = args.reason or "gate failure"
        dest = quarantine_staging(staging, reason=reason)
        if dest is not None:
            print(f"QUARANTINED: {staging} -> {dest}")
        else:
            print(f"QUARANTINE no-op: no staging artifact at {staging}")
        if args.receipt_out:
            candidate_path = dest if dest is not None else staging
            candidate_sha256 = (
                sha256_file(dest) if dest is not None and dest.exists() else "unknown"
            )
            receipt = build_receipt(
                status="rejected",
                candidate_path=candidate_path,
                candidate_sha256=candidate_sha256,
                reason=reason,
                **_receipt_kwargs_from_args(args),
            )
            write_receipt(Path(args.receipt_out), receipt)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
