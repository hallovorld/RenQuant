#!/usr/bin/env python3
"""Compose the weekly tournament retrain's final ntfy from a DURABLE receipt.

2026-08-30 incident: ``RenQuant 104 TOURNAMENT-RETRAIN ✓`` ("CERTIFIED")
fired ONE second after ``TOURNAMENT ACCEPTANCE WARN: rejected 2 per-ticker
candidate(s)``. Both were true — certification measures artifact coverage /
freshness / exit code, not acceptance — but the operator reads the ✓ as
"everything is fine" and the WARN scrolls out of sight. The verdict has to
carry the rejection count in its own title and body.

The rejection count reaches the shell through a RECEIPT, not a log grep:
``scripts/train_104.py`` writes ``$RENQUANT_TOURNAMENT_REJECTIONS_OUT`` after
the BaselineTournamentJob has run, bound to the wrapper's ``$RUN_ID`` via
``$RENQUANT_TOURNAMENT_RUN_ID`` — the same identity the completion marker's
no-change attestations are bound to. A receipt from another run (or none at
all) is reported as such, never read as "0 rejections".

Two functions, both pure, both tested (tests/test_tournament_verdict.py):

- ``write_rejection_receipt``  — the producer side (called by train_104.py).
- ``compose_tournament_verdict`` — the consumer side (called by the shell via
  the CLI below, which prints the title on line 1 and the body on line 2).
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA = 1
TITLE_OK = "RenQuant 104 TOURNAMENT-RETRAIN ✓"
TITLE_WARN = "RenQuant 104 TOURNAMENT-RETRAIN ⚠"
#: Names listed inline in the body; the rest are counted.
MAX_NAMES_IN_BODY = 10


def write_rejection_receipt(
    path: Path | str,
    *,
    run_id: str | None,
    trigger: str,
    rejected: dict[str, str],
    train_run_id: str | None = None,
) -> Path:
    """Atomically write the per-run rejection receipt. Always writes — a run
    with zero rejections leaves a receipt saying so, which is what lets the
    consumer tell "0 rejections" from "nobody wrote a receipt"."""
    path = Path(path)
    payload: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "run_id": run_id,
        "train_run_id": train_run_id,
        "trigger": trigger,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rejected": len(rejected),
        "rejected": {str(t): str(r) for t, r in sorted(rejected.items())},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return path


def load_rejection_receipt(path: Path | str, *, expected_run_id: str | None) -> tuple[dict | None, str | None]:
    """Return (receipt, problem). ``problem`` is a short reason when the
    receipt cannot be trusted for THIS run: missing, unreadable, malformed, or
    bound to a different run_id."""
    path = Path(path)
    if not path.exists():
        return None, f"rejection receipt MISSING ({path.name})"
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return None, f"rejection receipt UNREADABLE ({path.name}: {exc.__class__.__name__})"
    if not isinstance(raw, dict) or not isinstance(raw.get("rejected"), dict):
        return None, f"rejection receipt MALFORMED ({path.name})"
    if expected_run_id and raw.get("run_id") != expected_run_id:
        return None, (
            f"rejection receipt STALE ({path.name} is bound to run_id="
            f"{raw.get('run_id')!r}, this run is {expected_run_id!r})"
        )
    return raw, None


def compose_tournament_verdict(
    *,
    receipt_path: Path | str,
    expected_run_id: str | None,
    marker: str,
    log: str,
) -> tuple[str, str]:
    """Build (title, body) for the CERTIFIED branch of the weekly wrapper.

    - 0 rejections, receipt bound to this run → ✓ title, "0 rejections".
    - N > 0 → ⚠ title, body says "CERTIFIED WITH N REJECTIONS (A, B, …)".
    - receipt missing / stale / malformed → ⚠ title; the body says the
      count is UNKNOWN and why. Never silently downgraded to ✓.
    """
    receipt, problem = load_rejection_receipt(receipt_path, expected_run_id=expected_run_id)
    if receipt is None:
        return TITLE_WARN, (
            f"Weekly per-ticker tournament retrain CERTIFIED (see {marker}) but "
            f"the number of rejected per-ticker candidates is UNKNOWN: {problem}. "
            f"Check the log before trusting universe admission. Log: {log}"
        )
    rejected = receipt["rejected"]
    n = len(rejected)
    if n == 0:
        return TITLE_OK, (
            f"Weekly per-ticker tournament retrain CERTIFIED, 0 rejections "
            f"(see {marker}). Universe admission refreshed. Log: {log}"
        )
    names = sorted(rejected)
    shown = ", ".join(names[:MAX_NAMES_IN_BODY])
    more = f", +{n - MAX_NAMES_IN_BODY} more" if n > MAX_NAMES_IN_BODY else ""
    return TITLE_WARN, (
        f"Weekly per-ticker tournament retrain CERTIFIED WITH {n} "
        f"REJECTION{'S' if n != 1 else ''} ({shown}{more}) — previous models "
        f"kept for those names (see {marker}). Universe admission refreshed for "
        f"the rest. Log: {log}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--receipt", required=True, help="path train_104.py wrote via RENQUANT_TOURNAMENT_REJECTIONS_OUT")
    p.add_argument("--run-id", default=None, help="the wrapper's RUN_ID the receipt must be bound to")
    p.add_argument("--marker", required=True)
    p.add_argument("--log", required=True)
    args = p.parse_args(argv)
    title, body = compose_tournament_verdict(
        receipt_path=args.receipt, expected_run_id=args.run_id,
        marker=args.marker, log=args.log,
    )
    # Line 1 = title, line 2 = body. Neither contains a newline by construction.
    print(title)
    print(body.replace("\n", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
