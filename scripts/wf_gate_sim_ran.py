#!/usr/bin/env python3
"""Did the WF gate actually SIMULATE the candidate it just rejected?

2026-09-01..03: every retrain's three WF cuts died in
`WalkForwardModelLoader.model_as_of` (`ManifestUriResolutionError`, digest
compatibility window closed). `run_wf_gate` stamped
`wf_reason = "3/3 sim cuts failed execution"` with `cuts[*].returncode = 1`
and exited non-zero, `weekly_wf_promote.sh` took its ordinary reject branch,
consulted the RFC#210 freshness fallback (which never looks at the cuts), and
— prod being fresh — reported "Reject disposition: prod FRESH … governance
nominal, calm notify, exit 0" for three days. An infrastructure crash was
reported as the gate declining.

This helper answers ONE question from the staging artifact's stamped
`metadata.wf_gate_metadata`: did every cut execute (returncode 0)? It is a
gate, not a disposition:

    exit 0   every cut ran (a reject from here on is a verdict, whatever it is)
    exit 1   at least one cut did not run, or the evidence is missing/malformed
             (fail closed toward attention — an unproven "ran" is "did not run")

It prints one line to stdout naming the reason. It is called BEFORE the
freshness fallback is consulted, so a candidate whose simulation crashed is
never eligible for fallback promotion either.
"""
from __future__ import annotations

import json
import sys


def sim_ran(path: str) -> tuple[bool, str]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError) as exc:
        return False, f"staging artifact unreadable ({exc.__class__.__name__})"
    if not isinstance(d, dict):
        return False, "staging artifact is not an object"
    meta = d.get("metadata")
    wf = meta.get("wf_gate_metadata") if isinstance(meta, dict) else None
    if not isinstance(wf, dict):
        return False, "metadata.wf_gate_metadata missing — the gate stamped no evidence"
    cuts = wf.get("cuts")
    if not isinstance(cuts, list) or not cuts:
        return False, "wf_gate_metadata.cuts missing or empty — no cut was recorded"
    failed: list[str] = []
    for i, cut in enumerate(cuts):
        if not isinstance(cut, dict):
            failed.append(f"cut[{i}] malformed")
            continue
        rc = cut.get("returncode")
        # Explicit-sentinel rule: the int 0 proves execution; None/absent/bool/
        # anything else does not.
        if not (type(rc) is int and rc == 0):
            failed.append(f"cut[{i}] {cut.get('start')}..{cut.get('end')} returncode={rc!r}")
    if failed:
        reason = str(wf.get("wf_reason") or "").strip()
        return False, f"{len(failed)}/{len(cuts)} cuts did not execute ({'; '.join(failed)})" + (
            f" — gate said: {reason}" if reason else "")
    return True, f"all {len(cuts)} cuts executed (returncode 0)"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("WF-SIM-UNPROVEN|usage: wf_gate_sim_ran.py STAGING_ARTIFACT")
        return 1
    ok, reason = sim_ran(argv[1])
    print(("WF-SIM-RAN|" if ok else "WF-SIM-DID-NOT-RUN|") + reason)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
