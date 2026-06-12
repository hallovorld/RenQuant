"""DRPH — deterministic replay & parity harness (core substrate).

Design: renquant-orchestrator
doc/research/2026-06-12-engineering-architecture-deep-plan.md §IV quality
doctrine + S2 item 5 ("each extraction: behavior-identical PR gated by the
replay harness — one fixed historical day reproduced bit-identically
before/after"); prototype with self-proofs:
scripts/engineering/drph_core.py (orchestrator PR #112 batch).

Three pieces, deliberately tiny:

  * ``canonical_json``  — decision payloads canonicalized (sorted keys,
    floats rounded to PRECISION) so sub-precision wobble never diffs and
    any real behavior change always does.
  * ``run_fingerprint`` — the frozen-input identity (config/panel/state/
    artifact sha256s + pin digest + env sha). Two runs with equal
    fingerprints MUST produce byte-equal canonical decisions; a refactor
    PR is behavior-identical iff every corpus case still verifies.
  * ``ReplayCase``      — a content-addressed case directory on disk:
    ``inputs/`` (frozen input payloads + manifest of their hashes),
    ``expected/decisions.json`` (canonical), ``case_manifest.json``.
    ``verify`` byte-compares and, on mismatch, localizes the first
    diverging paths instead of dumping blobs.

The golden corpus lives in ``tests/drph_corpus/`` (version-controlled);
``scripts/drph_capture.py`` builds cases from the persisted run DBs (zero
runtime risk — capture reads what persistence already wrote).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PRECISION = 8


def _canon(obj):
    if isinstance(obj, float):
        return round(obj, PRECISION)
    if isinstance(obj, dict):
        return {k: _canon(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_canon(x) for x in obj]
    return obj


def canonical_json(decisions: dict) -> str:
    return json.dumps(_canon(decisions), sort_keys=True, separators=(",", ":"))


def sha(payload: bytes | str) -> str:
    b = payload.encode() if isinstance(payload, str) else payload
    return hashlib.sha256(b).hexdigest()[:16]


def run_fingerprint(*, config_sha: str, panel_sha: str, state_sha: str,
                    artifact_shas: dict, pin_digest: str, env_sha: str) -> dict:
    """Frozen-input identity. Equal fingerprint ⇒ byte-equal canonical
    decisions is the behavior-identity contract every refactor PR is
    gated on."""
    return {"config_sha": config_sha, "panel_sha": panel_sha,
            "state_sha": state_sha,
            "artifact_shas": dict(sorted(artifact_shas.items())),
            "pin_digest": pin_digest, "env_sha": env_sha}


class ReplayCase:
    """Content-addressed frozen case on disk."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def write(self, *, inputs: dict, expected_decisions: dict) -> str:
        """Freeze inputs + expected decisions; returns the case id
        (sha of the hash manifest — content-addressed)."""
        (self.root / "inputs").mkdir(parents=True, exist_ok=True)
        (self.root / "expected").mkdir(parents=True, exist_ok=True)
        manifest = {}
        for name, payload in sorted(inputs.items()):
            blob = (canonical_json(payload) if isinstance(payload, dict)
                    else str(payload))
            (self.root / "inputs" / f"{name}.json").write_text(blob)
            manifest[name] = sha(blob)
        expected_blob = canonical_json(expected_decisions)
        (self.root / "expected" / "decisions.json").write_text(expected_blob)
        manifest["expected"] = sha(expected_blob)
        manifest_blob = canonical_json(manifest)
        (self.root / "case_manifest.json").write_text(manifest_blob)
        return sha(manifest_blob)

    def read_input(self, name: str) -> dict:
        return json.loads((self.root / "inputs" / f"{name}.json").read_text())

    def expected(self) -> dict:
        return json.loads((self.root / "expected" / "decisions.json").read_text())

    def check_integrity(self) -> list[str]:
        """Re-hash everything against case_manifest.json. Catches corpus
        tampering / accidental edits before a verify is trusted."""
        manifest = json.loads((self.root / "case_manifest.json").read_text())
        problems = []
        for name, expected_sha in manifest.items():
            f = (self.root / "expected" / "decisions.json" if name == "expected"
                 else self.root / "inputs" / f"{name}.json")
            if not f.exists():
                problems.append(f"missing file for manifest entry {name!r}: {f}")
            elif sha(f.read_text()) != expected_sha:
                problems.append(f"hash mismatch for {name!r}: {f}")
        return problems

    def verify(self, actual_decisions: dict) -> tuple[bool, list[str]]:
        """Byte-compare canonical decisions; on mismatch return the first
        20 diverging paths (diff localization, not blob dumps)."""
        exp = (self.root / "expected" / "decisions.json").read_text()
        act = canonical_json(actual_decisions)
        if exp == act:
            return True, []
        e, a = json.loads(exp), json.loads(act)
        diffs: list[str] = []

        def walk(p, x, y):
            if len(diffs) >= 20:
                return
            if isinstance(x, dict) and isinstance(y, dict):
                for k in sorted(set(x) | set(y)):
                    walk(f"{p}.{k}", x.get(k), y.get(k))
            elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
                for i, (xi, yi) in enumerate(zip(x, y)):
                    walk(f"{p}[{i}]", xi, yi)
            elif x != y:
                diffs.append(f"{p}: expected={x!r} actual={y!r}")

        walk("$", e, a)
        return False, diffs[:20]
