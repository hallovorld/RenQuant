#!/usr/bin/env python3
"""Stamp per-entry content digests into a walk-forward corpus manifest.

Why
---
``kernel/manifest_uri_resolver.py`` closes its digest compatibility window on
``ARTIFACT_DIGEST_REQUIRED_AFTER`` (2026-09-01): on/after that date the
model-validation path (``WalkForwardModelLoader._resolve_model_uri``) passes
``require_digest=True`` and a manifest entry with no ``artifact_sha256``
fails closed. The committed WF corpus manifests were built before the digest
field existed (0/39 stamped — PR #497 review finding), so without a stamping
pass the weekly promote gate goes dark on the deadline.

What this script does
---------------------
For every entry in ``retrains``:

* resolves ``artifact_uri`` through the REAL bounded resolver
  (``kernel.manifest_uri_resolver.resolve_manifest_uri``) — identical root
  derivation, containment, and ambiguity semantics as enforcement — and
  stamps ``artifact_sha256`` = sha256 of the resolved file's raw bytes
  (exactly ``_digest`` in the resolver: ``sha256(path.read_bytes())``);
* where ``calibrator_uri`` is present, stamps ``calibrator_sha256`` the same
  way. Since task #82 (PR #499 review follow-up) the loader ENFORCES this
  field too: ``calibrator_as_of`` binds the resolved calibrator to the
  stamped digest under the same compatibility window as ``artifact_sha256``,
  and ``read_manifest``/``_parse_entry`` round-trip it;
* self-verifies each stamp by re-resolving with ``expected_digest=<stamp>``
  and ``require_digest=True`` — i.e. the exact post-window enforcement call.

Format preservation
-------------------
The output differs from the input by ADDED LINES ONLY: digest keys are
inserted immediately after their ``*_uri`` key, key order is otherwise
preserved, and the file's ``json.dumps(indent=2)`` shape and trailing-newline
convention are kept. The script refuses to run on a manifest whose bytes are
not exactly ``json.dumps(json.load(f), indent=2)`` (+ optional trailing
newline) — that guarantee is what makes the diff reviewable.

Usage
-----
    python3 scripts/stamp_wf_manifest_digests.py \
        --manifest backtesting/renquant_104/artifacts/sim/walkforward_manifest.json \
        [--resolve-as /path/to/deployed/.../walkforward_manifest.json] \
        [--check]

``--resolve-as`` sets the manifest path handed to the resolver for ROOT
derivation only (the stamped file itself is still ``--manifest``). It exists
because ``walkforward_manifest.json`` (v1) carries absolute URIs into the
deployed operator tree: resolving them from a scratch checkout must derive
roots from the deployed manifest location, exactly as the live loader will.
``--check`` verifies existing stamps instead of writing (exit 1 on any
missing/mismatched stamp).

This script only ever WRITES the ``--manifest`` file; referenced artifact
files are read-only inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.manifest_uri_resolver import resolve_manifest_uri  # noqa: E402

# (uri key, digest key) pairs stamped per entry. artifact_sha256 is the field
# the resolver enforces; calibrator_sha256 is forward-compatible provenance.
_DIGEST_FIELDS = (
    ("artifact_uri", "artifact_sha256"),
    ("calibrator_uri", "calibrator_sha256"),
)


def _sha256_file(path: Path) -> str:
    """Content digest, byte-for-byte the resolver's ``_digest``."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_preserving_format(manifest: Path) -> tuple[dict, str]:
    """Parse the manifest, refusing any format this script cannot reproduce.

    Returns ``(payload, trailing)`` where ``trailing`` is the file's
    trailing-newline suffix (kept identical on write).
    """
    raw = manifest.read_text()
    payload = json.loads(raw)
    body = json.dumps(payload, indent=2)
    if raw not in (body, body + "\n"):
        raise SystemExit(
            f"{manifest}: file bytes are not json.dumps(payload, indent=2) "
            f"(+ optional trailing newline); refusing to rewrite — the "
            f"added-lines-only diff guarantee would not hold."
        )
    return payload, raw[len(body):]


def _resolve_existing(resolve_as: Path, uri: str) -> Path:
    """Resolve ``uri`` via the bounded resolver to an existing local file."""
    resolved = resolve_manifest_uri(resolve_as, uri)
    if not isinstance(resolved, Path):
        raise SystemExit(f"{uri!r}: opaque scheme URI — cannot content-hash.")
    if not resolved.exists():
        raise SystemExit(f"{uri!r}: resolved to missing file {resolved}.")
    return resolved


def _stamped_entry(entry: dict, resolve_as: Path) -> tuple[dict, int]:
    """One entry with digest keys inserted right after their ``*_uri`` keys.

    Returns ``(new_entry, files_hashed)``. Fails loudly on a pre-existing
    stamp that disagrees with the recomputed digest.
    """
    digests: dict[str, str] = {}
    for uri_key, digest_key in _DIGEST_FIELDS:
        uri = entry.get(uri_key)
        if not uri:
            continue
        resolved = _resolve_existing(resolve_as, str(uri))
        digest = _sha256_file(resolved)
        prior = entry.get(digest_key)
        if prior and str(prior).strip().lower() != digest:
            raise SystemExit(
                f"{uri!r}: existing {digest_key} {prior!r} != recomputed "
                f"{digest}; refusing to overwrite a conflicting stamp."
            )
        # Self-verify with the exact post-window enforcement call.
        resolve_manifest_uri(
            resolve_as, str(uri), expected_digest=digest, require_digest=True
        )
        digests[uri_key] = digest
    out: dict = {}
    for key, value in entry.items():
        out[key] = value
        for uri_key, digest_key in _DIGEST_FIELDS:
            if key == uri_key and uri_key in digests:
                out[digest_key] = digests[uri_key]
    return out, len(digests)


def _check_entry(entry: dict, resolve_as: Path, idx: int) -> list[str]:
    """Problems (empty list = fully stamped and matching) for one entry."""
    problems = []
    for uri_key, digest_key in _DIGEST_FIELDS:
        uri = entry.get(uri_key)
        if not uri:
            continue
        stamped = str(entry.get(digest_key) or "").strip().lower()
        if not stamped:
            problems.append(f"entry[{idx}] {uri!r}: missing {digest_key}")
            continue
        actual = _sha256_file(_resolve_existing(resolve_as, str(uri)))
        if actual != stamped:
            problems.append(
                f"entry[{idx}] {uri!r}: {digest_key} {stamped} != file {actual}"
            )
    return problems


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--manifest", required=True, type=Path,
                    help="WF manifest JSON to stamp (read + rewritten).")
    ap.add_argument("--resolve-as", type=Path, default=None,
                    help="Manifest path used for resolver ROOT derivation "
                         "only (default: --manifest).")
    ap.add_argument("--check", action="store_true",
                    help="Verify existing stamps; write nothing.")
    args = ap.parse_args(argv)

    manifest: Path = args.manifest
    resolve_as: Path = args.resolve_as or manifest
    payload, trailing = _load_preserving_format(manifest)
    entries = payload.get("retrains")
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"{manifest}: no 'retrains' list to stamp.")

    if args.check:
        problems = [
            p for i, e in enumerate(entries)
            for p in _check_entry(e, resolve_as, i)
        ]
        for p in problems:
            print(p, file=sys.stderr)
        print(f"{manifest}: {len(entries)} entries checked, "
              f"{len(problems)} problem(s).")
        return 1 if problems else 0

    hashed = 0
    stamped_entries = []
    for entry in entries:
        new_entry, n = _stamped_entry(entry, resolve_as)
        stamped_entries.append(new_entry)
        hashed += n
    payload["retrains"] = stamped_entries
    manifest.write_text(json.dumps(payload, indent=2) + trailing)
    print(f"{manifest}: stamped {len(entries)} entries "
          f"({hashed} file digests).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
