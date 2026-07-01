"""Bounded manifest-URI resolver — one contract shared by every call site.

Historical bug (PR #421 review): a per-bar resolver walked *arbitrary*
ancestors of the manifest folder and returned the first path that happened to
exist. That made artifact identity a property of the machine's surrounding
filesystem rather than the manifest contract — a stale same-named file higher
in the checkout could be silently selected, relocating the checkout could
select a different model, and three duplicated resolver copies could drift.

This module (``kernel/manifest_uri_resolver.py``) replaces that with a BOUNDED
resolver used by all three call sites (``kernel/walk_forward/loader.py``,
``adapters/sim_artifacts.py``, ``scripts/run_wf_gate.py``). It lives directly
under ``kernel/`` (not ``kernel/walk_forward/``) on purpose: the sim adapter
imports it, and the ``kernel/walk_forward`` package ``__init__`` pulls heavier
pipeline modules the adapter/URI-resolution path has no need for.

* Absolute paths and ``scheme://`` URIs are returned untouched.
* Relative URIs resolve only against a small ORDERED set of KNOWN ROOTS:
    1. the manifest's own folder (manifest-relative — the legacy default);
    2. the strategy/repo root inferred from the manifest path (the parent of
       the outermost ``artifacts`` directory), where orchestrator-built WF
       manifests emit strategy-dir-relative URIs like
       ``artifacts/walkforward_.../panel-ltr.json``.
* Every candidate is NORMALIZED and CONTAINMENT-checked. A URI whose
  normalized join escapes *every* allowed root (``..`` traversal) is REJECTED,
  never silently walked.
* AMBIGUITY is REJECTED: if more than one root yields an existing file and
  those files have different content digests, we raise rather than guess.
* When an ``expected_digest`` is supplied (model-validation paths), the
  resolved file's sha256 must match or we raise — on a gate path "found a
  file" is not sufficient.
* When nothing exists, we fall back to the manifest-relative join so the
  downstream not-found error names the expected location (no surprise path).

Because both roots are derived from the manifest path itself (never from an
absolute machine prefix), resolution is deterministic across checkout
relocation: the same relative layout resolves to the same in-strategy artifact
regardless of where the tree lives.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

_ARTIFACTS_DIR_NAME = "artifacts"


class ManifestUriResolutionError(ValueError):
    """Raised when a manifest URI violates the resolution contract.

    Distinct subclass so callers can tell a contract violation (traversal,
    ambiguity, digest mismatch) apart from a plain missing-file error.
    """


def _normalize(path: Path) -> Path:
    """Lexical absolute normalization — no symlink/existence dependency.

    Containment is then a pure function of the path strings, and identical
    relative layouts resolve identically regardless of where the checkout is.
    """
    return Path(os.path.normpath(os.path.abspath(str(path))))


def _is_within(root: Path, candidate: Path) -> bool:
    root_n = _normalize(root)
    cand_n = _normalize(candidate)
    return cand_n == root_n or root_n in cand_n.parents


def _strategy_root(manifest_path: Path) -> Path | None:
    """Infer the strategy/repo root: parent of the outermost ``artifacts`` dir.

    Orchestrator-built manifests live at ``<strategy>/artifacts/sim/...`` and
    emit URIs relative to ``<strategy>`` (``artifacts/walkforward_.../...``).
    Returns None when the manifest path has no ``artifacts`` component (then
    the only known root is the manifest folder itself).
    """
    parts = _normalize(manifest_path).parts
    for i, part in enumerate(parts):
        if part == _ARTIFACTS_DIR_NAME:
            return Path(*parts[:i]) if i else None
    return None


def _known_roots(manifest_path: Path) -> list[Path]:
    """Ordered allowed roots: manifest folder first, strategy root second."""
    roots = [manifest_path.parent]
    strat = _strategy_root(manifest_path)
    if strat is not None and _normalize(strat) != _normalize(manifest_path.parent):
        roots.append(strat)
    return roots


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_digest(value: str | None) -> str:
    return str(value or "").strip().lower().removeprefix("sha256:")


def _enforce_digest(path: Path, expected_digest: str | None, uri: str) -> Path:
    """Bind the resolved file to the manifest's expected digest where present."""
    want = _normalize_digest(expected_digest)
    if want and path.exists():
        got = _digest(path)
        # Accept exact matches and historical short-sha prefixes (consistent
        # with the loader's fingerprint-matching convention).
        if not (got == want or got.startswith(want) or want.startswith(got)):
            raise ManifestUriResolutionError(
                f"manifest URI {uri!r} resolved to {path} whose digest "
                f"sha256:{got} does not match expected sha256:{want}."
            )
    return path


def resolve_manifest_uri(
    manifest_path: "str | Path",
    uri: str,
    *,
    expected_digest: str | None = None,
) -> "Path | str":
    """Resolve a manifest artifact URI to a filesystem path under the contract.

    Returns the original string for ``scheme://`` URIs (opaque), else a Path.
    Raises ManifestUriResolutionError on traversal-outside-roots, on ambiguity
    (multiple existing candidates with different digests), or on a digest
    mismatch against ``expected_digest``.
    """
    manifest_path = Path(manifest_path)
    text = str(uri)
    # scheme:// URIs (e.g. s3://) are opaque — untouched.
    if "://" in text:
        return text
    p = Path(text)
    if p.is_absolute():
        return _enforce_digest(p, expected_digest, text)

    roots = _known_roots(manifest_path)
    # Confine candidates to the allowed roots (containment enforced).
    contained: list[Path] = [
        root / p for root in roots if _is_within(root, root / p)
    ]
    if not contained:
        # Every allowed root rejected the join → traversal escape.
        raise ManifestUriResolutionError(
            f"manifest URI {text!r} escapes all allowed roots "
            f"{[str(r) for r in roots]} (traversal outside the contract)."
        )

    existing = [c for c in contained if c.exists()]
    if len(existing) > 1:
        # Reject ambiguity: distinct existing candidates with different digests.
        by_digest: dict[str, Path] = {}
        for c in existing:
            try:
                key = _digest(c)
            except OSError:
                key = f"path::{_normalize(c)}"
            by_digest.setdefault(key, c)
        if len(by_digest) > 1:
            raise ManifestUriResolutionError(
                f"manifest URI {text!r} is ambiguous: {len(by_digest)} candidates "
                f"with different digests exist under {[str(r) for r in roots]}: "
                f"{[str(c) for c in existing]}."
            )
    if existing:
        # Deterministic: first existing candidate in root order.
        return _enforce_digest(existing[0], expected_digest, text)

    # Nothing exists — fall back to the manifest-relative join so the
    # downstream not-found error names the expected location.
    return contained[0]
