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

Contract
--------
* ``scheme://`` URIs (e.g. ``s3://``) are opaque — returned untouched.
* Relative URIs resolve only against a small ORDERED set of KNOWN ROOTS:
    1. the manifest's own folder (manifest-relative — the legacy default);
    2. the strategy/repo root inferred from the manifest path (the parent of
       the outermost ``artifacts`` directory), where orchestrator-built WF
       manifests emit strategy-dir-relative URIs like
       ``artifacts/walkforward_.../panel-ltr.json``.
* Every candidate is LEXICALLY normalized and containment-checked (rejects
  ``..`` string traversal), THEN — for candidates that actually exist — the
  REAL path is resolved (symlinks followed) and realpath-containment is
  enforced under the trusted roots. An in-root symlink that points at a target
  outside every root is REJECTED, so the resolver never reads or digests a file
  outside the strategy tree (PR #421 review, blocking point 1).
* ABSOLUTE URIs are containment-checked the same way: an absolute path whose
  realpath is outside every allowed root is REJECTED unless the caller passes
  ``allow_external=True`` (the manifest schema explicitly declares an external
  root — PR #421 review, blocking point 2). They are never returned blindly.
* AMBIGUITY is REJECTED: if more than one root yields an existing file and
  those files have different content digests, we raise rather than guess.
* When ``expected_digest`` is supplied (model-validation paths), the resolved
  file's sha256 must match or we raise — on a gate path "found a file" is not
  sufficient. When ``require_digest`` is set (model-validation paths past the
  compatibility window) a MISSING digest fails closed: the manifest must carry
  ``artifact_sha256`` (PR #421 review, blocking point 3).
* When nothing exists, we fall back to the manifest-relative join so the
  downstream not-found error names the expected location (no surprise path).

Because both roots are derived from the manifest path itself (never from an
absolute machine prefix), resolution is deterministic across checkout
relocation: the same relative layout resolves to the same in-strategy artifact
regardless of where the tree lives.

Digest compatibility window
---------------------------
``ARTIFACT_DIGEST_REQUIRED_AFTER`` defines the date on/after which a
model-validation resolution MUST have a stamped ``artifact_sha256`` (see
``digest_required``). Before that date a missing digest is tolerated (the
loader warns) so existing unstamped manifests keep validating; on/after it,
resolution fails closed. Fail-closed on this weekly-promote gate means the
prior model stays pinned — the safe direction, no capital impact.
"""
from __future__ import annotations

import hashlib
import os
from datetime import date
from pathlib import Path

_ARTIFACTS_DIR_NAME = "artifacts"

# On/after this date a model-validation resolution requires a stamped
# ``artifact_sha256`` (schema v2). Chosen to give existing v2-recipe manifests a
# ~2-month restamp window from the 2026-07-01 fix. See ``digest_required``.
ARTIFACT_DIGEST_REQUIRED_AFTER = date(2026, 9, 1)


class ManifestUriResolutionError(ValueError):
    """Raised when a manifest URI violates the resolution contract.

    Distinct subclass so callers can tell a contract violation (traversal,
    symlink escape, external absolute path, ambiguity, digest mismatch, or a
    missing-but-required digest) apart from a plain missing-file error.
    """


def digest_required(now: "date | None" = None) -> bool:
    """Whether a model-validation resolution must carry a stamped digest now.

    Returns True on/after ``ARTIFACT_DIGEST_REQUIRED_AFTER``. ``now`` is
    injectable for deterministic tests; it defaults to today's date.
    """
    return (now or date.today()) >= ARTIFACT_DIGEST_REQUIRED_AFTER


def _normalize(path: Path) -> Path:
    """Lexical absolute normalization — no symlink/existence dependency.

    Used to build and containment-filter candidate strings so identical
    relative layouts compare identically regardless of where the checkout is.
    A separate realpath check (``_realpath_within``) guards actually-existing
    candidates against in-root symlink escapes.
    """
    return Path(os.path.normpath(os.path.abspath(str(path))))


def _realpath(path: Path) -> Path:
    """Fully resolved real path (symlinks followed for existing components)."""
    return Path(os.path.realpath(str(path)))


def _is_within(root: Path, candidate: Path) -> bool:
    """Lexical containment (string-level, symlink-blind)."""
    root_n = _normalize(root)
    cand_n = _normalize(candidate)
    return cand_n == root_n or root_n in cand_n.parents


def _realpath_within(roots: list[Path], candidate: Path) -> bool:
    """Realpath containment: is ``candidate``'s real target under any real root.

    Both sides are realpath-resolved so a symlinked checkout prefix (e.g.
    macOS ``/tmp`` -> ``/private/tmp``) compares consistently, while an in-root
    symlink that escapes to an external target is caught.
    """
    cand_r = _realpath(candidate)
    for root in roots:
        root_r = _realpath(root)
        if cand_r == root_r or root_r in cand_r.parents:
            return True
    return False


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


def _enforce_digest(path: Path, want: str, uri: str) -> Path:
    """Bind the resolved file to the manifest's expected digest where present.

    ``want`` is the already-normalized (lowercased, ``sha256:``-stripped)
    expected digest, or empty when none was supplied.
    """
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
    require_digest: bool = False,
    allow_external: bool = False,
) -> "Path | str":
    """Resolve a manifest artifact URI to a filesystem path under the contract.

    Returns the original string for ``scheme://`` URIs (opaque), else a Path.

    Raises ManifestUriResolutionError on:
      * a missing digest when ``require_digest`` is set (model-validation path
        past the compatibility window);
      * an absolute path outside all allowed roots when ``allow_external`` is
        not set;
      * ``..`` traversal that escapes every allowed root;
      * an in-root symlink whose real target escapes every allowed root;
      * ambiguity (multiple existing candidates with different digests);
      * a digest mismatch against ``expected_digest``.
    """
    manifest_path = Path(manifest_path)
    text = str(uri)
    want = _normalize_digest(expected_digest)

    # Model-validation paths past the compatibility window must carry a digest.
    if require_digest and not want:
        raise ManifestUriResolutionError(
            f"manifest URI {text!r} is on a model-validation path that requires "
            f"a stamped artifact_sha256 digest (compatibility window closed "
            f"{ARTIFACT_DIGEST_REQUIRED_AFTER.isoformat()}), but the manifest "
            f"entry does not carry one. Re-stamp the manifest with per-entry "
            f"artifact_sha256 (schema v2)."
        )

    # scheme:// URIs (e.g. s3://) are opaque — untouched.
    if "://" in text:
        return text

    p = Path(text)
    roots = _known_roots(manifest_path)

    if p.is_absolute():
        # An absolute URI is never returned blindly: its real target must be
        # under a trusted root, unless the caller declares an external root.
        if not allow_external and not _realpath_within(roots, p):
            raise ManifestUriResolutionError(
                f"absolute manifest URI {text!r} (realpath {_realpath(p)}) is "
                f"outside all allowed roots {[str(r) for r in roots]}. Pass "
                f"allow_external=True only when the manifest schema explicitly "
                f"declares an external root."
            )
        return _enforce_digest(p, want, text)

    # Confine candidates to the allowed roots (lexical containment enforced).
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

    # Symlink-escape guard: enforce realpath containment on every existing
    # candidate BEFORE we digest or select one, so an in-root symlink can never
    # make the resolver read/digest a file outside the trusted roots.
    for c in existing:
        if not _realpath_within(roots, c):
            raise ManifestUriResolutionError(
                f"manifest URI {text!r} resolves through an in-root symlink to "
                f"{_realpath(c)}, outside all allowed roots "
                f"{[str(r) for r in roots]} (symlink escape rejected)."
            )

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
        return _enforce_digest(existing[0], want, text)

    # Nothing exists — fall back to the manifest-relative join so the
    # downstream not-found error names the expected location.
    return contained[0]
