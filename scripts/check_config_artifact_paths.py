#!/usr/bin/env python3
"""Pre-deploy static gate: every configured artifact path in the strategy-104
config PROFILES resolves (via the CANONICAL runtime resolver) to an existing
file that carries scorer/calibrator identity metadata.

Incident this closes
--------------------
A strategy-104 config carried a PatchTST checkpoint ``artifact_path`` of
``../../artifacts/patchtst_shadow/.../model.pt``. After the pinned-subrepo
migration that relative path resolved OUTSIDE the repo tree
(``RenQuant/../../artifacts`` = ``/Users/renhao/git/artifacts``), the file was
never found, and the scorer silently failed to load for a long time. Nothing
statically verified that configured artifact paths point at real, identified
artifacts before deploy, so no gate caught it.

Design
------
1. CANONICAL resolver — the gate does NOT re-implement resolution. It CALLS
   renquant-pipeline PR #211's canonical contract
   ``renquant_pipeline.kernel.panel_pipeline.shadow_health.resolve_artifact_identity``
   (which delegates path resolution to ``kernel.artifact_resolver`` — the one
   runtime authority: absolute -> strategy_dir -> repo_root — and stamps the
   immutable ``content_sha256`` scoring actually uses). Same resolver as
   runtime, so CI and live agree. It is a SINGLE injected dependency
   (:class:`ArtifactContract`); :func:`default_contract` imports the canonical
   module and HARD-FAILS if it is unavailable (no silent mirror fallback — the
   pinned pipeline must provide it).
2. ALL profiles — validate every config profile / manifest the scheduled
   live / shadow / A-B paths can load, from a DECLARED registry
   (``scripts/config_artifact_gate_registry.json``), not one hardcoded file.
   The shadow / A-B profiles carry the ``../../`` checkpoint as their PRIMARY
   scorer, so validating only the live config misses the escape there.
3. Identity + swap-detection, not just existence — FAIL on ``resolved==false``
   OR absent required identity. Scorers require ``trained_date`` +
   ``config_fingerprint`` (provenance) with the canonical ``content_sha256``
   as the swap-detection anchor; calibrators require ``trained_date``. When a
   profile entry carries a config-pinned ``expected_content_sha256`` /
   ``expected_config_fingerprint`` (#211), a mismatch FAILS. Metadata field
   extraction reuses the umbrella snapshot contract
   (``render_strategy_104_snapshot.py``) — sidecar-aware (``.pt`` ->
   ``.metadata.json``) — the single canonical source for those fields.
4. ``../../`` escape lint (deterministic incident-class guard) — a relative
   path with a ``..`` segment is a HARD error EVEN IF the canonical resolver
   resolves it. Ground truth: on a machine where ``<umbrella>/artifacts``
   exists, the resolver resolves the ``../../`` checkpoint (``source=
   strategy_dir``) — so ``resolved==false`` alone is topology-dependent and
   would let the fragile escape through. The lint makes the incident class
   deterministically fail everywhere.

Read-only. Touches only config / artifact JSON + ``Path.stat``/``read_bytes``
via the canonical resolver — never mutates state.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple, Optional

# ── snapshot metadata-field extractor (REUSED, stdlib-only) ─────────────────

_SNAPSHOT_SCRIPT = Path(__file__).resolve().parent / "render_strategy_104_snapshot.py"
_SNAP_CACHE: object | None = None


def _snapshot():
    """Load and cache the umbrella snapshot renderer — the canonical source of
    artifact metadata-field extraction (sidecar resolution ``.pt`` ->
    ``.metadata.json``, whitelist extraction). stdlib-only, so it imports
    cheaply on a bare CI runner."""
    global _SNAP_CACHE
    if _SNAP_CACHE is None:
        spec = importlib.util.spec_from_file_location(
            "_render_strategy_104_snapshot_identity", _SNAPSHOT_SCRIPT
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"cannot load snapshot metadata contract from {_SNAPSHOT_SCRIPT}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _SNAP_CACHE = module
    return _SNAP_CACHE


# ── canonical resolver+identity contract (#211) ─────────────────────────────


class CanonicalContractUnavailable(RuntimeError):
    """Raised when renquant-pipeline's #211 artifact-identity contract cannot
    be imported — the gate refuses to run rather than fall back to a
    drift-prone in-repo resolver."""


def _import_canonical():
    """Import #211's canonical resolver+identity. Returns
    ``(resolve_artifact_identity, norm_digest_or_None, backend_label)``.
    Raises :class:`CanonicalContractUnavailable`.

    #211 module: ``renquant_pipeline.kernel.panel_pipeline.shadow_health``.
    ``resolve_artifact_identity(ref, *, strategy_dir, repo_root=None)`` returns
    an ``ArtifactIdentity(ref, resolved, resolved_path, source, content_sha256,
    error)`` and never raises; it delegates resolution to
    ``kernel.artifact_resolver`` (the one runtime authority)."""
    try:
        from renquant_pipeline.kernel.panel_pipeline.shadow_health import (  # noqa: PLC0415
            resolve_artifact_identity,
        )
    except Exception as exc:  # noqa: BLE001 — any import failure -> hard stop
        raise CanonicalContractUnavailable(
            "cannot import renquant_pipeline.kernel.panel_pipeline.shadow_health"
            f".resolve_artifact_identity (#211 canonical contract): {exc}. "
            "Install / point at a renquant-pipeline pin that contains #211."
        ) from exc
    try:
        from renquant_pipeline.kernel.panel_pipeline.shadow_health import (  # noqa: PLC0415
            _norm_digest,
        )
    except Exception:  # noqa: BLE001 — optional pin-compare helper
        _norm_digest = None

    # Adapt #211's keyword-only signature
    # ``resolve_artifact_identity(ref, *, strategy_dir, repo_root=None)`` to the
    # contract's positional ``(ref, strategy_dir, data_root)`` convention.
    def resolve_identity(ref, strategy_dir, data_root):
        return resolve_artifact_identity(
            ref, strategy_dir=strategy_dir, repo_root=data_root
        )

    backend = (
        "renquant_pipeline.kernel.panel_pipeline.shadow_health."
        "resolve_artifact_identity (#211 canonical)"
    )
    return resolve_identity, _norm_digest, backend


@dataclass(frozen=True)
class ArtifactContract:
    """The single injected dependency: the canonical resolver+identity function
    (#211) plus its digest-normalizer for pin comparison. Inject a faithful
    fake in tests; ``default_contract()`` binds the real #211 module."""

    resolve_identity: Callable[[str, Path, Path], object]
    norm_digest: Optional[Callable[[object], Optional[str]]]
    backend: str


def default_contract() -> ArtifactContract:
    resolve_identity, norm_digest, backend = _import_canonical()
    return ArtifactContract(
        resolve_identity=resolve_identity, norm_digest=norm_digest, backend=backend
    )


def _norm(contract: ArtifactContract, value: object) -> Optional[str]:
    """Normalize a content digest for pin comparison — the canonical
    ``_norm_digest`` when available (parity with runtime), else its recipe:
    strip an optional ``sha256:`` prefix and lowercase."""
    if contract.norm_digest is not None:
        return contract.norm_digest(value)
    if not value:
        return None
    return str(value).split(":", 1)[-1].strip().lower()


def escapes_repo(raw: str) -> bool:
    """True when a relative artifact path climbs out with a ``..`` segment.

    A deterministic incident-class guard, INDEPENDENT of whether the path
    happens to resolve on this machine. Ground truth: the canonical resolver
    resolves ``../../artifacts/...`` against ``strategy_dir`` on any machine
    where ``<umbrella>/artifacts`` exists — so ``resolved==false`` is
    topology-dependent and would let the fragile escape through. Configured
    artifact paths must be forward repo-relative (``artifacts/prod/...``) or
    absolute. Judged lexically — a symlink must not mask it."""
    p = Path(raw)
    if p.is_absolute():
        return False
    return ".." in p.parts


# ── metadata identity (snapshot-based field extraction) ──────────────────────


def _metadata_identity(
    resolved_path: Path, kind: str
) -> tuple[Optional[str], Optional[str], list[str]]:
    """Return ``(trained_date, config_fingerprint, missing)`` for a resolved
    scorer / calibrator artifact via the snapshot metadata contract (inline for
    JSON, ``<path>.metadata.json`` sidecar for ``.pt``). ``missing`` lists the
    required identity fields that are absent (``metadata_missing`` marks an
    unloadable metadata file)."""
    snap = _snapshot()
    meta_file = snap._metadata_file_for(resolved_path)
    doc = snap._load_json(meta_file)
    if not doc:
        return None, None, ["metadata_missing"]
    if kind == "calibrator":
        allowed = snap._extract_allowed_calibrator(doc)
        trained = allowed.get("trained_date") or doc.get("trained_date")
        return trained, None, [] if trained else ["trained_date"]
    # scorer (primary / shadow)
    allowed = snap._extract_allowed(doc)
    trained = allowed.get("trained_date")
    fingerprint = (
        allowed.get("config_fingerprint")
        or doc.get("scorer_model_content_fingerprint")
        or (doc.get("metadata") or {}).get("scorer_model_content_fingerprint")
    )
    missing = []
    if not trained:
        missing.append("trained_date")
    if not fingerprint:
        missing.append("config_fingerprint")
    return trained, fingerprint, missing


# ── extraction (per profile shape) ───────────────────────────────────────────


def _expected(entry: dict) -> dict:
    """Optional config-pinned expected identity for a profile entry (#211)."""
    return {
        "content_sha256": entry.get("expected_content_sha256"),
        "config_fingerprint": entry.get("expected_config_fingerprint"),
    }


def collect_paths_strategy_config(
    config: dict,
) -> list[tuple[str, str, str, dict]]:
    """``(dotted_field, kind, raw, expected)`` for every validated artifact
    path in a kernel strategy config. Absent fields skipped."""
    out: list[tuple[str, str, str, dict]] = []
    panel_scoring = (config.get("ranking") or {}).get("panel_scoring") or {}

    v = panel_scoring.get("artifact_path")
    if isinstance(v, str) and v:
        out.append(
            ("ranking.panel_scoring.artifact_path", "primary", v, _expected(panel_scoring))
        )

    # Composite blend scorer (pipeline#218): kind="blend" carries a
    # components[] list of TWO scorer legs, each with its own artifact path
    # and pinned identity. The activated profile must prove BOTH legs
    # resolve and match their pins (codex ruling on RenQuant#536) — the
    # top-level artifact_path only anchors component 0.
    for i, comp in enumerate(panel_scoring.get("components") or []):
        if not isinstance(comp, dict):
            continue
        cv = comp.get("artifact_path")
        if isinstance(cv, str) and cv:
            out.append(
                (f"ranking.panel_scoring.components[{i}].artifact_path",
                 "primary", cv, _expected(comp))
            )

    panel_ltr = config.get("panel_ltr") or {}
    v = panel_ltr.get("artifact_path")
    if isinstance(v, str) and v:
        out.append(("panel_ltr.artifact_path", "primary", v, _expected(panel_ltr)))

    global_calibration = panel_scoring.get("global_calibration") or {}
    v = global_calibration.get("artifact_path")
    if isinstance(v, str) and v:
        out.append(
            (
                "ranking.panel_scoring.global_calibration.artifact_path",
                "calibrator",
                v,
                _expected(global_calibration),
            )
        )

    for i, sm in enumerate(panel_scoring.get("shadow_models") or []):
        if not isinstance(sm, dict):
            continue
        v = sm.get("artifact_path")
        if isinstance(v, str) and v:
            out.append(
                (
                    f"ranking.panel_scoring.shadow_models[{i}].artifact_path",
                    "shadow",
                    v,
                    _expected(sm),
                )
            )
    return out


def collect_paths_artifact_manifest(
    config: dict,
) -> list[tuple[str, str, str, dict]]:
    """``(dotted_field, kind, raw, expected)`` for every validated artifact
    path in an operator artifact manifest (xgb_prod_artifact_manifest.json)."""
    out: list[tuple[str, str, str, dict]] = []
    prod = config.get("production_primary") or {}

    v = prod.get("artifact_path")
    if isinstance(v, str) and v:
        out.append(("production_primary.artifact_path", "primary", v, _expected(prod)))

    gc = prod.get("global_calibration") or {}
    v = gc.get("artifact_path")
    if isinstance(v, str) and v:
        out.append(
            (
                "production_primary.global_calibration.artifact_path",
                "calibrator",
                v,
                _expected(gc),
            )
        )

    rs = config.get("readonly_shadow") or {}
    v = rs.get("artifact_path")
    if isinstance(v, str) and v:
        out.append(("readonly_shadow.artifact_path", "shadow", v, _expected(rs)))
    return out


COLLECTORS: dict[str, Callable[[dict], list[tuple[str, str, str, dict]]]] = {
    "strategy_config": collect_paths_strategy_config,
    "artifact_manifest": collect_paths_artifact_manifest,
}


# ── checking ─────────────────────────────────────────────────────────────────


class PathCheck(NamedTuple):
    config: str
    field: str
    kind: str  # "primary" | "calibrator" | "shadow" | "profile"
    raw: str
    resolved: str
    ok: bool
    reason: str  # "" when ok, else an actionable failure message
    detail: str  # identity summary / provenance when ok


def _check_one(
    config_name: str,
    field: str,
    kind: str,
    raw: str,
    expected: dict,
    strategy_dir: Path,
    data_root: Path,
    contract: ArtifactContract,
) -> PathCheck:
    # (4) HARD incident-class guard first: a stray ``..`` escape (independent of
    # whether the canonical resolver resolves it on this machine).
    if escapes_repo(raw):
        return PathCheck(
            config_name, field, kind, raw, "", False,
            (
                f"path escapes the repo (contains '..'): {raw!r} — configured "
                f"artifact paths must be repo-relative (e.g. 'artifacts/prod/"
                f"model.json') or absolute; a '../../' resolves to a different "
                f"file (or nothing) per machine topology — the exact class that "
                f"silently killed the scorer"
            ),
            "",
        )

    # (1) canonical resolution + immutable content_sha256.
    ident = contract.resolve_identity(raw, strategy_dir, data_root)
    if not getattr(ident, "resolved", False):
        return PathCheck(
            config_name, field, kind, raw,
            str(getattr(ident, "resolved_path", "") or ""), False,
            (
                f"does not resolve to an existing file: {raw!r} "
                f"(canonical resolver source={getattr(ident, 'source', '?')}; "
                f"error={getattr(ident, 'error', None)})"
            ),
            "",
        )

    resolved_path = Path(ident.resolved_path)
    content_sha = getattr(ident, "content_sha256", None)

    # (3) identity — a resolvable file is insufficient. FAIL CLOSED on missing
    # required metadata.
    trained, fingerprint, missing = _metadata_identity(resolved_path, kind)
    if "metadata_missing" in missing:
        return PathCheck(
            config_name, field, kind, raw, str(resolved_path), False,
            (
                f"{kind} at {resolved_path} carries no loadable identity "
                f"metadata — a resolvable file is insufficient"
            ),
            "",
        )
    if missing:
        return PathCheck(
            config_name, field, kind, raw, str(resolved_path), False,
            (
                f"{kind} metadata at {resolved_path} missing required identity "
                f"field(s): {', '.join(missing)}"
            ),
            "",
        )
    if not content_sha:
        return PathCheck(
            config_name, field, kind, raw, str(resolved_path), False,
            (
                f"{kind} at {resolved_path} has no content_sha256 "
                f"(swap-detection anchor absent)"
            ),
            "",
        )

    # config-pinned expected identity (#211): a swapped/wrong artifact -> FAIL.
    exp_content = expected.get("content_sha256")
    if exp_content and _norm(contract, exp_content) != _norm(contract, content_sha):
        return PathCheck(
            config_name, field, kind, raw, str(resolved_path), False,
            (
                f"content_sha256 mismatch vs pinned expected_content_sha256: "
                f"got {content_sha}, expected {exp_content}"
            ),
            "",
        )
    exp_fp = expected.get("config_fingerprint")
    if exp_fp and fingerprint and str(exp_fp).strip() != str(fingerprint).strip():
        return PathCheck(
            config_name, field, kind, raw, str(resolved_path), False,
            (
                f"config_fingerprint mismatch vs pinned "
                f"expected_config_fingerprint: got {fingerprint}, expected "
                f"{exp_fp}"
            ),
            "",
        )

    detail = f"source={ident.source} content={content_sha} trained_date={trained}"
    if fingerprint:
        detail = f"{detail} fp={fingerprint}"
    return PathCheck(
        config_name, field, kind, raw, str(resolved_path), True, "", detail
    )


def check_config(
    config_path: Path,
    shape: str,
    strategy_dir: Path,
    data_root: Path,
    contract: ArtifactContract | None = None,
) -> list[PathCheck]:
    """Check every validated artifact path in ``config_path`` (of ``shape``)."""
    contract = contract or default_contract()
    if shape not in COLLECTORS:
        raise ValueError(
            f"unknown profile shape {shape!r}; known: {sorted(COLLECTORS)}"
        )
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    results: list[PathCheck] = []
    for field, kind, raw, expected in COLLECTORS[shape](config):
        results.append(
            _check_one(
                str(config_path), field, kind, raw, expected,
                strategy_dir, data_root, contract,
            )
        )
    return results


class RegistryRun(NamedTuple):
    results: list[PathCheck]
    validated_profiles: list[str]
    skipped_profiles: list[str]  # optional profiles absent from the checkout


def load_registry(registry_path: Path) -> list[dict]:
    doc = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    profiles = doc.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError(f"registry {registry_path} declares no profiles")
    return profiles


def check_registry(
    registry_path: Path,
    configs_dir: Path,
    strategy_dir: Path,
    data_root: Path,
    contract: ArtifactContract | None = None,
) -> RegistryRun:
    """Validate every profile the registry declares. A ``required`` profile
    absent from ``configs_dir`` FAILS; an optional one is skipped (reported)."""
    contract = contract or default_contract()
    results: list[PathCheck] = []
    validated: list[str] = []
    skipped: list[str] = []
    for prof in load_registry(registry_path):
        fname = prof["file"]
        shape = prof["shape"]
        config_path = configs_dir / fname
        if not config_path.exists():
            if prof.get("required", False):
                results.append(
                    PathCheck(
                        str(config_path), "<profile>", "profile", fname, "", False,
                        (
                            f"required profile missing from checkout: "
                            f"{config_path} (loaded_by: {prof.get('loaded_by', '?')})"
                        ),
                        "",
                    )
                )
            else:
                skipped.append(f"{fname} (optional; absent)")
            continue
        validated.append(f"{fname} [{shape}] — {prof.get('loaded_by', '?')}")
        results.extend(
            check_config(config_path, shape, strategy_dir, data_root, contract)
        )
    return RegistryRun(results, validated, skipped)


# ── CLI ──────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Static pre-deploy gate: verify every configured strategy-104 "
            "artifact path (across ALL declared profiles) resolves — via the "
            "canonical renquant-pipeline #211 resolver — to an existing, "
            "identified file. Fails on any unresolvable path, '../../' repo "
            "escape, missing identity metadata, or pinned-identity mismatch."
        )
    )
    parser.add_argument(
        "configs", nargs="*", type=Path,
        help="explicit config file(s) to check (ad-hoc mode; use --shape).",
    )
    parser.add_argument(
        "--registry", type=Path,
        help="declared profile registry JSON (registry mode; with --configs-dir).",
    )
    parser.add_argument(
        "--configs-dir", type=Path,
        help="directory holding the registry's profile files (registry mode).",
    )
    parser.add_argument(
        "--shape", choices=sorted(COLLECTORS), default="strategy_config",
        help="profile shape for explicit --configs (default: strategy_config).",
    )
    parser.add_argument(
        "--strategy-dir", type=Path, required=True,
        help=(
            "runtime _strategy_dir passed to the canonical resolver "
            "(e.g. backtesting/renquant_104)."
        ),
    )
    parser.add_argument(
        "--data-root", type=Path, required=True,
        help="runtime repo_root passed to the canonical resolver (umbrella root).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    strategy_dir = args.strategy_dir.resolve()
    data_root = args.data_root.resolve()

    try:
        contract = default_contract()
    except CanonicalContractUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    results: list[PathCheck] = []
    validated: list[str] = []
    skipped: list[str] = []

    if args.registry is not None:
        if args.configs_dir is None:
            print("ERROR: --registry requires --configs-dir", file=sys.stderr)
            return 2
        if not args.registry.exists():
            print(f"ERROR: registry not found: {args.registry}", file=sys.stderr)
            return 2
        run = check_registry(
            args.registry, args.configs_dir, strategy_dir, data_root, contract
        )
        results, validated, skipped = run
    elif args.configs:
        for config_path in args.configs:
            if not config_path.exists():
                print(f"ERROR: config not found: {config_path}", file=sys.stderr)
                return 2
            validated.append(f"{config_path.name} [{args.shape}]")
            results.extend(
                check_config(config_path, args.shape, strategy_dir, data_root, contract)
            )
    else:
        print(
            "ERROR: pass --registry --configs-dir, or explicit config file(s)",
            file=sys.stderr,
        )
        return 2

    failures = [r for r in results if not r.ok]
    checked = len(results)

    print(f"check_config_artifact_paths: resolver backend = {contract.backend}")
    print(f"  strategy_dir={strategy_dir}")
    print(f"  data_root(repo_root)={data_root}")
    print(f"  profiles validated ({len(validated)}):")
    for v in validated:
        print(f"    - {v}")
    for s in skipped:
        print(f"    - SKIP {s}")
    print(f"  {checked} artifact path(s) checked:")
    for r in results:
        status = "OK  " if r.ok else "FAIL"
        loc = f"{Path(r.config).name}:{r.field}"
        if r.ok:
            extra = f" [{r.detail}]" if r.detail else ""
            print(f"    [{status}] {loc} [{r.kind}] -> {r.resolved}{extra}")
        else:
            print(f"    [{status}] {loc} [{r.kind}]: {r.reason}")

    if failures:
        print(
            f"\nFAILED: {len(failures)} of {checked} configured artifact "
            f"path(s) did not pass. Fix the profile(s) at the candidate pin "
            f"(or the artifact tree) before deploying.",
            file=sys.stderr,
        )
        return 1

    print(f"\nOK: all {checked} configured artifact path(s) resolve + identify.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
