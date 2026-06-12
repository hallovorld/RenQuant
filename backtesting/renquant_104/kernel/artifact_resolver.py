"""ArtifactResolver — the ONE artifact path-resolution authority (umbrella copy).

Mirror of renquant-pipeline ``kernel/artifact_resolver.py`` (pipeline PR
#115); retires with the umbrella per the strangler-fig plan. Keep the two
copies byte-similar — divergence between them is exactly the bug class
this module exists to kill.

Design: renquant-orchestrator
doc/research/2026-06-12-engineering-architecture-deep-plan.md §III.5
"Artifact plane" (S1-PR5); prototype:
scripts/engineering/artifact_resolver_prototype.py (orchestrator PR #112).

Why one authority (incident #2, pipeline PR #114): the primary scorer
resolved config refs strategy_dir-first while the shadow scorer resolved
repo-root-first. The same ref string pointed at two different files, and
the shadow artifact was silently dead for a week. The umbrella had FOUR
ad-hoc resolvers before this module (artifact_contract._resolve_path's
prefix heuristic, LoadScorerTask._resolve_artifact_path,
the global-calibration closure, preflight._resolve_artifact_path).

Resolution order (fixed, never per-caller):
  1. absolute ref           → itself
  2. strategy_dir / ref     → the strategy bundle owns its artifacts
  3. repo_root / ref        → umbrella-level fallback (legacy layouts)

Two entry points for the two legitimate consumer shapes:
  * ``resolve_artifact``  — fail-closed loader path: raises
    FileNotFoundError listing every candidate tried, and returns a sha256
    prefix that feeds the run fingerprint (DRPH, eng plan §IV.2).
  * ``locate_artifact``   — preflight/check path: never raises; returns
    the first existing candidate, or the strategy_dir candidate when
    nothing exists so the caller can report a precise missing path.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import NamedTuple


class ResolvedArtifact(NamedTuple):
    path: Path
    sha256: str          # first 16 hex chars — feeds the run fingerprint
    source: str          # "absolute" | "strategy_dir" | "repo_root"
    ref: str


def default_repo_root(strategy_dir: Path) -> Path:
    """Umbrella repo root from the strategy dir.

    Same convention as persistence._db_path: strategy_dir is
    ``<repo_root>/backtesting/renquant_104`` → two levels up.
    """
    return Path(strategy_dir).parent.parent


def _candidates(
    ref: str | Path, strategy_dir: Path, repo_root: Path | None,
) -> list[tuple[Path, str]]:
    p = Path(ref)
    if p.is_absolute():
        return [(p, "absolute")]
    root = repo_root if repo_root is not None else default_repo_root(strategy_dir)
    return [(Path(strategy_dir) / p, "strategy_dir"), (root / p, "repo_root")]


def resolve_artifact(
    ref: str | Path,
    *,
    strategy_dir: Path,
    repo_root: Path | None = None,
) -> ResolvedArtifact:
    """Resolve a config artifact ref to exactly one existing file, fail-closed.

    Every artifact LOAD (primary / shadow / calibrator / gmm / gate
    metadata) must go through here. Missing artifact = FileNotFoundError
    naming every candidate tried — never a silent fallback.
    """
    tried: list[str] = []
    for cand, source in _candidates(ref, strategy_dir, repo_root):
        cand = cand.resolve()
        tried.append(str(cand))
        if cand.is_file():
            digest = hashlib.sha256(cand.read_bytes()).hexdigest()[:16]
            return ResolvedArtifact(cand, digest, source, str(ref))
    raise FileNotFoundError(
        f"artifact unresolvable (fail-closed): {str(ref)!r} — tried {tried}"
    )


def locate_artifact(
    ref: str | Path,
    *,
    strategy_dir: Path,
    repo_root: Path | None = None,
) -> Path:
    """Best-effort variant for preflight checks: first existing candidate,
    else the strategy_dir candidate (so 'missing artifact' findings report
    the canonical expected location). Never raises."""
    cands = _candidates(ref, strategy_dir, repo_root)
    for cand, _source in cands:
        if cand.exists():
            return cand
    return cands[0][0]
