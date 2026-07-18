"""AC4 migration P0 invariants (census doc/design/2026-07-18-ac4-migration-census.md
§6 P0: "build, no live change"; RFC #492 §3 rollback invariant).

Three pins:

1. the DECLARED bundle-store location (`deploy/bundle_store_location.json`)
   is well-formed and points at the flat pair's own directory (the RFC
   §2.1 "alongside the flat files" layout), which still contains the pair
   — the flat pair stays authoritative in P0;
2. the store paths can never enter the git index (`.gitignore` carries
   the three store patterns — pre-empting census blocker B1);
3. ZERO serving change / revert-cleanliness: no serving-surface code
   (kernel, 104 kernel/adapters/training_panel, scripts, dagster) refers
   to the bundle store — reverting the P0 commit therefore restores the
   previous serving behavior with no artifact surgery, because nothing
   that serves ever learned the store exists.

File-based only: no git subprocesses, no network, no store creation.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DECLARATION = REPO / "deploy" / "bundle_store_location.json"

#: The census §2.1 flat-pair members (serving stays on these in P0).
FLAT_PAIR = ("panel-ltr.alpha158_fund.json", "panel-rank-calibration.json")

#: Store path fragments / module names that must NOT appear on any
#: serving surface in P0 (reader/writer redirection is P1+ per census §6).
FORBIDDEN_MARKERS = (
    "prod/bundles",
    "prod/ACTIVE",
    "renquant_artifacts.bundle",
    "bundle_store_location",
    "bundle_breakglass",
    "bundle_store_init",
)

#: Serving/ops surfaces per census §2/§3 (umbrella side).
SERVING_DIRS = (
    "kernel",
    "backtesting/renquant_104/kernel",
    "backtesting/renquant_104/adapters",
    "backtesting/renquant_104/training_panel",
    "scripts",
    "dagster_renquant",
)

GITIGNORE_PATTERNS = (
    "backtesting/*/artifacts/prod/bundles/",
    "backtesting/*/artifacts/prod/ACTIVE",
    "backtesting/*/artifacts/prod/ACTIVE.tmp",
)


def test_declaration_is_wellformed_and_points_at_the_pair_directory() -> None:
    payload = json.loads(DECLARATION.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    store_root = payload["store_root"]
    assert store_root == "backtesting/renquant_104/artifacts/prod"
    unknown = [
        k for k in payload if k not in {"schema_version", "store_root"}
        and not k.startswith("_")
    ]
    assert not unknown, f"declaration schema v1 allows no extra fields: {unknown}"

    prod = REPO / store_root
    assert prod.is_dir()
    for member in FLAT_PAIR:  # the flat pair stays authoritative in P0
        assert (prod / member).is_file(), (
            f"flat pair member {member} missing from {prod} — P0 must not "
            "move or remove the served flat files"
        )


def test_store_paths_are_gitignored_never_indexed() -> None:
    ignore_lines = {
        line.strip()
        for line in (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    for pattern in GITIGNORE_PATTERNS:
        assert pattern in ignore_lines, (
            f".gitignore must carry {pattern!r}: a git-tracked bundle store "
            "recreates census blocker B1 (git as an unmediated pair writer)"
        )


def test_no_serving_surface_references_the_bundle_store() -> None:
    offenders: list[str] = []
    for rel in SERVING_DIRS:
        base = REPO / rel
        if not base.is_dir():
            continue
        for suffix in ("*.py", "*.sh"):
            for path in base.rglob(suffix):
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for marker in FORBIDDEN_MARKERS:
                    if marker in text:
                        offenders.append(f"{path.relative_to(REPO)}: {marker}")
    assert not offenders, (
        "P0 forbids any reader/writer redirection — serving surfaces "
        "referencing the bundle store:\n" + "\n".join(sorted(offenders))
    )
