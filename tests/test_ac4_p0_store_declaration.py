"""AC4 migration P0 invariants (census doc/design/2026-07-18-ac4-migration-census.md
§6 P0: "build, no live change"; RFC #492 §3 rollback invariant).

Three pins:

1. the DECLARED bundle-store location (`deploy/bundle_store_location.json`)
   is well-formed and points at the flat pair's own directory (the RFC
   §2.1 "alongside the flat files" layout), which still contains the pair
   — the flat pair stays authoritative in P0;
2. the store paths are ABSENT FROM THE GIT INDEX (the load-bearing
   check: `git ls-files` on the store paths must be empty — `.gitignore`
   neither evicts already-tracked entries nor stops `git add -f`, so the
   ignore patterns are ergonomics, not proof) and `.gitignore` carries
   the three store patterns — together pre-empting census blocker B1;
3. ZERO serving change / revert-cleanliness: no serving-surface code
   (kernel, 104 kernel/adapters/training_panel, scripts, dagster) refers
   to the bundle store — reverting the P0 commit therefore restores the
   previous serving behavior with no artifact surgery, because nothing
   that serves ever learned the store exists.

No network, no store creation. One read-only git subprocess (`ls-files`)
for the index assertion; everything else file-based.
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
    # 2026-09-03: the flat pair is live-mutated run-surface state and no
    # longer git-tracked (deploy/live_mutated_prod_artifacts.json). On the
    # serving machine it is present at exactly this path; in a fresh checkout
    # it is absent BY DESIGN and must be DECLARED there instead. P0's
    # invariant — serving stays on the flat pair, at this directory — is
    # pinned either way; what changed is who owns the bytes (the promote
    # jobs, not git).
    declared = {
        Path(a["path"]).name
        for a in json.loads(
            (REPO / "deploy" / "live_mutated_prod_artifacts.json").read_text(encoding="utf-8")
        )["artifacts"]
    }
    for member in FLAT_PAIR:  # the flat pair stays authoritative in P0
        if (prod / member).exists():
            assert (prod / member).is_file(), f"flat pair member {member} is not a file"
        else:
            assert member in declared, (
                f"flat pair member {member} missing from {prod} and not declared "
                "live-mutated — P0 must not move or remove the served flat files"
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


def test_store_paths_absent_from_git_index() -> None:
    """The load-bearing B1 pre-emption: no store path is git-tracked.

    `.gitignore` cannot evict already-tracked entries and does not stop
    `git add -f`; only the index itself proves the store is outside
    git's write authority. Regression guard: if any store path ever
    lands in the index, this fails loudly.
    """
    import subprocess

    store_paths = [
        "backtesting/renquant_104/artifacts/prod/bundles",
        "backtesting/renquant_104/artifacts/prod/ACTIVE",
        "backtesting/renquant_104/artifacts/prod/ACTIVE.tmp",
    ]
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "--", *store_paths],
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = [line for line in out.stdout.splitlines() if line.strip()]
    assert not tracked, (
        "bundle-store paths are git-tracked (census blocker B1 — git as an "
        "unmediated writer):\n" + "\n".join(tracked)
    )
