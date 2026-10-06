"""The live-mutated momentum shadow ledgers are outside git's write authority.

2026-08-31 07:17: the live-tree `git pull --ff-only` reset the git-tracked
`artifacts/momentum/momentum_artifact_ledger.jsonl` to its committed 1-row
version and wiped the 08-08/15/22/29 refit rows the Saturday
`momentum-train-weekly` job had appended (file mtime 07:17:50 vs reflog
07:17:54); the UNTRACKED `momentum_fast` ledger kept all six rows. RenQuant#638.

Two pins, mirroring tests/test_ac4_p0_store_declaration.py:
1. the index does NOT contain any path under the two momentum artifact
   directories (`git ls-files` empty — the load-bearing check; `.gitignore`
   neither evicts tracked entries nor stops `git add -f`);
2. `.gitignore` carries the two directory patterns AND `git check-ignore`
   confirms the exact ledger + dated-artifact paths resolve as ignored, so a
   writer recreating them leaves the tree clean rather than dirty.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

GITIGNORE_PATTERNS = (
    "backtesting/*/artifacts/momentum/",
    "backtesting/*/artifacts/momentum_fast/",
)
LIVE_PATHS = (
    "backtesting/renquant_104/artifacts/momentum/momentum_artifact_ledger.jsonl",
    "backtesting/renquant_104/artifacts/momentum/2026-08-02/momentum_residual_v0.json",
    "backtesting/renquant_104/artifacts/momentum/2026-08-29/anything.json",
    "backtesting/renquant_104/artifacts/momentum_fast/momentum_artifact_ledger.jsonl",
    "backtesting/renquant_104/artifacts/momentum_fast/2026-08-29/anything.json",
)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def test_momentum_artifact_dirs_absent_from_git_index() -> None:
    out = _git("ls-files", "--",
               "backtesting/renquant_104/artifacts/momentum",
               "backtesting/renquant_104/artifacts/momentum_fast")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", (
        "live-mutated momentum artifacts are git-tracked — a live-tree pull can "
        "reset them (2026-08-31 incident):\n" + out.stdout
    )


def test_gitignore_carries_the_two_momentum_patterns() -> None:
    ignore_lines = {
        line.strip()
        for line in (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    for pattern in GITIGNORE_PATTERNS:
        assert pattern in ignore_lines, f".gitignore must carry {pattern!r}"


def test_exact_live_paths_resolve_as_ignored() -> None:
    """`git check-ignore` on the exact paths the live jobs write (present or
    not — check-ignore resolves patterns, not files): a recreated ledger or a
    new dated refit dir must never show up as an untracked change."""
    out = _git("check-ignore", "-v", "--", *LIVE_PATHS)
    assert out.returncode == 0, out.stdout + out.stderr
    matched = {line.split("\t")[-1] for line in out.stdout.splitlines()}
    assert set(LIVE_PATHS) <= matched, f"not ignored: {set(LIVE_PATHS) - matched}"
