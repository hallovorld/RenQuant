"""The served flat pair is live-mutated run-surface state, outside git's write authority.

`weekly_wf_promote.sh` (Step 5 / Step 4b via `fallback_pair_promote.py`),
`manual_promote.sh` and `monthly_calibrator_refresh.sh` `os.replace()` the
served scorer + calibrator on every promotion, so the git-tracked copies were
permanently "modified" on the live tree (both since the 2026-08-31 promotion)
and any live-tree pull either refuses on them or — when the working copy
happens to equal the last commit — resets them. The 2026-08-31 07:17 pull did
exactly that to the git-tracked momentum ledger (RenQuant#638); this PR
removes the same hazard from the pair the live book actually trades on.

Three pins, mirroring tests/test_live_momentum_ledgers_untracked.py:
1. the index does NOT contain either pair member (`git ls-files` empty — the
   load-bearing check; `.gitignore` neither evicts tracked entries nor stops
   `git add -f`);
2. `.gitignore` carries the pair patterns AND `git check-ignore` confirms the
   exact live paths — and the promotion side-files the scripts write beside
   them — resolve as ignored, so a promotion leaves the tree clean;
3. the deployment declaration `deploy/live_mutated_prod_artifacts.json` names
   exactly the two pair members, so the config-artifact-path gate's INFO
   waiver covers precisely what was untracked and nothing else.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DECLARATION = REPO / "deploy" / "live_mutated_prod_artifacts.json"

PAIR = (
    "backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json",
    "backtesting/renquant_104/artifacts/prod/panel-rank-calibration.json",
)
GITIGNORE_PATTERNS = (
    "backtesting/*/artifacts/prod/panel-ltr.alpha158_fund.json",
    "backtesting/*/artifacts/prod/panel-rank-calibration.json",
)
#: What the promotion scripts write beside the pair (observed on the live tree
#: 2026-09-03: .previous.json, weekly_*.staging.json, weekly_rollback_*.json,
#: .json.accepted_receipt-*, .json.bak-*, monthly_rollback_*, _rejected_calibrators/).
SIDE_FILES = (
    "backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.previous.json",
    "backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.weekly_20260903T201006Z.staging.json",
    "backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.weekly_rollback_2026-09-03.json",
    "backtesting/renquant_104/artifacts/prod/panel-rank-calibration.json.bak-20260714-143011",
    "backtesting/renquant_104/artifacts/prod/panel-rank-calibration.weekly_20260903T201006Z.staging.json",
    "backtesting/renquant_104/artifacts/prod/panel-rank-calibration.monthly_rollback_2026-09-01.json",
    "backtesting/renquant_104/artifacts/prod/_rejected_calibrators/anything.json",
)
#: Frozen, consumed-only calibrator variants that stay TRACKED — the patterns
#: must not swallow them.
TRACKED_SIBLINGS = (
    "backtesting/renquant_104/artifacts/prod/panel-rank-calibration.bull_calm.json",
    "backtesting/renquant_104/artifacts/prod/panel-rank-calibration.recent-12mo.json",
    "backtesting/renquant_104/artifacts/prod/panel-rank-calibration.pre-2026-05-15-clip.json",
)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def test_served_pair_absent_from_git_index() -> None:
    out = _git("ls-files", "--", *PAIR)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", (
        "the live-mutated served pair is git-tracked — a live-tree pull can "
        "refuse on or reset it (2026-08-31 class):\n" + out.stdout
    )


def test_gitignore_carries_the_pair_patterns() -> None:
    ignore_lines = {
        line.strip()
        for line in (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    for pattern in GITIGNORE_PATTERNS:
        assert pattern in ignore_lines, f".gitignore must carry {pattern!r}"


def test_exact_live_paths_and_promotion_side_files_resolve_as_ignored() -> None:
    paths = PAIR + SIDE_FILES
    out = _git("check-ignore", "-v", "--", *paths)
    assert out.returncode == 0, out.stdout + out.stderr
    matched = {line.split("\t")[-1] for line in out.stdout.splitlines()}
    assert set(paths) <= matched, f"not ignored: {set(paths) - matched}"


def test_frozen_tracked_calibrator_variants_stay_tracked_and_unignored() -> None:
    tracked = set(_git("ls-files", "--", *TRACKED_SIBLINGS).stdout.split())
    assert tracked == set(TRACKED_SIBLINGS), f"tracked siblings changed: {tracked}"
    out = _git("check-ignore", "-v", "--no-index", "--", *TRACKED_SIBLINGS)
    # exit 1 == none of the paths is ignored
    assert out.returncode == 1, "a pair pattern swallows a frozen tracked sibling:\n" + out.stdout


def test_declaration_names_exactly_the_untracked_pair() -> None:
    payload = json.loads(DECLARATION.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    strategy_dir = payload["strategy_dir"]
    declared = {f"{strategy_dir}/{a['path']}" for a in payload["artifacts"]}
    assert declared == set(PAIR), declared
    for a in payload["artifacts"]:
        assert a["writers"], f"{a['path']}: a live-mutated artifact must name its writers"
