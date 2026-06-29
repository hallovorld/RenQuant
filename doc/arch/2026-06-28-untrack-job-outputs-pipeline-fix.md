# Untrack job-written outputs: a clobber-proof live tree

Date: 2026-06-28
Status: implemented (this PR) + flagged follow-ups
Related: PR #415 (the live-state NameError fix, already on `main`)

## Root cause

The live umbrella checkout is **perpetually dirty** — 166 uncommitted paths on
a quiet day. The dirt is not human edits; it is **job-written runtime output
that happens to be git-tracked**. Every daily/weekly run rewrites these tracked
files in place, so `git status` is never clean.

A perpetually-dirty tree is not a cosmetic annoyance — it is a **production
hazard**. Any routine git operation on the tree (`git checkout`, `git reset
--hard`, `git stash`, a botched `git pull`) silently overwrites or discards the
job-written state, because git treats those files as version-controlled content
it is free to restore to `HEAD`. This exact pattern already caused a multi-day
trading outage: a recovery `git reset/checkout` reverted live state and code
hotfixes back to a committed-buggy snapshot.

The fix is structural: **job outputs must not be git-tracked.** A clean tree is
a safety property — when the only dirty paths are untracked runtime files, git
ops can no longer clobber live state.

## The specific .gitignore bug (live state)

`.gitignore` had:

```
backtesting/*/live_state.json
```

Live trading state is written to **broker-suffixed** paths
(`live_state.alpaca.json`, `live_state.alpaca_shadow.json`) by
`adapters/runner.py` via `kernel/state_paths.py::live_state_path`. The bare
`live_state.json` pattern does **not** match the suffixed names, so the two
live-state files stayed tracked and dirty on every run — directly in the
clobber blast radius. (The legacy `live_state.json` was correctly ignored and
is not tracked.)

## What this PR untracks (and why each is safe)

Each was verified read-only against the umbrella sources and the pinned subrepo
runtime sources (`.subrepo_runtime/repos/*`) to confirm **no consumer reads it
from the committed git path**. These are also exactly the paths the existing
`scripts/check_ops_deployment_ready.py::RUNTIME_DIRTY_PATTERNS` already
classifies as non-blocking runtime output — untracking is the durable form of
that classification.

| Path | Producer | Consumed from committed path? | Action |
| --- | --- | --- | --- |
| `backtesting/renquant_104/live_state.alpaca.json` | `adapters/runner.py` (every run) | No. Only `tests/test_live_state_v2.py::test_roundtrip_byte_identical_real_committed_snapshot` read it — now guarded to skip when absent; the lossless contract is still proved unconditionally against `REPRESENTATIVE_STATE`. | `git rm --cached` + ignore `backtesting/*/live_state.*.json` |
| `backtesting/renquant_104/live_state.alpaca_shadow.json` | shadow run | No. | same |
| `doc/dashboard.md` | `scripts/build_dashboard.py` (regenerated every run; described in `doc/STATUS.md` as an "auto-generated per-build artifact") | No code parses it; no CI/README link depends on it. | `git rm --cached` + ignore `doc/dashboard.md` |
| `subrepos.lock.json.promote-bak.<ts>` | `scripts/promote_pin.py` rollback snapshots | No (only re-read by an explicit `promote_pin.py revert`; `system_doctor.py` only *counts* stale ones). Not tracked in `main`. | ignore `subrepos.lock.json.promote-bak.*` (no `rm` needed) |

`subrepos.lock.json` itself (the pin manifest) stays tracked — it is load-bearing
and read everywhere (`repos.py`, `runtime_paths.py`, `live_multirepo.py`,
`system_doctor.py`).

### Test consumer handled

`test_roundtrip_byte_identical_real_committed_snapshot` read the canonical
`live_state.alpaca.json` from the working tree. After untracking, a fresh clone
/ CI has no such file, so the test now `pytest.skip`s when it is absent. The
round-trip contract it proved is also proved unconditionally by
`test_roundtrip_byte_identical_indent2_representative` against
`REPRESENTATIVE_STATE`, so coverage is preserved.

## What is DEFERRED (consumed-from-git — must NOT untrack here)

These are dirty for the same root reason (jobs rewrite them in place), but they
**are read back from their committed path**, so untracking them in this PR would
break consumers. They are flagged for a follow-up that relocates them to a
git-ignored data directory *and* updates the consumers in the same change.

1. **Walk-forward calibrators** —
   `backtesting/renquant_104/artifacts/sim/walkforward_calibrators/<date>/panel-rank-calibration.json`
   (43 files), and the companion scorer artifacts
   `backtesting/renquant_104/artifacts/walkforward_gbdt_prod_recipe_v2/<date>/panel-ltr.json`
   (43 files).
   - Consumed via the **committed manifest**
     `artifacts/sim/walkforward_manifest_gbdt_prod_recipe_v2.calibrated.json`,
     whose 43 retrain entries carry relative `calibrator_uri` /
     `artifact_uri` pointing at those per-date files.
   - `renquant-backtesting/.../walk_forward/loader.py::WalkForwardModelLoader`
     resolves those URIs and calls `GlobalPanelCalibration.load(...)` /
     `PanelScorer.load(...)` →
     `renquant-pipeline/.../panel_scorer.py` `json.loads(path.read_text())`,
     which raises `FileNotFoundError` if the file is missing.
   - Verified: every `artifact_uri` / `calibrator_uri` in the committed manifest
     resolves to an existing committed file. **CONSUMED → keep tracked.**

2. **LEAN USA equity dataset** — `backtesting/data/equity/usa/**`
   (`daily/*.zip`, `factor_files/*.csv`, `map_files/*.csv`).
   - These are the bundled LEAN sample/reference tickers (aapl, spy, goog, ibm,
     bac, …) **read from the committed path** by
     `backtesting/renquant_104/training_panel/data_scan.py` (e.g. the
     `…/equity/usa/daily/{ticker}.zip` fallback) under the training preflight
     `ScanTrainingDataTask`, which `strategy_config.json` runs with
     `data_scan.enabled=true, strict=true`.
   - Also a write target for the daily LEAN export
     (`scripts/export_lean_watchlist.py`) which updates these same files in
     place — the source of the in-place modifications.
   - `.gitignore` already lists `backtesting/data/`, but these files predate the
     ignore (committed before / with `-f`), so the rule never untracked them and
     does not prevent the in-place job edits from showing as dirty.
   - **CONSUMED → keep tracked.** Untracking would break training preflight and
     backtests in a fresh clone.

### Follow-up plan for the deferred categories

The correct separation is: a single committed, point-in-time reference snapshot
(consumed) vs. job-regenerated outputs (ignored), with consumers pointed at a
git-ignored data dir. Proposed sequencing (each its own reviewed PR so a consumer
update always lands with its untracking):

1. **WF artifacts:** relocate per-date `walkforward_*` artifacts +
   `walkforward_calibrators` under a git-ignored runtime dir (e.g.
   `backtesting/renquant_104/artifacts/runtime/…`); rewrite the manifest URIs
   and `WalkForwardModelLoader` base path; gitignore + `git rm --cached` the old
   path *in the same PR*. If the WF gate needs a committed corpus for CI, pin a
   single frozen manifest+artifacts snapshot under a clearly-named immutable dir
   that jobs never rewrite.
2. **LEAN data:** split the consumed reference dataset (frozen, committed, small)
   from the daily-export target. Point `export_lean_watchlist.py` and
   `data_scan.py` at a git-ignored data dir; keep only the minimal frozen sample
   needed by tests under a non-job-written path.

## runner.py durability note

The working-tree hotfix on the live machine —
`save_live_state_atomic(state_file, self._state, config)` →
`self._config` (a `NameError` fix) in
`backtesting/renquant_104/adapters/runner.py` (~line 1692) — is **already on
`origin/main`** via PR #415 (`fix(live-state): pass self._config to
save_live_state_atomic (NameError)`, 2026-06-26). No code change is needed in
this PR; the live tree only needs to **sync to `main`** to pick it up. This is
called out because the live-state untracking above is precisely what makes such
a sync safe: once `live_state.*.json` is untracked, syncing the tree to `main`
can no longer clobber live state.
