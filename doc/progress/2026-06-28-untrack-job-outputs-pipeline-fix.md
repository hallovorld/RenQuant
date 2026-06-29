# untrack job-written outputs — clobber-proof live tree

STATUS: delivered
WHAT: fixes the `.gitignore` live-state bug and untracks the job-written
outputs that made the live umbrella tree perpetually dirty (166 paths). Adds
`backtesting/*/live_state.*.json` (the broker-suffixed live-state files that the
bare `live_state.json` pattern missed), `doc/dashboard.md`, and
`subrepos.lock.json.promote-bak.*` to `.gitignore`, and `git rm --cached`s the
two live-state files + `doc/dashboard.md` (working copies kept). Guards
`tests/test_live_state_v2.py::test_roundtrip_byte_identical_real_committed_snapshot`
to skip when the now-untracked snapshot is absent (the lossless contract is
still proved against `REPRESENTATIVE_STATE`).
WHY/DIR: a perpetually-dirty tree lets any routine git op (`checkout`/`reset`/
botched `pull`) clobber live trading state — the exact pattern behind a prior
multi-day outage. Untracking job outputs makes a clean tree a safety property:
git can no longer restore/clobber files it does not track.
EVIDENCE: read-only consumer analysis against the umbrella sources + the pinned
subrepo runtime sources (`.subrepo_runtime/repos/*`). Untracked items confirmed
NOT consumed-from-committed-path: `live_state.*.json` (only a test read it →
guarded), `doc/dashboard.md` (`build_dashboard.py` writes it; nothing parses it),
`promote-bak.*` (throwaway rollback snapshots). DEFERRED as consumed-from-git
(left tracked): the 43 `walkforward_calibrators` + 43
`walkforward_gbdt_prod_recipe_v2/panel-ltr.json` (read via the committed
`*.calibrated.json` manifest URIs → `PanelScorer.load` /
`GlobalPanelCalibration.load`; verified every URI resolves to a committed file)
and `backtesting/data/equity/usa/**` (LEAN sample dataset read by
`training_panel/data_scan.py` under the strict training preflight + the daily
LEAN export target). Design note:
`doc/arch/2026-06-28-untrack-job-outputs-pipeline-fix.md`.
NEXT: ship the two deferred relocations (each as its own PR that moves the
artifacts to a git-ignored runtime dir AND updates the consumer in the same
change) per the follow-up plan in the design note. The runner.py NameError
hotfix is already on `main` (PR #415); the live tree only needs to sync — which
this untracking now makes clobber-safe.
