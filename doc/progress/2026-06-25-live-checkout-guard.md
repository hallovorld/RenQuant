# Live-checkout guard + agent-reset postmortem

2026-06-25.

## What & why
A sub-agent ran `git reset --hard` in the shared live umbrella checkout while fixing a PR,
reverting the demean-deploy pins + shadow re-stamp in the working tree and leaving the tree on a
feature branch. Impact was NONE (the runtime the daily job runs from was untouched; caught before
the next run), but it was a P0 near-miss. Recovered to ✓ system green. This PR adds the missing
guard + records the postmortem.

## Deliverables
- `scripts/system_doctor.py` — new **`check_live_checkout_branch`**, **opt-in**: it SKIPs unless
  `RENQUANT_DOCTOR_EXPECT_BRANCH` is set (so running `make doctor` on a PR/worktree feature branch
  stays green); when the live path sets `RENQUANT_DOCTOR_EXPECT_BRANCH=main` it REDs off-main.
  By default the **active** off-main protection is the pre-pin-align daily guard (#414), not this
  heartbeat. `runtime_at_pin` separately flags a reverted lockfile (runtime≠pin).
- `tests/test_system_doctor.py` — test that a `feat/*` checkout trips the guard while `main`
  passes. 5 tests pass.
- `doc/retro/2026-06-25-live-tree-agent-reset-incident.md` — the full postmortem (timeline, why
  impact was nil, root cause, the exact recovery, prevention, lessons).

## Recovery already done (live tree)
`git checkout -B main origin/main` → `promote_pin bump` pipeline 42d6205 + strategy-104 a15a64b
(verify `admits=4 demean=True mu_floor=0.03`) → `system_doctor` ✓ green. Shadow fp restored to
f8fb2259 via the merged #410.

## Follow-up (not in this PR)
Wire `make doctor` into the daily preflight with an ntfy on RED, so a stray checkout / pin drift
is alerted automatically before the next trade.
