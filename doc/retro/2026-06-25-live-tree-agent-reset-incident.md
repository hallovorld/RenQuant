# Postmortem: a sub-agent ran `git reset --hard` on the live umbrella checkout

2026-06-25. Severity: P0 (live trading tree). Customer impact: **none** (caught + recovered
before the next daily run; the runtime the live job runs from was never touched).

## What happened
While fixing PR #409, a launched sub-agent ran git commands in the **shared live checkout**
`/Users/renhao/git/github/RenQuant` instead of its own isolated worktree, including
`git reset --hard origin/feat/fmp-harvest-plan`. A second agent (fixing #408) checked out its
feature branch in the same shared tree. Net result, observed afterward:
- Live tree left **on `feat/finnhub-analyst-cron`** (not `main`); local `main` moved to a
  harvest commit.
- `git reset --hard` **discarded the uncommitted live overlay**: the demean-deploy pins in
  `subrepos.lock.json` (pipeline `42d6205` / strategy-104 `a15a64b`) reverted to the committed
  `fa2c47de` / `1fe312b4`, and the shadow-artifact re-stamp reverted (`14586756`).

## Why impact was NONE
- The **runtime the daily job actually assembles from** (`.subrepo_runtime/repos/*`) was NOT
  touched by the reset — it stayed at the demean pins (`42d6205` / `a15a64b`), clean. The live
  model was correct the entire time.
- The incident happened AFTER the 14:06 daily-full ran — today's trades were unaffected.
- The shadow re-stamp was already committed to `origin/main` via the merged **#410**, so it was
  recoverable for free.

The latent risk (not realized): the reverted `subrepos.lock.json` would have made the **next**
`preflight_pin_align` re-align the runtime to the wrong pins → silently reverting the demean fix
live. Caught before that.

## Root cause
Sub-agents were given `isolation: worktree`, but nothing stopped them from `cd`-ing into the
shared live checkout and running git there. The live umbrella tree carries critical **uncommitted
live state** (pins, artifact stamps, `live_state.alpaca.json`) that `git reset --hard` destroys.

## Recovery (executed, verified green)
1. `git checkout -B main origin/main` — restored the clean canonical tree (a4f2c79); this also
   restored the shadow re-stamp (`f8fb2259`, committed via #410).
2. `promote_pin.py bump --subrepo renquant-pipeline --commit 42d6205... --apply` then
   `--subrepo renquant-strategy-104 --commit a15a64b... --apply` — restored the demean overlay,
   each with the default `check_conviction_admits` verify gate. (The pipeline bump's sync briefly
   pulled strategy-104 to its stale pin — the strategy bump restored it; verify then showed
   `admits=4 / n=81 demean=True mu_floor=0.03`, the correct deployed state.)
3. `system_doctor.py` → **✓ system green** (all 9 runtimes at pin + clean), backups pruned.
4. `live_state.alpaca.json` self-heals from the broker on the next run.

## Prevention (this PR)
- **`system_doctor` now checks `live_checkout_branch`** — RED if the umbrella checkout is not on
  `main` (the symptom this incident left that nothing flagged). `runtime_at_pin` already catches a
  reverted lockfile (runtime≠pin), so doctor now covers both halves of this incident class.
- **Run `make doctor` in the daily preflight and ntfy on RED** (follow-up) so a stray checkout /
  pin drift is alerted before the next trade, not discovered by hand.
- **Agent rule**: sub-agents MUST operate only in their isolated worktree and NEVER run git in
  `/Users/renhao/git/github/RenQuant`. Agent prompts that touch git now state this explicitly.

## Lessons
1. The live umbrella checkout is sacred shared state — give agents worktrees AND tell them never
   to touch the live tree's git ([[never-touch-production-inputs-on-live-tree]]).
2. Defense-in-depth worked partially: the runtime/pin split meant the live model survived a
   working-tree reset. The missing piece was an automated branch/drift alert.
