# Pin renquant-orchestrator to the G3 F-8 fail-closed bootstrap fix

**Date**: 2026-07-16
**Companions**: renquant-orchestrator PR #514 (merged 2026-07-14), renquant-pipeline
PR #199 (merged 2026-07-14)

## What this changes

Bumps the `renquant-orchestrator` entry in `subrepos.lock.json`:

- `e8fe46206025b6432eb6e0db7a898d9cfd27bead` (merge of orchestrator PR #451,
  2026-07-10) →
- `bfb935e4a38d3e7653f6576c5e6a461731cb4acc` (merge of orchestrator PR #514,
  2026-07-14)

`renquant-pipeline`'s pin is untouched — it is already at `2b1b70d...`
(pipeline's current `main`), which contains PR #199 as an ancestor, so the
pipeline side of this fix has been live in the pin since a routine fleet
bump landed after PR #199 merged.

## Why this pin was stale

Orchestrator PR #514 closed audit finding G3 F-8 (`bootstrap_multirepo`'s
kernel-alias loop silently falling back to stale umbrella copies of pipeline
modules on import failure) across 5 review rounds:

1. Force-alias + `UMBRELLA_ONLY_STEMS` allowlist (rejected: name-keyed, not
   contract-keyed).
2. Pipeline-declared `NON_OWNED_KERNEL_STEMS` (pipeline #198) — the
   non-owned-stem exemption moves to the pinned, versioned pipeline package
   instead of an orchestrator-local name list.
3. `OWNED_KERNEL_STEMS` (pipeline #199) as the positive-side companion,
   replacing an arbitrary `_MIN_PIPELINE_KERNEL_MODULES = 10` path-identity
   heuristic with a real structural equivalence check.
4. Round 5 (Codex): the r4 landing still fell back to the coarse "nonzero
   owned modules" guard whenever `OWNED_KERNEL_STEMS` was absent — fail-open
   with respect to the very identity contract it introduced. Fixed in
   commit `bb6ecf99` (`fix(bridge): require OWNED_KERNEL_STEMS, no
   fallback`): absence of **either** `OWNED_KERNEL_STEMS` or
   `NON_OWNED_KERNEL_STEMS` now raises `RuntimeError` unconditionally, with
   no "count what's on disk instead" escape hatch.

That final fix (commit `bb6ecf99`, merged into orchestrator `main` via
`bfb935e4`) is the fix this pin bump deploys. Between the last routine
fleet pin bump (2026-07-10, orchestrator PR #451) and PR #514 merging
(2026-07-14), the orchestrator pin was never advanced, so the live/pinned
system was still running the pre-G3-F8 fail-open bootstrap despite the fix
being merged to `main` days earlier ("merged is not deployed").

This bump is scoped to exactly the PR #514 merge commit (`bfb935e4`), not
orchestrator's current `main` tip — `main` has since accumulated ~150
unrelated commits since the stale pin, none of which are needed to close
this gap, and pulling them all in is a separate decision for a routine
fleet bump, not bundled into this fix.

## Verification

All performed in isolated worktrees under
`/private/tmp/.../scratchpad/g3f8/` (never against the shared live
checkouts) with a fresh `git fetch` immediately before each check:

- **renquant-orchestrator @ `bfb935e4` (the pin target)**: full suite —
  3915 passed, 3 skipped.
- **renquant-pipeline @ `main` (`2b1b70d`, the current pin)**: full suite —
  1735 passed, 8 skipped, 3 failed. The 3 failures (2 in
  `test_replay_d6_conventions.py`, 1 in `test_xgboost_scorer_contract.py`)
  are pre-existing and unrelated to G3 F-8 — documented as such across
  rounds 1-4 in pipeline's own `doc/progress/2026-07-14-g3-f8-kernel-ownership-contract.md`
  and reproduced identically here.
- **Real, non-mocked `bootstrap_multirepo()` smoke test** pairing the two
  exact pin-target commits (orchestrator `bfb935e4`, pipeline `2b1b70d`),
  both src roots on `sys.path`, no mocks:
  1. Clean pins: succeeds, aliases 53 kernel modules (49 owned + `meta_label`
     alias + `preflight`/`panel_pipeline`/`panel_scoring` force-aliases).
  2. `OWNED_KERNEL_STEMS` deleted from the real imported `renquant_pipeline.
     kernel` module → fails closed: `"...does not declare OWNED_KERNEL_STEMS
     — pin a pipeline version >= #199"`.
  3. `OWNED_KERNEL_STEMS` tampered to include a stem absent from disk
     (`totally_made_up_stem_xyz`) → fails closed citing that exact stem
     name.
  4. `NON_OWNED_KERNEL_STEMS` deleted → fails closed:
     `"...does not declare NON_OWNED_KERNEL_STEMS — cannot verify
     ownership contract"`.

This is the "real bootstrap smoke result against those exact pins" Codex's
round-5 review asked for on orchestrator PR #514 — attached here rather
than on that PR because the pin-level pairing only became possible once
this PR exists.

## Not in scope here

- No other subrepo pins touched (pipeline's is already current).
- No `.subrepo_runtime` materialization / `promote_pin.py --apply` run —
  that tool syncs against the live local-path checkouts on this machine,
  which is out of scope for a worktree-only pin-bump PR. The umbrella
  operator should run the normal pin-align step against this merged commit
  before/at the next scheduled daily-full.
- Branch protection on RenQuant requires a separate reviewer approval;
  this PR is not self-merged.
