# Fleet pin bump — materializes the 2026-07-02/03 burst (all behavior flags OFF)

STATUS: chore (lock-only change). Authored directly by the leader loop after the delegated
agent died to a spend limit mid-task; pin table verified against each repo's origin/main
head at authoring time.

## Old → new pins

| Subrepo | Old | New | Delta highlights |
|---|---|---|---|
| renquant-common | 1d10aaf7 | 19cba70f | 0.9.x fingerprint API + 0.9.1 back-compat shims (#19–#21) |
| renquant-strategy-104 | 1fe312b4 | dd337d45 | sleeve keys (inert, #39), cap fix (#40); resolves the #437-flagged lock↔runtime drift |
| renquant-model | 19919ec9 | 775804db | current main |
| renquant-pipeline | fa2c47de | df7bc073 | sleeve shadow (#157), admission shadow logger (#161, observe-only), BL-1 recenter (#162, flag OFF), intraday decisioning (#163, unreferenced without the scheduler), fingerprint adaptations (#159/#160) |
| renquant-execution | f7c5cde8 | bad04155 | order state machine (#20, inert module) |
| renquant-backtesting | 50149e63 | 34fd4edd | S1/S2 gate repair, #59/#60/#61 (v2 stays enforcing; v3 shadow-only) |
| renquant-base-data | 0bbb5349 | bb69f5e2 | the TRUE transformer-corpus recipe (#31) — unblocks the S12 refresh |
| renquant-artifacts | 538b5c70 | c09d66f8 | cap fix (#11-class) |
| renquant-orchestrator | 65402735 | 4b8af94e | the whole 07-02/03 burst (pairing fix #253, S8 driver, monitors, S12 B3, prestamp tool) |

## Safety argument

- Every behavior-bearing change rides a default-OFF flag or an observe-only, fail-isolated
  task: sleeve OFF, one-share floor OFF, BL-1 recenter OFF, intraday decisioning gated by a
  separate uninvoked process + env + kill-file, M5 admission logger observe-only.
- The 0.9.x fingerprint convergence detonations were DEFUSED before this bump: step-0
  legacy pre-stamp 47/47 applied 2026-07-03 (orchestrator #280), verified idempotent.
- The prod scorer/calibrator pairing is fresh and PAIRED (#437 + the 07-03 refit).
- The WF gate v2 remains the sole enforcing gate (bt#61's v3 is shadow diagnostics).
- What Monday's daily run does differently: pin-align materializes these pins and stamps
  artifact_hashes into run bundles (healing the batch-scores exporter); trading behavior
  is UNCHANGED by construction.

## Landing note

Merged ≠ deployed: the machine picks this up at the next pin-align (Monday 13:55 PT daily
run, or a manual align in a safe window). The S12 shadow refresh becomes runnable
immediately after alignment (its TRUE-recipe resolver needs the new base-data pin).

## Round 2 (CI fix)

`verify-pinned-declaration` failed: the committed `doc/arch/strategy-104-snapshot.md` was
still generated from the prior strategy-104 pin (c019b256), not the dd337d45 this PR bumps
to — the snapshot generation step was skipped as part of the pin bump itself. Regenerated
via the same command CI's verify step runs (checkout renquant-strategy-104 at the exact
lock pin, render from its `configs/`); `--verify-pinned-declaration` now passes locally.
204/205 relevant tests pass; the one failure is pre-existing and reproduces identically on
the unmodified checkout (unrelated cwd-sensitivity quirk in an unrelated test).
