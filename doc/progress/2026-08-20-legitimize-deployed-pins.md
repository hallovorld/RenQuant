# Legitimize the pins the 2026-08-18 deploy left uncommitted

STATUS:   delivered. `subrepos.lock.json` only — no code, no config, no
          behaviour change. The pins it records are ALREADY what production
          runs; this makes the reviewed surface say so.

WHAT:     The 2026-08-18 vol-window deploy advanced three subrepo pins using
          `scripts/promote_pin.py bump --apply`, which rewrites
          `subrepos.lock.json` in the working tree and does **not** commit it.
          The live umbrella has carried the change uncommitted ever since, so
          two different answers to "what is pinned" have coexisted:

            subrepo                committed   working = RUNTIME HEAD
            renquant-strategy-104  e00d9356    8a395e49
            renquant-pipeline      4aec0e35    3d9c7fb1
            renquant-backtesting   8c2c4456    e5f9bae3

          The runtime follows the WORKING copy (all three
          `.subrepo_runtime/repos/<name>` HEADs match it exactly), so
          production has been running pins that exist only in an uncommitted
          file, while CI's `verify-pinned-*` jobs and any fresh clone read the
          pre-deploy commits.

          Each pin is a legitimately merged commit — verified as an ancestor of
          its own `origin/main` FROM THE RUNTIME CHECKOUT. (The dev checkouts
          do not have the objects and report a false "NOT ON MAIN"; that
          reading was checked and discarded rather than reported.)
          `renquant-pipeline` sits one merge past the value recorded on 08-18
          (`f9f488d5` -> `3d9c7fb1`) — PR #296, the vol-window
          Series-truthiness crash fix landed as a same-day fast-follow.

WHY/DIR:  CLAUDE.md's containment protocol: "if the change is meant to persist,
          the reviewed surface [is] updated in the same batch … otherwise the
          daily run-surface drift scan alarming on it is the DESIGNED reminder
          to lift or legitimize it." This is the "legitimize" branch of that
          choice, and the same ritual `fdc5933` performed for the previous
          batch.

          CORRECTION (2026-08-20, before merge): an earlier draft of this doc
          claimed the reminder "has been firing — `run-surface-drift` last exit
          1 — and it is correct", i.e. that the drift scan was the alarm
          pointing here. **That was false and is withdrawn.** I ran the scanner
          instead of continuing to assert it; it never mentions the pins. Its
          actual findings are the two uncommitted working-tree files
          (`renquant-model README.md`, `orchestrator-run
          run_session_scheduler.sh`), three import-resolution lines already
          diagnosed as a scanner-environment artifact, and five rq105 jobs
          resolving `renquant-common` by filesystem fallback (filed as
          renquant-orchestrator#1016).

          What actually surfaced this was `verify-pinned-declaration` failing
          on the pin bump — a gate doing its job — plus reading the lock
          against the runtime while investigating something else. The
          containment-protocol reasoning stands on its own: a reviewed surface
          that disagrees with what runs is worth fixing whether or not a
          scanner happens to say so. Recorded rather than quietly amended,
          because a PR whose whole point is that reviewed surfaces should state
          the truth cannot ship a false one.

          The concrete risk is not theoretical: a `git checkout` or
          `reset --hard` on the umbrella would silently revert production to
          three older pins, including the one carrying the vol-window lane's
          crash fix. Nothing would report it, because the committed lock would
          look consistent with itself.

EVIDENCE:
  artifact:      `subrepos.lock.json` (three `commit` fields), this doc.
  prod or exp:   **exp** — the lock content was copied out of the live tree and
                 pushed to a branch via the contents API; the live umbrella
                 working tree was not written by this PR. Merging it makes the
                 committed lock agree with what is already deployed; it changes
                 no runtime behaviour because the runtime already uses these
                 pins.
  existing data: measured, not assumed —
                 - committed vs working lock diffed field by field [VERIFIED]
                 - all three `.subrepo_runtime/repos/<name>` HEADs read back and
                   matched against the working lock [VERIFIED]
                 - each pin `merge-base --is-ancestor <pin> origin/main` from the
                   runtime checkout after an explicit fetch → ancestor [VERIFIED]
                 - `git log f9f488d5..3d9c7fb1` → PR #296 only [VERIFIED]
  best-known?:   yes. The alternative — reverting the deployed pins to match the
                 committed lock — would remove the vol-window crash fix from a
                 running lane, which is strictly worse.
  scope:        the lock file. Deliberately does NOT touch
                `ops/launchd_manifest.json` or any job definition.

NEXT:      Do NOT expect `run-surface-drift` to clear after this merges — per
           the correction above, it was never reporting these pins, and an
           earlier draft of this section told operators to watch exactly that.
           It will keep reporting the two uncommitted working-tree files and
           the rq105 fallback until those are dealt with on their own terms
           (renquant-orchestrator#1016).

           `shadow-ab-daily` (exit 3) IS worth watching: its expired ack names
           "run-checkout pins synced to the run manifest (orch#747 item 5) and
           the 14:35 two-arm run passes PRECHECK" as its clear condition, and
           its PRECHECK reads the pins. If it still refuses after this merges,
           the remaining blocker is the dirty `renquant-model` working tree
           (`M README.md`), which its ack also names — not the pins.
