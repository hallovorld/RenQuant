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
          to lift or legitimize it." That reminder has been firing —
          `run-surface-drift` last exit 1 — and it is correct. This is the
          "legitimize" branch of that choice, and it is the same ritual
          `fdc5933` performed for the previous batch.

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

NEXT:      Watch `run-surface-drift` on its next firing; if it stays exit 1
           after this merges, the drift it reports is broader than these three
           pins and needs its own investigation rather than another
           legitimization. Same for `shadow-ab-daily` (exit 3), whose ack names
           "run-checkout pins synced to the run manifest" as its clear
           condition.
