# A reviewed input was also a runtime output, so the live tree could never be clean

STATUS:   delivered. One writer redirected + its tests rewritten to the new
          contract + one stale rationale corrected. No live path touched, no
          config value changed, nothing deployed.

WHAT:     `scripts/recalibrate_scores.py` stamped `ranking.blend_updated` and
          `ranking.blend_n_symbols` into `backtesting/renquant_104/
          strategy_config.json` — a git-TRACKED, reviewed input. It now writes
          them to `backtesting/renquant_104/logs/blend_calibration_state.json`,
          a gitignored runtime sidecar, and does not touch the config at all.

WHY/DIR:  orch#1024. A file that is simultaneously a reviewed input and a
          runtime output leaves exactly two outcomes and no third:

          * every deploy touching the path aborts — `git merge --ff-only`
            refuses against a dirty tracked file. This is not hypothetical: it
            blocked the deploy of RenQuant#602, the change that unfreezes the
            weekly tournament; or
          * somebody clears it with `git checkout --` / `reset --hard` and the
            recorded blend state is gone with no trace.

          It also defeats `ops/run_surface_drift_check.py`, which treats a
          tracked modification in a runtime checkout as an alarm — correctly —
          but this one is expected, so it is a permanent false positive, and a
          permanent false positive is how a real one gets ignored.

EVIDENCE:
  artifact:      scripts/recalibrate_scores.py (`_write_blend_state`,
                 `BLEND_STATE_RELPATH`), tests/test_recalibrate_scores.py,
                 scripts/check_config_drift.py (comment only).
  prod or exp:   neither. The script is not run by this change; the next
                 scheduled run writes the sidecar instead of the config.
  existing data: repo-wide sweep for consumers of either key, across all
                 sibling checkouts [VERIFIED 2026-08-24]:
                   writer  — scripts/recalibrate_scores.py
                   tests   — tests/test_recalibrate_scores.py
                   ignores — renquant-strategy-104 config_drift.DEFAULT_IGNORES
                             and this repo's scripts/check_config_drift.py
                   docs    — orchestrator doc/research/2026-08-10-stale-config-
                             surface-audit.md
                 **No trading-logic consumer.** The only code that names them
                 does so to IGNORE them in a drift check. They are telemetry.
  best-known?:   yes for the destination: `logs/` is already gitignored and
                 already holds this strategy's runtime outputs, so the sidecar
                 cannot dirty a tracked path by construction rather than by
                 convention. Placement is asserted by test, including that
                 `.gitignore` really covers it.
  scope:         the writer and its contract. The config's CURRENT values are
                 not edited, and the live tree's existing modification is not
                 touched — that is an operator-gated live-tree action, called
                 out under NEXT.

VERIFICATION:
  Mutation-verified. Re-adding the config write on top of the fix:
      2 failed, 3 passed
      FAILED test_the_config_is_byte_identical_across_a_run
      FAILED test_concurrent_edit_survives_write
  With the fix: 5 passed [VERIFIED 2026-08-24].

  The 2026-04-22 concurrent-edit regression test is KEPT, not deleted. Its
  guarantee — an edit landing mid-run is not wiped — now holds in a stronger
  form, because a write that does not happen cannot wipe anything. The test
  still drives the identical race; only the assertion about the script's own
  two fields moved to the sidecar. Deleting it would have thrown away the
  evidence that the older bug stays fixed.

  Migration is LOSSLESS by test. Until the first post-deploy run, the only copy
  of the live values (`2026-08-16` / `141`) is the untracked modification in the
  config; an operator clearing that dirt first would destroy them. So the first
  run SEEDS the sidecar from the config and marks `source: config-seed`, and a
  second run must not re-seed — both pinned.

  TWO DEFECTS FOUND IN REVIEW (codex on #606), both mine, both real:

  1. **Unbounded recursive history.** `previous` was the entire prior PAYLOAD,
     which contains its own `previous`. Depth grew by one every run: measured
     51 levels and ~10KB of duplicated `note` after a year of weekly
     calibrations. `previous` now carries the prior run's STATE_FIELDS and a
     one-word `source` marker — constant schema, constant depth, bounded size.

     Worth naming why it survived my own testing: my second-run test checked a
     top-level VALUE and never the SHAPE. So the replacement asserts the shape
     over eight runs — key set, nesting depth, and size — which is the only form
     of the assertion that can see accumulation at all.

  2. **A stale contract in the comment I kept.** The retained 2026-04-22 text
     still said "Also drop any stale blend_weights", which this change no longer
     does. Not doing it is CORRECT — `blend_weights` is a decision input, legacy
     and zero-weighted at the current 104 seam but an input, and a rule that
     runtime must not edit the reviewed config cannot have one silent exception.
     The comment now says so, and a test asserts an existing `blend_weights`
     survives a run. Removing the key is a reviewed config change, not a side
     effect of calibrating.

  Both are mutation-verified: re-nesting the payload turns 2 tests red;
  re-adding the `blend_weights` deletion turns 1 red. 7 pass with the fixes.

  Wider scope (every config/drift/calibration test file): 434 passed, 10 failed.
  The same 10 fail identically on pristine origin/main here (431 passed, 10
  failed — the +3 is this change's new tests), so they are not this change.

NEXT:     Two follow-ups, neither attempted here:
          1. The live umbrella tree still carries the historical modification on
             `backtesting/renquant_104/strategy_config.json`. After this ships
             and one run has seeded the sidecar, restoring that path becomes
             safe — but it is a live-tree mutation and therefore operator-gated.
          2. Once that is reconciled, the three `DEFAULT_IGNORES` entries in
             `scripts/check_config_drift.py` and in renquant-strategy-104's
             `config_drift.py` should be DROPPED, so a difference in those paths
             starts reading as the real drift it would then be. The umbrella
             copy's comment says so; the strategy-104 copy is another repo and
             was deliberately not edited from here.
