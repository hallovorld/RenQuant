# One shadow lane had no callsign, so its title shouted its own tag

STATUS:   delivered. One dict entry + a coverage guard. No behaviour change
          beyond the ntfy title of a single shadow lane; no config, no
          launchd, no live path.

WHAT:     `LANE_CALLSIGNS` (live/runner.py) covered five of the **six** shadow
          lanes `scripts/daily_104.sh` launches. The sixth,
          `alpaca_shadow_vol_window` (added 2026-08-18), fell through
          `LANE_CALLSIGNS.get(tag, tag.upper())` and rendered as

              [READONLY][ALPACA_SHADOW_VOL_WINDOW]

          — 36 characters of title, against `[READONLY][RC]`'s 14, before the
          reader reaches a single decision. This PR gives the lane a callsign
          (`V`) and adds a guard that asserts coverage against the launching
          script rather than against a second hand-kept list.

WHY/DIR:  found while closing orch#1014 by measurement. That issue is the same
          harm by a different route: a shadow alert the operator could not
          distinguish from real money. #1014's cause was a dropped body
          sentence; this one pushes the lane marker out of a phone's title
          budget. Both make the pager mean less, which is the only thing a
          pager has.

          The reason it ran that way for six days is the shape worth naming:
          `.get(tag, tag.upper())` means **adding a lane cannot fail**. It
          degrades, silently, into the form that is worst for the reader. The
          fallback is kept — a title must never be *missing* a lane marker —
          but it is no longer the de-facto path for a new lane.

EVIDENCE:
  artifact:      live/runner.py (`LANE_CALLSIGNS` + a comment on the fallback)
                 and tests/test_broker_readonly_tag.py (`TestCallsignCoverage`).
  prod or exp:   neither. The vol-window lane is a READONLY shadow lane; the
                 only observable change is its ntfy/log title. The prod lane
                 (broker `alpaca`) is not prefixed here at all.
  existing data: the six launched lanes, read from the authority:
                 `grep -o 'RENQUANT_READONLY_TAG=[a-z_]*' scripts/daily_104.sh`
                 → blend, blend_mom, blend_mom_fast, blend_rb_mom,
                 blend_rb_fast, **vol_window**; the map had the first five
                 [VERIFIED 2026-08-24]. Observed in the wild:
                 `[READONLY][ALPACA_SHADOW_VOL_WINDOW]` in
                 logs/daily_104/2026-08-2[01]*.log alongside `[READONLY][RC]`
                 and `[READONLY][RCS]` [VERIFIED].
  best-known?:   yes for the guard. The alternative — a hand-maintained list
                 of expected lanes in the test — rots by the identical
                 mechanism it is meant to catch, so the test parses
                 daily_104.sh, and asserts the parse found something so it
                 cannot pass vacuously if that idiom is ever refactored.
                 The callsign LETTER is a judgment call, flagged as freely
                 renameable in the comment: the scheme's letters (R/C/S/f)
                 name a scoring composition, and this lane's scoring is the
                 prod blend — what it varies is the volatility gate
                 (vol_window_license), so it gets its own letter rather than a
                 composition string.
  scope:         one map entry, one comment, one test class. No change to
                 `_notify_decision`, to `is_shadow` classification, or to any
                 lane's state/db routing.

VERIFICATION:
  Mutation-verified — the guard is red on the state that actually shipped.
  With the new entry REMOVED (i.e. main as of 2026-08-18..08-24):
      3 failed, 32 passed
      FAILED …TestCallsignCoverage::test_every_running_shadow_lane_has_a_callsign
      FAILED …TestCallsignCoverage::test_the_vol_window_lane_specifically
      FAILED …TestCallsignCoverage::test_coverage_holds_for_the_prefix_function_not_just_the_map
  With it restored: 35 passed [VERIFIED 2026-08-24].

  Wider scope (every ntfy/notify/runner/readonly/shadow test file):
  711 passed, 5 failed. The same 5 fail identically on **pristine origin/main**
  (43fd40b) in this worktree — 706 passed, 5 failed, same test IDs — so they
  are not this change. They are also not a main-is-red finding: the cause is
  `ModuleNotFoundError: No module named 'renquant_pipeline'`, i.e. a detached
  worktree without the sibling subrepo runtime the live tree resolves through.
  main's CI is green (config-artifact-path-gate / job-outcome-contract /
  strategy-104-snapshot-fresh, all success 2026-08-24T06:01) [VERIFIED].

NEXT:     none required. Noted, not attempted: `LANE_CALLSIGNS` is consulted
          only for shadow lanes, so this guard says nothing about the prod
          title; that is by design (live titles keep their existing format).
