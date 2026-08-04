# 2026-08-04 — Step 5b: the S1 momentum-blend e2e rail (dormant until its profile lands)

STATUS:    rail + allowlist + guard-test repair; dormant by design
WHAT:      scripts/daily_104.sh gains Step 5b, cloned from Step 5 per its
           own resurrect-a-second-lane instruction: gates on
           strategy_config.shadow_blend_momentum.json in the PINNED
           configs (INFO-skip while absent — the same lands-before-the-
           profile shape Step 5 shipped with), tag
           alpaca_shadow_blend_mom (state/db/log/ntfy prefix all disjoint
           from prod, legacy shadow, and the clf-blend lane; prefix
           derivation is the generic alpaca_shadow* rule in live/runner.py
           — no runner change), non-fatal, distinct FAIL/TIMEOUT titles,
           same buy-side-preflight suppression pattern. The Step 5 slot
           STAYS with the certified z(prod)+z(clf) profile.
           backtesting/renquant_104/kernel/state_paths.py ALLOWED_BROKERS
           gains the new tag — the second, enumerated validator; without
           it the lane would fail-closed at state-path derivation.
WHY/DIR:   GOAL-8 S1 (prereg orch#777, in review) requires the consuming
           lane to exist so the s104 profile PR activates a REVIEWED rail
           instead of shipping wiring and profile in one blob. Direction:
           this rail (dormant) → #777 freeze → s104 profile PR (reviewed
           against the frozen prereg, records the deployment boundary) →
           20-session S1 clock.
FOUND+FIXED (same batch): tests/test_daily_104_shadow_notify.py is
           enforced by NO CI enumeration, and had rotted: two Step-4-era
           guards asserted a block RETIRED 2026-08-03, one Step 5 guard
           asserted a quoted literal that left the script, and the
           ordering guard compared against a vanished "--- Step 4:" echo —
           find() = -1 made it pass VACUOUSLY. All repaired against the
           current script with non-vacuous anchors; the Step-4-era guards
           now guard the retirement itself (no half-present legacy lane).
EVIDENCE:  bash -n clean; test_broker_readonly_tag.py +
           test_daily_104_shadow_notify.py + test_state_paths.py:
           61 passed (5 new Step 5b mirrors + 1 new tag test + repairs).
CI:        [codex round 1 — not deferred] the repaired guard file is now
           ENUMERATED in .github/workflows/strategy-104-snapshot-fresh.yml
           (unit job, triggers on every PR) and the same workflow's shell
           step gains `bash -n scripts/daily_104.sh`. The tag test stays a
           local/make-test guard: it imports live.runner, which is not
           bare-hosted-runner-safe; the notify guard is pure stdlib.
NEXT:      #777 freeze → s104 shadow_blend_momentum profile PR (+ pipeline
           pin 3ecd9880 in that batch).
