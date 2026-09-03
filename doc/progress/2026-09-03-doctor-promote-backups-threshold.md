# DOCTOR promote_backups alarms only above the retention policy's keep count   (PR #637)

STATUS:    delivered — ops-truth fix (G-F AC4), one constant + one test.
WHAT:      `scripts/system_doctor.py` `check_promote_backups` default threshold
           3 → `PROMOTE_BACKUPS_KEEP = 5`, the number of newest lock backups the
           reviewed retention policy (renquant-orchestrator `retention_policy.py`)
           deliberately keeps. Test: 5 backups OK, 6 alarm.
WHY/DIR:   After `prune-artifacts --repo … --execute` (2026-09-03, 104 files) the
           doctor still reported `[RED ] promote_backups: 5 stale backup(s) (>3,
           prune)` — the policy keeps 5, the doctor wanted ≤3, so the RED could
           never clear by the sanctioned tool. A check that stays red on the
           policy's own steady state trains the operator to ignore the DOCTOR
           page. Direction: G-F (every ntfy page has a fix or a by-design note).
EVIDENCE:  artifact:      `system_doctor.py --json` 2026-09-03 after prune: promote_backups ok=False "5 stale backup(s) (>3, prune)"; bundle_consistency ok=True [VERIFIED — read-only run]
           prod or exp:   prod ops check; no trading path
           existing data: `tests/test_system_doctor.py` passes with the new test [VERIFIED — at the reviewed head]
           best-known?:   n/a — ops check
           scope:         "one threshold aligned to the reviewed policy; nothing else in the doctor changes"
NEXT:      merge → live ff-only → the next DOCTOR page reports promote_backups green at the policy's steady state.
