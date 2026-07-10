# backup_to_github.sh: stop swallowing the multirepo backup rc   (PR #457)

STATUS:    delivered
WHAT:      Fix an if-construct rc-capture bug in `backup_to_github.sh` that
           silently converted real backup failures (rc=1) into a reported
           rc=0, hiding them from launchd.
WHY/DIR:   Incident response — ntfy "Multirepo backup pipeline failed rc=0"
           at 2026-07-10T14:00:05Z while the backup was actually failing
           rc=1 (oversized `data/runs.alpaca.db` > GitHub 100MB limit).
           Companion fix in renquant-orchestrator (`#452`, oversized-SQLite
           gzip policy) addresses the rc=1 root cause itself; this PR fixes
           the reporting bug that hid it.
EVIDENCE:  n/a (ops/infra fix, not a model/data claim)
NEXT:      Merged ≠ deployed — unblocks only after this machine syncs the
           umbrella pin, and after renquant-orchestrator#452 lands (the
           actual rc=1 root cause).

## Problem [VERIFIED]

`scripts/backup_to_github.sh` wrapped the multirepo runner in an
if-construct and read `$?` afterwards:

```sh
if run_multirepo_backup; then
    exit 0
fi
BACKUP_RC=$?          # <- status of the if-construct: 0 when no branch ran
```

When the function returned nonzero, `BACKUP_RC` became 0, the notification
said "failed rc=0", and the script exited 0 — launchd recorded success, so
the hourly backup failed invisibly.

## Fix

- Capture the function rc directly in a checked context (no ERR-trap
  double-notify): `run_multirepo_backup && BACKUP_RC=0 || BACKUP_RC=$?`.
- Notify with the real rc and exit with the real rc so launchd sees the
  failure.
- Tee the module output to a temp file; the failure notification now carries
  the last line of the module's JSON summary (truncated to 1000 chars) for
  one-glance triage.
- Legacy shell path and the 127 fall-through (module unavailable → legacy
  backup) are unchanged.

The root cause of the rc=1 itself (oversized SQLite → gzip policy) is fixed
in renquant-orchestrator (`state_backup` oversized-SQLite compression PR).

## Tests

`tests/test_backup_rc_propagation.py` (new):

- functional: real script run against an isolated fake repo tree with a
  stubbed `.venv/bin/python` that exits 1 → script exits 1, notification
  contains `rc=1` and the module's JSON line; never `rc=0`.
- functional: stubbed module success → script exits 0, no notification.
- static guard: the if-construct rc-capture pattern is banned; the direct
  capture form must be present.

Pre-existing failures on `main` (unrelated, not touched):
`test_operator_script_env.py::test_manual_promote_uses_project_venv` and
`::test_multirepo_shell_wrappers_use_shared_strict_helper` fail identically
on a clean `origin/main` checkout.
