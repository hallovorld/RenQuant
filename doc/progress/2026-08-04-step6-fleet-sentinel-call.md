# 2026-08-04 — Step 6: the daily wrapper calls the fleet lane sentinel

Companion to renquant-orchestrator#801. The watcher's LOGIC lives in the
orchestrator (daily orchestration is its declared role); this PR is the
umbrella's one-line-of-responsibility half: call it, with the session date,
as the daily run's last step.

## Why here and not on a clock

The lanes the sentinel inspects are Steps 5–5e immediately above it, so
**daily completion IS the correct trigger**. The first attempt at a
clock-scheduled job picked 15:30 PT from a MANUAL run's wall clock and would
have paged MISSING on a still-running fleet (codex on orch#801). Coupling to
completion removes the cadence guess entirely — and removes the plist,
the install grant, and the pending-install bookkeeping with it.

## Contract

- **Non-fatal by construction**: every decision above is already made and
  executed; a watcher must never turn its own finding into a failed daily run.
- **One page, not two**: the alarm belongs to the orchestrator wrapper (whose
  body carries the offending lane lines); Step 6 deliberately sends no ntfy of
  its own, and a test asserts `notify ` never appears in the section.
- **Explicit session date**: `"$DATE"` is passed positionally so a
  post-midnight finish still classifies its own session.
- **Absent checker skips LOUDLY** with the remedy named (sync the run checkout
  to a pin carrying orch#801) — never silently.

## Verification

`bash -n` clean; `tests/test_daily_104_shadow_notify.py` 37 passed, including
four new Step-6 guards: session-date passthrough + orchestrator-owned path,
ordering after every inspected lane, non-fatal + no-duplicate-page, and the
loud skip with its remedy.
