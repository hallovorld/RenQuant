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
- **Two distinct cases, exactly one channel each** (codex on RQ#582 — the
  first draft got this half-wrong): a sentinel FINDING is already paged by the
  orchestrator wrapper, so Step 6's finding branch must NOT double-page; but
  the sentinel's ABSENCE is paged HERE, because nobody else can — the missing
  component IS the pager, and an INFO line about a stale run-checkout deploy
  removing the fleet watcher is exactly the silence this watcher exists to
  end. Tests assert the finding branch contains no `notify`, the absence
  branch does, and Step 6 as a whole pages in exactly ONE branch.
- **Explicit session date**: `"$DATE"` is passed positionally so a
  post-midnight finish still classifies its own session.
- **Absent checker pages** `FLEET-SENTINEL-MISSING` naming the date, the path,
  the fact that the five lanes ran UNWATCHED, and the remedy (sync the run
  checkout to a pin carrying orch#801) — non-fatal for the run, actionable for
  the operator.

## Verification

`bash -n` clean; `tests/test_daily_104_shadow_notify.py` **38 passed**,
including five Step-6 guards: session-date passthrough + orchestrator-owned
path, ordering after every inspected lane, finding-branch non-fatal with no
duplicate page, absence-is-actionable (pages, names UNWATCHED + remedy), and
"exactly one notify in the whole step" so the two channels can never collapse
into zero or two.
