# Pin renquant-pipeline to the G4 training-metadata persistence change

**Date**: 2026-07-16
**Companions**: renquant-pipeline PR #202 (merged 2026-07-16), RenQuant PR #482
(merged 2026-07-16)

## What this changes

Bumps the `renquant-pipeline` entry in `subrepos.lock.json`:

- `2b1b70dad369abf4f518f9d899061f75cdad6198` →
- `28be76c604c8361113495f7d346bc1ac20e57717` (merge of pipeline PR #202,
  2026-07-16)

Exactly one commit between the pins (verified via compare API): the
`pipeline_runs` schema change adding `training_cutoff` and
`model_content_sha256` columns with additive migration and two optional
`record_pipeline_run()` keyword parameters.

## Why

G4 Phase A requires admissible score evidence from `runs.alpaca.db`. The
canonical admissibility validator rejects records without `training_cutoff`
and `model_content_sha256`. Pipeline #202 added the persistence columns;
umbrella #482 (already on main) wires the values from the panel artifact
through `build_run_bundle()` into `record_pipeline_run()` with a
try/except TypeError fallback for pin-order independence.

With this pin bump, the wiring in #482 stops falling back and starts
persisting real values: every new live/sim run records its model's
training cutoff and content fingerprint.

## Risk

- Additive schema change only; migration runs on next `ensure_schema()`
- All existing callers unaffected (optional kwargs, NULL default)
- Pipeline CI green at the pinned commit (1748 tests)
- Live machine pin sync remains a separate operator-gated landing action
