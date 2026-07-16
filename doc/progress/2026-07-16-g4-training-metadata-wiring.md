# G4: Wire training_cutoff + model_content_sha256 through run bundle to persistence

Date: 2026-07-16

## Problem

Pipeline PR #202 adds `training_cutoff` and `model_content_sha256` columns to
`pipeline_runs`, but the umbrella callers (runner.py, sim.py) never populate
them. Every new row still stores NULL for both fields, so the G4 canonical
admissibility validator continues to reject all score evidence.

## Solution

Three files changed:

1. **artifact_contract.py** (`build_run_bundle`): extract `trained_date` from
   the panel artifact payload into `bundle["training_cutoff"]`, and compute
   `model_content_sha256` via `renquant_common.model_fingerprint` (fail-safe
   to `None` if the import is unavailable).

2. **runner.py** (live adapter): build `_record_kw` dict, conditionally add
   `training_cutoff` and `model_content_sha256` from run_bundle, then call
   `record_pipeline_run` with try/except TypeError fallback for pin-order
   independence (the pinned pipeline may not yet have the new params).

3. **sim.py** (sim adapter): same `_record_kw` + try/except pattern as
   runner.py, ensuring backtest runs also persist training metadata.

## Pin-order independence

The try/except TypeError pattern ensures the umbrella code works regardless of
whether the pinned pipeline version has been bumped to include the new
`training_cutoff`/`model_content_sha256` parameters. When the pipeline pin is
advanced past PR #202, the new columns are populated; before that, the call
falls back to the existing signature transparently.

## Depends on

- renquant-pipeline PR #202 (schema + migration + API)
