# Fix: system_doctor bundle consistency check PYTHONPATH + --repo

Date: 2026-07-14

## Problem

`make doctor` bundle_consistency check was RED even when the model bundle
was actually consistent (verified by running the checker manually with
correct environment).

Root cause: `check_bundle()` in `scripts/system_doctor.py` shells out to
the orchestrator's `check_model_bundle_consistency.py` but:

1. The checker imports from `renquant_orchestrator` (needs orchestrator
   `src/` on PYTHONPATH) -- without it, `ModuleNotFoundError`.
2. The checker resolves `--repo` from cwd by default -- when invoked from
   the umbrella's `scripts/` directory, it resolves the wrong path
   (inside `.subrepo_runtime` instead of the umbrella root).

## Fix

- Set `PYTHONPATH` to include the orchestrator runtime's `src/` directory
  before shelling out.
- Pass `--repo <REPO>` explicitly so the checker resolves the umbrella
  root correctly regardless of cwd.

## Verification

`make doctor` all green after this fix + calibrator re-stamp (V-003).
