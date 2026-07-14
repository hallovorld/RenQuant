# G3 F-6: Kernel parity CI check

Date: 2026-07-13
Finding: F-6 from 2026-07-04 architecture compliance audit
PR: g3/f6-kernel-parity-ci

## Problem

The §3.5 byte-equivalence invariant between `backtesting/renquant_104/kernel/`
(umbrella) and `renquant_pipeline/kernel/` (pinned pipeline) had no automated
enforcement. As of the audit, 78 of 169 shared files had drifted with 111
commits touching the umbrella kernel since 2026-05-25.

## Fix

Added `scripts/check_kernel_parity.py` — compares both kernels after
normalizing import paths (`renquant_pipeline.kernel.` ↔ `kernel.`). Maintains
a `KNOWN_DRIFT_ALLOWLIST` with the 91 currently-drifted files. Fails only on
NEW drift (a previously-identical file diverges).

`tests/test_kernel_parity.py` wraps the script as a pytest test, skipping
cleanly when the pipeline repo is not available locally.

## Current state

- 169 common files: 78 identical, 91 allowlisted drift, 0 new drift
- 48 umbrella-only files, 32 pipeline-only files
- As files are ported/unified, remove them from the allowlist so re-drift
  is caught automatically
