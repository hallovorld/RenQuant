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

Added `.github/workflows/kernel-parity-ci.yml` — the CI job that checks out
`renquant-pipeline` at its `subrepos.lock.json` pin as a sibling and sets
`RENQUANT_KERNEL_PARITY_STRICT=1`. Without a wired-in job that actually
provides the sibling checkout, "skip cleanly when unavailable" meant this
guard could go green on every PR without ever comparing the two kernel
trees — a genuine no-op dressed up as a passing check (Codex review on
PR #468). `check_kernel_parity.py` now returns a distinct exit code (3) for
"skipped" instead of folding it into the 0 ("passed") case, and
`test_kernel_parity.py` fails (not skips) on that code when
`RENQUANT_KERNEL_PARITY_STRICT=1` is set — i.e. only in the one job that is
supposed to have the sibling present. `_resolve_pipeline_kernel()` also
gained an explicit `RENQUANT_PIPELINE_KERNEL_PATH` override and a
sibling-directory fallback, since a CI runner's checkout layout has nothing
to do with a developer machine's `subrepos.lock.json` `local_path`.

## Current state

- 169 common files: 78 identical, 91 allowlisted drift, 0 new drift
  (verified via `scripts/check_kernel_parity.py -v` against the pinned
  `renquant-pipeline` commit)
- 48 umbrella-only files, 32 pipeline-only files
- As files are ported/unified, remove them from the allowlist so re-drift
  is caught automatically
