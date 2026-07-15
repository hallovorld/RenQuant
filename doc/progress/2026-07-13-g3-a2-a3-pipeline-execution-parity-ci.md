# G3 Phase A: pipeline/execution parity CI (A2 + A3 relocation)

Date: 2026-07-13
Registry items: A2, A3 (see `doc/arch/2026-07-13-g3-refactoring-plan.md` Phase A)
PR: g3/a2-a3-pipeline-execution-parity-ci
Relates to: renquant-orchestrator#515, umbrella#468 (F-6, same pattern)

## Problem

`renquant-orchestrator#515` added 6 read-only pytest tests
(`tests/test_cross_repo_parity.py`) that imported `renquant_pipeline` and
`renquant_execution` from local sibling directories to check:

- A2: `MIN_FRACTIONAL_NOTIONAL_USD` value equality + `compute_parent_intent_id`
  golden-vector parity (3 vectors + signature match) between the two repos.
- A3: an upper-bound inventory of non-canonical `pandas_market_calendars`
  imports in the pipeline kernel (baseline 7).

Codex review on that PR flagged the same no-op-green pattern already fixed
here for F-6 (umbrella PR #468): renquant-orchestrator is the wrong owner
for a pipeline/execution contract test, and its normal CI is allowed to skip
these tests (no job checks out the sibling repos), so a green orchestrator
build proved none of the actual invariants — orchestrator can merely *see*
the siblings as local directories on a developer machine.

## Fix

Relocated the comparison to this repo, which owns `subrepos.lock.json` (the
exact pipeline/execution pins) and already has the F-6 precedent for a
strict, sibling-checkout CI job:

- `scripts/check_pipeline_execution_parity.py` — new script performing both
  checks. Resolves each of the 5 required sibling `src` dirs (renquant-common,
  renquant-base-data, renquant-artifacts, renquant-pipeline,
  renquant-execution — pipeline transitively needs the first three; a plain
  `import renquant_pipeline...` triggers `renquant_pipeline/__init__.py`,
  which pulls in the whole decisioning stack) via the same 3-step resolution
  order as `check_kernel_parity.py`: env var override → `subrepos.lock.json`
  `local_path` → `../<repo>` filesystem-sibling fallback. Returns exit code
  0 (pass) / 1 (drift) / 2 (setup error) / 3 (skipped — sibling(s) missing).
- `tests/test_pipeline_execution_parity.py` — pytest wrapper, same
  strict/non-strict skip-vs-fail contract as `test_kernel_parity.py`, gated
  on `RENQUANT_PIPELINE_EXECUTION_PARITY_STRICT=1`.
- `.github/workflows/pipeline-execution-parity-ci.yml` — the ONE job that
  checks out all 5 sibling repos at their `subrepos.lock.json`-pinned
  commits (same checkout-with-token pattern as `kernel-parity-ci.yml`) and
  runs the wrapper test under the strict flag, so a green run genuinely
  proves the comparison happened rather than skipping.

## Renquant-orchestrator#515

`tests/test_cross_repo_parity.py` and its progress doc were removed from
that PR entirely (no non-authoritative stub kept) — the checks now live
here, where a real CI job enforces them; keeping a duplicate in orchestrator
would re-create the exact wrong-owner problem Codex flagged. Orchestrator
PR #515 was updated to link here instead.

## Verification

Ran directly against real pinned-commit sibling checkouts (git worktrees at
the exact `subrepos.lock.json` commits, not the possibly-ahead local dev
checkouts), using the project's `.venv` (Python 3.10.20, matching this
repo's `Path | None` syntax; the system `python3` here is 3.9 and fails on
that syntax for unrelated reasons):

```
MIN_FRACTIONAL_NOTIONAL_USD: OK — pipeline=1.0 execution=1.0
compute_parent_intent_id: OK — [] (0 golden-vector mismatches / 3 vectors)
calendar_import_inventory: OK — 7 non-canonical pandas_market_calendars import(s) (baseline 7)
PASS: pipeline/execution parity checks all OK (3 checks)
```

Also verified, in an isolated directory with no `subrepos.lock.json` and no
sibling checkouts present:
- non-strict mode → real `SKIP` (not a false pass)
- `RENQUANT_PIPELINE_EXECUTION_PARITY_STRICT=1` mode → real `FAIL` with a
  message pointing at the CI job's checkout steps

`.github/workflows/pipeline-execution-parity-ci.yml` passes `actionlint`
with no errors.

### Real CI run caught a real gap

The first push of this branch triggered `pipeline-execution-parity-ci` for
real on the PR and it FAILED — not skipped — with
`SETUP ERROR: No module named 'pydantic'`. Unlike `check_kernel_parity.py`
(pure text-file diffing, no imports), this check actually imports
`renquant_pipeline`/`renquant_execution`, which transitively pulls in every
third-party dependency declared across `renquant-common` (`arch`, `numpy`,
`pandas`, `pandas_market_calendars`, `pyarrow`, `pydantic`, `scipy`,
`statsmodels`), `renquant-base-data` (`requests`), and `renquant-pipeline`
itself (`cvxpy`) — none of which the job installed (only `pytest`). Fixed by
adding an explicit `pip install` step for that dependency set (installed by
name/version-range, not `pip install ./renquant-common` etc., so pip never
tries to resolve the *private* renquant-* packages from PyPI — those stay
resolved via `sys.path`, as the script already does). Re-verified locally in
a from-scratch virtualenv with only that dependency list installed (no
system-wide packages): all 4 sibling-module imports succeed. This is exactly
the kind of gap RENQUANT_PIPELINE_EXECUTION_PARITY_STRICT=1 is supposed to
surface as a hard CI failure instead of a silent skip — working as intended.

## Current state

- All 3 checks pass against the pins recorded in `subrepos.lock.json` as of
  2026-07-13 (`renquant-pipeline` 289b9199, `renquant-execution` 42e5d7d7).
- The calendar-import baseline (7) is a temporary migration guard per Phase
  B2 of the G3 plan — it does not block PRs on the existing count, only on
  growth beyond it.
- Codex review (COMMENTED, positive) confirms the relocation is correct and
  the CI job genuinely executes. Awaiting formal APPROVE to merge.

## Codex hardening follow-up (2026-07-14, blocking review)

Codex's second, blocking pass raised four security/reproducibility points
about the CI design plus an out-of-band identity problem (this branch's
commits were authored under the `haorensjtu-dev` identity despite the PR
being opened by `hallovorld` — a known credential-sharing artifact on the
shared dev machine, same class as RenQuant#471). Fixes, in order:

1. **Push-capable credential removed from a job that only needs read.**
   `pipeline-execution-parity-ci.yml` previously fell back to
   `secrets.AGENT_GIT_PUSH_TOKEN` (a contents:write + pull-requests:write PAT
   scoped to this repo, documented in `.github/workflows/README.md` as
   existing for the G3 agent-fix template to push fix commits — copy-pasted
   into this job's checkout-token fallback chain from
   `kernel-parity-ci.yml` / `subrepo-pin-ci-green.yml` /
   `strategy-104-snapshot-fresh.yml`, none of which need push either).
   Confirmed this job is pure read-plus-report (no step ever pushes or
   mutates a checkout) and removed the `AGENT_GIT_PUSH_TOKEN` fallback
   entirely; the token chain is now `secrets.SUBREPO_CI_READ_TOKEN ||
   github.token` only. Added `persist-credentials: false` to all 6
   `actions/checkout` steps (umbrella + 5 pins) — confirmed no later step
   needs the checkout token to remain in `.git/config` (every later step
   only reads files or runs Python; nothing pushes or re-authenticates).
   Two-stage design: this repo has no prior example of a fully separate
   no-secret-PR-stage / trusted-main-stage split (checked all 4 existing
   workflows here — `kernel-parity-ci.yml`, `subrepo-pin-ci-green.yml`,
   `strategy-104-snapshot-fresh.yml` all use the same read-token on
   `pull_request` too), and a literal zero-secret PR run isn't achievable
   without dropping pre-merge signal (the 5 sibling repos are private, so
   *some* credential is required to check them out at all). Given that
   constraint, the read-only-token change is what actually shrinks blast
   radius for a `pull_request` run (can read already-intentionally-checked-
   out private repos; cannot push anywhere); the `push`-to-`main` run of the
   same job — only reachable after a lock-changing PR is reviewed and
   merged — is the trusted integration execution, and its run record (point
   3) is tagged `trigger`/`trusted` so a pre-merge validation run's record
   is distinguishable from a post-merge trusted one. This is a proportionate
   approximation of Codex's ask, not a from-scratch two-workflow split —
   disclosed as such rather than overclaimed.

2. **Hashed, pinned dependency environment.** Added
   `.github/workflows/pipeline-execution-parity-requirements.lock.txt`: every
   package the job installs (the 10 third-party packages
   `renquant_pipeline`/`renquant_execution` transitively need, plus `pytest`
   to run the wrapper test) pinned to an exact version with a `sha256`
   `--hash=` for every transitive dependency too, installed via
   `pip install --require-hashes` (refuses anything not listed with a
   matching hash). Generated by `pip download --platform
   manylinux2014_x86_64/manylinux_2_17_x86_64/manylinux_2_28_x86_64
   --python-version 311 --implementation cp --abi cp311 --only-binary=:all:`
   for the exact runner target (`actions/setup-python@v5`
   `python-version: "3.11"`, newly pinned — the job previously relied on
   whatever `python3` ubuntu-latest happened to ship). Kept the original
   ranges' *intent* rather than grabbing newest-available: capped
   `pandas<3.0` and `arch<8` (in addition to the pre-existing
   `pandas_market_calendars<5`, `pydantic<3.0` ceilings) because an
   unconstrained resolve during lock generation picked pandas 3.0.3 and
   arch 8.0.0 — untested new majors over the 2.x/7.x lines this repo's own
   dev environment (`requirements.lock.txt`, `environment.yml`) is actually
   validated against; capping landed the resolver back on pandas==2.3.3 and
   arch==7.2.0, matching that known-good environment exactly. Digest of the
   lock file is computed fresh every run (`sha256sum`) and recorded in the
   run record / step summary — not hardcoded anywhere, so it can't go stale.

3. **SHA-equality assertion + machine-readable run record.** Added a
   "Verify checkouts match subrepos.lock.json exactly" step that runs `git
   rev-parse HEAD` in each of the 5 checked-out sibling dirs and fails
   closed (before any pinned code executes, before dependency install even)
   if any doesn't equal the lock file's declared commit for that repo. Added
   a "Build machine-readable run record" step (`if: always()`, so a FAILING
   parity check still produces an audit record) that assembles JSON via
   `jq`: umbrella SHA, all 5 pin SHAs, the dependency-environment digest,
   `trigger`/`trusted` (event name), and the parity script's own structured
   per-check payload (new `--json-out` flag on
   `scripts/check_pipeline_execution_parity.py`, additive — the existing
   pytest wrapper's `--verbose` text-output contract is unchanged). Written
   to `$GITHUB_STEP_SUMMARY` and uploaded as a
   `pipeline-execution-parity-run-record` artifact (90-day retention).

4. **Calendar-import baseline reframed as tracked temporary debt.** The
   script's module docstring, the `CALENDAR_BASELINE = 7` constant, and the
   check's failure-detail string now explicitly say this is a TEMPORARY
   debt-inventory ceiling (blocks growth only; does NOT prove the 7 existing
   sites are compliant, and does NOT prove pipeline/execution behavior is
   integrated end-to-end), reference owning issue
   [hallovorld/RenQuant#475](https://github.com/hallovorld/RenQuant/issues/475)
   (opened for this), and state the retirement condition: once G3 Phase B
   task B2 migrates all 7 sites to `renquant_common.market_calendar`, lower
   `CALENDAR_BASELINE` to 0 and close #475.

### Verification

- `actionlint .github/workflows/*.yml` — clean, no errors.
- `pip install --no-index --find-links <downloaded wheels> --require-hashes
  -r pipeline-execution-parity-requirements.lock.txt` inside a clean
  `python:3.11-slim` (`--platform linux/amd64`) container — succeeded, all
  44 packages installed, every hash matched.
- `tests/test_pipeline_execution_parity.py` re-run locally: non-strict mode
  → real SKIP (unchanged behavior); `RENQUANT_PIPELINE_EXECUTION_PARITY_STRICT=1`
  → real FAIL (unchanged contract).
- `scripts/check_pipeline_execution_parity.py --json-out` verified to write
  a correctly-shaped JSON record (including on the exit-code-2 setup-error
  path).
- **Disclosed gap:** full import-level verification of every direct package
  (`pandas`, `pandas_market_calendars`, `pyarrow`, `statsmodels`, `arch`)
  inside the linux/amd64 container hit `qemu: uncaught target signal 11
  (Segmentation fault)` under this machine's QEMU cross-arch user-mode
  emulation (arm64 host emulating x86_64) — a known limitation of binary
  translation for SIMD-heavy compiled extensions, not a real issue on actual
  x86_64 hardware (`numpy`, `pydantic`, `scipy`, `requests`, `cvxpy` imported
  fine in the same container). The authoritative check is the real
  `pipeline-execution-parity-ci` run on GitHub's ubuntu-latest (genuine
  x86_64) runner, triggered by pushing this branch.

### Identity rebuild

Per the now-merged single-identity-branch policy
(`doc/agent-pr-workflows.md`, renquant-orchestrator#517): this branch's
existing 3 commits were authored under `haorensjtu-dev` despite the PR being
opened by `hallovorld`. Rebuilt as one `hallovorld`-authored commit on top of
current `main`, no `Co-Authored-By` trailers. Old branch tip:
`ac8fc503dd3d7f10fa051fe617df857f677d5110`; new tip recorded in the PR
comment for this fix round.
