# G3 F-7: sim strict-pinned default + manifest-based experiment mode (r5)

Date: 2026-07-14
PR: #471

## Problem

r4 added `--experiment-config` + `--experiment-id` as a free-form escape hatch
for experiment mode, but:
1. `--experiment-id` accepted any string with no lookup against a registered
   manifest — no config-digest binding, no pin tracking, no status gating.
2. `_exploratory_only` was only set in-memory; no durable classification file
   existed and no promotion/live guard consumed it.
3. Two completed-experiment scripts (`_doe_orchestrate_bb.sh` and
   `run_parallel_after_trail015.sh`) used `--strategy-config-name` with local
   sweep configs, which breaks the strict-pinned default.

## Fix

### 1. Experiment manifest schema

Replaced `--experiment-config` + `--experiment-id` with `--experiment-manifest`.
The manifest is a JSON file with required fields:
- `experiment_id`: registered experiment/run identifier
- `config_path`: path to the strategy config file
- `config_digest`: `sha256:`-prefixed digest of the config file at registration
- `status`: one of `ACTIVE`, `COMPLETED`, `RETIRED`

Optional: `pins` (dict of subrepo→commit for reproducibility).

`load_experiment_manifest()` validates all required keys, status, config file
existence, and verifies the config_digest against the actual file content.
RETIRED manifests are rejected. Digest mismatch (config modified since
registration) is rejected.

### 2. Durable EXPLORATORY_ONLY classification + promotion guard

`write_experiment_classification()` writes
`_experiment_classification.json` alongside experiment outputs, containing
the classification, experiment_id, manifest_path, and config_digest.

`reject_exploratory_promotion()` checks a directory for this marker and
raises ValueError if found. Promotion and live-admission code must call this
before accepting sim output.

### 3. Legacy script retirement

Both `_doe_orchestrate_bb.sh` and `run_parallel_after_trail015.sh` are from
completed experiments (BB DOE sweep, re-evaluation queue). Both now exit 1
immediately with a message pointing to the manifest-based migration path.
Original code preserved below the exit for reference.

## Tests (r5)

43 tests pass across both test files:
- `test_resolve_strategy_config.py`: 14 tests (pin verification + resolver)
- `test_run_sim_104_config_resolution.py`: 29 tests (strict-pinned default,
  manifest validation, digest verification, status gating, RETIRED rejection,
  relative config paths, pins support, durable classification file,
  promotion guard rejection + acceptance, legacy script retirement,
  fingerprint format)

## r6: pins required + manifest path restriction + atomic classification

Codex round-5 review kept the block: `reject_exploratory_promotion()` was an
unreferenced helper, `pins` was optional and unverified, and any absolute
`--experiment-manifest` path was accepted. r6 (commit 3bbceaa) addressed the
schema-level parts:
- `pins` became a required manifest field with the 5 category keys required
  present (`data_snapshot`, `model_artifact`, `strategy_config`,
  `pipeline_version`, `calendar_universe`).
- `--experiment-manifest` must resolve under `experiments/manifests/`.
- The `_experiment_classification.json` marker moved earlier (before
  `run_backtest()`) and gained a `manifest_digest` field, written atomically
  (tmp+rename).

Codex's round-5 review (quoted in full in the PR#471 comment thread) kept
the block anyway: pins being *required* is not the same as pins being
*verified* against reality, "lives under the right directory" is not the
same as "registered," and nothing outside `run_sim_104.py` consumed the
marker — so it still wasn't enforcement.

## r7: verified pins + manifest registry + real promotion-boundary wiring

Resolves all 3 remaining round-5 findings by moving the governance contract
into `renquant_artifacts` (the canonical artifact-registry/validation
package every real promotion path already imports) and making
`run_sim_104.py` a caller of that contract rather than a second
implementation of it. Companion PR: renquant-artifacts g3/f7-experiment-registry.

**Finding 1 (marker is not enforcement).** `reject_exploratory_promotion`
now lives in `renquant_artifacts.experiment_registry` and is wired into
`renquant_artifacts.validation.ValidateArtifactManifestTask.run()` — the one
function every real promotion/admission caller across the multirepo funnels
through (`renquant_pipeline.inference.ValidateRuntimeInputsTask` for live/
shadow/sim runtime; `renquant_artifacts.registry.{load,resolve}_artifact_manifest`
for registry resolution). A candidate artifact manifest that declares
`"provenance_dir": "<a run_sim_104.py experiment output dir>"` is rejected if
that directory carries an EXPLORATORY_ONLY classification.
`renquant-artifacts/tests/test_experiment_registry.py::TestPromotionBoundaryIntegration`
and this repo's `tests/test_run_sim_104_config_resolution.py::TestPromotionBoundaryIntegration`
both prove this against the REAL (non-mocked) `validate_artifact_manifest`/
`load_artifact_manifest`/`resolve_artifact_manifest`.

**Finding 2 (pins optional and unverified).** `run_sim_104.py` now requires
two additional manifest fields, `data_manifest_path` and
`model_artifact_path`, and calls the new
`renquant_artifacts.verify_experiment_pins()` (in
`verify_and_classify_experiment()`, right after the config is loaded and
before classification is written) to check all 5 pins against the ACTUAL
environment:
- `strategy_config` / `pipeline_version`: the caller's `subrepos.lock.json`
  entry, cross-checked against the live checkout's HEAD/dirty/remote state
  (same discipline as `_verify_pin`, generalized).
- `data_snapshot`: the `fingerprint` field of the data manifest at
  `data_manifest_path` (schema: `renquant_base_data.validate_data_manifest`).
- `model_artifact`: the full-file hash of `model_artifact_path`
  (`renquant_common.model_fingerprint.artifact_sha256`).
- `calendar_universe`: a stable hash of the resolved strategy config's
  `watchlist`.

A category whose supporting evidence is missing is a hard error, not a
silent skip. Fails closed (`sys.exit(1)`) BEFORE the classification marker
is written if any pin fails to verify.

**Finding 3 (arbitrary manifest path).** Added
`experiments/manifests/INDEX.json` — an immutable, git-tracked registry
index mapping `experiment_id -> {"digest": ..., "path": ...}`.
`load_experiment_manifest()` now requires the manifest's own file digest to
match a registered entry (`renquant_artifacts.verify_manifest_registered`),
in addition to the existing directory restriction — living under
`experiments/manifests/` is necessary but not sufficient; the content must
also match a deliberately-registered digest. See
`experiments/manifests/README.md` for the registration process.

## Tests (r7)

- `renquant-artifacts` (companion PR): 85 tests pass, 0 regressions
  (`tests/test_experiment_registry.py` is new, 40 tests).
- This repo: 103 tests pass across `test_run_sim_104_config_resolution.py`
  (57 tests, including the new `TestManifestRegistry`,
  `TestVerifyAndClassifyExperiment`, and `TestPromotionBoundaryIntegration`
  classes), `test_resolve_strategy_config.py` (14 tests, one pre-existing
  test updated to register its manifest per the new contract),
  `test_monthly_jobs_multirepo_fail_closed.py`, and `test_wf_gate_cli_contract.py`.
- Full `tests/` suite run for regressions: baseline (r6, unmodified) vs.
  this branch, same environment. The whole-suite parallel run (`pytest -n
  auto`) is order/worker-dependent and noisy independent of this change
  (repeat baseline runs varied 66-137 failures depending on PYTHONPATH
  completeness alone); after normalizing the environment
  (`renquant-pipeline` importable, which the ambient `.venv` was missing on
  both baseline and branch), branch failures (68) are a strict subset-ish
  improvement over baseline (137) with zero failures attributable to this
  change — the sole failure unique to the branch run
  (`test_software_stops.py::TestStalenessWatchdog::test_cli_exit_codes`,
  unrelated to run_sim_104/renquant_artifacts) passes cleanly in isolation,
  confirming xdist-order flakiness rather than a regression.

## r8: provenance made a required, immutably-bound record (closes the r7 gap Codex found on artifacts#24)

r7 was pushed to `origin/g3/f7-sim-pinned-config` (commit `8accd7f`) claiming
the promotion boundary was real enforcement. Codex reviewed the companion
artifacts#24 PR separately and found r7's own claim still bypassable
(`CHANGES_REQUESTED`, 2026-07-14T16:42:29Z, quoted in full in the artifacts
repo's `doc/progress/2026-07-14-experiment-registry-promotion-gate.md`):
`ValidateArtifactManifestTask` only checked `reject_exploratory_promotion()`
when a manifest happened to supply `provenance_dir` — an experiment-derived
candidate could be promoted by simply omitting it, and even when supplied,
a missing classification marker at that path was silently accepted as "not
exploratory." Codex separately left the same finding as an issue comment on
THIS PR (2026-07-14T16:42:46Z), plus a distinct dependency-ordering concern
(see below).

**Fix (entirely in renquant-artifacts#24 — see that PR's progress doc for
the full design):** `provenance` is now a REQUIRED, typed field on every
candidate artifact manifest, bound to the immutable experiment-manifest
registry index (`experiments/manifests/INDEX.json`) rather than a bare
caller-supplied path, and `reject_exploratory_promotion()` fails closed on
a missing/unregistered/falsified marker instead of silently passing.

**What changed in THIS repo (r8):**

- `verify_and_classify_experiment()` now builds the canonical `provenance`
  reference via `renquant_artifacts.build_experiment_provenance_reference()`
  (using the registry index path it already resolved in
  `load_experiment_manifest()`, now threaded through as
  `manifest_data["_registry_index_path"]`) and logs it verbatim, so any
  future code that builds an artifact manifest from this run's output has
  the exact, correctly-shaped reference to copy — not a hand-typed
  `provenance_dir` string.
- `tests/test_run_sim_104_config_resolution.py`:
  `TestExploratoryClassification`'s two "accept when no marker" tests are
  renamed and now assert the new fail-closed `ValueError` (documented in
  each docstring). `TestPromotionBoundaryIntegration` gained two new
  end-to-end negative tests against the REAL (non-mocked)
  `renquant_artifacts.validate_artifact_manifest`: one where provenance is
  omitted entirely from a candidate built from a genuinely EXPLORATORY_ONLY
  run, and one where provenance points at an empty decoy directory instead
  of the real run's output — both now correctly rejected.

**Dependency-ordering note (Codex's separate finding on THIS PR, not yet
addressed — flagging, not fixing, since it requires a coordinated
multi-repo pin bump outside this PR's diff):** this repo's
`subrepos.lock.json` still pins `renquant-artifacts` at `c09d66f8...`,
whose `main` branch does not export the symbols `run_sim_104.py` now
imports (`verify_experiment_pins`, `verify_manifest_registered`,
`write_experiment_classification`, `reject_exploratory_promotion`,
`build_experiment_provenance_reference`). Merging this PR before
artifacts#24 merges AND the umbrella pin is bumped would break the pinned
environment. Required order: merge/fix artifacts#24 → bump this repo's
`renquant-artifacts` pin → merge this PR → materialize a clean pinned
checkout and re-run the experiment-mode + fail-closed integration tests.
Not resolved in this commit; do not merge out of this order.

## Tests (r8)

- `renquant-artifacts` (companion PR): 93 tests pass (0 regressions; new
  `TestProvenanceBypassClosed` class, 8 tests, proves the exact
  missing/falsified-provenance scenarios Codex asked for).
- This repo: 105 tests pass across the same 4 F-7-relevant test files (was
  103; net +2 from the new negative tests, with 2 pre-existing tests'
  assertions updated in place for the new fail-closed semantics).
- Full `tests/` suite: ran twice against an UNMODIFIED r7 baseline (same
  environment, `renquant-pipeline` on `PYTHONPATH`) to establish the noise
  floor — 76 failed/35 errors, then 64 failed/35 errors on a second
  identical run of the SAME code. The r8 branch run (93 failed/1 error)
  diffed against both baseline runs by test name: every differing test
  belongs to files this PR does not touch (`test_walkforward_loader.py`,
  `test_wf_loader_fingerprint_dispatch.py`, `test_umbrella_gates_ledger.py`,
  `test_sim_pipeline_smoke.py`, `test_walkforward_manifest.py`, etc.), and
  the same tests already flip between the two identical-code baseline
  runs — confirming `pytest -n auto` order/worker flakiness, not a
  regression from this change.

## Round 9 (2026-07-14): connect experiment output to artifact publication

**Trigger:** Codex round-3 follow-up review on companion PR
renquant-artifacts#24, quoted verbatim (full context in that repo's
progress doc, `2026-07-14-experiment-registry-promotion-gate.md`):

> `run_sim_104.py` only logs the reference returned by
> `build_experiment_provenance_reference()`; it does not emit an artifact
> manifest or bind that reference into the registry publication path...
> Fix the ownership boundary rather than strengthening another
> caller-supplied dict.

Confirmed by reading the code: `verify_and_classify_experiment()` built
the `provenance` reference and only `log.info`'d it — nothing in this
script ever wrote an actual artifact manifest file. That's the real gap:
manifest construction was left entirely to some future, disconnected
caller, which is exactly how a dishonest `kind="none"` substitution became
possible in the first place (renquant-artifacts#24 round 3 closes the
validator side of that; this round closes the producer side).

**What changed in THIS repo (r9):**

- `verify_and_classify_experiment()` now returns its `output_dir` (was
  computed but discarded by the caller); `main()` captures it as
  `experiment_output_dir`.
- New `write_candidate_artifact_manifest(output_dir, *, manifest_data,
  sim_metrics)`: builds the candidate-artifact manifest for this run's sim
  output — `fingerprint`/`local_artifact_path`/`uri` derived from the
  experiment manifest's own `_model_artifact_path` (via
  `renquant_common.model_fingerprint.artifact_sha256`, the one canonical
  hash impl), `metrics` from the sim result (`apy`/`sharpe`/`max_dd`/
  `n_trades`, `accepted: False`), and — the point of this round —
  `provenance` set directly from `build_experiment_provenance_reference()`
  at the source, not left for a separate caller to reconstruct or assert.
  Written atomically (`.tmp` + rename, same discipline as
  `write_experiment_classification`) to
  `<output_dir>/candidate_artifact_manifest.json`.
- `main()` calls it right after `result.print_summary()` for any
  `EXPLORATORY_ONLY` run.
- `tests/test_run_sim_104_config_resolution.py`: new
  `TestCandidateArtifactManifestEmission` class (3 tests): (1) the written
  manifest's `provenance` matches
  `build_experiment_provenance_reference()`'s output exactly; (2) the real
  `renquant_artifacts.validate_artifact_manifest` correctly refuses to
  promote this manifest (EXPLORATORY_ONLY, as intended — the manifest is
  supposed to be un-promotable, not un-writable); (3) THE proof this round
  closes the round-3 bypass end-to-end: a dishonestly hand-built manifest
  that reuses this SAME output's `local_artifact_path` but declares
  `provenance={"kind": "none"}` is rejected by renquant-artifacts' round-3
  fix (`_verify_none_provenance`), i.e. wiring this repo's real output into
  a real, on-disk-verifiable location is what makes the "none" lie
  detectable.

## Tests (r9)

- `renquant-artifacts` (companion PR, round 3): 96 tests pass (was 93; net
  +3, 1 renamed) — see that repo's progress doc for the full before/after
  proof of the `kind="none"` bypass closure.
- This repo: `test_run_sim_104_config_resolution.py` 49/49 (was 46; +3 new
  in `TestCandidateArtifactManifestEmission`). The 5 F-7-adjacent test
  files together: 116/116.
- Full `tests/` suite (`pytest -n auto`, ~22k tests) run against the r9
  branch: 93 failed/2 errors. Two identical-code baseline runs (this
  branch's `scripts/run_sim_104.py` and
  `tests/test_run_sim_104_config_resolution.py` reverted via `git stash`,
  everything else unchanged): 73 failed/2 errors, then a re-run of the SAME
  reverted code with the fix restored/re-diffed showed the failing-test SET
  itself differs between runs of IDENTICAL code (confirmed by diffing test
  names, not just counts) — the same `pytest -n auto` order/worker noise
  round 8 already documented. Extra check this round (not done in r8):
  every test name that differs between the branch run and either baseline
  was re-run STANDALONE (`pytest <file>`, no xdist) and passes 100% in
  isolation — e.g. `tests/test_shadow_scoring.py`,
  `tests/test_umbrella_gates_ledger.py`, `tests/test_sim_pipeline_smoke.py`,
  `tests/test_training_modules.py` all green alone, confirming the
  divergence is suite-order/global-state pollution under parallel workers,
  not a regression introduced by this round's diff (which touches exactly
  `scripts/run_sim_104.py` and its own test file).

## Sequencing (Codex's explicit order, not yet complete)

renquant-model's `BuildArtifactManifestTask`/`BuildPatchTstArtifactManifestTask`
were updated FIRST (branch `fix/f7-provenance-none`, not yet a PR at time
of writing — full 796-test suite passes against renquant-artifacts#24
round 3), then renquant-artifacts#24 round 3 (this round's companion), then
this repo's r9. None of the three are merged. The umbrella
`subrepos.lock.json` pin bump for `renquant-model`/`renquant-artifacts` is
an explicit follow-up once all three have Codex's approval — do not merge
or pin out of that order.
