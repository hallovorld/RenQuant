# renquant_104 Codex Fix Batch 1

Status: implemented, tested, not promoted.

This batch implements the first production-hardening slice from the Codex
review while leaving PatchTST / transformer in shadow mode.

## What Changed

1. Artifact contract and run provenance
   - New module: `backtesting/renquant_104/kernel/artifact_contract.py`.
   - Validates panel artifacts for required training/evaluation metadata.
   - Builds a run bundle containing config hash, watchlist hash, artifact
     paths/hashes, panel contract status, pipeline flags, and OHLCV max dates.

2. Daily run persistence
   - `pipeline_runs` now has `run_bundle_json`.
   - `RunnerAdapter` and `SimAdapter` persist the run bundle on each run.
   - This makes future daily e2e audits reconstructable without trusting docs.

3. Training dry-run gate
   - `scripts/train_104.py --dry-run` runs preflight and artifact-contract
     checks without training or writing artifacts.
   - `--strict-contract` hard-fails legacy artifacts that lack OOS evidence.

4. QP mu semantics guard
   - New `ValidateQPMuContractTask` in the QP pipeline.
   - Default config is `rotation.joint_actions.qp_mu_contract = "warn"`.
   - It warns and increments counters when raw rank/panel scores reach QP as
     mu without `ranking.alpha_to_mu.enabled`.
   - Promotion/dry-run configs can set `"strict"` to stop QP before solve.

## Current Finding

The current production panel artifact is legacy-compatible but not strict:

- Missing `train_run_id`
- Missing `oos_mean_ic`
- Missing `oos_std_ic`
- Missing `oos_per_fold_ic`
- Missing `cv_method`
- Missing `cv_embargo_days`

Non-strict dry-run passes so daily operations are not broken. Strict dry-run
correctly fails until a clean retrain stamps these fields.

## Commands Run

```bash
.venv/bin/python -m pytest \
  tests/test_artifact_contract.py \
  tests/test_persistence.py \
  tests/test_qp_grinold_kahn_transform.py \
  tests/test_qp_force_mu_source.py -q

.venv/bin/python scripts/train_104.py --dry-run --skip-baseline --skip-recalibrate

.venv/bin/python scripts/train_104.py --dry-run --strict-contract \
  --skip-baseline --skip-recalibrate
```

Results:

- Targeted pytest: 36 passed.
- Non-strict dry-run: passed, with legacy panel-contract warnings.
- Strict dry-run: failed as intended on missing panel OOS metadata.

## Next Required Step

Run the next clean panel retrain only after confirming the training pipeline
stamps all strict contract fields. Promotion should require:

1. `python scripts/train_104.py --dry-run --strict-contract`
2. Full targeted pytest for changed areas.
3. Existing acceptance / weekly WF promotion gates.

Do not promote PatchTST / transformer out of shadow mode in this batch.
