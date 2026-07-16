# G4: active-scorer provenance fallback for non-JSON panel artifacts

Date: 2026-07-16

## Finding that motivated this (corrects part of model#58)

The model#58 audit concluded "PatchTST per-date score persistence does not
exist" from `data/runs.alpaca.db` (1 hf_patchtst date). That DB is the PROD
arm only. The daily shadow e2e (daily_104.sh Step 4, since 2026-05-19) runs
the FULL pipeline a second time with the shadow scorer and persists to the
isolated `data/runs.alpaca_shadow.db` via the same runner/persistence path:

- 40 distinct live-run dates (2026-05-19 → 2026-07-15)
- `active_scorer=hf_patchtst`: 15 dates, continuous every session since
  2026-06-25 — accruing daily
- `active_scorer=panel_ltr_xgboost`: 11 dates (pre-swap window)

So blocker ① from model#58 (PatchTST per-date persistence wiring) already
exists — no new persistence surface is needed, and building one would
duplicate the existing shadow-arm contract.

## The real remaining gap

#482 extracts `training_cutoff`/`model_content_sha256` in
`build_run_bundle()` by JSON-reading the config's panel artifact path. The
latest shadow run bundle shows its panel path is the HF PatchTST **`.pt`
checkpoint** (`hf_patchtst_all_seed44_model.pt`) — not JSON — so
`panel_payload` is None and the shadow arm would keep persisting NULL
provenance forever, even after the pipeline pin sync. The PatchTST expert's
forward evidence would stay inadmissible for G4 Phase A.

## Fix

In `build_run_bundle()`: when the config-path read yields no
`training_cutoff` and a ctx is present, fall back to the ACTIVE scorer's
runtime metadata contract (`ctx._panel_scorer.metadata`, stamped by
`stamp_artifact_metadata()` — the same contract the calibrator/scorer
fingerprint check already relies on):

- `training_cutoff` ← `effective_train_cutoff_date` (preferred) or
  `trained_date`
- `model_content_sha256` ← `model_content_fingerprint`

This also attributes provenance to the scorer that actually produced the
scores — the config panel path and the active scorer can disagree across
shadow/prod swaps, so the runtime contract is the more truthful source for
the fallback.

Fail-safe: the whole block is wrapped in try/except — a provenance
extraction failure can never break a trading run. The JSON path shipped in
#482 is unchanged (fallback fires only when it yielded nothing).

## Tests

`tests/test_artifact_contract.py` — new `TestRunBundleTrainingProvenance`
(4 tests): JSON artifact sets metadata (covers the #482 path, previously
untested); non-JSON checkpoint + scorer metadata fallback; effective-cutoff
preference; absent scorer metadata is harmless. 12/12 pass.

## What remains for Phase A (blocker chain update)

- ② PatchTST PIT parity ledger (§5.1) — still missing, next work item.
- ③ Evidence volume: shadow PatchTST accrues per-session since 06-25
  (15 dates), so the second expert IS accumulating — but the ~560-session
  frozen-design requirement still makes Phase A a re-registration-or-wait
  decision (separate design PR).
- Phase A tooling must read the SHADOW DB for the PatchTST expert
  (model-repo `backfill_scores.py` currently assumes one DB).
