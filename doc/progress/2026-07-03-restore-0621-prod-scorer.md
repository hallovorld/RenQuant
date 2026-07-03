# Fix: restore the 06-21 promoted prod scorer (reverts the silent 06-26 rollback)

2026-07-03. Remediation for the incident diagnosed in orchestrator PR #274
(`doc/research/2026-07-03-raw-jump-0626-diagnosis.md`).

## Incident chain

1. **06-22**: operator promoted the 2026-06-21-trained XGB panel scorer
   (booster `a9b1a075…`, oos_mean_ic 0.0533) to prod. Promotion lives as
   working-tree state by design — uncommitted.
2. **06-25**: the agent-reset recovery (`git checkout -B main origin/main`,
   postmortem #412) reverted the prod artifact to the committed 05-18 blob
   (`5a1e4e14…`, booster `a6f5a22f…`, oos_mean_ic 0.0447). The revert was
   noticed but misread as a stale config-fingerprint stamp.
3. **06-25**: the fingerprint was re-stamped (`14586756` → `f8fb2259`) over
   the wrong (05-18) booster and PR #413 was opened to commit it "for
   durability". #413 was **closed unmerged** — so origin/main still carries
   the even-staler `5a1e4e14…` blob (fingerprint `14586756`, which would
   fail P-CONFIG-FP), while the live tree runs the restamped 05-18 copy
   (`5ce63326…`) as uncommitted state.
4. **06-26 onward**: prod ran the rolled-back 05-18 scorer. Raw-score
   cross-sectional median jumped −0.297 → −0.047 at exactly that boundary
   (#274 §1). The 05-18 model is 46 days old at 2026-07-03 — violating the
   2026-06-30 model-freshness directive (28-day cap).

## What this PR does

Restores `backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json`
to the exact promoted bytes of the 06-21 model, recovered from the live
tree's `panel-ltr.alpha158_fund.weekly_rollback_2026-06-23.json` snapshot,
and commits it (the durability lesson from #412/#413, this time with the
right booster).

## Verification chain (all byte-verified before commit; see
`scripts/verify_prod_scorer_restore_20260703.py`)

| check | value |
|---|---|
| 06-25 run bundle stamped panel sha (`pipeline_runs['2026-06-25-live-6c3aa3fa'].artifact_hashes.panel`, `runs.alpaca.db` mode=ro) | `sha256:04d7a381cd6df84721dd938ce74a297cbf3eda9d5bc3385515bc155014dd5b08` |
| `weekly_rollback_2026-06-23.json` file sha256 (recovery source) | `04d7a381cd6df847…` — **identical** |
| committed prod artifact file sha256 (this PR) | `04d7a381cd6df847…` — **byte-identical restore** |
| booster content sha256 (#274 identity: sha256 over `json.dumps(booster_raw_json, sort_keys=True)`) | `a9b1a07533a028588f7fe12b9917108bf9b31af35a388fc8249fcca8ea970bfe` (= #274 family A) |
| trained_date / oos_mean_ic / label_col / n_features | 2026-06-21 / 0.0533246294… / fwd_60d_excess / 172 |
| stamped `config_fingerprint` | `sha256:f8fb2259b2bf1537` |
| `fingerprint_config` recomputed over the PINNED strategy-104 config (`.subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.json`, runtime checkout c019b256) | `sha256:f8fb2259b2bf1537`, `_model_relevant_fields` diff = `[]` |
| rolled-back model being replaced | file `5ce63326…`, booster `a6f5a22f…`, trained 2026-05-18, oos 0.0447 |

**No re-stamp was needed**: the 06-22 promotion had already stamped the
snapshot against the grown (watchlist-145) pinned config — verified zero
field diff with the same canonical `kernel.config_consistency` the runtime
uses (the #413 convention). Deliberately NOT re-stamped, preserving byte
identity with the run-bundle sha. P-CONFIG-FP passes as-is.

## Calibrator ruling (flagged, deliberately not fixed here)

The live calibrator (`panel-rank-calibration.json`, trained_date
**2026-07-01**) was fit against the **rolled-back 05-18 scorer**: its
`scorer_model_content_fingerprint` `sha256:9c4bbd74…` equals
`model_content_sha256` of the 05-18 artifact (verified with the pinned
renquant-pipeline implementation — the one the runtime checks). The restored
06-21 artifact's content fingerprint is `sha256:6fc9985e6e53e3883a13d572b1e11b7dea484e14df23819dab5474f2cd592372`.

- With `strict_scorer_match: true` (pinned config),
  `LoadGlobalCalibrationTask` **fail-closes buys** on the mismatch — so
  post-restore there is **no silent mis-paired scoring window**; the window
  is safe-dark (sell-only) until the calibrator is refit.
- **A refit is therefore required as part of the landing**, not passively
  "next weekly cycle": run the weekly calibrator refit against the restored
  prod artifact (it fits whatever is at the prod path and stamps its content
  fingerprint), then verify with
  `verify_prod_scorer_restore_20260703.py --require-calibrator-parity`.
- Do NOT re-stamp the calibrator instead of refitting: per #274 §3 the two
  boosters differ by +0.13 ± 0.12 raw-units per row (identical-rows test)
  with rank agreement only Spearman 0.765 — a re-stamped-but-not-refit
  pairing would read μ optimistic by ~+0.13 raw units through the ER slope.
- Note the **current** pairing is the mis-calibrated one: calibrator neutral
  raw ≈ −0.290 vs the live 05-18 cross-section center ≈ −0.05 (+0.24 above
  neutral) — the source of #274's +2% μ intercept and 44-of-45
  sign-laundering. The restored model's live center (median −0.297,
  06-22..25 runs) sits at the neutral; restore + refit removes the artifact.

## Freshness compliance

Restored model trained 2026-06-21 → 12 days old at restore, within the
28-day directive; the next weekly retrain supersedes it through the normal
promote gate. The 05-18 model it replaces is 46 days old (violates the
directive).

## Deploy note (merged ≠ deployed)

The daily run consumes the live tree, not origin/main. Landing sequence
(operator-machine actions under the landing batch grant, ask-first):

1. Sync the live umbrella tree. The live tree has **uncommitted state at the
   prod artifact path** (the restamped 05-18 copy `5ce63326…`); the sync
   must take the merged file at this path — targeted, no blanket
   reset/checkout in the live tree (#412 lesson).
2. Run the calibrator refit against the restored scorer.
3. `python3 scripts/verify_prod_scorer_restore_20260703.py --require-calibrator-parity`
   (all six checks must pass on the live tree).
4. Readonly daily-full: expect P-CONFIG-FP ✓ `f8fb2259`, calibrator parity ✓,
   buys unblocked.

## Out of scope / follow-ups

- **Snapshot regen blocked by pre-existing pin drift**:
  `doc/arch/strategy-104-snapshot.md` still declares the rolled-back scorer
  (`5ce63326…`, trained 05-18), but regeneration refuses because
  `subrepos.lock.json` pins strategy-104 at `1fe312b4` while the live
  runtime checkout is at `c019b256` (carries the XGB override #32, watchlist
  145, demean rollback — deployed-but-uncommitted ahead of the lock). Lock
  bump + `make snapshot` is a separate operator-visible action.
- **Monitor gap**: orchestrator issue #276 — scorer-identity change without
  a promote event should alarm; PSI baseline should re-anchor per scorer
  vintage (#274 §4(ii)).
- Weekly retrain + promote gate supersede this restore on the normal cadence.
