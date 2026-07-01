# stamp panel effective_train_cutoff_date — freshness-monitor provenance

STATUS: delivered (PR to main)
WHAT: the full-history production XGB panel now stamps its binding DATA
cutoff. `scripts/train_production_model.py:build_artifact` previously wrote
`effective_train_cutoff_date` ONLY on the walk-forward (`--train-cutoff`) path;
the daily full-history prod retrain left it unstamped, so the panel carried only
`trained_date` (wall-clock) and a human-readable contract string. This change
stamps, on BOTH paths:
- `effective_train_cutoff_date` — the binding information-set cutoff. Prod
  (full-history) path = the max LABELED training date (`train["date"].max()`
  after the fwd_60d `dropna` clip), derived from the training frame, NEVER
  `datetime.now()`/`trained_date`. Walk-forward path keeps its existing
  pre-embargo boundary (`cutoff_date - embargo`).
- `effective_selection_cutoff_date` — mirrored to the same value. The panel has
  no separate held-out selection window (CV is purged walk-forward WITHIN the
  clipped panel; the final model trains on the full labeled panel), so selection
  cutoff == train cutoff. Stamped because the freshness monitor's selection
  anchor (`kernel/walk_forward/lean_guard.py:_selection_anchor`) reads
  `effective_selection_cutoff_date` FIRST, then falls back to
  `effective_train_cutoff_date`.
Metadata provenance only — model weights, training logic, the fwd_60d clip, and
the `config_fingerprint` are all unchanged. Backward-compatible additive fields.

WHY/DIR: the freshness monitor (orchestrator PR #213) and the renquant-pipeline
P-MODEL-STALENESS preflight gate must key on the DATA cutoff, not `trained_date`
— a fresh trained_date over stale labeled data is NOT fresh (the #210/#212
lesson). Because the prod panel had no `effective_train_cutoff_date`,
P-MODEL-STALENESS SOFT-SKIPPED ("effective_train_cutoff_date unstamped —
decay-curve rail unmeasurable; xgb does not stamp it") and could not catch a
stale panel. Now it can measure it.

EVIDENCE: unit tests in `tests/test_train_production_model_cutoff.py`
(`.venv/bin/python -m pytest`, Python 3.10):
- `TestFullHistoryDataCutoffStamp` — full-history path stamps
  `effective_train_cutoff_date` == the fwd-clipped max labeled date (a 2024 date
  on an artifact trained "today"), the selection alias equals it, and the value
  is DATA-derived not wall-clock; the walk-forward path still stamps both.
- `TestPromotePreservesDataCutoff` — `kernel/model_acceptance.promote()`
  (RQ_ALLOW_NO_WF override) preserves both fields (it `shutil.copy2`s the whole
  artifact; no metadata whitelist).
- `TestRestampPreservesDataCutoff` — `scripts/restamp_prod_fingerprint.py`
  (sector_map re-stamp) rewrites the WHOLE artifact dict, mutating only the
  fingerprint fields, so both cutoff fields survive.
8 targeted tests + the existing `TestArtifactStampedCutoff`/`test_model_acceptance`
(47) pass. Pre-existing environmental failure unrelated to this change:
`TestStrictContractStamp::test_walk_forward_cv_purges_embargo_before_validation`
reads `data/alpha158_qlib_dataset.stats.json`, a generated data artifact absent
in a fresh clone.

Path-to-live: `training_panel/daily_retrain_alpha158_fund.py` runs
`train_production_model.py --output-path <prod dst>` directly, so the next daily
retrain writes the field into the live prod artifact; `promote()` and the
sector_map re-stamp both preserve it. The committed prod artifact is NOT
hand-edited (its true max labeled date is only knowable from the training panel);
it acquires the field on the next retrain/promote.

NEXT (follow-up, NOT in this PR): once panels stamp
`effective_train_cutoff_date`, the renquant-pipeline P-MODEL-STALENESS gate can
be hardened from soft-skip to enforce for panels (currently it fail-opens on the
provenance gap). Gate the hardening on a grace window so the field has propagated
to the live prod artifact via a retrain first.
