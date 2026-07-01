# stamp panel freshness axis + label provenance — freshness-monitor provenance

STATUS: revised after Codex CHANGES_REQUESTED round 3 (PR #423)
WHAT: the full-history production XGB panel stamps TWO DISTINCT information-set
fields so the freshness monitor (orchestrator #213) + the renquant-pipeline
P-MODEL-STALENESS gate can measure panel staleness on the CORRECT axis.
`scripts/train_production_model.py:build_artifact` previously wrote
`effective_train_cutoff_date` only on the walk-forward path; round 1 stamped it
on the full-history path too, but set it to the max LABELED row date — which is
structurally ~60 business days behind the raw data frontier (the fwd_60d
`dropna` clip). Round 2 introduced two split fields but keyed model FRESHNESS on
the raw feature frontier (`max_feature_anchor_date`). Round-3 Codex review
rejected that: those trailing frontier rows are excluded from `train` because
their fwd labels are unobservable, so the weights/normalization/CV never
consumed them — keying freshness on the frontier lets fresh UNLABELED rows make
a frozen model read healthy ("fresh metadata over stale trained information"
under a new field name). This revision flips which field is which axis and
removes the fallback that gave one field two meanings:

- `label_observation_cutoff` — the fwd-label-clipped max FULLY-LABELED training
  row (`train["date"].max()`). ALWAYS the observed max on BOTH paths (consistent
  meaning — no path overloading). THIS is the MODEL-FRESHNESS axis: it is the
  latest information that actually affected fitting, and only moves when the
  labeled training frame moves. The monitor keys P-MODEL-STALENESS on it and
  accounts for the ~60 BD fwd-label horizon lag EXPLICITLY (a fresh retrain,
  labeled anchor ≈ today − horizon, reads HEALTHY once the lag is subtracted
  out; a globally frozen panel BREACHES). Derived from the frame, never
  `datetime.now()`/`trained_date` (the #210/#212 lesson).
- `max_feature_anchor_date` — the RAW feature/data frontier: the latest date
  with FEATURE rows in the (watchlist/window-filtered) panel, BEFORE the
  fwd-label `dropna` clip (so it includes rows whose forward label is not yet
  observable). For a fresh daily retrain this is ≈ today−1. This is
  DATA-PIPELINE HEALTH provenance ONLY (proof the feed is current) — NOT model
  freshness. Threaded out of `load_and_slice_panel` (new
  `return_feature_frontier=True` → 4-tuple; default keeps the legacy 3-tuple,
  zero blast radius on other callers). Stamped ONLY when the caller supplies
  it — when absent the field is OMITTED, never backfilled from the label
  cutoff (a fallback would give the field two meanings depending on caller).

Fixed contract issues Codex raised:
- `effective_selection_cutoff_date` is NO LONGER stamped. The panel has no
  DERIVABLE held-out model-selection information date (CV validation, recipe /
  factor / hyperparameter choice, acceptance, promotion all use information
  LATER than the training-row boundary). Copying the train cutoff would BACKDATE
  the anchor that `kernel/walk_forward/lean_guard.py:_selection_anchor` reads
  FIRST, making a static backtest appear point-in-time before the selection
  evidence existed. Omitting it lets the guard fall through to the conservative
  `trained_date`.
- The walk-forward vs full-history overloading is removed.
  `effective_train_cutoff_date` keeps its EXISTING documented contract — the
  upper *exclusive* feature-row cutoff (`kernel/walk_forward/loader.py`:
  "upper exclusive feature-row cutoff"), consumed by the manifest/loader
  leakage checks — and is stamped ONLY on the walk-forward path. It is NOT
  reused for the observed-max label on the full-history path (Codex: "Do not
  overload one field with observed max on one path and an exclusive boundary on
  another"); there it is OMITTED.

Metadata provenance only — model weights, training logic, the fwd_60d clip, and
the `config_fingerprint` are all unchanged. Additive/backward-compatible.

WHY/DIR: model freshness must key on the latest information that ACTUALLY
affected fitting (the labeled training anchor), not `trained_date` (a fresh
trained_date over stale data is NOT fresh, #210/#212) and not the raw feature
frontier (unused trailing rows the model never saw cannot refresh it). #213 /
P-MODEL-STALENESS key panel FRESHNESS on `label_observation_cutoff` with the
~60 BD fwd-label horizon lag accounted for EXPLICITLY; `max_feature_anchor_date`
is a separate data-pipeline-health signal (proof the feed is current), not
freshness. #213's SLA/policy update is a SEPARATE orchestrator change (I own
that); this PR is the producer half of the contract.

EVIDENCE: `.venv/bin/python -m pytest tests/test_train_production_model_cutoff.py
tests/test_model_acceptance.py -q` (Python 3.10 + xgboost 3.2.0) → 78 passed, 1
FAILED. The single failure is pre-existing + unrelated (reproduces identically
on the unmodified base commit via `git stash`):
`TestStrictContractStamp::test_walk_forward_cv_purges_embargo_before_validation`
reads `data/alpha158_qlib_dataset.stats.json`, a generated data artifact absent
in a fresh clone (the whole `data/` dir is gitignored). Tests:
- `TestFullHistoryDataCutoffStamp` — distinct fields (feature anchor LEADS the
  label cutoff), label cutoff is DATA not wall-clock, feature anchor is OMITTED
  (never backfilled from the label cutoff) when the caller does not supply it,
  `effective_selection_cutoff_date` is NEVER fabricated, the full-history path
  OMITS the exclusive `effective_train_cutoff_date`, and the walk-forward path
  keeps the exclusive bound while still not fabricating a selection cutoff.
- `TestModelFreshnessAxisIntegration` — INTEGRATION producer→monitor: a CURRENT
  panel (feature rows to ≈today, last 60 BD labels null) reads HEALTHY on the
  LABEL axis once the ~60 BD horizon lag is accounted for explicitly; a
  globally FROZEN panel reads BREACH; and the round-3 anti-regression test
  appends fresh UNLABELED feature rows to a frozen panel (same labeled training
  frame, same mocked booster) and asserts `label_observation_cutoff` and the
  freshness read are UNCHANGED (still BREACH) even though `max_feature_anchor_date`
  correctly advances — proving the two fields are decoupled and the raw
  frontier cannot launder a stale model into looking fresh.
- `TestPromotePreservesDataCutoff` / `TestRestampPreservesDataCutoff` — the
  active-swap `promote()` and the sector_map re-stamp both preserve the new
  fields (whole-artifact copy / whole-dict rewrite).

Path-to-live: `training_panel/daily_retrain_alpha158_fund.py` runs
`train_production_model.py --output-path <prod dst>` directly, so the next daily
retrain writes the fields into the live prod artifact; `promote()` and the
sector_map re-stamp both preserve them. The committed prod artifact is NOT
hand-edited (its true frontier + max labeled date are only knowable from the
training panel); it acquires the fields on the next retrain/promote.

NEXT (follow-up, NOT in this PR): (1) update orchestrator #213 / P-MODEL-STALENESS
to key panel FRESHNESS on `label_observation_cutoff` with the ~60 BD fwd-label
horizon lag subtracted out explicitly (fast-axis SLA on the lag-adjusted label
axis, not the raw feature frontier); (2) once the fields propagate to the live
prod artifact via a retrain, harden the gate from soft-skip to enforce, gated
on a grace window.
