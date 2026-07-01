# stamp panel freshness axis + label provenance — freshness-monitor provenance

STATUS: revised after Codex CHANGES_REQUESTED (PR #423)
WHAT: the full-history production XGB panel now stamps TWO DISTINCT
information-set fields so the freshness monitor (orchestrator #213) + the
renquant-pipeline P-MODEL-STALENESS gate can measure panel staleness on the
CORRECT axis. `scripts/train_production_model.py:build_artifact` previously
wrote `effective_train_cutoff_date` only on the walk-forward path; the first
revision stamped it on the full-history path too, but set it to the max LABELED
row date — which is structurally ~60 business days behind the raw data frontier
(the fwd_60d `dropna` clip). Codex flagged that keying #213's fast-axis policy
(healthy <=14d, breach >28d) on that value makes every fresh retrain born
permanently BREACH, and that copying it into `effective_selection_cutoff_date`
fabricates a point-in-time claim. This revision stamps, on BOTH paths:

- `max_feature_anchor_date` — the RAW feature/data frontier: the latest date
  with FEATURE rows in the (watchlist/window-filtered) panel, BEFORE the
  fwd-label `dropna` clip (so it includes rows whose forward label is not yet
  observable). For a fresh daily retrain this is ≈ today−1. THIS is the
  FRESHNESS axis #213 must key on. Threaded out of `load_and_slice_panel`
  (new `return_feature_frontier=True` → 4-tuple; default keeps the legacy
  3-tuple, zero blast radius on other callers). Falls back to the label max
  when a caller does not supply it (per-regime / legacy) — a model can be no
  fresher than the labeled rows it trained on.
- `label_observation_cutoff` — the fwd-label-clipped max FULLY-LABELED training
  row (`train["date"].max()`). ALWAYS the observed max on BOTH paths (consistent
  meaning — no path overloading). This is PROVENANCE (which labels the model
  saw), NOT a freshness axis: with fwd_60d it structurally lags the feature
  frontier by ~60 business days. Derived from the frame, never
  `datetime.now()`/`trained_date` (the #210/#212 lesson).

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

WHY/DIR: freshness must key on the DATA frontier, not `trained_date` (a fresh
trained_date over stale data is NOT fresh) AND not the label-observation max
(whose ~60 BD label lag is EXPECTED, not staleness). #213 keys panel FRESHNESS
on `max_feature_anchor_date`; the 60d fwd label lag on `label_observation_cutoff`
is provenance. #213's SLA/policy update is a SEPARATE orchestrator change (I own
that); this PR is the producer half of the contract.

EVIDENCE: `.venv/bin/python -m pytest tests/test_train_production_model_cutoff.py
tests/test_model_acceptance.py -q` (Python 3.10 + xgboost 2.1.4) → 77 passed, 1
FAILED. The single failure is pre-existing + unrelated:
`TestStrictContractStamp::test_walk_forward_cv_purges_embargo_before_validation`
reads `data/alpha158_qlib_dataset.stats.json`, a generated data artifact absent
in a fresh clone. Tests:
- `TestFullHistoryDataCutoffStamp` — distinct fields (feature anchor LEADS the
  label cutoff), label cutoff is DATA not wall-clock, feature anchor falls back
  to the label max when absent, `effective_selection_cutoff_date` is NEVER
  fabricated, the full-history path OMITS the exclusive `effective_train_cutoff_date`,
  and the walk-forward path keeps the exclusive bound while still not fabricating
  a selection cutoff.
- `TestFreshnessAxisIntegration` — INTEGRATION producer→monitor: a CURRENT panel
  (feature rows to ≈today, last 60 BD labels null) → `max_feature_anchor_date`
  reads HEALTHY under #213's fast-axis policy while `label_observation_cutoff`
  would BREACH (the bug this fixes); a globally FROZEN panel → BREACH.
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
to key panel FRESHNESS on `max_feature_anchor_date` with a per-axis SLA
(fast-axis on the feature frontier; the label lag is expected, not staleness);
(2) once the fields propagate to the live prod artifact via a retrain, harden
the gate from soft-skip to enforce, gated on a grace window.
