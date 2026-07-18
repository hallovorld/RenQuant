# Monthly meta-label retrain wrapper: consumer gate + WF snapshot config + corpus asserts (§5.1)

STATUS: delivered (wrapper + tests; runtime ACs deferred by design — see below)
RFC: `doc/design/2026-07-18-metalabel-monthly-retrain-redesign.md` (r2, merged)
SCOPE: `scripts/monthly_meta_label_retrain.sh` ONLY (+ tests). Implementation
plan item 1 of the RFC. No subrepo changes, no launchd changes, no pipeline/
backtesting code changes — exactly the §2.4 "wrapper-only" contract.

## WHAT

Three changes to the monthly wrapper, per §2.1/§2.2/§2.3:

1. **Step-0 consumer gate (§2.1).** Immediately after resolving the PINNED
   strategy config (the existing `renquant_strategy_config "$SUBREPO_ROOT"`
   resolution → `.subrepo_runtime/repos/renquant-strategy-104/configs/
   strategy_config.json` on the production box), the wrapper reads
   `ranking.meta_label.enabled`; false or absent → exit 0 with the single
   log line
   `meta-label consumer dark — retrain skipped by design (see doc/design/2026-07-18-metalabel-monthly-retrain-redesign.md)`.
   No training compute, no artifact churn, no ntfy. An unreadable config
   still fails closed (exit 1). The gate sits BEFORE the multirepo import
   preflights: while the consumer is dark by decision, a missing subrepo
   must not page anyone for a job whose output nothing reads.
2. **Walk-forward snapshot config (§2.2).** The snapshot side-config now
   sets `walkforward = {"enabled": true, "manifest_path": <REPO_DIR>/
   backtesting/renquant_104/artifacts/sim/walkforward_manifest_v2_20260602.json,
   "fail_on_no_model": true}` as a WHOLESALE dict replacement — the prod
   pointer (dead `dropsenti_v3` reference, trap 2a) is structurally
   unreachable, never inherited. The v2 (calibrator-bound) manifest is the
   RFC-mandated override; `fail_on_no_model` stays true.
3. **Corpus-coverage + scorer-family asserts (§2.3/§2.2), before the sim.**
   Loader-parity feature cutoff (`effective_train_cutoff_date`, else
   `cutoff_date`) per manifest row; requires newest feature cutoff ≥
   TRAIN_END − 60 business days − 35 calendar days, else fail closed with
   `wf corpus stale for window (newest cutoff <date>; need >= <threshold> …)`.
   Scorer-family parity: every vintage artifact metadata `kind` must map to
   the pinned `ranking.panel_scoring.kind` through the explicit allowlist
   `{"panel_ltr_xgboost": "xgb"}` (the v2 manifest rows carry no kind
   field — r2 nit); violations fail closed with
   `wf corpus scorer-family mismatch` / `wf corpus scorer-family unmapped`.
   All three named errors go to the log AND the ntfy body (sentinel
   patterning is RFC §5.3, a follow-up orchestrator PR).

Mechanical enabler: `REPO_DIR` gains a test-only sandbox override
(`RQ_META_LABEL_REPO_DIR`, same convention as `RQ_WEEKLY_PROMOTE_REPO_DIR`),
and `VENV_DIR` derives from it. Production launchd invokes with the env
unset → identical live paths.

## EVIDENCE

`tests/test_monthly_meta_label_retrain_redesign.py` (7/7 pass) — the script
is COPIED into a fabricated umbrella tree under tmp_path (live tree only
ever read), curl stubbed, python shimmed; the armed-path sandbox installs a
stub `sim_driver` that refuses to run, so no real sim can ever execute:

- AC-1: consumer-dark run with the REAL pinned config (copied) exits 0 in
  <1s (bound 5s), prints exactly the skip line, writes nothing outside
  `logs/`, sends no ntfy; plus an absent-block variant (gate treats a
  missing `ranking.meta_label` as dark).
- AC-3: staleness injection (fabricated manifest, newest cutoff 2025-01-05)
  → nonzero exit, named `wf corpus stale for window` error, no snapshot
  config written; artifacts intentionally absent on disk, proving staleness
  fires before family parity reads them.
- AC-4: execution-level — armed sandbox run constructs the snapshot config
  and it carries exactly the explicit v2 override (`enabled`/
  `fail_on_no_model` true, v2 path) with no `dropsenti` anywhere; plus a
  source-level rot-guard (wholesale `src["walkforward"] = {` replacement,
  gate precedes the sim machinery, named errors present).
- AC-6: family-mismatch injection (pinned `hf_patchtst` vs corpus
  `panel_ltr_xgboost`) and unmapped-kind injection → nonzero exit with the
  named errors.

Existing guards untouched and green: `test_monthly_jobs_multirepo_fail_closed.py`,
`test_subrepo_ops_contract.py` (all required literals preserved).
Pre-existing failures in `test_operator_script_env.py`
(`manual_promote.sh` venv-literal drift) reproduce on clean main with this
change stashed — unrelated.

## Runtime-gated ACs (stated, not claimed)

AC-2 (live walk-forward sim end-to-end, per-bar predicate + calibrator
bindings) and AC-5 (resolver digest acceptance; stamping already landed via
`doc/progress/2026-07-18-wf-manifest-digest-stamp.md`) are validated at the
first green ARMED run post-deploy. Per the RFC double gate, that run cannot
happen yet: the consumer is dark (§2.1 exits first) AND the corpus is stale
TODAY (newest feature cutoff 2025-12-15 < threshold ≈ TRAIN_END − 60bd −
35d), so the §2.3 assert currently fails BY DESIGN if armed. The job steady
state after this PR is the honest §2.1 skip. Merged-is-not-deployed: this
lands on `main` and reaches the box at the next checkout sync.
