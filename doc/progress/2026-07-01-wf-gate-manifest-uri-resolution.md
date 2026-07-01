# WF-gate per-bar artifact resolution — strategy-dir-relative manifest URIs

STATUS: open PR (fix/wf-gate-config-path-parity). Code + tests done; awaiting
review. Severity P2 (weekly automation, not live trading): the GBDT primary
panel (`panel-ltr.alpha158_fund`) could not pass the weekly WF-promote gate, so
the production XGB panel refresh has been frozen at 2026-05-18 (~43d). No bad
orders, no capital impact — the gate fail-closes and keeps the prior model.

## The mechanical bug (bug 1): per-bar scorer ARTIFACT-NOT-FOUND (rc=1)

The orchestrator-built WF manifests
(`built_by=renquant_orchestrator.build_wf_manifest`) live under
`backtesting/renquant_104/artifacts/sim/` but emit **strategy-dir-relative**
artifact/calibrator URIs, e.g.

    "artifact_uri":   "artifacts/walkforward_gbdt_prod_recipe_v2/2023-10-02/panel-ltr.json"
    "calibrator_uri": "artifacts/sim/walkforward_calibrators/2023-10-02/panel-rank-calibration.json"

All three manifest-URI resolvers, however, joined a relative URI onto the
manifest's **parent** directory (`artifacts/sim/`), doubling the prefix into a
path that does not exist:

    artifacts/sim/ + artifacts/walkforward_gbdt_prod_recipe_v2/... 
      = artifacts/sim/artifacts/walkforward_gbdt_prod_recipe_v2/...   (missing)

During a WF cut, `WalkForwardModelLoader.model_as_of()` then handed that
non-existent path to `PanelScorer.load()` → `FileNotFoundError` → the gate
fail-closed every bar (rc=1). The validated corpus
(`artifacts/walkforward_gbdt_prod_recipe_v2/<date>/panel-ltr.json`, 43 cuts) and
the per-fold calibrators (`artifacts/sim/walkforward_calibrators/<date>/...`)
both exist at the **strategy-dir-relative** location. Confirmed empirically in a
fresh clone: parent-relative resolution → `exists=False`; strategy-dir-relative
→ `exists=True`.

The older manifests used absolute URIs (which resolve fine as pass-through), so
only the newer orchestrator-built v2 manifests tripped the doubling.

## Fix

Make the three resolvers tolerant of BOTH conventions: resolve against the
manifest folder first (unchanged default), and if that path does not exist walk
up the manifest folder's ancestors and return the first join that exists;
fall back to the manifest-folder join when nothing exists so the downstream
not-found error stays meaningful. Absolute paths and `scheme://` URIs are
unchanged. Sites fixed (all on the GBDT weekly-gate path):

- `kernel/walk_forward/loader.py::WalkForwardModelLoader._resolve_uri` — the
  per-bar scorer + per-fold calibrator load (the `FileNotFoundError` source) and
  `_scorer_fingerprints_for_entry`.
- `backtesting/renquant_104/adapters/sim_artifacts.py::_resolve_manifest_uri` —
  the sim's scorer-kind probes (`_alpha158_cache_required` etc.; the GBDT kind
  `panel_ltr_xgboost` is alpha158-cache-eligible, so a mis-probe silently skips
  the feature cache the scorer needs).
- `scripts/run_wf_gate.py::_manifest_uri_to_path` — the manifest-scope §5.2
  sanity per-cut scorer resolution (`_score_manifest_sanity`).

No manifest data was rewritten (a regenerated manifest would re-break it); the
umbrella's loader now defines a robust read contract for orchestrator-emitted
URIs.

## Bug 2 (scorer-kind parity) was already fixed — no change needed

The paired diagnosis (derived WF config inheriting `kind=hf_patchtst` while
pointing at a GBDT JSON → parity FAIL) was already resolved by the
candidate-matched prod-reference selection in
`scripts/run_wf_gate.py::_prod_config_path` (see
`doc/progress/2026-06-28-wf-gate-derive-config-honor-env.md`, commit
`d1277333`). Re-verified in this clone: for the GBDT candidate (`kind=
panel_ltr_xgboost`) the derived config gets `panel_scoring.kind=xgb` from the
kind-matched `strategy_config.shadow.json`, and `evaluate_wf_config_parity`
returns PASS. No further change made here — the remaining mechanical blocker was
bug 1 alone.

## Evidence / verification

- Reproduced in a fresh blobless clone of `hallovorld/RenQuant` (never the live
  tree): both v2 manifest URIs resolve to existing files after the fix
  (loader, sim_artifacts, run_wf_gate resolvers all confirmed against the real
  `walkforward_manifest_gbdt_prod_recipe_v2.calibrated.json`).
- `tests/test_sim_artifacts.py` — 8 passed, including two new regression tests:
  `test_strategy_dir_relative_uri_resolves_to_existing_corpus` (the bug) and
  `test_missing_relative_uri_falls_back_to_manifest_parent` (sensible-error
  fallback). Existing `TestResolveManifestUri` (absolute passthrough +
  manifest-parent default) stays green, as does the loader regression guard
  `test_relative_artifact_uri_resolved_against_manifest_parent` (materialized
  file → level-0 short-circuit).
- Scope: no gate-threshold change, no placebo/§5.2 statistic change, no model
  weights, no promotion criteria, no retrain. Path-resolution only.
- `git diff --check` clean.

## Next

Merge after review, then the weekly `weekly_wf_promote.sh` GBDT gate can reach
the per-bar scorers and re-validate/promote the frozen XGB primary. The parallel
active package path lives in renquant-backtesting (`_resolve_uri` there, if it
carries the same manifest-parent assumption, needs the identical fix).
