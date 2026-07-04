# B2: re-point the umbrella fingerprint call-sites to the canonical impl

DATE: 2026-07-04
CAMPAIGN: compliance fix campaign B2 (orchestrator PR #297; findings RQ#444
F-2/F-10, orchestrator#296 BT-1). Coordinated with renquant-backtesting B1
(`fix/wf-loader-unify`). Design authority: orchestrator
`doc/design/2026-07-03-m6-stage2-fingerprint-migration.md` §2a/§3 step 1.

## What changed

Three umbrella call-sites verified/derived the scorer content identity via
STALE LOCAL copies of `model_content_sha256` (the umbrella kernel
`panel_pipeline/panel_scorer.py` copy — one of the three divergent verifiers
behind the 2026-05-27 / 06-22 / 07-01 fail-closed no-trade incidents):

1. `backtesting/renquant_104/kernel/walk_forward/loader.py` — the LIVE
   promote-gate leg (`run_wf_gate.py` `_score_manifest_sanity` /
   `run_walk_forward`, `adapters/sim.py`). Its local 12-char-prefix matcher +
   stale recompute are REMOVED; verification now routes through the pipeline
   M6 dispatch (`renquant_pipeline.kernel.panel_pipeline.fingerprint_dispatch`
   — schema-version dispatch, fail-closed `verify()` on v1 stamps, legacy
   route byte-for-byte behind the `accept_legacy_stamps` window flag).
   Resolution of the pipeline module follows the wrapper order
   (importable → `RENQUANT_SUBREPO_ROOT` → `.subrepo_runtime/repos` →
   sibling checkouts) and FAILS LOUD if absent — never a silent local
   fallback. The #421 bounded URI resolver + `artifact_sha256` digest binding
   are untouched. NOTE: `kernel/walk_forward/__init__.py` already imports
   `renquant_pipeline` (correlation_guard), so this adds no new runtime
   requirement; the live `.subrepo_runtime/repos/renquant-pipeline` already
   carries the dispatch (verified 2026-07-04).
2. `scripts/fit_calibrator_alpha158_fund.py` — recompute fallback re-pointed
   to the EXPLICIT legacy engine (`renquant_common.model_fingerprint`
   imports only); stamped-value precedence unchanged; now also propagates
   `scorer_fingerprint_schema_version` when the scorer is v1-stamped (dead
   path today).
3. `scripts/stamp_walkforward_fingerprints.py` — `_scorer_identity` is now
   stamped-value-first with the explicit legacy engine as unstamped fallback
   (previously an UNCONDITIONAL venv-coupled recompute: on a 0.9.x venv the
   bare name is the v1 hasher — stamping v1 hashes into fold calibrators, the
   §1a.3 corpus-pollution arming path). Propagates the schema version into
   calibrator bindings for v1-stamped folds (dead path today).

## Behavior-invariance proof (protection contract, run 2026-07-04)

Read-only A/B against the REAL live-tree inventory (fingerprint census
GREEN: 47/47 legacy-stamped, 69 bindings, 0 red):

| Leg | old | new | delta |
|---|---|---|---|
| umbrella WF loader, 2 in-scope manifests x 43 folds (real corpus, real `_assert_calibrator_matches_entry`) | 43 PASS / 43 NO_CALIBRATOR | 43 PASS / 43 NO_CALIBRATOR | NONE |
| `_artifact_fingerprint` (fit_calibrator) x 47 artifacts | — | — | 47/47 identical values |
| `_scorer_identity` (WF stamper) x 47 artifacts | — | — | 47/47 identical values |
| 12-char-prefix reliance among green matches | — | — | ZERO (every green match is exact) |
| hash-engine equivalence (umbrella kernel copy vs `renquant_common` legacy engine) | — | — | tables identical, hashes identical, error types identical |

Suites: targeted umbrella tests (`test_walkforward_loader`,
`test_manifest_uri_resolver`, `test_wf_gate_recipe_scope`,
`test_walkforward_eval_config`) A/B vs pristine main — identical failure set
(1 pre-existing environmental failure both sides); +13 new pins in
`tests/test_wf_loader_fingerprint_dispatch.py`.

## Deploy note

No venv or pin change required: the pinned runtime already resolves the
dispatch. Live sync follows the normal pin-advance flow; nothing here is
config- or enablement-gated.
