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

## Fix — a BOUNDED resolver (revised after Codex review)

The first cut resolved by walking *arbitrary* ancestors of the manifest folder
and returning the first path that happened to exist. Codex correctly blocked
that: on a model-validation path it makes artifact identity a property of the
machine's surrounding filesystem — a stale same-named file higher in the
checkout could be silently picked, relocating the checkout could select a
different model, and three duplicated resolver copies could drift.

Replaced with a single shared **bounded** resolver,
`kernel/manifest_uri_resolver.py::resolve_manifest_uri`, used by all three call
sites. Contract:

- Resolve a relative URI only against a small ORDERED set of KNOWN ROOTS:
  (1) the manifest's own folder (legacy default), then (2) the strategy/repo
  root inferred from the manifest path (parent of the outermost `artifacts`
  directory) — where orchestrator-built manifests emit their strategy-dir-
  relative URIs. No arbitrary parent walking.
- NORMALIZE each candidate and ENFORCE CONTAINMENT: a URI whose normalized
  join escapes *every* allowed root (`..` traversal) is REJECTED
  (`ManifestUriResolutionError`), not walked.
- REJECT AMBIGUITY: if more than one root yields an existing file and those
  files have different content digests, raise rather than guess. Identical
  bytes under both roots are not ambiguous (manifest-folder root wins the tie,
  deterministically).
- Bind to the manifest's expected digest WHERE PRESENT: the resolver accepts an
  optional `expected_digest`; on a mismatch it raises. A new optional
  `RetrainEntry.artifact_sha256` (round-tripped by `manifest.py`, default absent
  → unchanged for pre-digest manifests) feeds it on the loader path, so "found a
  file" is not sufficient to run the gate.
- Absolute paths and `scheme://` URIs are returned untouched.
- When nothing exists, fall back to the manifest-relative join so the
  downstream not-found error names the expected location.

Because both roots are derived from the manifest path itself (never an absolute
machine prefix), resolution is deterministic across checkout relocation.

The module lives directly under `kernel/` (not `kernel/walk_forward/`) on
purpose: the sim adapter imports it, and the `kernel/walk_forward` package
`__init__` pulls heavier pipeline modules the URI-resolution path does not need.

Sites consolidated onto the shared resolver (all on the GBDT weekly-gate path):

- `kernel/walk_forward/loader.py::WalkForwardModelLoader._resolve_uri` — the
  per-bar scorer + per-fold calibrator load (the `FileNotFoundError` source) and
  `_scorer_fingerprints_for_entry`; both now pass the entry's `artifact_sha256`
  as `expected_digest`.
- `backtesting/renquant_104/adapters/sim_artifacts.py::_resolve_manifest_uri` —
  the sim's scorer-kind probes (`_alpha158_cache_required` etc.; the GBDT kind
  `panel_ltr_xgboost` is alpha158-cache-eligible, so a mis-probe silently skips
  the feature cache the scorer needs).
- `scripts/run_wf_gate.py::_manifest_uri_to_path` — the manifest-scope §5.2
  sanity per-cut scorer resolution (`_score_manifest_sanity`).

No manifest data was rewritten (a regenerated manifest would re-break it); the
umbrella now defines a single robust read contract for orchestrator-emitted
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

- New `tests/test_manifest_uri_resolver.py` pins the bounded contract:
  manifest-relative resolves, strategy-dir-relative resolves, traversal-outside
  → rejected, two conflicting candidates → error, deterministic across
  simulated checkout relocation (ignores a same-named decoy above the strategy
  root), digest-mismatch → rejected, digest-match accepted, absolute/`scheme://`
  untouched, and loader delegation.
- `tests/test_sim_artifacts.py` extended: strategy-dir-relative resolves and
  manifest-parent fallback (kept from the first cut), plus new
  traversal-rejected and conflicting-candidates-rejected cases through the sim
  wrapper.
- Ran in a fresh blobless clone of `hallovorld/RenQuant` (never the live tree),
  Python venv: `tests/test_sim_artifacts.py` + `tests/test_manifest_uri_resolver.py`
  → 24 passed, 1 skipped (the loader-delegation case skips only where the
  pipeline subrepo is not assembled). `tests/test_walkforward_loader.py` +
  `tests/test_walkforward_manifest.py` → 27 passed incl. the relative/absolute
  URI regression guards and the manifest round-trip; the single remaining
  failure is an unrelated Python-3.9-vs-3.10 `X | None` runtime-annotation issue
  in `kernel/config.py` (the real gate venv is 3.10+), not this change.
- Scope: no gate-threshold change, no placebo/§5.2 statistic change, no model
  weights, no promotion criteria, no retrain. Path-resolution + optional
  digest-binding only.
- `git diff --check` clean.

## Next

Merge after review, then the weekly `weekly_wf_promote.sh` GBDT gate can reach
the per-bar scorers and re-validate/promote the frozen XGB primary. The parallel
active package path lives in renquant-backtesting (`_resolve_uri` there, if it
carries the same manifest-parent assumption, needs the identical fix).
