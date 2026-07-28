# Umbrella kernel copies accept the blend scorer kind (rehearsal-caught)   (PR #540)

STATUS:    delivered (rehearsal-caught fix, needed pre-13:55 PT)

WHAT:      The umbrella-local kernel fork under `backtesting/renquant_104/kernel/`
now dispatches `ranking.panel_scoring.kind="blend"` (pipeline#218 mirror):

- `panel_pipeline/model_registry.py`: `BlendHandler` registered as
  `"blend"` — DELEGATES to the pinned renquant-pipeline implementation
  (`renquant_pipeline.kernel.panel_pipeline.blend_scorer.load_blend_scorer`,
  fail-closed two-pin loader) instead of forking a second copy of the
  scorer. Import resolves via the daily run's PYTHONPATH
  (`.subrepo_assembly/current.env` carries the pinned pipeline src), with
  the same `RENQUANT_SUBREPO_ROOT`/sibling fallback the gate-registry
  import uses; an unresolvable import raises and LoadScorerTask converts
  it into `panel_scorer_load_failed` (never a silent skip).
- `panel_pipeline/job_panel_scoring.py`: `LoadScorerTask` gains the
  `_blend_component0_path` anchor (strict-consistency gate anchored on
  component 0 when no top-level `artifact_path` is configured; wired in
  BOTH the fresh-load and preloaded branches, mirroring the #218 review
  fix); `ApplyScoresTask` routes kind `"blend"` down the alpha158-rebuild
  path and SKIPS the outer raw→model transform (the composite consumes
  the RAW union matrix; each leg's transform is applied inside
  `BlendPanelScorer` — one outer transform cannot be correct for two legs
  with different feature_means/stds).
- `panel_pipeline/tasks_feature_matrix.py`: `"blend"` joins the
  alpha158 kind tuples in `ResolveInferenceFramesTask` (target-only
  matrix) and `DriftGuardTask` (structural-drift skip).
- `panel_pipeline/shadow_scoring.py`: shadow entries may carry a
  `components` sub-config (parity with the pipeline copy).
- `tests/test_blend_kind_umbrella.py`: 9 tests — real pinned
  shadow_blend profile loads through the umbrella registry against the
  real artifacts read-only (both pins re-checked), scores a synthetic
  6-name raw union frame (finite z-sums, no degraded legs), missing-pin
  fail-closed, bogus kind still `panel_scorer_invalid_kind`, component-0
  anchor in both LoadScorerTask branches, DriftGuard/ResolveFrames kind
  branches.

WHY/DIR:   Today's full-lane rehearsal ran the shadow_blend profile e2e through
`live.runner` and fail-closed at scoring with `panel_scorer_invalid_kind`
from `kernel.panel_pipeline.scoring`: pipeline#218 landed the blend kind
in the renquant-pipeline kernel copies — which the strategy#68 acceptance
exercised via the sim/worktree smoke — while `live.runner` executes THIS
umbrella-local fork, whose registry has no `"blend"`.

This is the THIRD fork-divergence surfaced today by the probe/rehearsal
surfaces, after umbrella#537 (umbrella shadow_scoring resolved artifacts
repo-root-only while the pinned pipeline copy resolved strategy_dir-first)
and the sim-adapter divergence (the blend acceptance passed on the
pipeline copies because the sim harness runs them — the same config
fail-closed on the fork the live runner actually executes). Same
duplicated-kernel divergence class as the calibrator/scorer fingerprint
triple-impl playbook. The durable fix is FORK RETIREMENT (F-2 class, as
already flagged in the #537 doc): the umbrella copies should delegate to
the pinned `renquant_pipeline` kernel outright. This PR moves in that
direction by importing the blend loader rather than porting it — the only
fork-side additions are dispatch/kind-branch wiring.

EVIDENCE:
artifact:      backtesting/renquant_104/kernel/panel_pipeline/{model_registry.py, job_panel_scoring.py, tasks_feature_matrix.py, shadow_scoring.py} (this PR, diffed against origin/main); strategy_config.shadow_blend.json component pins (prod panel-ltr.alpha158_fund.json content=sha256:04d7a381cd6df847… fp=sha256:f8fb2259b2bf1537 trained=2026-06-21; clf panel-clf.top-decile.fwd60.json content=sha256:1e644354e0981f47… fp=sha256:1d8f167f…e41b trained=2026-07-28)
prod or exp:   experiment — acceptance rerun in an isolated worktree of origin/main + this branch (live venv, pinned-runtime PYTHONPATH, live data/ and backtesting/renquant_104/models/ APFS-cloned read-identical into the worktree, all writes isolated to the worktree); `--broker readonly-alpaca --once`, no live-tree file touched
existing data: on unmodified origin/main the same full-lane rehearsal fail-closed at scoring with `panel_scorer_invalid_kind` (pipeline#218 landed kind="blend" in the renquant-pipeline copies only; the strategy#68 acceptance never exercised this umbrella fork). With this PR's branch, `grep -c panel_scorer_invalid_kind` over the full rerun log = 0; the log instead shows both component pins verified, `LoadScorerTask: loaded blend artifact`, `ApplyScoresTask[blend]: passing RAW union matrix`, 87/87 candidates scored, 5/5 holdings, 3 orders sized, `[READONLY][ALPACA_SHADOW_BLEND]` SHADOW-ACTION emitted, rc=0
best-known?:   this is the only umbrella-fork variant that dispatches kind="blend" at all; pipeline#218 already carries the blend kind in the renquant-pipeline kernel (the best-known upstream implementation) and this PR brings the umbrella fork to parity via delegation to that same loader rather than porting a second copy
scope:         claim is scoped to the umbrella kernel fork (`backtesting/renquant_104/kernel/panel_pipeline/`) exercised via an isolated-worktree rehearsal rerun of the shadow_blend readonly-alpaca profile, vs. the immediately-prior state on the same fork (fail-closed, rc reaching `panel_scorer_invalid_kind` before any decision)

Tests: `tests/test_blend_kind_umbrella.py` 9/9 passed. Regression subset
on the touched modules (`test_model_registry.py`,
`test_apply_scores_panel_linear_dispatch.py`, `test_panel_scoring_job.py`,
`test_panel_scoring_drift.py`, `test_artifact_resolver_umbrella.py`,
`test_panel_scoring_specialist_wiring.py`): 69 passed, 3 failed — the 3
(`test_panel_scoring_drift.py` NGBoost drift cases) reproduce IDENTICALLY
on the unmodified live tree (origin/main), pre-existing and unrelated.

kind != blend paths: registry registration is additive; the
`_blend_component0_path` hooks only fire when `kind=="blend"` AND the
resolver returned None; the kind tuples only gained the `"blend"` member;
the transform branch is `if scorer_kind == "blend"` with the original
body verbatim in the else.

NEXT:      Fork retirement (F-2): replace the umbrella-local panel_pipeline fork
with delegation to the pinned `renquant_pipeline` kernel so the pipeline
and live-runner surfaces cannot diverge again — three same-day incidents
(#537, sim-adapter, this) justify scheduling it now. Until then, every
pipeline kernel-kind change MUST land a same-batch umbrella mirror.
