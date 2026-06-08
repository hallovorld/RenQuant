# 2026-06-07 PatchTST prod/shadow status audit

**Status**: PatchTST is the current production primary scorer by strategy
configuration. Older research notes that describe PatchTST as shadow-only are
historical evidence, not the live config state.

## Current pinned state

- Umbrella `subrepos.lock.json` pins `renquant-strategy-104` at
  `c6b868576acd0ae6b5b8c7cb8b60cb47d85ae226`.
- `renquant-strategy-104/configs/strategy_config.json` sets:
  - `ranking.panel_scoring.enabled = true`
  - `ranking.panel_scoring.kind = "hf_patchtst"`
  - `ranking.panel_scoring.artifact_path =
    "../../artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt"`
  - `ranking.panel_scoring.global_calibration.strict_scorer_match = true`
  - `ranking.panel_scoring.global_calibration.artifact_path =
    "artifacts/shadow/panel-rank-calibration.hf_patchtst_seed44_trainfit_20230103_20240409.json"`
- `renquant-strategy-104/configs/strategy_config.shadow.json` now runs the
  previous XGB primary:
  - `ranking.panel_scoring.kind = "xgb"`
  - `ranking.panel_scoring.artifact_path =
    "artifacts/prod/panel-ltr.alpha158_fund.json"`
- `renquant-strategy-104/tests/test_strategy_configs.py` asserts the same
  prod/shadow split, so this is not only a comment-level change.

## Operator-directed promotion boundary

The active config records the promotion as:

`2026-06-05 operator-directed prod/shadow switch: HF PatchTST pt07 strict seed44 promoted to primary scorer; XGB moved to readonly shadow config.`

This means the live policy has switched, but the research acceptance story has
not become cleaner retroactively. Treat the switch as an operator override with
explicit residual controls, not as proof that every earlier PatchTST promotion
gate passed.

## Residual risks

1. Artifact path hygiene remains confusing. The primary scorer and calibrator
   still live under `patchtst_shadow` / `artifacts/shadow` naming even though
   they are now production primary. This is a traceability risk for reviews,
   dashboards, and incident response.

2. Runtime regime admission is intentionally disabled for the HF checkpoint.
   The active config notes that the checkpoint does not yet carry strict
   walk-forward regime-admission metadata. Preflight, scorer contract,
   calibrator strict matching, QP, freshness, drawdown, and sell-only gates
   remain active, but this is not equivalent to strict regime admission.

3. The 2026-06-02 sequence-boundary audit remains relevant for future
   PatchTST training. It ruled in a split-purity violation in sequence window
   construction. Do not use new PatchTST training or promotion evidence unless
   that path is fixed or the resulting artifact carries an explicit accepted
   override.

4. Historical docs before the 2026-06-05 switch still say PatchTST should stay
   shadow-only. They should be read as evidence snapshots taken before the
   operator-directed promotion, not as the current runtime truth.

## Required follow-up PRs

- `renquant-pipeline`: stamp active scorer identity into runtime telemetry,
  order attribution, score distribution, and decision trace so rows say
  `hf_patchtst` when PatchTST is primary and do not silently inherit stale
  per-ticker XGB labels.
- `renquant-strategy-104` or `renquant-artifacts`: promote/alias the PatchTST
  checkpoint and calibrator into an explicit production artifact registry, or
  add a manifest that explains why production references intentionally point at
  shadow-named paths.
- `renquant-model`: close the PatchTST sequence-boundary split-purity follow-up
  before producing the next candidate PatchTST primary artifact.
- Dashboards/reports: label the current A/B relationship as
  `prod=hf_patchtst`, `shadow=xgb_alpha158_fund_previous_primary`.

## Decision

Current prod answer: **yes, PatchTST is in production primary config now**.

Engineering answer: **keep refactoring the surrounding telemetry, artifact
promotion, and training acceptance path so the production state is auditable
without relying on tribal memory or shadow-named paths.**
