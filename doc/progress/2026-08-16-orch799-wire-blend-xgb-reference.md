# orch#799: wire the blend→xgb reference derivation into weekly_wf_promote.sh

STATUS:    feat — the umbrella half of the approved orch#799 design (#982). Unfreezes
           the weekly WF promote gate for a blend prod. Paired with renquant-backtesting
           #112 (the derivation function). Deploy order: pin backtesting #112 first, then
           this. Operator-gated live-tree deploy.

WHAT:      `scripts/weekly_wf_promote.sh`: `_find_gbdt_config` gains a `kind=blend`
           branch — instead of failing closed (exit 2) when the pinned primary is a
           blend, it calls the new helper `_derive_xgb_ref_from_blend`, which invokes
           `renquant_backtesting.wf_gate.wf_config_builder.derive_xgb_reference_from_blend`
           to write a temp xgb-shaped reference (from the blend's component[0]) and echoes
           its path. Both GBDT_PROD_CONFIG consumers (stamp_walkforward_fingerprints L396,
           run_wf_gate L406) then receive a genuine xgb config — identical shape to the
           pre-blend-switch world, so neither consumer changes.

WHY/DIR:   Operator-directed ("解决所有问题" 2026-08-16). retrain_panel/wf-promote fired
           WEEKLY-BLOCKED again today (`ERROR: no PINNED strategy config declares kind=xgb
           … orch#799`). The pinned strategy_config.json is kind=blend and
           strategy_config.shadow.json is kind=hf_patchtst → no xgb reference → exit 2.
           This is the root of the wf-promote/retrain-panel104/silent-refusal cluster.

EVIDENCE:
  artifact:      `scripts/weekly_wf_promote.sh` (the helper + the blend branch) + this doc.
  prod or exp:   neither — the change was VALIDATED end-to-end but no production/live gate
                 was run and nothing was deployed.
  existing data: [VERIFIED end-to-end] running the derivation against the REAL pinned blend
                 config + real artifact yields kind=panel_ltr_xgboost (→xgb), artifact_path
                 `artifacts/prod/panel-ltr.alpha158_fund.json`, components removed,
                 provenance fingerprint sha256:f8fb2259b2bf1537; and
                 `select_prod_reference_for_candidate("panel_ltr_xgboost", …)` ACCEPTS the
                 derived reference (parity passes) — i.e. the gate unfreezes for an xgb
                 candidate. `bash -n` clean. SUBREPO_ROOT + renquant_subrepo_pythonpath are
                 sourced at L69 (before the L148 call), so the helper's PYTHONPATH resolves.
  best-known?:   yes — FAIL-SAFE by construction: any derivation failure (module not yet
                 pinned, component not xgb, missing artifact) echoes nothing / returns
                 non-zero, so `_find_gbdt_config` falls through to the pre-existing exit-2
                 fail-closed — the change can only REACH the (already-safe) gate, never
                 fabricate a reference. Non-blend path is byte-unchanged (the branch is
                 added AFTER the xgb check). The adversarial kind-check (component must
                 DECLARE xgb) lives in the tested backtesting function.
  scope:         "adds a blend→xgb-reference derivation path to the weekly promote gate's
                 config resolver so a blend prod can supply the xgb reference the gate
                 requires. Changes NO gate threshold, does NOT run the gate here, does NOT
                 touch production, does NOT alter the non-blend path. Inert until
                 backtesting #112 is pinned; then operator-gated live-tree deploy. Does NOT
                 by itself restore 104 buying (no-bull-edge is separate) and does NOT gate
                 the served z-sum (booster-blind gate + blend-level gating are deferred per
                 #982)."

TESTS:     `bash -n` clean; the derivation function's own 18 unit tests pass in
           renquant-backtesting #112 (behaviour-invariance + 4 fail-closed paths); the
           end-to-end derive+parity proof above ran against the real pinned config.

NEXT:      codex review (this + #112) → merge → pin backtesting from the umbrella →
           operator-gated deploy of the -run checkout → observe the next weekly promote run
           produce a gate verdict (pass/reject) instead of exit 2, and the
           wf-promote/retrain/silent-refusal alarm cluster clear.
