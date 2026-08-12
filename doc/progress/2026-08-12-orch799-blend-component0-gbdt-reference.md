# Progress: the weekly WF-promote gate derives its xgb reference from the blend's component[0] (orch#799)

STATUS:   delivered. Fail-closed preserved; production untouched by the change.

WHAT:     `scripts/weekly_wf_promote.sh` `_find_gbdt_config()` — after the
          existing top-level `kind=xgb` scan over the PINNED runtime configs
          finds nothing, it now attempts a SECOND resolution: derive a
          `kind=xgb` production reference from the pinned blend's
          `ranking.panel_scoring.components[0]` (orch#799 option A). The
          derivation lives in the new `scripts/derive_gbdt_wf_reference.py` and
          reads the PINNED runtime `strategy_config.json` ONLY — never the
          umbrella working copy or the sibling checkout. `scripts/subrepo_ops_
          contract.py` weekly check updated to assert the new safe behaviour and
          to add two more `--pinned-config` bans on the banned sources.
          Tests: `tests/test_derive_gbdt_wf_reference.py` (new, 11 cases) +
          `tests/test_gate_prod_reference_fail_closed.py` (source-level guards
          updated for the derivation + the reworded fail-closed message).

WHY/DIR:  orch#799. After the full-book z-blend switch the pinned primary is
          `kind=blend`; the gate only accepted a top-level `kind=xgb` config as
          the GBDT reference, so every weekly retrain hit the
          `WEEKLY-BLOCKED` / `exit 2` path — the retrain runs, then the gate
          cannot even score it. The refusal message itself named the fix: derive
          the xgb reference from the blend's `component[0]`. Implemented that.

EVIDENCE (§4(b)):
  what:           A blend-primary pinned config now resolves the GBDT reference
                  from `components[0]` and REACHES the gate; a non-xgb
                  `components[0]` (or an absent leg artifact) still fails closed;
                  the banned working-copy / sibling sources are never consulted.
  claim:          `[VERIFIED]` — dry-run against the CURRENT served blend, unit
                  tests, and the reference-selection guard replica below.
  evidence:       `[VERIFIED-now]` DRY-RUN, current served blend
                  (`.subrepo_runtime/.../strategy_config.json`, `kind=blend`),
                  reading live READ-ONLY, output only to scratch — resolves:
                    derived kind        = xgb
                    derived artifact    = artifacts/prod/panel-ltr.alpha158_fund.json
                    components dropped   = True
                    provenance source   = the PINNED strategy_config.json
                  Reference-selection guard replica (run_wf_gate.py:165-175):
                    BEFORE (blend cfg as RENQUANT_STRATEGY_CONFIG): ref_kind
                      'blend' != candidate 'xgb'  -> FAIL CLOSED (exit 2)
                    AFTER  (derived cfg):          ref_kind 'xgb' == 'xgb'
                      -> gate PROCEEDS
                  config_fingerprint  blend = derived = `sha256:f8fb2259b2bf1537`
                  (EQUAL), which also equals `component[0].expected_config_
                  fingerprint` — independent confirmation the derived reference
                  IS the GBDT leg's recipe, on the SAME WF manifest discipline.
                  `[VERIFIED-now]` `_model_relevant_fields`
                  (backtesting/renquant_104/kernel/config_consistency.py:51-101)
                  reads only `panel_ltr`/`watchlist`/`sector*` — NOT
                  `panel_scoring.kind` / `.components` — so flipping kind and
                  dropping `components` is fingerprint-invariant by construction.
                  `[VERIFIED-now]` file suites, `python3 -m pytest`:
                    tests/test_derive_gbdt_wf_reference.py           11 passed
                    tests/test_gate_prod_reference_fail_closed.py     6 passed
                    tests/test_subrepo_ops_contract.py (contract ok)  passed
                  `scripts/subrepo_ops_contract.py` CLI: ok=True, 43 checks,
                  0 failures, `weekly_retrain_delegates_to_orchestrator_wrapper`
                  in passed.
  artifact:       Reference source is the PINNED runtime
                  `.subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.json`
                  only. The derived reference is written to a run-scoped scratch
                  path `logs/weekly_wf_promote/derived_gbdt_reference.<RUN_ID>.json`
                  — NOT a production config, NOT `.subrepo_runtime`, NOT
                  `backtesting/renquant_104/*`.
  prod or exp:    Production script change, behaviour-EXPANDING only on the
                  previously-dead blend branch: a run that used to `exit 2` now
                  reaches the gate on a kind-matched reference. The candidate the
                  gate scores (the freshly-retrained xgb leg) is unchanged; ONLY
                  reference-resolution gained the component[0] path. No scoring,
                  sizing, admission, WF/sanity/placebo/parity gate touched.
  existing data:  Yes — pinned config + prod GBDT leg artifact already on disk.
                  No compute, no spend. The change itself promotes nothing and
                  writes no production artifact/config.
  best-known?:    Yes. Option A was the code's own sanctioned option 1. The
                  banned sources (umbrella working copy `strategy_config.shadow.json`
                  = the A8 known-diverged `kind=xgb` file; sibling checkout) stay
                  banned — verified the live working-copy shadow is still
                  `kind=xgb` and would recreate the incident if consulted.
  scope:          One production script edited, one new helper script, one
                  contract edit, one new test file, one updated test file, this
                  doc. No config edited, no pin advanced, no artifact touched, no
                  promotion performed, no live-tree / `.subrepo_runtime` write.

VERIFICATION (how to reproduce the dry-run, promotes nothing):
  python3 scripts/derive_gbdt_wf_reference.py \
    --pinned-config .subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.json \
    --strategy-dir  backtesting/renquant_104 \
    --out           /tmp/derived_gbdt_ref.json
  # exit 0, stdout = the derived path (kind=xgb, artifact_path = component[0] leg).
  # Non-xgb component[0] or absent leg artifact -> exit != 0, nothing written.

NEXT:
  - Companion consideration for the ops team: if a future blend's `component[0]`
    is NOT the xgb leg, the gate stays `WEEKLY-BLOCKED` by design — the standing
    orch#799 alternative ("gate blend prods on a blend-kind candidate") remains
    the decision if that state ever becomes intentional.
  - No orchestrator emitter-contract line changed (the `WEEKLY-BLOCKED` and
    promote/reject action literals are unchanged), so no companion orchestrator
    PR is required for this change.
