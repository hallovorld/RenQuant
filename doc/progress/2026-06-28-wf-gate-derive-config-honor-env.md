# WF-gate prod-reference selection (rollback/umbrella path) — converged parity contract

STATUS: open PR (fix/wf-gate-derive-config-honor-env). Code + tests done; awaiting
review. Severity P2 (weekly automation, not live trading): the GBDT/XGBoost
shadow track had no path through the weekly WF-promote gate. No bad orders, no
capital impact.

WHAT: `scripts/run_wf_gate.py::_prod_config_path()` now SELECTS the production
reference whose scorer kind MATCHES the candidate's declared kind (read from
artifact metadata), and routes BOTH the `--derive-config-from-prod` derivation
site and the parity-check site through it. A GBDT/xgb candidate is compared
against the GBDT/shadow config (`strategy_config.shadow.json`, `kind=xgb`); a
PatchTST candidate against the PatchTST primary (`strategy_config.json`,
`kind=hf_patchtst`). `RENQUANT_STRATEGY_CONFIG` (exported by
`weekly_wf_promote.sh`) is honored but VALIDATED against the candidate kind; a
mismatch or an unknown kind FAILS CLOSED. No mutation of
`ranking.panel_scoring.kind` (that would defeat parity). No retrain.

WHY/DIR: the Python derive path hardcoded the prod reference as the PatchTST
PRIMARY at two sites, so a GBDT candidate derived an eval config with
`kind=hf_patchtst` pointing at a GBDT JSON artifact and the scorer-kind/artifact
parity guard fired on every GBDT run — the GBDT shadow track could never
graduate. Selecting the kind-matched reference (rather than forcing kind to
match) keeps a genuine prod-vs-candidate mismatch failing, as it should.

EVIDENCE:
  §4(b)
  - artifact: `scripts/run_wf_gate.py` (`_prod_config_path`, both derive +
    parity sites); `tests/test_wf_gate_derive_prod_config_env.py` (10 passed);
    real configs `backtesting/renquant_104/strategy_config.json` (hf_patchtst)
    and `strategy_config.shadow.json` (xgb, artifact
    `artifacts/prod/panel-ltr.alpha158_fund.json`).
  - prod or exp: ROLLBACK / umbrella-of-record path (`scripts/run_wf_gate.py`,
    invoked by `scripts/weekly_wf_promote.sh`). The ACTIVE package path
    (`renquant_backtesting.wf_gate.runner` /`pipelines`) is fixed separately in
    renquant-backtesting #58. Same selection contract on both.
  - existing data: env unset → `strategy_config.json` (hf_patchtst); env=
    `strategy_config.shadow.json` → kind=xgb, artifact
    `artifacts/prod/panel-ltr.alpha158_fund.json` — exactly the GBDT pairing the
    weekly wrapper expects. Branch `fix/wf-gate-derive-config-honor-env` off
    `main`.
  - best-known?: this fixes ONLY the rollback/umbrella path. renquant-backtesting
    #58 separately claims (and fixes) that the active package path is the live
    failure; both now share one candidate-matched selection contract. Merge #58
    first (active path), then #417 (rollback path).
  - scope: no gate-threshold change, no promotion, no retrain. Selection +
    fail-closed validation only; the builder/parity helpers are unchanged.

  Verification:
  - `tests/test_wf_gate_derive_prod_config_env.py`: kind→reference mapping
    (gbdt→shadow, patchtst→primary); validated env override; unknown kind and
    env-kind mismatch FAIL CLOSED; POSITIVE same-reference (GBDT candidate +
    GBDT reference) passes; NEGATIVE (derived PatchTST kind vs the correct GBDT
    reference) diverges on `ranking.panel_scoring.kind` and fails; the
    selection-layer blocks a GBDT-vs-PatchTST mismatch before parity; AST guard
    pins both call sites to the helper with the candidate kind.
  - Existing suites pass: `test_wf_config_parity`,
    `test_wf_config_builder_propagation`, `test_wf_gate_cli_contract`,
    `test_weekly_wf_promote_wrapper_guard`, `test_weekly_wf_recipe_guard`
    (60 passed). `git diff --check` clean.

NEXT: merge #58 (active path) first, then #417 (rollback path); both carry the
identical selection contract so the GBDT shadow track can graduate through both
the active and rollback gate entry points. No further change required here.
