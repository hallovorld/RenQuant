# WF-gate `--derive-config-from-prod` honors `RENQUANT_STRATEGY_CONFIG` (GBDT parity)

2026-06-28. Severity: P2 (weekly automation, not live trading). Customer impact:
the weekly WF-promote gate could never admit a GBDT/XGBoost candidate — every
GBDT run fail-closed on a config parity guard — so the GBDT shadow track had no
path to graduation. No bad orders, no capital impact.

## Symptom
`scripts/weekly_wf_promote.sh` evaluates a GBDT candidate by exporting
`RENQUANT_STRATEGY_CONFIG=$GBDT_PROD_CONFIG`
(`backtesting/renquant_104/strategy_config.shadow.json`, `kind=xgb`) and calling
`scripts/run_wf_gate.py … --derive-config-from-prod`. The run fail-closed on the
WF config parity guard with a scorer-kind/artifact mismatch
(`ranking.panel_scoring.kind=hf_patchtst` vs a GBDT `panel-ltr*.json` artifact).

## Root cause
The Python derive path ignored the env var the wrapper set. In
`scripts/run_wf_gate.py::main()`, `--derive-config-from-prod` hardcoded the
production reference as `STRATEGY_DIR / "strategy_config.json"` — the PatchTST
PRIMARY config (`kind=hf_patchtst`) — at two sites:

- the config-derivation site (feeds `build_wf_config_from_prod`), and
- the parity-check site (`evaluate_wf_config_parity(prod, derived, …)`).

`build_wf_config_from_prod` (`scripts/wf_config_builder.py`) deep-copies the prod
config and overwrites `walkforward` + `ranking.panel_scoring.artifact_path`, but
**not** `ranking.panel_scoring.kind`. So when the candidate is GBDT, the derived
eval config kept `kind=hf_patchtst` (inherited from the PatchTST primary) while
`artifact_path` pointed at a GBDT JSON — the parity guard correctly fired. The
wrapper had already pointed `RENQUANT_STRATEGY_CONFIG` at the GBDT/shadow config
(`kind=xgb`), but `run_wf_gate.py` never read it.

## Fix
`scripts/run_wf_gate.py`:
- Add `_prod_config_path()` — returns `RENQUANT_STRATEGY_CONFIG` (absolute, or
  relative to `STRATEGY_DIR`) when set, else the hardcoded PatchTST
  `strategy_config.json`. Default behaviour is unchanged when the env is unset.
- Route both the derivation site and the parity site through it (and fail-closed
  with a clear error if the resolved prod config is missing).

Minimal and orchestrator-of-record-side only: the builder still copies `kind`
from whatever prod config it is given, so selecting the GBDT prod config is
sufficient — the derived `kind` becomes `xgb`, matching the GBDT candidate, and
parity (prod `xgb` vs derived `xgb`) passes. No change to
`wf_config_builder.py`/`wf_config_parity.py` was needed; no retrain.

## Verification
- `tests/test_wf_gate_derive_prod_config_env.py` (new): env-unset falls back to
  the primary; relative/absolute env values resolve correctly; an end-to-end
  derive+parity over a GBDT (`xgb`) prod config + `panel_ltr_xgboost` candidate
  **passes**; a regression case proves that deriving from the PatchTST prod while
  the correct reference is the GBDT prod makes `ranking.panel_scoring.kind`
  diverge and parity **fail**; AST guard pins both call sites to the helper.
- Against the real config files: env unset → `strategy_config.json`
  (`hf_patchtst`); env=`strategy_config.shadow.json` → `kind=xgb`, artifact
  `artifacts/prod/panel-ltr.alpha158_fund.json` — exactly the GBDT pairing the
  weekly wrapper expects.
- Existing parity/builder/wrapper suites pass (`test_wf_config_parity`,
  `test_wf_config_builder_propagation`, `test_wf_gate_cli_contract`,
  `test_weekly_wf_promote_wrapper_guard`, `test_weekly_wf_recipe_guard`). The 4
  `test_wf_gate_recipe_scope.py` failures in this sandbox are a pre-existing
  `pyarrow`-not-installed environment issue (identical on stock `origin/main`),
  unrelated to this change.
