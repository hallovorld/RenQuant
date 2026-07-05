# fix(promote): dynamic GBDT config resolution in weekly_wf_promote

DATE: 2026-07-05

## What changed

`scripts/weekly_wf_promote.sh` hardcoded `strategy_config.shadow.json` as the
GBDT reference config. After the 06-23 operator lineup reversal (XGB back to
primary in `strategy_config.json`), the shadow config became PatchTST, so every
weekly XGB promote failed at the WF gate kind-parity check:

```
scorer kind ('hf_patchtst') does not match candidate kind ('xgb')
```

Replaced the hardcoded path with `_find_gbdt_config()` — scans both
`strategy_config.json` and `strategy_config.shadow.json`, reads their declared
`ranking.panel_scoring.kind`, and picks the one that declares `xgb`. Survives
future primary/shadow swaps without code changes.

Companion: renquant-backtesting#69 (same dynamic approach in the WF gate's
`_resolve_prod_reference_by_kind()`).

## Round 2 (codex review)

STATUS: fixed
WHAT: `_find_gbdt_config()`'s umbrella-mode branch only checked
`backtesting/renquant_104/<name>` — the umbrella WORKING-COPY config.
`render_strategy_104_snapshot.py`'s own header documents that this exact path
is NOT what production actually consumes and had already gone stale once
across the 2026-06-23 lineup reversal: the authoritative source is the
pin-aligned runtime checkout at
`.subrepo_runtime/repos/renquant-strategy-104/configs/` (synced to
`subrepos.lock.json`). The prior code never verified this — it only assigned
a path string, so the staleness risk was invisible until this PR made the
resolution actually READ the file, which then correctly failed under
`RQ_WF_GATE_RUNNER=umbrella` (`tests/test_weekly_wf_promote_snapshot_backstop.py`,
"no strategy config declares kind=xgb; cannot resolve GBDT reference" at
Step 3.5) since the umbrella snapshot-backstop fixture — like real
production — only populates the pin-aligned location.
WHY-DIR: an emergency-rollback path (`RQ_WF_GATE_RUNNER=umbrella`) silently
depending on a documented-stale location means a real rollback scenario
could fail closed for the wrong reason.
EVIDENCE: `_find_gbdt_config()` now checks the pin-aligned location FIRST
(the real, current source of truth), falling back to the umbrella working
copy only if the pin-aligned tree isn't present. Added
`test_umbrella_resolves_gbdt_config_regardless_of_lineup_slot` — swaps
which config slot declares `kind=xgb` (the pre-06-23 orientation) with no
config at all under the umbrella working-copy path, confirming resolution
still succeeds; confirmed via `git stash` that all 4
`test_weekly_wf_promote_snapshot_backstop.py` tests fail against the
pre-fix code with the exact symptom codex described, and pass after.
Also fixed `scripts/subrepo_ops_contract.py`'s
`weekly_retrain_delegates_to_orchestrator_wrapper` check, which asserted the
OLD hardcoded `GBDT_PROD_CONFIG="$(renquant_strategy_config "$SUBREPO_ROOT"
strategy_config.shadow.json)"` string verbatim — already broken by this PR's
original commit (dca70d4), before this round's fix, since that exact string
no longer exists once resolution became dynamic. Updated the check to assert
the current `renquant_strategy_config "$SUBREPO_ROOT" "$cfg_name"` pattern.
Full snapshot-backstop + wrapper-guard + ops-contract test group: 19/19
pass (1 pre-existing unrelated WF-manifest fingerprint-mismatch test
deselected, confirmed failing identically on clean `origin/main`; 2
unrelated `test_operator_script_env.py` failures about `conditional_retrain_
104.sh`/`manual_promote.sh` confirmed pre-existing on clean `origin/main`,
untouched by this PR).
NEXT: none.
