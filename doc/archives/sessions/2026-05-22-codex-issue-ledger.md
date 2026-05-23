# 2026-05-22 Codex Issue Ledger

Source of truth: code and tests first, docs second. This ledger exists so the
session's repeated issues cannot be lost in chat history.

## Fixed Today

| Issue | Invariant | Verification |
|---|---|---|
| Failed WF evidence killed the whole daily run | A failed buy-side WF gate blocks new risk, but sell-only risk exits still run | `tests/test_preflight.py::TestCheckWFGateMetadata::test_failed_wf_metadata_allows_sell_only_risk_exits`; live `daily_104.sh` exited 0 with sell-only fallback |
| Daily full-mode probe sent urgent false ERROR before fallback | Daily wrapper owns expected fallback notifications; inner preflight ntfy can be suppressed | `tests/test_runner_trade_ntfy.py::TestSourceLevel::test_daily_wrapper_can_suppress_inner_preflight_ntfy`; probe log shows warning + suppressed |
| Legacy live DB crashed on `trade_date` index before migration | Column migrations run before indexes that depend on new columns | `tests/test_persistence.py::TestConnectionLifecycle::test_legacy_trades_table_migrates_before_trade_date_index`; `data/runs.alpaca.db` has `trades.trade_date` and `idx_trades_date` |
| Shadow e2e could hang daily indefinitely | Shadow is read-only and non-fatal; it must have a wall-clock budget | `tests/test_smoke_test_model.py::TestNoRetrainInDailyShell::test_shadow_e2e_has_wall_clock_timeout`; daily verification with `RENQUANT_SHADOW_TIMEOUT_SEC=30` exited 0 |
| Daily log double-wrote audit/dashboard lines | After `exec >> "$LOG"`, do not pipe to `tee -a "$LOG"` | `tests/test_smoke_test_model.py::TestNoRetrainInDailyShell::test_daily_does_not_double_append_log_after_exec_redirect` |
| Live QP optimized names with missing sector metadata | Every buyable watchlist ticker must have `sector_map`; every sector must have `sector_etf_map`; sector metadata is part of the model/config fingerprint | `tests/test_candidate_sector_map_gate.py`; `tests/test_strategy_config_sector_map.py`; targeted suite `92 passed`; real preflight now reports `P-SECTOR-MAP` OK and `P-CONFIG-FP` blocks old artifacts until retrained/stamped |
| Shadow preflight/timeout spammed ntfy with non-fatal errors | Preflight checks must use the active scorer artifact, not stale `panel_ltr` JSON; daily wrapper suppresses inner preflight ntfy; shadow timeout/fail logs by default and alerts only with `RENQUANT_SHADOW_ALERT_NTFY=1` | `tests/test_preflight.py` sequence-artifact cases; `tests/test_runner_trade_ntfy.py::TestSourceLevel::test_daily_shadow_wrapper_suppresses_inner_preflight_ntfy`; `tests/test_smoke_test_model.py::TestDailyShellInvariants::test_shadow_e2e_has_wall_clock_timeout`; shadow config preflight full pass |

Commit: `d315b65 Fix daily WF fallback and schema migration` pushed to `origin/main`.

## Operator Override Run

At the user's explicit instruction on 2026-05-22, Codex executed a one-time
LIVE Alpaca full e2e run with preflight `strict=false` in a side config
(`strategy_config.live_no_wf_gate_once.json`). This did not change the
production config.

Result: the active failed-WF artifact was allowed to pass through the full
buy path once and submitted three live market buy orders after hours:
`BAC x13`, `D x8`, and `WFC x7`, all accepted by Alpaca. This is evidence
that the e2e flow can execute, not evidence that the model is production-safe.
Production default should remain WF-gated until a passing artifact is promoted.

## Still Open / Must Close

| Priority | Issue | Required close condition |
|---|---|---|
| P0 | Active prod panel artifact carries failed WF evidence | Promote only a WF-passing artifact, or keep production in sell-only/no-new-buy mode. Close with strict WF report, per-regime metrics, SPY benchmark, and stamped `wf_gate_metadata.passed=true`. |
| P0 | Active prod panel artifact was trained/stamped before sector metadata coverage was fixed | Retrain and promote only if acceptance + WF pass under the complete sector schema; until then `P-CONFIG-FP` blocks full-buy and sell-only remains available. |
| P0 | XGB and PatchTST positive IC but weak/negative APY/Sharpe | Produce trade-level attribution by pipeline stage: signal rank, gate, selection, sizing, exit, tax/friction. Close only when the losing stage is identified with per-trade evidence and a regression/acceptance test. |
| P0 | PatchTST shadow is too slow for daily e2e | Keep timeout now; close by profiling and moving heavy feature/scorer setup out of the live critical path, or by running shadow as detached research with its own monitor. |
| P0 | Strict calibrator/scorer contract work is partly dirty | Finish/commit scorer artifact fingerprints, calibrator source fingerprints, and no-cross-model calibrator reuse guards. Close with tests in `tests/test_hf_patchtst_scorer.py`, `tests/test_regime_calibrator.py`, and `tests/test_shadow_scoring.py`. |
| P0 | Daily decision tree completeness in DB | Verify every ticker has a daily state row and every executed order has `score_snapshot_json` + `decision_inputs_json`. Close with DB invariant tests plus one live-run query. |
| P0 | Tax/gross accounting trust | Verify gross, net, tax, and after-tax P&L cannot violate basic accounting. Close with per-trade lot-level examples and tests around HIFO/ST/LT/wash-sale reporting. |
| P1 | PatchTST DOE conclusion vs shadow config | Reconcile DOE best parameter set with `strategy_config.shadow.json`; if different, retrain/recalibrate shadow under purged train-fit and report IC + placebo sanity. |
| P1 | Docs may overstate model trust | Update docs to say current prod artifact is buy-blocked by failed WF until a passing artifact is promoted. |

## Operating Rules For The Remaining Work

1. Every fix gets a regression guard before or with the patch.
2. Every model/training number must state split, embargo, regime breakdown, SPY benchmark, and placebo/shuffle status.
3. No production buy is re-enabled from a failed-WF artifact.
4. Shadow/read-only work cannot block primary risk exits.
5. Commit small, pushed chunks; do not mix code fixes with generated experiment artifacts.
