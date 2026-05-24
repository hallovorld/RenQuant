# 2026-05-23 RenQuant 104 Current State Ledger

Purpose: prevent future agents from mixing incompatible result streams. This
file is a factual ledger, not a promotion memo. Code and artifacts remain the
source of truth.

## Do Not Mix These Four Things

### 1. Active production artifact stamp

- Artifact:
  `backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json`
- Trained: 2026-05-18.
- Embedded WF stamp:
  - APY mean: `+0.63%`
  - Sharpe mean: `-1.3233`
  - SPY APY mean: `+16.94%`
  - SPY Sharpe mean: `+1.0808`
  - `passed=false`
- Current strict preflight treats this as failed evidence and blocks full/buy
  paths. It is stale failed metadata, not the repaired research state.

### 2. 2026-05-22 old full-window resim

- Window: 2024-07-02 to 2026-02-10.
- XGB strict-cutoff:
  - Total return `+1.88%`
  - APY `+1.17%`
  - Sharpe `+0.20`
  - MaxDD `8.90%`
  - Closed win rate `61.98%`
  - Avg hold `49.1d`
- PatchTST clean diagnostic:
  - Total return `+2.40%`
  - APY `+1.49%`
  - Sharpe `+0.23`
  - MaxDD `7.39%`
  - Closed win rate `58.16%`
  - Avg hold `36d`
- SPY same window:
  - Total return `+26.07%`
  - APY `+15.59%`
  - Sharpe `+0.91`
- Use this stream only as failure forensics. It predates later QP/tax contract
  repairs and should not be quoted as the current production verdict.

### 3. 2026-05-23 short-window style comparison

- Handoff doc:
  `doc/research/2026-05-23-pending-research-and-patchtst-xgb-style.md`.
- Window: 2026-05-06 to 2026-05-22, only 13 trading days, zero sells.
- PatchTST primary:
  - Total return `+3.21%`
  - Sharpe `+6.61`
  - Buys/sells `7/0`
  - Style: more aggressive; bought ORCL, SPOT, HON, GM, LLY, DUK, IBM.
- XGB WF primary:
  - Total return `+0.69%`
  - Sharpe `+12.10`
  - Buys/sells `3/0`
  - Style: more conservative; bought ABBV, MCD, PH.
- SPY same short window: `+1.61%`.
- This is useful for decision-tree behavior and model style. It is not a
  stable APY/Sharpe acceptance result.

### 4. 2026-05-23 strict XGB WF rerun claim

- Handoff doc:
  `doc/research/2026-05-23-strict-wf-xgb-patchtst-rerun.md`.
- Claimed controls: walk-forward XGB manifest, complete sector metadata,
  `tax.cash_debit_mode=reporting_only`, 2024-07-02 to 2026-02-10.
- Claimed result:
  - XGB + per-regime NGBoost overlay: event APY `+14.83%`, Sharpe `+1.67`,
    annual-net APY `+9.53%`, annual-net Sharpe `+0.94`, buys/sells `86/55`.
  - Pure XGB: event APY `+15.35%`, Sharpe `+1.69`, annual-net APY `+8.99%`,
    annual-net Sharpe `+0.84`, buys/sells `101/75`.
  - SPY: APY `+15.48%`, Sharpe `+0.91`.
- I have not yet reconciled the exact raw equity/trade artifact path behind
  this table. Treat it as a documented result requiring artifact-path
  verification, not as a promotion decision by itself.

## Current Fix Status

- Strict model-contract gates were implemented and pushed in commit `81bd338`.
- Current full/buy paths fail closed when WF/SPY/regime-IC/calibration/config
  evidence is missing or failed.
- Sim tax accounting now uses `tax.cash_debit_mode=reporting_only` in current
  production semantics: estimated capital-gains tax is reported separately and
  is not subtracted from broker cash during the event path.
- A regression test now asserts that WF configs generated from production keep
  prod `tax.cash_debit_mode=reporting_only` rather than inheriting an old
  event-cash-debit side config.

## Trade-Level Diagnosis So Far

The IC-to-alpha failure is not simply "model has no signal." The failure mode
seen in old and repaired traces is conversion:

- Trade-domain score monotonicity is weak. Higher entry rank_score does not
  reliably produce higher realized trade P/L.
- Final decision-tree actions materially reshape the raw signal. QP buys,
  top-ups, QP sells/closes, stop-losses, and trailing stops decide much of the
  realized distribution.
- Stop-loss exits are the main gross loss bucket in several runs.
- Trailing-stop winners are strong but require longer holds.
- Low exposure/beta can make a strategy look stable while still failing to
  participate enough in bull markets.
- Tax cash corruption appears fixed, but tax drag remains economically real
  whenever turnover realizes short-term gains.

## Current Running Validation

As of this ledger, a fresh current-contract WF gate is running:

```bash
.venv/bin/python scripts/run_wf_gate.py \
  --artifact backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.foreground_20260523-094050.staging.json \
  --strategy-config strategy_config.sim_wl200_172_sentiment.calibrated_causal.json \
  --derive-config-from-prod \
  --jobs 3 \
  --trace-dir artifacts/diagnostics/wf_trade_traces/codex_current_contract_20260523-190033
```

This is the next authoritative acceptance stream because it derives WF config
from current production semantics and runs three cuts concurrently.

## Pending Work

1. Reconcile the raw artifacts behind
   `2026-05-23-strict-wf-xgb-patchtst-rerun.md`.
2. Finish the current-contract WF gate and inspect per-cut APY/Sharpe, SPY
   comparison, tax mode, and trade-monotonicity metadata.
3. Add a dedicated decision-tree contract doc with expected input/output ranges
   for data freshness, regime, gates, scoring, calibration, QP, rotation,
   persistence, and ntfy.
4. Implement PatchTST walk-forward manifest generation before quoting PatchTST
   portfolio APY/Sharpe as true OOS.
5. Continue stop-loss and no-trade-region research with per-regime tests and
   transaction-cost/tax-aware acceptance metrics.
