# RenQuant 104 Mainline Memory — 2026-05-23

This is the first file to read when continuing the current repair campaign.
It exists because multiple result streams were mixed together during the
session. Do not infer current state from a single stale metric.

## Mission

Make RenQuant 104 scientifically trustworthy end to end:

1. A model is accepted only with leak-safe WF evidence, SPY comparison,
   per-regime IC/Sharpe, calibration health, and config/sector fingerprint.
2. IC must convert into tradeable alpha after the decision tree, QP, exits,
   turnover, and tax reporting.
3. Every buy/sell decision must be explainable from persisted decision-tree
   fields, not reconstructed from scattered logs.
4. XGB remains primary unless a challenger passes the same strict acceptance.
   PatchTST stays shadow/router research until it has a true WF manifest.
5. No silent fallback: missing metadata, weaker score fallback, missing sector
   metadata, or failed artifact evidence blocks buy/full paths.

## Current Truth

- The active production artifact still carries an old failed WF stamp:
  APY `+0.63%`, Sharpe `-1.3233`, SPY Sharpe `+1.0808`, `passed=false`.
  This is stale failed metadata, and strict preflight now blocks full/buy on it.
- That `-1.3233` stamp is not the whole repaired research state. It must not be
  quoted as "current prod performance" without the above context.
- Tax cash corruption is fixed in current production semantics:
  `tax.cash_debit_mode=reporting_only`. Sim reports estimated tax separately
  and does not debit broker cash for estimated capital-gains tax.
- QP/strict-contract safety improved, but alpha conversion is not solved. The
  remaining main problem is whether signal survives the final decision tree
  into realized APY/Sharpe after turnover, exits, and annual-net tax.
- The best current short-window style evidence says PatchTST and XGB differ:
  PatchTST bought more names and outperformed over 2026-05-06 to 2026-05-22,
  but that 13-trading-day, zero-sell window is not promotion evidence.
- A 2026-05-23 strict WF rerun doc claims XGB event APY around `+15%` and
  Sharpe around `+1.7`, but the exact raw artifact path still needs
  reconciliation. Treat that claim as promising but not authoritative until the
  equity/trade artifacts are verified.

## Pushed Progress

- `81bd338 fix(renquant104): enforce strict model contracts`
  - Hard-fails buy/full preflight on bad or missing WF/SPY/regime
    IC/calibration/config evidence.
  - Stops global calibrator fallback to raw score.
  - Persists QP target/delta/status fields.
  - Full test suite passed at that checkpoint:
    `12654 passed, 8791 skipped, 1 xfailed`.
- `6e68f09 docs(renquant104): record current evaluation state`
  - Adds `doc/research/2026-05-23-current-state-ledger.md`.
  - Adds a WF-config regression test proving prod `tax.cash_debit_mode` wins
    over stale side-config event-cash debit.
  - Targeted test passed: `tests/test_wf_config_parity.py`.

## Active Validation

Current-contract WF gate is running with three cuts in parallel:

```bash
.venv/bin/python scripts/run_wf_gate.py \
  --artifact backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.foreground_20260523-094050.staging.json \
  --strategy-config strategy_config.sim_wl200_172_sentiment.calibrated_causal.json \
  --derive-config-from-prod \
  --jobs 3 \
  --trace-dir artifacts/diagnostics/wf_trade_traces/codex_current_contract_20260523-190033
```

Log:
`logs/wf_gate_104/current_contract_20260523-190033.log`.

Generated WF config already checked:
`tax.cash_debit_mode=reporting_only`.

This WF run is the next authoritative acceptance stream. Do not promote,
declare fixed, or run live buy/full based only on old docs while this is
unresolved.

## Mainline Queue

1. Poll the running current-contract WF to completion.
2. Parse each cut:
   - APY, Sharpe, MaxDD, turnover, buys/sells;
   - SPY APY/Sharpe on the same dates;
   - annual-net APY/Sharpe and tax estimate;
   - trade monotonicity by regime and score decile;
   - stop-loss, QP sell/close, trailing-stop P/L buckets.
3. If WF fails, diagnose the first failing contract or conversion point:
   model evidence, calibration, universe/sector metadata, QP, exits, turnover,
   or exposure.
4. If WF passes, verify raw artifacts and only then consider promotion or daily
   full. Production buy/full must remain preflight-gated.
5. Write a decision-tree contract doc with expected input/output ranges for
   data freshness, regime, gates, candidates, scoring, calibration, QP,
   rotation, persistence, and ntfy.
6. Build PatchTST true WF manifest before quoting PatchTST portfolio APY/Sharpe
   as OOS. Static PatchTST full-window sims are style diagnostics only.
7. Continue after-tax/no-trade-region and stop-loss research per regime, using
   literature-backed hypotheses and paired A/B sims.

## Known Failure Modes To Keep Front And Center

- Signal IC does not automatically become alpha. Trade-domain monotonicity must
  be measured after the full decision tree.
- QP must size/rebalance qualified alpha; it must not turn weak candidates into
  trades.
- Bull markets punish low exposure. Low beta can look safe while failing to
  participate.
- Event-level tax stress and annual-net economic tax are different metrics.
  Current headline should use reporting-only cash plus annual-net tax estimate.
- Stop-loss exits have been the main gross loss bucket in multiple traces.
  Do not change thresholds blindly; split by regime, score decile, hold age,
  volatility, and drawdown path first.
- PatchTST evidence is positive but weaker and not yet strict-WF accepted.
  Treat it as shadow/router candidate, not replacement primary.
- Noisy ntfy/reopen-cancel alerts are partly fixed but wrapper success alerts
  may still duplicate runner alerts.

## Stop Conditions

Stop and fix before reporting performance if any of these happen:

- WF config loses `tax.cash_debit_mode=reporting_only`.
- A calibrator/scorer fingerprint mismatch is detected.
- Sector metadata is missing for a buyable ticker.
- A buy/full path silently falls back to raw score or a weaker score.
- Trade logs lack `blocked_by`, model type, sector, score snapshot, QP
  target/delta/status, or sell P/L/tax/net for emitted orders.
- A metric is not labeled as event-level, annual-net, short-window style, or
  acceptance-grade WF.

## Companion Docs

- Detailed stream separation:
  `doc/research/2026-05-23-current-state-ledger.md`.
- PatchTST/XGB style handoff:
  `doc/research/2026-05-23-pending-research-and-patchtst-xgb-style.md`.
- Strict WF rerun claim needing artifact reconciliation:
  `doc/research/2026-05-23-strict-wf-xgb-patchtst-rerun.md`.
- Decision-tree and sim audit:
  `doc/research/2026-05-23-decision-tree-and-sim-audit.md`.
