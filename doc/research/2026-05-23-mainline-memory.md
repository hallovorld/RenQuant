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
- Current-contract WF has now completed and failed acceptance. This replaces
  the earlier "running" state below.

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
- `00fdf70 fix(renquant104): fail closed on unavailable sanity gates`
  - `run_wf_gate.py` no longer skip-passes when the rawlabel panel is missing,
    the scorer kind is unsupported, or sanity prediction fails.
  - Existing-model sanity now stamps `sanity_method` and a multi-shift placebo
    profile into `wf_gate_metadata`.
  - Targeted tests passed:
    `tests/test_wf_gate_cli_contract.py tests/test_promote_wf_gate.py`.
- Latest forensic update:
  - `scripts/sim_trade_ledger.py` now rebuilds forensic round trips using the
    configured tax-lot method (`hifo` for current 104) instead of hard-coded
    FIFO.
  - New `scripts/analyze_wf_trade_forensics.py` gives a repeatable WF trace
    report for exit/source/regime/score/tax attribution.
  - Targeted tests passed:
    `tests/test_sim_trade_ledger.py tests/test_wf_trade_forensics.py`.
- Latest decision-tree repair:
  - QP buy/top-up emission now has a strict alpha-admission gate:
    finite `rank_score >= 0.55`, finite raw `panel_score >= 0`, and available
    slot capacity for new tickers.
  - Production/golden configs disable forced QP cash deployment
    (`qp_min_invested_pct=0`, `qp_cash_drag_lambda=0`) and enable conviction
    caps.
  - Standalone TopUp floor raised from `0.20` to `0.55`.
  - Targeted tests passed:
    `tests/test_qp_admission_gate.py tests/test_qp_conviction_cap.py
    tests/test_buy_quality_gates.py tests/acceptance/jobs/test_split_jobs_e2e.py
    tests/test_qp_grinold_kahn_transform.py tests/test_p0_fixes_regression_guards.py
    tests/test_wf_config_parity.py`.

## Active Validation

Current-contract WF gate completed:

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

Generated WF config checked:
`tax.cash_debit_mode=reporting_only`.

Verdict: `FAIL`.

- Annual-net WF Sharpe mean: `+0.816`.
- Annual-net WF APY mean: `+7.55%`.
- SPY Sharpe mean: `+1.081`.
- Strategy minus SPY Sharpe: `-0.265`.
- SPY APY mean: `+16.94%`.
- Strategy minus SPY APY: `-9.39pt`.
- Positive Sharpe cuts: `3/3`.
- Beat SPY Sharpe: `1/3`.
- Beat SPY APY: `0/3`.
- Benchmark-lag regimes: `HIGH_CALM`, `LOW_SPIKED`.
- Trade ledger contract: passed.
- Trade monotonicity gate: passed only because BULL_CALM was eligible and
  passed; pooled Spearman was `-0.002`, so single-trade score monotonicity is
  still weak.
- Sanity battery: failed. Real IC `+0.0750`, shuffled IC `-0.0020`, placebo IC
  `+0.0462`; placebo must be `< +0.0375`.
- Follow-up diagnostic: shuffled labels are clean across 10 seeds (max |IC|
  about `0.0047`), but future-shift labels remain correlated at many horizons:
  shift 5d `+0.0734`, 20d `+0.0670`, 60d `+0.0462`, 120d `+0.0835`,
  252d `+0.0741`. Treat this as unresolved slow-factor/placebo methodology
  risk, not proof of clean alpha.

Do not promote this candidate. Do not run production buy/full from this
evidence. The immediate research question is why the time-shift placebo keeps
too much signal, and why positive event-level returns still fail benchmark and
annual-net acceptance.

Current-contract cut-level metrics:

| cut | event APY | event Sharpe | annual-net APY | annual-net Sharpe | SPY Sharpe | Δ Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| 2024-01-02 to 2024-12-31 | `+15.71%` | `+1.497` | `+9.54%` | `+0.848` | `+1.778` | `-0.931` |
| 2024-07-01 to 2025-06-30 | `+11.51%` | `+1.152` | `+4.67%` | `+0.462` | `+0.715` | `-0.253` |
| 2025-04-01 to 2026-03-28 | `+13.36%` | `+2.140` | `+8.44%` | `+1.139` | `+0.749` | `+0.390` |

Current-contract trade forensics, after rebuilding from raw trade events with
the current config's `hifo` tax-lot method:

- Round trips: `191` total, `158` closed, `33` open.
- Closed gross P/L: `+$32.78k`.
- Closed tax estimate: `+$27.16k`.
- Closed tax-estimated net P/L: `+$5.63k`.
- Closed win rate: `62.0%`.
- Median hold: `60d`; average hold: `86.2d`.
- Tax integrity is clean under current semantics: `tax_cash_debited=$0`, no
  positive closed row has `tax > gross_pnl`, and no losing row has positive tax.
- Exit buckets:
  - `stop_loss`: 36 exits, gross/net `-$17.59k`, win rate `0%`.
  - `trailing_stop`: 59 exits, gross `+$39.23k`, net `+$19.08k`.
  - `qp_sell`: 39 exits, gross `+$8.79k`, net `+$3.65k`.
  - `panel_conviction`: 9 exits, gross `-$0.29k`, net `-$0.34k`.
- Entry source:
  - QP buys: 89 closed, gross `+$21.53k`, net `+$2.46k`, win `57.3%`.
  - Top-ups: 69 closed, gross `+$11.26k`, net `+$3.17k`, win `68.1%`.
- Entry rank-score versus realized `pnl_pct` Spearman: `-0.0028`.
- Score deciles are not cleanly monotonic; the 8th decile lost money and the
  6th/9th deciles carried much of the gross P/L.

Important correction: the earlier "tax greater than gross" forensic symptom
was caused by replaying HIFO-configured sim trades with a FIFO round-trip
matcher. It was an attribution bug, not a broker-cash debit regression.

## 2026-05-23 Panel-Exit Ownership Fix

- Found one more decision-tree ownership issue while the QP-gated WF rerun was
  running: `TickerSellJob` still executed legacy `PanelConvictionExitTask`,
  while `InferencePipeline` also runs `CrossSectionalPanelExitTask` immediately
  after `PanelScoringJob`.
- Fix: production/golden config now sets
  `risk.panel_exit.legacy_enabled=false`. The legacy task still defaults on for
  backward-compatible unit tests and explicit A/B configs, but prod 104 has one
  owner for raw panel/NGBoost exits: `CrossSectionalPanelExitTask`.
- Added a regression test that proves the legacy task is a no-op when
  `legacy_enabled=false`.
- Hardened `scripts/wf_config_parity.py` to compare `risk.panel_exit` as a
  semantic path. A WF side config can no longer drift on panel-exit ownership
  or sell thresholds without failing before simulation.
- Also corrected stale exit docs: trailing stop, stop loss, and single-day loss
  are regime-configured, not BULL_CALM-only.

## Mainline Queue

1. Diagnose the sanity failure: time-shift placebo IC `+0.0462` is too high.
   Check splitter/label horizon, feature timestamping, regime persistence, and
   market beta/momentum leakage before trusting the reported real IC.
2. Diagnose benchmark/annual-net failure: event Sharpe is positive, but
   annual-net and SPY-relative metrics fail. Split by regime, exposure, tax,
   stop-loss bucket, and QP/top-up source.
3. Re-run strict WF after the QP/TopUp admission repair and compare:
   event-level, annual-net, SPY-relative, regime cuts, score monotonicity,
   stop-loss bucket, and QP/TopUp source buckets.
4. Implement stop-loss changes only after paired A/B acceptance; leading
   candidates are non-BULL volatility-aware stops and earlier panel/mu soft
   exits for positions whose model thesis deteriorates before hard stop.
5. Build PatchTST true WF manifest before quoting PatchTST portfolio APY/Sharpe
   as OOS. Static PatchTST full-window sims are style diagnostics only.
6. Continue after-tax/no-trade-region and stop-loss research per regime, using
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
- Decision-tree contract:
  `doc/research/2026-05-23-decision-tree-contract.md`.
- HIFO-aligned WF trade forensics:
  `doc/research/2026-05-23-wf-trade-forensics.md`.
