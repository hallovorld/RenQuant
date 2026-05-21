# renquant_104 Sim Trade Forensics - 2026-05-21

## Verdict

The current 172-feature WL200 walk-forward simulation is not tradable as configured. It has positive gross closed PnL across the sampled windows, but after-tax returns are crushed by short-term tax drag, stop losses, a broken QP expected-return contract, and a rank score that is not monotonic with realized trade outcomes. The prior lack of durable per-trade evidence was itself a bug; it has been fixed and pushed.

## Why this had to be rerun

The previous WF and acceptance runs did not leave a trustworthy, replayable per-trade ledger for the exact 172-feature WL200 manifest. That is not acceptable for a model promotion pipeline.

Observed before the fix:

- `scripts/run_sim_104.py` only emitted equity JSON, not trade logs, round trips, or a forensic report.
- `scripts/run_wf_gate.py` parsed stdout and did not persist per-cut trade ledgers by default.
- `data/sim_runs.db` had old sim traces, but no sufficient manifest/config fingerprint for the current 172-feature WL200 walk-forward artifacts.
- Candidate score provenance in old rows was often blank or incomplete.

Fixes already committed and pushed:

- `2af90b2 fix(renquant104): persist sim trade ledgers`
- `87be440 fix(renquant104): stamp sim trade regime context`

The second commit fixed two attribution bugs discovered during the audit:

- QP-generated buy trades did not carry the entry regime.
- Open lots were not marked to end-of-window prices in the report.

The expensive sims were not rerun after the second fix. Instead, their existing raw ledgers were enriched from the immutable run logs for regime attribution and from OHLCV end prices for open-lot MTM. This did not change trade execution or PnL; it fixed observability.

## Provenance

Simulation config:

- `backtesting/renquant_104/strategy_config.sim_wl200_172_sentiment.json`

Walk-forward manifest:

- `backtesting/renquant_104/artifacts/sim/walkforward_manifest_172_sentiment.json`

Key facts:

- Watchlist size: 142.
- Manifest rows: 43.
- Sampled artifacts have 172 features.
- Sentiment features are present, including `mean_sentiment`, `n_articles_log`, and `sentiment_pos_share`.
- `walkforward.enabled = true`.
- `ranking.panel_scoring.ngboost.enabled = false`.
- `ranking.kelly_sizing.enabled = true`.
- Tax config used by sim: short-term rate `0.50`, long-term rate `0.32`, long-term threshold `365` days.

Per-trade artifacts:

- `backtesting/renquant_104/artifacts/diagnostics/wf_trade_traces/172_sentiment_20260521_forensics/2024-01-02_to_2024-12-31.trades.enriched.csv`
- `backtesting/renquant_104/artifacts/diagnostics/wf_trade_traces/172_sentiment_20260521_forensics/2024-01-02_to_2024-12-31.round_trips.enriched.csv`
- `backtesting/renquant_104/artifacts/diagnostics/wf_trade_traces/172_sentiment_20260521_forensics/2024-01-02_to_2024-12-31.report.enriched.md`
- `backtesting/renquant_104/artifacts/diagnostics/wf_trade_traces/172_sentiment_20260521_forensics/2024-07-01_to_2025-06-30.trades.enriched.csv`
- `backtesting/renquant_104/artifacts/diagnostics/wf_trade_traces/172_sentiment_20260521_forensics/2024-07-01_to_2025-06-30.round_trips.enriched.csv`
- `backtesting/renquant_104/artifacts/diagnostics/wf_trade_traces/172_sentiment_20260521_forensics/2024-07-01_to_2025-06-30.report.enriched.md`
- `backtesting/renquant_104/artifacts/diagnostics/wf_trade_traces/172_sentiment_20260521_forensics/2025-04-01_to_2026-03-28.trades.enriched.csv`
- `backtesting/renquant_104/artifacts/diagnostics/wf_trade_traces/172_sentiment_20260521_forensics/2025-04-01_to_2026-03-28.round_trips.enriched.csv`
- `backtesting/renquant_104/artifacts/diagnostics/wf_trade_traces/172_sentiment_20260521_forensics/2025-04-01_to_2026-03-28.report.enriched.md`

## Window Summary

| Window | Strategy final | APY | Sharpe | MaxDD | SPY APY | SPY Sharpe | SPY MaxDD | Raw buys/sells | Closed/open lots |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024-01-02 to 2024-12-31 | 93,543 | -6.48% | -0.673 | 16.9% | +24.11% | +1.778 | 8.4% | 132 / 165 | 178 / 7 |
| 2024-07-01 to 2025-06-30 | 96,735 | -3.30% | -0.367 | 15.9% | +13.47% | +0.715 | 19.0% | 126 / 159 | 168 / 17 |
| 2025-04-01 to 2026-03-28 | 109,826 | +9.99% | +0.342 | 15.8% | +13.26% | +0.749 | 12.1% | 117 / 158 | 163 / 1 |

Interpretation:

- The strategy underperformed SPY in all three sampled windows.
- The third window is profitable, but still loses to SPY on APY and Sharpe.
- The first two windows are outright negative after tax.
- No unmatched sells were found in the enriched round-trip reconstruction.

## Money Attribution

| Window | Realized gross | Tax | Realized net | Open MTM | Net plus open |
|---|---:|---:|---:|---:|---:|
| 2024-01-02 to 2024-12-31 | +19,083.39 | 24,202.48 | -5,119.09 | -1,308.03 | -6,427.12 |
| 2024-07-01 to 2025-06-30 | +12,540.02 | 21,387.21 | -8,847.20 | +5,607.18 | -3,240.02 |
| 2025-04-01 to 2026-03-28 | +40,177.45 | 32,790.44 | +7,387.01 | +2,468.17 | +9,855.18 |
| Combined overlapping cuts | +71,800.86 | 78,380.13 | -6,579.27 | +6,767.31 | +188.04 |

This is the core result: pre-tax alpha exists in the closed trades, but the live taxable objective does not survive taxes and stop-loss asymmetry. Winners are usually short-term gains taxed at 50%; losses create limited benefit in this sim accounting. A system promoted on after-tax Sharpe cannot treat this as acceptable.

## Exit Reason Attribution

Across all enriched closed lots:

| Exit reason | Lots | Gross PnL | Tax | Net PnL | Win rate | Median hold |
|---|---:|---:|---:|---:|---:|---:|
| stop_loss | 75 | -58,740.15 | 0.00 | -58,740.15 | 0.00% | 15d |
| single_day_loss | 32 | -10,682.68 | 481.31 | -11,163.99 | 18.75% | 20d |
| trailing_stop | 36 | +14,467.27 | 7,535.26 | +6,932.01 | positive | short-term dominated |
| qp_close | 82 | +20,457.57 | 11,834.16 | +8,623.41 | positive | short-term dominated |
| model_sell | 98 | +27,644.59 | 15,694.64 | +11,949.95 | positive | short-term dominated |
| qp_sell | 97 | +34,517.24 | 19,850.35 | +14,666.88 | positive | short-term dominated |
| max_hold | 89 | +44,137.01 | 22,984.41 | +21,152.60 | positive | longer |

Interpretation:

- The stop stack is the largest negative driver.
- The profitable exit reasons are real, but they are taxed hard.
- The system is taking too many short-term taxable wins while letting a smaller number of losers hit large realized losses.
- The gross-to-net conversion is not a rounding issue; it changes the sign of the strategy in two windows.

## Regime Attribution

All closed entries in the enriched sampled windows were `BULL_CALM`.

| Entry regime | Closed lots | Realized net |
|---|---:|---:|
| BULL_CALM | 509 | -6,579.27 |

This matters. A pooled "all-regime" average is not the right diagnostic here, because the trades are not diversified across regimes in these windows. The live behavior tested here is essentially a BULL_CALM-only long book. Under the project principles, the correct comparison is BULL_CALM strategy behavior versus BULL_CALM SPY/passive baseline and versus the strategy's own entry thesis.

## Rank Score Validity

The panel rank score is not behaving like a tradable expected-return ordering in these trades.

Spearman correlation between entry rank score and realized lot PnL percent:

| Window | Spearman(rank_score, PnL pct) |
|---|---:|
| 2024-01-02 to 2024-12-31 | -0.0546 |
| 2024-07-01 to 2025-06-30 | +0.0204 |
| 2025-04-01 to 2026-03-28 | -0.0158 |
| Combined | -0.0380 |

Combined rank-score quintiles:

| Quintile | Lots | Realized net | Mean PnL pct |
|---|---:|---:|---:|
| Q1 low | 94 | -2,603.30 | +2.48% |
| Q2 | 93 | +1,074.28 | +3.37% |
| Q3 | 97 | +4,690.11 | +7.48% |
| Q4 | 90 | -4,100.03 | +4.59% |
| Q5 high | 94 | -8,039.35 | +3.14% |

The top score quintile is the worst after tax. For a cross-sectional learning-to-rank system used for top-N selection, sizing, and rotation, this is a theory-level failure. If rank is the proxy for expected excess return, higher rank must show positive monotonic relation to forward return or realized trade PnL after reasonable costs. Here it does not.

This does not prove the model has zero signal in all formulations. It proves the current score-to-trade pipeline has no demonstrated monotonic economic edge in these sampled out-of-sample trades.

## Ticker Attribution

Worst net tickers across enriched closed lots:

| Ticker | Lots | Net PnL | Gross PnL | Win rate | Mean PnL pct |
|---|---:|---:|---:|---:|---:|
| NVDA | 6 | -5,145.82 | -5,145.82 | 0.00% | -11.02% |
| SNOW | 9 | -3,975.09 | negative | weak | negative |
| LRCX | 15 | -3,868.11 | negative | weak | negative |
| MSFT | 9 | -2,894.63 | negative | weak | negative |
| AMAT | 10 | -2,548.63 | negative | weak | negative |
| MPWR | 9 | -2,319.40 | negative | weak | negative |
| PANW | 13 | -1,540.34 | negative | weak | negative |
| DLR | 4 | -1,333.17 | negative | weak | negative |
| AMZN | 1 | -1,250.01 | negative | 0.00% | negative |
| CVS | 13 | -1,155.29 | negative | weak | negative |

Best net tickers:

| Ticker | Net PnL |
|---|---:|
| PLTR | +5,748.03 |
| AVGO | +2,098.25 |
| WFC | +2,017.54 |
| LITE | +1,816.07 |
| FTNT | +1,804.82 |
| WDC | +1,578.68 |
| NET | +1,559.33 |
| AMD | +1,556.99 |

The losers are not a single-symbol edge case. Several liquid mega/large-cap tech names lose under the current selection and exit system.

## Log Warnings That Matter

Warning counts by window:

| Window | missing_mu / QP contract | insufficient cash | calibrator saturated | drawdown logs | buy_blocked true | panel veto |
|---|---:|---:|---:|---:|---:|---:|
| 2024-01-02 to 2024-12-31 | 174 / 174 | 175 | 180 | 17 | 43 | 152 |
| 2024-07-01 to 2025-06-30 | 186 / 186 | 170 | 195 | 58 | 107 | 120 |
| 2025-04-01 to 2026-03-28 | 166 / 166 | 174 | 192 | 28 | 73 | 142 |

Important examples from the logs:

- `ApplyKellySizingTask` reports zero non-zero Kelly candidates because `mu` is missing.
- `ValidateQPMuContract` warns that QP is using raw score semantics while `alpha_to_mu` was not applied.
- Sim execution often rejects QP buy orders as insufficient cash.
- `CALIBRATOR-SATURATED` appears repeatedly, with compressed rank-score dispersion.
- `PANEL_VETO` often keeps model-sell exits when panel rank is above a static threshold, despite compressed scores.

These warnings are not harmless. They explain why the economic behavior does not match the intended design.

## Bugs And Design Failures

### Fixed: missing per-trade evidence

This was a pipeline observability bug. WF/acceptance must leave enough evidence to reconstruct every trade, not just final APY and Sharpe. Fixed by adding default trade traces to `run_wf_gate.py` and explicit trade-log outputs to `run_sim_104.py`.

### Fixed: missing regime on QP buys

QP-generated buy orders were missing entry regime in the sim ledger. Fixed by stamping the current context regime into sim buy trade logs, with report-side enrichment for older raw outputs.

### Fixed: open-lot MTM missing from forensic report

Open positions previously had no end-of-window MTM in the report. Fixed by passing end prices into the report writer.

### Not fixed: QP expected-return contract

This is the most important remaining bug-class issue. The sim config has Kelly/QP enabled, but the logs show `missing_mu` and a raw-score fallback. An optimizer must not mix arbitrary rank score units with covariance/risk/cash constraints. QP needs an expected return in consistent units, or it should fail strict.

The correct contract is one of:

- candidate and holding `mu` is present and in forward-return units; or
- `alpha_to_mu.enabled = true` transforms calibrated rank/alpha into forward-return units before QP; or
- QP/Kelly is disabled for that acceptance path.

Warn-and-continue is not a promotion-safe behavior.

### Not fixed: optimizer orders are not executable

The logs show many `insufficient cash` skips after QP order generation. That means the optimizer can propose a portfolio that the execution adapter cannot actually implement. This invalidates portfolio-level claims because the realized strategy is not the optimized strategy.

The order generator must enforce self-financing/cash constraints before emitting orders, or a deterministic post-optimizer prune must be part of the modelled optimization objective and test contract.

### Not fixed: score monotonicity failure

The rank score has near-zero/negative relation to realized trade PnL, and the top quintile is worst after tax. This is a promotion blocker. A cross-sectional rank model may still have IC on a raw label, but the current trading pipeline has not shown economic monotonicity after gates, sizing, exits, and taxes.

### Not fixed: after-tax objective mismatch

The system has positive gross closed PnL but fails after tax in two windows and loses to SPY in all three. If this strategy is meant for a taxable account, model selection, exit tuning, and promotion gates must optimize after-tax outcomes. If it is meant for a non-taxable account, the sim config should disable tax and the acceptance criteria should say so explicitly.

### Not fixed: static panel-veto threshold under compressed scores

The panel veto appears to use a static threshold around `0.5`, while calibrated scores are compressed and saturated. Under compression, `> 0.5` may not mean "strong enough to override an exit." This should be distribution-aware, regime-aware, or tied to expected return and uncertainty.

## Theory Check

The current design violates several basic principles of model-based portfolio construction:

1. Cross-sectional rank principle: if a score is used to select top names, size positions, or rotate holdings, it must have monotonic relation to forward excess return or realized economic PnL. The sampled trades fail this check.

2. Optimizer unit principle: mean vector, covariance, costs, and constraints must be in compatible units. Feeding raw or compressed rank scores into QP as if they were expected returns is not theoretically valid.

3. Execution feasibility principle: a portfolio optimizer's emitted trades must be executable under the same cash, lot, and holding constraints used in simulation. Post-hoc `insufficient cash` skips create a different strategy than the one optimized.

4. Regime conditioning principle: claims must be evaluated conditional on the regimes that actually generated trades. In these windows, the strategy is effectively BULL_CALM-only.

5. Tax-aware objective principle: a taxable strategy cannot promote on pre-tax gross edge if short-term realization turns it negative after tax.

## Required Next Fixes

P0 fixes before trusting another acceptance result:

1. Make QP mu strict.
   - Set acceptance/sim QP contract to fail, not warn, when `mu` is missing.
   - Ensure `alpha_to_mu.enabled = true` or write expected-return `mu` for every candidate and holding before QP.
   - Add a regression test that acceptance configs cannot run QP with `mu_none > 0`.

2. Make QP orders executable.
   - Enforce cash/self-financing constraints before emitting buys.
   - Add a test where optimizer output cannot produce `insufficient cash` skips in sim execution.

3. Add economic monotonicity gates.
   - For each active entry regime, require rank-score quintile monotonicity or at least Q5 materially better than Q1-Q3.
   - Require positive Spearman or another prespecified monotonic metric.
   - Evaluate after-tax and pre-tax separately, with an explicit account assumption.

4. Add regime-specific baseline comparison.
   - Compare BULL_CALM trades against SPY over the same BULL_CALM dates.
   - Do not average across empty or inactive regimes.

P1 design fixes:

1. Rework tax/turnover objective.
   - Penalize short-term churn in model promotion and exit tuning.
   - Consider holding winners toward long-term tax treatment when risk allows.

2. Replace static panel-veto threshold.
   - Use rank percentile, expected return, or expected return minus uncertainty/cost instead of raw `rank_score > 0.5`.

3. Re-evaluate stop stack.
   - Stop-loss and single-day-loss rules are the largest negative bucket.
   - Retune with regime-conditioned drawdown control, not just headline Sharpe.

4. Separate model signal quality from trading-system quality.
   - Report label IC, rank monotonicity, raw pre-tax trade PnL, after-tax PnL, and execution slippage/cash skips independently.

## Bottom Line

The answer is not "all models have no signal." The sharper answer is: this current renquant_104 172-feature WL200 trading system is not yet trustworthy, because the model score, optimizer contract, execution adapter, exit rules, and tax objective are not aligned. The per-trade evidence now exists, and it points to specific repair work rather than a vague model failure.

