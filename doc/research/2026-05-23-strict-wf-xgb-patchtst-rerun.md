# 2026-05-23 strict WF rerun: XGB, NGBoost overlay, PatchTST shadow

Purpose: rerun the RenQuant 104 simulation after the tax cash-debit and QP
churn fixes, using point-in-time walk-forward model artifacts and complete
sector metadata. This file is written for audit handoff.

## Controls used

- Window: 2024-07-02 to 2026-02-10, 404 trading days.
- Manifest: `backtesting/renquant_104/artifacts/sim/walkforward_manifest_172_sentiment.calibrated_causal.json`.
- Manifest contract: 43/43 retrain folds have `calibrator_uri`.
- Regime/correlation artifacts: `artifacts/sim/spy-gmm-regime.json` and
  `artifacts/sim/watchlist-correlation.json`, both `as_of_date=2023-12-29`.
- Sector metadata: generated from current production config; 0 buyable
  watchlist tickers missing `sector_map`, 0 unmapped sectors.
- Tax cash mode: `reporting_only`; per-trade tax is estimated, but cash is not
  debited inside the sim.
- Shadow telemetry was disabled for speed in these comparison configs. It is
  read-only and does not alter primary decisions.

Temporary configs used:

- `backtesting/renquant_104/strategy_config.codex_xgb_wf_calibrated_qpfix_noshadow_20260523.json`
- `backtesting/renquant_104/strategy_config.codex_xgb_pure_wf_calibrated_qpfix_noshadow_20260523.json`

## Result summary

| run | event APY | event Sharpe | max DD | annual-net APY | annual-net Sharpe | tax cash debited | tax estimate | buys/sells |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| XGB + prod per-regime NGBoost overlay | +14.83% | +1.67 | 6.58% | +9.53% | +0.94 | $0 | $14,297 event / $9,068 annual-net | 86 / 55 |
| Pure XGB, NGBoost fully disabled | +15.35% | +1.69 | 5.90% | +8.99% | +0.84 | $0 | $17,677 event / $10,893 annual-net | 101 / 75 |
| SPY benchmark | +15.48% | +0.91 | 18.99% | n/a | n/a | n/a | n/a | n/a |

Interpretation:

- The prior "all models lose money" conclusion is not supported under the
  corrected sim semantics. Both XGB variants are positive and have much higher
  risk-adjusted return than SPY in this window.
- Tax is no longer corrupting cash accounting: `tax_cash_debited=0`.
- Tax remains economically material. Event-level return is roughly SPY-like
  with much lower drawdown; annual-net APY trails SPY because realized gains
  create estimated tax drag.
- NGBoost overlay is not a catastrophic break here. Pure XGB has slightly
  higher event APY/Sharpe, but higher turnover and tax estimate. The production
  per-regime NGBoost overlay lowers annual-net tax drag and ends with better
  annual-net Sharpe in this run.

## Trade forensics

XGB + NGBoost overlay:

- Round trips: 99 total, 88 closed, 11 open.
- Gross win rate: 57.6%.
- Average closed hold: 76.0 days; median 58 days.
- Gross P/L: +$24,755; tax-estimated net P/L: +$10,458.
- Loss bucket: stop-loss exits lost -$10,146 gross across 18 exits.
- Positive buckets: trailing-stop exits +$13,424 gross; QP sells +$8,727
  gross; open marks +$13,156 gross.
- Entry source: QP buys outperformed top-ups in this run:
  - QP: +$19,542 gross, +$8,502 net, 62.1% win.
  - TopUp: +$5,213 gross, +$1,956 net, 51.2% win.

Pure XGB:

- Round trips: 122 total, 110 closed, 12 open.
- Gross win rate: 61.5%.
- Average closed hold: 73.8 days; median 62 days.
- Gross P/L: +$25,665; tax-estimated net P/L: +$7,988.
- Stop-loss exits lost -$17,415 gross across 31 exits.
- Trailing-stop exits were much stronger (+$22,133 gross), but the extra
  churn increased estimated tax drag.

Score monotonicity on realized trades remains weak:

- XGB + NGBoost: entry rank-score vs realized `pnl_pct` Spearman +0.018;
  `entry_mu` +0.054; `entry_sigma` -0.107.
- Pure XGB: entry rank-score vs realized `pnl_pct` Spearman +0.038;
  `entry_sigma` -0.211.

This means the signal is portfolio-useful but noisy at individual trade level.
The decision tree still relies heavily on exit/risk management to convert weak
cross-sectional edges into a stable equity curve.

## PatchTST status

The strict-contract PatchTST shadow artifact currently wired in config is:

`artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt`

Its validation evidence:

- `best_val_ic=+0.030657`
- Per-regime IC:
  - BULL_VOLATILE: +0.0524
  - BEAR: +0.1916
  - CHOPPY: +0.0307

The earlier DOE result (`artifacts/patchtst_doe_hf/summary_full.md`) found
point 1 best by bull-regime IC:

- lr 1e-4, weight_decay 1e-2, warmup 4, seq_len 24
- bull_ic_mean +0.0580 across 3 cuts
- PBO 0.33, but DSR -0.702

Current shadow is not the DOE point-1 hyperparameter set. It is a stricter
full-window contract artifact with weaker but positive validation IC.

Do not report a PatchTST portfolio APY/Sharpe from the current static shadow
artifact as true OOS. The artifact was selected using validation data ending
2026-02-10; with a 60-business-day forward label, the last label reaches into
early May 2026. A portfolio sim over 2024-2026 would be selection-leaky unless
PatchTST is rebuilt as a proper per-cut walk-forward manifest with causal
calibrators, analogous to the XGB manifest above.

## Open design issues

1. Production docs/config wording still need reconciliation: global
   `ranking.panel_scoring.ngboost.enabled=false`, but per-regime overlays turn
   NGBoost on in BULL_VOLATILE/CHOPPY/BEAR and hysteresis can carry it into
   BULL_CALM. This is not necessarily wrong, but it is not "dormant."
2. Annual-net APY trails SPY despite much better drawdown and Sharpe. The next
   design work should optimize after-tax holding period and turnover, not just
   event-level Sharpe.
3. Stop-loss exits remain the main loss bucket. This should be analyzed by
   entry regime, entry score decile, and holding age before changing stop
   thresholds.
4. PatchTST needs a walk-forward acceptance path before promotion: per-cut
   artifacts, causal calibrators, and the same tax/QP sim harness.

