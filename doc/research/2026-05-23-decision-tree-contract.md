# RenQuant 104 Decision-Tree Contract — 2026-05-23

This contract describes expected inputs, outputs, and failure behavior for the
RenQuant 104 daily/full decision tree. Code is still the source of truth; this
file is the audit checklist future agents should use when deciding whether the
pipeline output is scientifically usable.

## Global Invariants

- No buy/full path may silently fall back from a missing/failed score,
  calibrator, sector map, or model artifact to a weaker default.
- Every emitted order must carry `order_type`, `source_job`, `source_task`,
  `order_source`, `attribution_version`, `score_snapshot`, and
  `decision_inputs`.
- Every watchlist ticker must be explainable each bar through
  `ticker_daily_state` with `blocked_by`, `model_type`, `sector`, scores, and
  QP delta/target/status where applicable.
- Every metric must say which basis it uses: event-level, annual-net,
  short-window style, or acceptance-grade WF.
- A model score qualifies alpha; QP sizes/rebalances qualified alpha. QP must
  not turn a weak or metadata-incomplete ticker into a buy.

## Component Contract

| phase | component | required inputs | expected outputs | hard-fail / block conditions |
|---|---|---|---|---|
| Preflight | Strict model contract | active panel artifact, WF metadata, SPY refs, regime IC/monotonicity, calibrator health, config/sector fingerprint | full/buy allowed only when all hard gates pass | missing/failed WF, missing SPY refs, failed eligible-regime evidence, missing/bad calibrator, nondiffable or missing config fingerprint |
| Adapter | Runner/Sim/LEAN context build | config, today, OHLCV, models, sector map, broker holdings, cash, prices, state | populated `InferenceContext` | stale OHLCV, missing SPY, missing buyable sector metadata, pending broker order excluded from buy universe |
| Data freshness | `DataFreshnessGateTask` | per-symbol OHLCV max date, today/run timestamp, freshness config | pass/fail freshness state | stale market data in live/full/sell-only |
| Regime | `RegimeJob` | SPY OHLCV/returns, GMM/regime state, Hurst/CUSUM inputs | `ctx.regime`, `ctx.confidence`, `ctx.bear_only` | nonfinite confidence, impossible regime label, detector drift versus objective bear/volatile periods |
| Drawdown | `DrawdownJob` | portfolio value, HWM, risk config | updated HWM, `skip_buys` | drawdown halt/flatten thresholds breached |
| Buy gates | `BuyGatesJob` | regime, SPY trend/velocity, transition window, config | `buy_blocked`, `bear_only`, counters | SPY crash/trend block, transition uncertainty, disabled new buys by regime |
| Sell | `TickerSellJob` | holding, price, feature frame, model, exit params | `ExitSignal` with reason and threshold context | stop-loss, trailing stop, single-day loss, max hold, model sell streak; earnings blackout may veto model exits |
| Candidate build | `TickerCandidateJob` | ticker OHLCV, SPY OHLCV, model, sector map, earnings/wash state | `CandidateResult` or `blocked_by` | earnings blackout, wash sale, missing sector map, missing input, empty/nonfinite features, non-buy model signal when ticker gate is not bypassed |
| Panel scoring | `PanelScoringJob` | candidates/holdings, panel artifact, feature contract, inference frames | finite `panel_score`; calibrated `rank_score`; optional `mu`, `sigma`; blocked weak buys | scorer load failure, feature drift, row coverage failure, missing calibrator when enabled, scorer/calibrator fingerprint mismatch |
| Calibration | `LoadGlobalCalibrationTask` / `ApplyGlobalCalibrationTask` | calibrator artifact, scorer fingerprint, raw panel score | `rank_score in [0,1]`, expected-return estimate when configured | missing calibrator path, load failure, fingerprint mismatch, flat calibrator region when strict |
| Quality gates | weak-buy, vol, concentration, momentum/deep-drawdown gates | scored candidates, realized vol, current positions, regime | candidate list pruned with `blocked_by` | weak calibrated score, realized vol above cap, over-concentration, regime/momentum mismatch |
| Ranking | `RankingJob` | scored candidates | ordered `ctx.ranked` | empty candidates skip |
| Selection | `SelectionJob` | ranked candidates, held tickers, wash/sector/correlation rules, tiered thresholds | selected tickers and NEW_BUY orders | buy blocked, skip buys, wash-sale, sector cap, correlation cap, tier threshold, defensive outside BEAR |
| QP | `JointPortfolioQPJob` | holdings + qualified candidates, strict μ, positive σ, current weights, covariance, sector/correlation metadata, tax/wash masks | `ctx._qp_solution`, QP buy/sell orders, exits, `qp_delta_w`, `qp_target_w`, `qp_status` | disabled/non-QP solver, BEAR branch, calibrator contract failed, missing sector metadata, raw score as μ under strict contract, μ/σ horizon mismatch, duplicate sell for ticker already exiting |
| Top-up/trim | Kelly top-up and trim tasks | current holdings, latest scores, Kelly target, cash | TOP_UP or trim orders with attribution | buy blocked/skip buys, over target, insufficient cash, missing score |
| Persistence | SQLite decision trace | context, candidates, holdings, orders, exits, QP solution | `candidate_scores`, `ticker_daily_state`, `trades` rows | selected row with `blocked_by`; trade missing score snapshot or decision inputs; missing watchlist ticker row |
| Notification | ntfy/macOS alert helpers | run outcome, trade list, holdings, error taxonomy | deduped success/failure/trade alerts | duplicate wrapper success alerts, false reopen-cancel alerts, swallowed failure alerts |

## Numeric Ranges

- `rank_score`: calibrated probability-like score in `[0, 1]`.
- `panel_score`: finite raw panel score; not directly comparable to
  probability unless calibrated.
- `mu`: expected return on the configured QP/model horizon; must not be a raw
  rank score under strict QP contract.
- `sigma`: positive volatility on the same horizon as `mu` after alignment.
- `confidence`: finite regime confidence, effectively bounded to `[0, 1]`.
- `qp_delta_w` / `qp_target_w`: portfolio weights; finite and consistent with
  QP constraints.
- `tax_cash_debited`: `0` in current `reporting_only` sim semantics.
- `annual_net_tax_estimate`: reporting/evaluation overlay, not broker-cash
  debit.

## Current Acceptance Findings

The current-contract WF gate completed and failed:

- Annual-net mean APY `+7.55%`, Sharpe `+0.816`.
- SPY mean APY `+16.94%`, Sharpe `+1.081`.
- Beat SPY Sharpe `1/3`; beat SPY APY `0/3`.
- Sanity failed because time-shift placebo IC `+0.0462` exceeded the allowed
  half-real-IC threshold `+0.0375`.
- Trade ledger contract passed: `192` auditable rows, `159` closed rows,
  no missing entry μ/σ, no missing exit regime.
- Trade-domain monotonicity is weak overall: pooled Spearman `-0.002`.

This means the system is no longer grossly broken in the old tax-cash sense,
but it is not accepted. The next fixes must target placebo/split causality,
benchmark-relative performance, stop-loss loss bucket, and after-tax turnover.
