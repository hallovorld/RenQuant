# renquant_104 Non-Tax Root Cause Audit

Date: 2026-05-21

This note continues `doc/audits/renquant104_tax_gross_root_cause_2026-05-21.md`.
The tax model explains why `tax > net gross` is possible. It does not explain
why the strategy lost money in a rising market. The deeper failure is that the
current WF evidence does not show a tradable, monotone edge after the decision
tree, sizing, stops, and portfolio constraints touch the signal.

## Evidence Scope

Primary diagnostic data:

- `backtesting/renquant_104/artifacts/diagnostics/wf_trade_traces/172_sentiment_20260521_forensics/`
- 3 WF cuts:
  - `2024-01-02_to_2024-12-31`
  - `2024-07-01_to_2025-06-30`
  - `2025-04-01_to_2026-03-28`

Important caveat: these ledgers are diagnostic, not promotion-grade evidence.
They were produced before the current strict QP mu contract path was fully
stamped into the artifact.

Specific evidence gaps:

- The prod artifact's `wf_gate_metadata.trade_monotonicity` is `null`.
- The same stamp says `passed=false`, mean WF Sharpe `-0.233`, SPY mean Sharpe
  `+1.081`, and beat-SPY cuts `0/3`.
- The historical run log contains `ValidateQPMuContract ... continuing in warn
  mode`; current code/config now require `qp_mu_contract=strict`.
- The round-trip ledgers have `entry_mu`, `entry_sigma`, and `entry_sigma_mult`
  empty for all closed trades.
- Default tracked `strategy_config.sim_wl200.json` points at an old 169-feature
  manifest, while current prod is a 172-feature sentiment artifact.
- The 172-feature sim config validates the recipe, but it is untracked and uses
  FIFO tax lots while prod config uses HIFO.

Conclusion: old WF ledgers are useful for forensics, but not a clean
"post-fix accepted/rejected" run.

## Root Cause 1: Trade-Level Score Edge Is Not Monotone

The most damaging result is not tax. It is that entry score barely predicts the
actual closed trade outcome.

Closed trades with finite `entry_rank_score`: 468.

Score correlation:

| target | Pearson | Spearman |
|---|---:|---:|
| gross PnL dollars | +0.0069 | +0.0068 |
| after-tax PnL dollars | -0.0530 | +0.0060 |
| trade return pct | +0.0131 | -0.0114 |

`scripts/trade_monotonicity.py` on the same ledgers returns:

```text
passed: false
reason: score monotonicity failed in active regime(s): BULL_CALM
pooled spearman: -0.0114
```

Score quintiles:

| score bucket | n | score range | gross | tax | net after tax | mean pnl pct | stop-loss n | SDL n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 low | 94 | 0.5697-0.5833 | 4,762.47 | 7,365.77 | -2,603.30 | +2.48% | 13 | 5 |
| Q2 | 93 | 0.5833-0.5898 | 14,801.63 | 13,727.35 | 1,074.28 | +3.37% | 12 | 5 |
| Q3 | 94 | 0.5899-0.5967 | 22,252.22 | 17,423.81 | 4,828.41 | +7.59% | 13 | 4 |
| Q4 | 93 | 0.5967-0.6040 | 12,568.79 | 16,807.12 | -4,238.33 | +4.58% | 16 | 7 |
| Q5 high | 94 | 0.6041-0.6353 | 8,771.78 | 16,811.12 | -8,039.35 | +3.14% | 20 | 8 |

The highest-score bucket is the worst after tax and has the most stop-loss /
single-day-loss exits. That is incompatible with treating `rank_score` as a
valid portfolio optimizer input or a direct buy-confidence proxy.

Scientific basis: a ranking model can have positive panel IC while failing as a
trading policy after costs, stop rules, sizing, and turnover. Acceptance must
therefore test realized trade monotonicity, not just panel IC. This is the same
principle behind purged/embargoed finance validation in Lopez de Prado's
financial ML work: the validation target must match the deployed decision.

## Root Cause 2: Stop-Loss Tail Losses Dominate

Exit reason decomposition:

| exit reason | n | gross | tax | net after tax | mean pnl pct | median hold | median score |
|---|---:|---:|---:|---:|---:|---:|---:|
| stop_loss | 75 | -58,740.15 | 0.00 | -58,740.15 | -10.42% | 15 | 0.5961 |
| single_day_loss | 32 | -10,682.68 | 481.31 | -11,163.99 | -4.11% | 20 | 0.5993 |
| trailing_stop | 36 | 14,467.27 | 7,535.26 | 6,932.01 | +9.05% | 42.5 | 0.5905 |
| qp_close | 82 | 20,457.57 | 11,834.16 | 8,623.41 | +3.92% | 13 | 0.5892 |
| model_sell | 98 | 27,644.59 | 15,694.64 | 11,949.95 | +6.07% | 24 | 0.5940 |
| qp_sell | 97 | 34,517.24 | 19,850.35 | 14,666.88 | +6.63% | 19 | 0.5943 |
| max_hold | 89 | 44,137.01 | 22,984.41 | 21,152.60 | +13.52% | 53 | 0.5951 |

Soft exits are gross-positive. The loss is dominated by hard path exits:

```text
stop_loss + single_day_loss = -69,904.14 net
```

This says the system is not mainly "selling winners too early." It is entering
clusters that later hit path-risk exits. The stop logic is revealing bad entry
selection and concentration; loosening stops alone would likely hide the
problem until drawdown is larger.

Worst clusters:

| entry month | n | gross | tax | net after tax | stop-loss n | SDL n |
|---|---:|---:|---:|---:|---:|---:|
| 2024-07 | 49 | -18,771.90 | 1,374.42 | -20,146.33 | 22 | 3 |
| 2024-11 | 30 | -2,976.60 | 3,986.79 | -6,963.39 | 9 | 2 |
| 2025-12 | 16 | -3,598.58 | 1,485.71 | -5,084.29 | 3 | 1 |
| 2026-02 | 7 | -2,873.13 | 232.77 | -3,105.90 | 6 | 0 |

Example: `2024-07-31` bought ANET/LRCX/TSLA/WDC exposure, then every closed lot
from that entry date exited by stop loss within 1-5 days for `-5,936.11`. The
entry scores were not low; LRCX had `0.6007`, TSLA `0.6001`, ANET `0.5970`.

Local market context:

| period | SPY return | XLK return | NVDA return |
|---|---:|---:|---:|
| 2024-07-01 to 2024-08-05 | -5.13% | -13.18% | -19.19% |
| 2026-01-26 to 2026-02-27 | -0.97% | -5.01% | -4.98% |

The strategy was not losing to "the whole bull market." It was repeatedly long
crowded tech/AI-beta clusters during local corrections. A SPY benchmark hides
that sector/factor exposure.

## Root Cause 3: BULL_CALM-Only Entry Flow

All 509 closed round trips entered in `BULL_CALM`.

That means the active trade generator is effectively a BULL_CALM-only entry
engine. Other regimes mostly affect buy blocks, current-position risk, or
sentiment/NGBoost overlays, but they did not create meaningful new-entry
diversity in these WF cuts.

The sell path uses current regime params:

```python
regime_p = ctx.config.get("regime_params", {}).get(ctx.regime, {})
exit_params = _build_exit_params(regime_p, ctx.config)
```

So a position entered in BULL_CALM can later be evaluated using BULL_VOLATILE,
CHOPPY, or BEAR stops. That may be intended, but the diagnostic ledger records
only `entry_regime`, not `exit_regime` or the exact exit thresholds used.

This creates a blind spot: the audit can see "entered BULL_CALM and stopped
out," but cannot prove whether the stop fired under BULL_CALM's 15% rule,
BULL_VOLATILE's 5% rule, CHOPPY's 8% rule, or sigma-scaled SDL.

Required fix: round-trip ledgers must stamp `exit_regime`, `exit_confidence`,
`exit_stop_loss_pct`, `exit_sdl_threshold`, and whether the stop threshold came
from absolute, sigma, or ATR logic.

## Root Cause 4: Buy Floor Is Scale-Mismatched

The configured buy floor is:

```json
"buy_floor": "adaptive_mean_std_cap",
"buy_floor_adaptive_cap": 0.30,
"buy_floor_min": 0.20
```

The implementation clamps `mean(rank_score) + std(rank_score)` into `[0.20,
0.30]`. But current calibrated entry scores are in a narrow high-probability
band:

```text
min=0.5697, p25=0.5842, median=0.5934, p75=0.6012, max=0.6353
```

Therefore a 0.30 floor is effectively a no-op. It does not select the top edge;
it allows nearly everything scored by the calibrator through to downstream QP
and stops.

This is a scale bug, not a philosophy issue. A floor that made sense on an old
score scale became meaningless after calibration moved the score distribution
to roughly 0.59.

Required fix: replace this with a scale-aware gate, for example:

- per-bar top quantile or top-k within the eligible universe,
- `rank_score >= median + k * IQR` on that day's cross-section,
- positive calibrated expected return above costs and risk buffer,
- per-regime trade monotonicity gate before promotion.

## Root Cause 5: Portfolio Concentration Is Too Permissive

`max_concurrent_positions=8` and `max_positions_per_sector=6` permits up to 75%
of the book in one sector bucket before price movement. That is too loose for a
strategy whose worst losses are clustered in AI-chip / giant-tech names.

Sector PnL from the WF ledgers:

| sector | n | gross | tax | net after tax | stop-loss n | unique tickers |
|---|---:|---:|---:|---:|---:|---:|
| ai_chip | 98 | 8,531.70 | 16,692.40 | -8,160.70 | 21 | 11 |
| giant_tech | 33 | -540.01 | 4,642.39 | -5,182.40 | 9 | 7 |
| consumer | 11 | -1,263.86 | 225.41 | -1,489.28 | 1 | 3 |
| software | 77 | 23,629.04 | 21,584.89 | 2,044.15 | 18 | 10 |
| datacenter_hw | 66 | 17,825.34 | 14,315.18 | 3,510.16 | 8 | 9 |

Markowitz-style portfolio construction is only as good as its expected-return
vector and constraints. Markowitz (1952) requires expected returns and covariance
to be meaningful; Kelly sizing requires a real edge estimate and variance; and
Ledoit-Wolf shrinkage helps covariance stability but cannot repair a bad or
non-monotone mu vector.

Required fix: add benchmark-relative and sector/factor constraints, not just a
loose sector count:

- cap sector/factor active weight vs SPY/XLK/sector ETF exposure,
- add special group caps for AI-chip / semis / mega-tech,
- require per-sector trade monotonicity before that sector can consume multiple
  slots,
- add portfolio-level beta and drawdown-at-risk telemetry to acceptance.

## Root Cause 6: QP Evidence Was Not Runtime-Stamped

Current code has moved in the right direction:

- `rotation.joint_actions.qp_mu_contract = "strict"`
- `ranking.kelly_sizing.use_calibrator_mu = true`
- `ranking.kelly_sizing.use_realized_vol_fallback = true`
- `ValidateQPMuContractTask` now returns `False` in strict mode when QP falls
  back to raw score semantics.

But the historical ledgers still show:

```text
entry_mu: all empty
entry_sigma: all empty
ApplyKellySizingTask: zero_reasons[mu_none=...]
ValidateQPMuContract ... continuing in warn mode
```

So the old WF result cannot prove that QP/Kelly consumed a valid expected return
and volatility vector.

Required fix:

- promotion ledgers must require finite `entry_mu` and `entry_sigma` whenever
  QP/Kelly is enabled,
- stamp `_qp_mu_contract` into every sim result / run metadata,
- fail acceptance if any strict-QP bar logs fallback, missing mu, or missing
  sigma,
- remove or hard-disable `BuildMuVectorTask`'s fallback from `mu` to
  `panel_score` in promotion mode; raw score fallback should be explicit
  experimental mode only.

## Root Cause 7: Validation Gates Were Still Too High-Level

The prod artifact stamp includes panel/sanity/WF metrics, but it did not include
trade monotonicity for the old run. That allowed a positive panel IC to coexist
with a non-tradable realized policy.

Promotion should require all of these separately:

1. Recipe scope: candidate artifact or matching walk-forward manifest.
2. Strict QP mu/sigma contract.
3. Panel IC and placebo sanity.
4. Trade-level score monotonicity by active regime.
5. Score bucket PnL after costs.
6. SPY and sector benchmark comparison.
7. Exit-family concentration: no single hard-exit family can dominate total
   loss.
8. Tax reporting in pre-tax, immediate-tax stress, and annual-net-tax modes.

## Immediate Fix Plan

P0:

- Treat existing 172-feature forensics as diagnostic only.
- Rerun WF with the post-strict code path, tracked 172-feature manifest, prod
  HIFO tax-lot setting, and trade monotonicity gates enabled.
- Add acceptance failure when `entry_mu` or `entry_sigma` is missing for any
  QP/Kelly-driven entry.
- Stamp `exit_regime` and exit threshold provenance in trade ledgers.

P1:

- Replace `adaptive_mean_std_cap` with a scale-aware cross-sectional gate.
- Add sector/factor exposure caps and active-risk telemetry.
- Add per-regime and per-sector score monotonicity reports.
- Split tax reporting into pre-tax, immediate-tax stress, and annual-net-tax.

P2:

- Re-evaluate stop rules only after P0/P1. The stop losses are currently a
  symptom of bad entries and concentration. Loosening them before fixing entry
  quality would make the backtest look better by taking more unreconciled risk.

## References

- IRS Topic 409, Capital Gains and Losses:
  https://www.irs.gov/taxtopics/tc409
- IRS Schedule D instructions:
  https://www.irs.gov/instructions/i1040sd
- Markowitz, H. (1952), "Portfolio Selection", Journal of Finance.
- Kelly, J. L. (1956), "A New Interpretation of Information Rate", Bell System
  Technical Journal.
- Ledoit, O. and Wolf, M. (2004), "Honey, I Shrunk the Sample Covariance
  Matrix", Journal of Portfolio Management.
- Lopez de Prado, M. (2018), "Advances in Financial Machine Learning", finance
  cross-validation and leakage-control framework.

## Bottom Line

The project should not be archived solely because one after-tax WF metric is
negative. But the current 172-feature XGB/panel-LTR candidate is not deployable.
It fails the more important test: higher scores do not reliably produce better
trades after the actual decision tree. Until trade monotonicity, strict mu/sigma
stamping, and concentration controls pass, this model should remain demoted or
shadow-only.
