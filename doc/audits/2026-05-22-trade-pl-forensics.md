# RenQuant 104 Trade P/L Forensics

Date: 2026-05-22
Dataset: `backtesting/renquant_104/artifacts/diagnostics/post_fix_20260522/wf_traces_172_sentiment/*.round_trips.csv`
Scope: 164 closed walk-forward round trips from the recipe-matched 172-feature sentiment manifest.

## Summary

The model is not failing because every trade loses. It is failing because the edge is weak and noisy: win rate is high enough to look tempting, but the realized return path is too small, too short-term, and too exposed to regime transitions.

Overall closed-trade stats:

| Metric | Value |
|---|---:|
| closed round trips | 164 |
| gross win rate | 64.63% |
| after-tax win rate | 64.63% |
| gross PnL | +16,351.08 |
| tax | 13,916.32 |
| net after tax | +2,434.76 |
| avg gross / trade | +99.70 |
| avg net / trade | +14.85 |
| avg gross winner | +265.48 |
| avg gross loser | -203.27 |
| gross payoff ratio | 1.31 |
| gross profit factor | 2.39 |
| avg hold | 29.13d |
| median hold | 22d |
| avg winner hold | 33.13d |
| avg loser hold | 21.83d |

The problem: after-tax expectancy is only about `$14.85/trade`. That is not enough to produce a robust portfolio Sharpe once mark-to-market volatility and drawdowns are included.

## Exit Timing

| Exit reason | n | Win rate | Gross PnL | Tax | Net PnL | Avg hold |
|---|---:|---:|---:|---:|---:|---:|
| stop_loss | 12 | 0.00% | -5,482.75 | 0.00 | -5,482.75 | 21.50d |
| qp_close | 1 | 100.00% | +120.02 | 60.01 | +60.01 | 26.00d |
| qp_sell | 25 | 64.00% | +3,690.28 | 2,102.69 | +1,587.59 | 16.28d |
| panel_conviction | 91 | 65.93% | +8,964.49 | 6,657.47 | +2,307.02 | 24.77d |
| max_hold | 35 | 82.86% | +9,059.03 | 5,096.15 | +3,962.88 | 52.37d |

Interpretation:

- The stop-loss bucket is catastrophic: only 12 exits, but it erases more than the net profit of all other buckets combined.
- `max_hold` is the best-performing exit class, which suggests the strategy benefits from letting winners run rather than frequent short-term conviction exits.
- `qp_sell` and `panel_conviction` are not gross-negative, but they crystallize many short-term gains, creating heavy tax drag.

## Holding-Time Pattern

| Hold bucket | n | Win rate | Gross PnL | Tax | Net PnL |
|---|---:|---:|---:|---:|---:|
| <=5d | 9 | 55.56% | -319.16 | 118.80 | -437.96 |
| 6-10d | 18 | 38.89% | +570.84 | 815.27 | -244.43 |
| 11-20d | 42 | 59.52% | +85.03 | 1,733.76 | -1,648.73 |
| 21-30d | 40 | 65.00% | +3,925.82 | 3,602.57 | +323.25 |
| 31-45d | 30 | 76.67% | +5,632.66 | 3,743.48 | +1,889.18 |
| 46-60d | 8 | 62.50% | +289.14 | 728.69 | -439.55 |
| 61-90d | 13 | 84.62% | +3,547.47 | 1,910.03 | +1,637.43 |
| >90d | 4 | 100.00% | +2,619.28 | 1,263.71 | +1,355.57 |

Short holding periods are the weak zone. Trades held 11-20 days are gross flat and deeply tax-negative. Longer holds, especially 31-45d and 61d+, are much better.

## Regime Timing

| Entry -> Exit regime | n | Win rate | Gross PnL | Tax | Net PnL |
|---|---:|---:|---:|---:|---:|
| BULL_CALM -> BEAR | 9 | 11.11% | -2,367.88 | 343.15 | -2,711.03 |
| BULL_CALM -> BULL_VOLATILE | 11 | 63.64% | +524.42 | 591.24 | -66.81 |
| BULL_CALM -> BULL_CALM | 84 | 64.29% | +8,504.16 | 6,363.41 | +2,140.75 |
| BULL_CALM -> CHOPPY | 48 | 70.83% | +9,153.49 | 6,216.57 | +2,936.92 |
| BULL_VOLATILE -> BULL_VOLATILE | 1 | 0.00% | -336.18 | 0.00 | -336.18 |
| BULL_VOLATILE -> BULL_CALM | 5 | 80.00% | +268.90 | 116.46 | +152.44 |
| BULL_VOLATILE -> CHOPPY | 6 | 100.00% | +604.15 | 285.49 | +318.67 |

The weakest pattern is `BULL_CALM -> BEAR`: only 9 trades, but net loss is `-2,711.03`. This is direct evidence that the model lacks transition-risk awareness. It buys during apparently favorable regimes but does not know when those regimes are fragile.

## Score Quality

Spearman correlations between entry scores and realized P/L are weak:

| Entry field | rho vs pnl_pct | rho vs gross PnL | rho vs net PnL |
|---|---:|---:|---:|
| entry_rank_score | +0.113 | +0.116 | +0.112 |
| entry_mu | +0.113 | +0.116 | +0.112 |
| entry_sigma | +0.096 | +0.113 | +0.105 |

Rank-score quintiles:

| Rank bucket | n | Win rate | Gross PnL | Tax | Net PnL |
|---|---:|---:|---:|---:|---:|
| Q1 low | 33 | 60.61% | +2,520.82 | 2,211.79 | +309.03 |
| Q2 | 33 | 57.58% | -431.77 | 1,383.03 | -1,814.80 |
| Q3 | 32 | 75.00% | +7,005.85 | 3,934.52 | +3,071.33 |
| Q4 | 33 | 63.64% | +2,811.95 | 2,583.93 | +228.02 |
| Q5 high | 33 | 66.67% | +4,444.23 | 3,803.05 | +641.18 |

This is the central model-capability problem. The top score bucket is not reliably the best bucket, and the second bucket is deeply negative. The model has some signal, but it is not monotonic or strong enough to drive capital allocation.

## Ticker Concentration

Worst net contributors:

| Ticker | n | Win rate | Net PnL |
|---|---:|---:|---:|
| MSFT | 13 | 61.54% | -1,504.72 |
| MO | 9 | 22.22% | -1,411.70 |
| KO | 5 | 20.00% | -725.38 |
| PANW | 1 | 0.00% | -693.50 |
| T | 2 | 0.00% | -633.04 |
| WELL | 3 | 0.00% | -502.12 |

Best net contributors:

| Ticker | n | Win rate | Net PnL |
|---|---:|---:|---:|
| FTNT | 5 | 100.00% | +1,847.58 |
| KMI | 20 | 85.00% | +1,460.52 |
| MS | 10 | 80.00% | +956.02 |
| JNJ | 7 | 71.43% | +776.37 |
| DUK | 6 | 83.33% | +748.02 |

This is another model weakness: it is mixing real winners with persistent losers and does not learn stable ticker/regime conditional failure modes.

## Root-Cause Hypotheses

1. Entry alpha is weak. Entry scores have only about +0.11 Spearman correlation with realized outcomes.
2. The score is not monotonic enough for sizing. Q5 is not best; Q2 is sharply bad.
3. Regime transition risk is under-modeled. `BULL_CALM -> BEAR` trades lose heavily.
4. Exit timing is too eager for many winners. `max_hold` performs best, while short holding buckets are weak after tax.
5. Stop-loss is too late or too large. Only 12 stop-loss trades produce `-5,482.75` gross PnL.
6. The model does not identify persistent ticker-specific failure modes, e.g. MSFT/MO/KO/T/PANW in this WF sample.

## Practical Improvement Plan

Do not start with a tax optimizer. First prove pre-tax alpha:

1. Add a pre-tax WF acceptance gate alongside after-tax WF.
2. Add entry-score monotonicity as a hard promotion gate: top score bucket must outperform bottom buckets, and Q5 must be positive net of realistic costs.
3. Train a transition-risk overlay: probability of `BULL_CALM -> BEAR/volatile/choppy` within 20-30 trading days.
4. Replace one-size stop-loss with volatility/ATR-aware stop and test whether catastrophic stop-loss loss shrinks without destroying winners.
5. Add ticker/regime failure memory: penalize ticker/regime pairs whose rolling WF contribution is negative.
6. Reduce short-horizon exits unless expected edge clears tax and cost hurdle; current 11-20d bucket is gross flat and tax-negative.

Promotion should remain blocked until a model passes recipe-matched WF against SPY on both Sharpe and APY.
