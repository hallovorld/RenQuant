# 2026-05-13 — QP transaction-cost penalty sweep

## Method

Single-knob sweep of `rotation.joint_actions.qp_cost_kappa` (the L1 penalty
on Δw in the joint-actions QP solver). Range-finding informed by analysis
in `2026-05-13-baseline-structural-diagnosis.md` — baseline kappa=0.0001 is
20-30× below realized per-trade friction; theory predicts a larger penalty
recovers lost alpha by suppressing low-edge trades.

Same 16 non-overlapping 3-month windows + paired-daily HAC + bootstrap +
DSR/PBO framework as `evaluation-protocol.md`.

## Q3-2025 range-find (single window, Q14)

SPY Q14 raw return = +8.22% (annualised +35.9%).

| κ | turn/bar | buys / sells | APY | Sharpe | Tax$ | avg P&L/trade | avg hold |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0001 (baseline) | 2.73 | 148 / 27 | +0.80% | -0.20 | 5,824 | 1.9% | 22d |
| 0.001 | 2.70 | 146 / 27 | +2.57% | -0.08 | 5,702 | 1.9% | 22d |
| 0.01 | 2.73 | 148 / 27 | +0.80% | -0.20 | 5,824 | 1.9% | 22d |
| **0.1** | **1.55** | **89 / 10** | **+41.50%** | **+1.98** | 5,374 | **4.3%** | **27d** |

κ ∈ [0.0001, 0.01] is below the QP's effective-binding threshold for this
μ-scale; κ = 0.1 is decisively binding. Avg per-trade edge jumps 1.9 → 4.3%
because the QP becomes selective (only crosses friction when conviction is
high).

## Full 16-window panel (κ = 0.1)

| Q | n | meanΔ_ann | SE_NW | t | p | SR_Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 01 | 62 | +1.19% | 0.004% | +1.07 | 0.285 | +1.17 |
| 02 | 63 | +2.64% | 0.024% | +0.44 | 0.657 | +0.64 |
| 03 | 62 | −5.28% | 0.033% | −0.64 | 0.521 | −1.38 |
| 04 | 61 | **−23.88%** | 0.057% | −1.66 | 0.097 | −2.67 |
| 05 | 61 | −0.11% | 0.053% | −0.01 | 0.994 | −0.02 |
| 06 | 62 | +7.56% | 0.037% | +0.80 | 0.422 | +1.68 |
| 07 | 62 | −6.83% | 0.072% | −0.38 | 0.708 | −0.85 |
| 08 | 61 | **+20.45%** | 0.101% | +0.80 | 0.422 | +1.82 |
| 09 | 63 | +8.74% | 0.050% | +0.70 | 0.484 | +1.39 |
| 10 | 64 | −15.85% | 0.060% | −1.06 | 0.291 | −2.28 |
| 11 | 63 | −9.31% | 0.053% | −0.69 | 0.489 | −1.25 |
| 12 | 60 | +3.39% | 0.064% | +0.21 | 0.833 | +0.43 |
| 13 | 62 | +4.45% | 0.028% | +0.62 | 0.535 | +0.99 |
| 14 | 64 | **+33.91%** | 0.060% | **+2.26** | **0.024** | **+3.78** |
| 15 | 63 | +0.31% | 0.046% | +0.03 | 0.979 | +0.05 |
| 16 | 57 | −13.76% | 0.057% | −0.95 | 0.341 | −2.00 |

### Pooled
- n_days = 990, n_windows = 16
- mean Δ annualised : **+0.58%**
- Newey-West SE     : 0.014% (daily, lag = 6)
- t-statistic       : **+0.17**
- p-value           : 0.869
- 95% bootstrap CI  : [−6.32%, +7.36%]
- Sharpe of Δ       : +0.08 (95% CI [−0.91, +1.08])
- Deflated Sharpe   : +0.514 (K_trials=100, n=100 multi-test correction)
- Window consistency: 9/16 positive (56%)

### Verdict: **NEITHER**

DSR = 0.514 just barely clears 0.5, but t-stat = +0.17 contradicts it. Mean
Δ +0.58% is dominated by Q14 alone (+33.9%) offset by Q04 (−23.9%). The
candidate creates **variance** without **edge** under proper paired
inference.

## Regime breakdown

| regime | n | wins / total | mean Δ |
|---|---:|---:|---:|
| BEAR (SPY APY < 0): Q01,Q02,Q06,Q12,Q16 | 5 | 4 / 5 (80%) | −0.34% |
| BULL (SPY APY > 0): Q03,Q04,Q05,Q07-Q11,Q13-Q15 | 11 | 5 / 11 (45%) | +0.99% |

κ = 0.1 helps in BEAR (don't churn losing positions) but blocks
bull-market deployment in 6/11 bull windows. Counter to the initial
hypothesis from Q14 alone.

## Implication

κ = 0.1 is **too aggressive**. Suppresses bear-regime churn (good) but
also suppresses bull-regime deployment (bad). The QP's μ-scale is in raw
panel-rank units (~±2), so κ = 0.1 means a single buy at Δw = 0.075 pays
~0.0075 in QP-objective units — comparable to the expected return
contribution (μ ≈ 1.5 × 0.075 = 0.1125 per name). Penalty ≈ 7% of benefit,
which is REAL friction-equivalent but is too high relative to single-name
signal precision.

The right κ is somewhere between baseline 0.0001 and 0.1. Currently
running:
- κ = 0.003 (intermediate, 30× baseline)
- κ = 0.05  (intermediate, 500× baseline)
- min_dw_pct = 0.05 (alternative friction mechanism — raise minimum
  trade size from 2% to 5%)

Predicted: a milder κ may preserve bull deployment while still suppressing
the bottom of the trade-distribution. min_dw_pct may be a cleaner
mechanism (size-based threshold) than κ (signal-magnitude-based).

## κ=0.003 panel — verdict (post-full-16-window analysis)

| metric | value |
|---|---:|
| mean Δ annualised | +0.48% |
| Newey-West SE | 0.009% (daily, lag=6) |
| t-statistic | +0.22 |
| p-value | 0.828 |
| 95% bootstrap CI | [−3.79%, +4.89%] |
| Sharpe of Δ | +0.11 (CI [−0.81, +1.15]) |
| Deflated Sharpe | +0.690 (K_trials=100) |
| Window consistency | 8/16 (50%) |

### Verdict: **NEITHER**

5/16 windows (Q03, Q04, Q07, Q12, Q14) returned **exactly +0.00pt**
— κ=0.003 below binding threshold there (identical to baseline trades).
Wins: Q08 +10.26%, Q09 +12.00%. Losses: Q05 −8.80%, Q11 −11.31%.

Net: variance not edge — same failure mode as κ=0.1, just smaller
magnitude. The κ knob alone is not the answer.

## EMA50-off panel — verdict (16-window)

Tests the second mechanism from the baseline diagnosis doc:
disabling the SPY < EMA50 buy-block gate (now config-flag-able).

| metric | value |
|---|---:|
| mean Δ annualised | **−3.21%** |
| Newey-West SE | 0.020% (daily, lag=6) |
| t-statistic | −0.63 |
| p-value | 0.531 |
| 95% bootstrap CI | [−13.14%, +6.28%] |
| Sharpe of Δ | −0.33 (CI [−1.35, +0.67]) |
| Deflated Sharpe | +0.000 (K_trials=100) |
| Window consistency | 6/16 (38%) |

### Verdict: **TIER 1 REJECT**

Pattern by regime:
- **BEAR wins LOST**: Q01 −21pt, Q03 −19pt, Q05 −19pt, Q11 −19pt, Q16 −16pt
- **Bull wins** (modest): Q07 +15pt, Q13 +25pt, Q06 +9pt

The EMA50 gate's bear-regime protection (~25pt in Q01) outweighs
its bull-regime drag (~12pt in Q04). Net effect of disabling: −3.21pt/yr.

### Implication

EMA50 gate is **NOT the chronic-lag culprit.** The bull-market alpha
gap has a different root cause. Disabling the gate loses more than it
recovers — keep enabled. Theory falsified by data per CLAUDE.md §5.2
sanity protocol.

**Action taken**: EMA50 gate stays enabled in production. The new
`gates.ema50_gate.enabled` flag remains for future research (per the
bug-bounty fix that introduced it).
