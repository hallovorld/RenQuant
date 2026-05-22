# APY/Sharpe root-cause note (2026-05-22)

Question: why are APY and Sharpe still weak even when model IC is positive?

Short answer: the signal is not being converted into a high-return portfolio.
The current stack produces a low-beta, underinvested, high-turnover, short-term
tax-heavy portfolio, and the PatchTST shadow path is using an invalid
model-mismatched calibration artifact.

## 1. Event-tax headline is crushing the reported curve

The event-level after-tax equity curve is a cash-stress metric, not a clean
alpha metric. Adding back event-level tax gives:

| Model | Event-tax APY | Event-tax Sharpe | Event-tax-free APY | Event-tax-free Sharpe |
|---|---:|---:|---:|---:|
| XGB strict-cutoff | +1.17% | +0.20 | +7.90% | +1.20 |
| PatchTST clean diagnostic | +1.49% | +0.23 | +11.51% | +1.83 |

Annual-net tax reporting is less punitive but still leaves weak APY:
XGB +4.17%, PatchTST +6.07%.

Conclusion: tax explains most of the bad headline Sharpe, but not the full
underperformance versus SPY.

## 2. The portfolio is underinvested and very low beta

Same window SPY: +15.59% APY, +0.91 Sharpe, +26.07% total return.

| Model | Mean exposure | Median exposure | Days below 50% exposure | Beta vs SPY |
|---|---:|---:|---:|---:|
| XGB strict-cutoff | 51.5% | 46.5% | 215 / 404 | 0.101 |
| PatchTST clean diagnostic | 59.3% | 67.7% | 101 / 404 | 0.109 |

In a bull-market window, a long-only strategy with beta around 0.10 cannot
compete with SPY APY unless it has very strong idiosyncratic alpha. It does not.

## 3. Score ordering is not strong in the actual trade domain

The full-panel IC can be positive while the final decision tree still destroys
trade-level monotonicity.

| Model | Corr(entry rank_score, realized excess vs SPY) | Worst symptom |
|---|---:|---|
| XGB strict-cutoff | -0.039 | top rank-score quintile has negative mean excess (-0.90%) |
| PatchTST clean diagnostic | +0.069 | weak/non-monotonic; best bucket is middle, not top |

This means the pipeline after raw scoring - gates, adaptive floor, calibration,
QP, top-up, exits - is not preserving a reliable score-to-return ordering.

## 4. QP/top-up creates too much short-term turnover

| Model | Buy turnover / year | QP buys | Top-up buys | QP sells | Median QP-sell hold |
|---|---:|---:|---:|---:|---:|
| XGB strict-cutoff | 3.47x | 61 | 39 | 23 | 9d |
| PatchTST clean diagnostic | 5.52x | 89 | 58 | 57 | 12d |

Top-up is a large share of buy notional and creates extra lots. QP exits are
often short-term rebalancing sells, which makes the tax drag worse. PatchTST's
event-level QP-sell bucket is especially suspicious: gross +$1.57k, tax $1.92k.

## 5. Stop-loss losses are not compensated by entry edge

| Model | Stop-loss closed P&L |
|---|---:|
| XGB strict-cutoff | -$5.67k |
| PatchTST clean diagnostic | -$8.14k |

Trailing stop and max-hold exits are profitable; stop-loss is pure drag. If the
entry signal were strong enough, stop losses would be acceptable insurance. Here
they consume too much of the edge.

## 6. Calibration artifact mismatch is a real structural bug

The PatchTST shadow calibration file is byte-identical to the prod XGB
calibration file:

- `artifacts/prod/panel-rank-calibration.json`
- `artifacts/shadow/panel-rank-calibration.shadow.json`
- SHA256: `1baaf489cae3175fa03a13336d35b4875da57bc4ea2186d8cd1bcd2c02ae9990`

Both metadata blocks point to:

`backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json`

Therefore PatchTST raw scores are being interpreted with an XGB-trained
calibrator. The strict-cutoff XGB diagnostic sim also used the prod calibration,
not a strict-cutoff model-specific calibration. This means the exact QP μ,
rank_score thresholds, and Kelly sizes are not acceptance-grade.

## Current conclusion

Do not archive the project based on this run. The event-tax-free Sharpe says
there is a usable low-volatility signal. But do not promote PatchTST or trust
the current APY/Sharpe as final acceptance either.

The next correct fix is:

1. train/stamp model-specific calibration artifacts for XGB strict-cutoff and
   PatchTST strict artifacts;
2. make the sim/live loader reject calibration artifacts whose scorer fingerprint
   does not match the active scorer;
3. rerun the same comparison with annual-net tax as the headline, event-tax as
   stress, and no-tax as the alpha diagnostic;
4. add trade-domain IC checks after the full decision tree, not only raw model IC;
5. reduce QP/top-up turnover and enforce a bull-calm target exposure/beta policy.

