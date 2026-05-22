# RenQuant 104 Raw-Signal Top-K Baseline

Produced: 2026-05-22T15:30:38.483423+00:00

## Purpose

This evaluates the model before QP sizing, top-ups, stop-losses, soft exits, rotation, and broker/tax lot handling. It is a signal audit, not a production backtest.

## Literature / Mature Scheme Anchors

- Qlib TopkDropoutStrategy: cross-sectional scores choose TopK holdings and replace low-ranked names with high-ranked names.
- PatchTST, Nie et al. 2023: sequence model remains shadow-only here; the same top-K evaluator can test it without changing execution.
- Bailey and Lopez de Prado 2014 DSR/PBO: treat good backtests as suspect until controls reduce false-discovery risk.
- cvxportfolio/Boyd transaction-cost framing: optimizer economics must include costs; this script intentionally removes optimizer effects to isolate alpha first.

## Configuration

- scorer_kind: `xgb`
- artifact: `backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json`
- panel: `data/alpha158_291_fundamental_dataset.parquet`
- window: `2024-01-02` to `2026-03-28`
- top_k: `10`
- bottom_k: `10`
- hold_days: `60`
- rebalance_days: `20`
- short-term tax stress rate: `0.5`
- events evaluated: `27`

## Event Geometry

Overlapping event returns: APY and Sharpe are diagnostic only and are likely overstated versus a self-financing portfolio.

## Regime-First Results

| regime | n | APY | Sharpe | after-tax APY | alpha vs SPY | long-short | win rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| HIGH_CALM | 9 | +216.92% | +4.421 | +81.15% | +5.71% | +7.56% | +88.89% |
| HIGH_NORMAL | 3 | +84.82% | +6.816 | +36.55% | -0.10% | +1.26% | +100.00% |
| HIGH_SPIKED | 6 | +144.82% | +2.754 | +41.51% | +3.57% | +6.64% | +83.33% |
| LOW_SPIKED | 3 | +926.88% | +5.647 | +242.35% | +9.84% | +19.64% | +100.00% |
| MED_CALM | 2 | +28.31% | +0.797 | -2.45% | +2.43% | -1.49% | +50.00% |
| MED_NORMAL | 2 | -50.99% | -1.587 | -55.31% | -4.48% | -1.18% | +50.00% |
| MED_SPIKED | 2 | +95.89% | +1.624 | +28.27% | +7.92% | +17.65% | +50.00% |
| POOLED | 27 | +152.43% | +2.808 | +49.71% | +4.21% | +7.43% | +81.48% |

## Controls

- A/A max absolute return diff: `0`
- shuffle: pooled APY +49.40%, Sharpe +1.359, alpha vs SPY -0.22%, long-short -1.91%
- reverse: pooled APY +6.09%, Sharpe +0.374, alpha vs SPY -3.21%, long-short -7.43%
- time_shift: pooled APY +171.45%, Sharpe +2.597, alpha vs SPY +5.26%, long-short +6.61%

## Cross-Sectional IC

- pooled mean IC: +0.078
- pooled positive IC rate: +70.37%

| regime | n days | mean IC | positive IC rate |
|---|---:|---:|---:|
| HIGH_CALM | 9 | +0.081 | +77.78% |
| HIGH_NORMAL | 3 | +0.045 | +66.67% |
| HIGH_SPIKED | 6 | +0.043 | +66.67% |
| LOW_SPIKED | 3 | +0.252 | +100.00% |
| MED_CALM | 2 | +0.006 | +50.00% |
| MED_NORMAL | 2 | +0.028 | +50.00% |
| MED_SPIKED | 2 | +0.082 | +50.00% |

## Interpretation Contract

If actual top-K does not beat shuffle/reverse/time-shift in the regimes where it trades, the model edge is not trustworthy. If it does beat controls but production WF still loses, the loss belongs to downstream sizing, churn, exits, or tax-aware execution.
