# RenQuant 104 Raw-Signal Top-K Baseline

Produced: 2026-05-22T15:30:38.057128+00:00

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
- rebalance_days: `60`
- short-term tax stress rate: `0.5`
- events evaluated: `9`

## Event Geometry

Non-overlapping event returns: APY and Sharpe are interpretable as a coarse fixed-hold strategy proxy.

## Regime-First Results

| regime | n | APY | Sharpe | after-tax APY | alpha vs SPY | long-short | win rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| HIGH_CALM | 5 | +28.79% | +1.745 | +13.73% | +2.84% | +1.85% | +80.00% |
| HIGH_SPIKED | 1 | -37.56% | nan | -37.56% | -11.45% | -8.87% | +0.00% |
| LOW_SPIKED | 1 | +124.70% | nan | +52.85% | +14.10% | +21.19% | +100.00% |
| MED_CALM | 1 | +45.32% | nan | +21.05% | +1.98% | +6.16% | +100.00% |
| MED_SPIKED | 1 | +79.49% | nan | +35.34% | +9.12% | +21.42% | +100.00% |
| POOLED | 9 | +32.94% | +1.517 | +12.88% | +3.11% | +5.46% | +77.78% |

## Controls

- A/A max absolute return diff: `0`
- shuffle: pooled APY +16.10%, Sharpe +1.144, alpha vs SPY -0.52%, long-short +1.69%
- reverse: pooled APY +7.76%, Sharpe +0.610, alpha vs SPY -2.35%, long-short -5.46%
- time_shift: pooled APY +49.73%, Sharpe +3.106, alpha vs SPY +6.73%, long-short +8.68%

## Cross-Sectional IC

- pooled mean IC: +0.059
- pooled positive IC rate: +66.67%

| regime | n days | mean IC | positive IC rate |
|---|---:|---:|---:|
| HIGH_CALM | 5 | +0.017 | +60.00% |
| HIGH_SPIKED | 1 | -0.098 | +0.00% |
| LOW_SPIKED | 1 | +0.227 | +100.00% |
| MED_CALM | 1 | +0.136 | +100.00% |
| MED_SPIKED | 1 | +0.187 | +100.00% |

## Interpretation Contract

If actual top-K does not beat shuffle/reverse/time-shift in the regimes where it trades, the model edge is not trustworthy. If it does beat controls but production WF still loses, the loss belongs to downstream sizing, churn, exits, or tax-aware execution.
