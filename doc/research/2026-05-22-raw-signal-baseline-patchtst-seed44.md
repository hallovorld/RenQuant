# RenQuant 104 Raw-Signal Top-K Baseline

Produced: 2026-05-22T15:31:06.562600+00:00

## Purpose

This evaluates the model before QP sizing, top-ups, stop-losses, soft exits, rotation, and broker/tax lot handling. It is a signal audit, not a production backtest.

## Literature / Mature Scheme Anchors

- Qlib TopkDropoutStrategy: cross-sectional scores choose TopK holdings and replace low-ranked names with high-ranked names.
- PatchTST, Nie et al. 2023: sequence model remains shadow-only here; the same top-K evaluator can test it without changing execution.
- Bailey and Lopez de Prado 2014 DSR/PBO: treat good backtests as suspect until controls reduce false-discovery risk.
- cvxportfolio/Boyd transaction-cost framing: optimizer economics must include costs; this script intentionally removes optimizer effects to isolate alpha first.

## Configuration

- scorer_kind: `hf_patchtst`
- artifact: `artifacts/patchtst_shadow/canonical_5seed_mps/seed_44/hf_patchtst_all_seed44_model.pt`
- panel: `data/alpha158_291_fundamental_dataset.parquet`
- window: `2025-02-06` to `2026-02-10`
- top_k: `10`
- bottom_k: `10`
- hold_days: `60`
- rebalance_days: `60`
- short-term tax stress rate: `0.5`
- events evaluated: `5`

## Event Geometry

Non-overlapping event returns: APY and Sharpe are interpretable as a coarse fixed-hold strategy proxy.

## Regime-First Results

| regime | n | APY | Sharpe | after-tax APY | alpha vs SPY | long-short | win rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| HIGH_CALM | 1 | +123.41% | nan | +52.37% | +13.95% | +17.41% | +100.00% |
| HIGH_NORMAL | 1 | +32.55% | nan | +15.40% | +5.21% | +3.05% | +100.00% |
| LOW_SPIKED | 1 | -11.95% | nan | -11.95% | -15.15% | -11.36% | +0.00% |
| MED_NORMAL | 2 | -18.93% | -2.256 | -18.93% | -2.73% | -8.07% | +0.00% |
| POOLED | 5 | +11.37% | +0.553 | +0.35% | -0.29% | -1.41% | +40.00% |

## Controls

- A/A max absolute return diff: `0`
- shuffle: pooled APY +29.57%, Sharpe +1.689, alpha vs SPY +3.24%, long-short +1.46%
- reverse: pooled APY +20.00%, Sharpe +2.527, alpha vs SPY +1.12%, long-short +1.41%
- time_shift: pooled APY +70.48%, Sharpe +2.526, alpha vs SPY +7.99%, long-short +7.81%

## Cross-Sectional IC

- pooled mean IC: -0.016
- pooled positive IC rate: +40.00%

| regime | n days | mean IC | positive IC rate |
|---|---:|---:|---:|
| HIGH_CALM | 1 | +0.068 | +100.00% |
| HIGH_NORMAL | 1 | -0.048 | +0.00% |
| LOW_SPIKED | 1 | +0.145 | +100.00% |
| MED_NORMAL | 2 | -0.123 | +0.00% |

## Interpretation Contract

If actual top-K does not beat shuffle/reverse/time-shift in the regimes where it trades, the model edge is not trustworthy. If it does beat controls but production WF still loses, the loss belongs to downstream sizing, churn, exits, or tax-aware execution.
