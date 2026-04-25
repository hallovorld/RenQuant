# Rust Transformer IC — what's good vs what's mediocre

## TL;DR

| Model              | val_IC on synthetic | val_IC on real |
|--------------------|--------------------:|---------------:|
| Linear regression (theoretical max on synthetic) | **+0.3228** | n/a |
| Rust transformer (50 epochs) | +0.0683 (~21% of max) | TBD |
| Python XGBoost (post-audit) | n/a (would also crush it) | **+0.0372** |
| Python transformer (overfit, shelved) | n/a | +0.0062 |

## What the synthetic test proved

The 0.0683 IC on synthetic data with `label = 0.3*f0 + 0.2*f1 + N(0,1)`
proves only that the **training pipeline works**: forward → loss →
backward → AdamW → weights save. It does NOT prove the architecture
is good — a linear regression hits the theoretical ceiling of 0.32 on
the same data trivially.

The 0.0683 number tells us:
- Gradients flow correctly (loss decreases monotonically 6.5 → 5.8)
- Multi-core CPU saturation works (369% peak vs Python 1-core GIL)
- Hyperparameters (lr=5e-4, batch=32, dropout=0.3, wd=1e-4) are
  reasonable but the architecture is over-specified for a 2-feature
  linear signal

## Why real data should be different

Real production panel:
- Signal-to-noise dramatically worse than the 0.13/1.0 synthetic
- Cross-sectional structure (sector, regime, size effects)
- Non-linear feature interactions (vol × momentum)
- Time-series autocorrelations the panel residualization tries to remove

A linear model on real data won't outperform XGBoost — that's why the
Python production setup uses XGB. The fair test is:
**Python XGB OOS IC = 0.0372** vs **Rust transformer OOS IC on real**.

## Comparison context

Cross-sectional Spearman IC scale (typical quant equity):
- < 0  : worse than random
- 0.00 - 0.02 : barely-trading edge, costs eat it
- 0.02 - 0.05 : publishable academic; tradable with low costs
- 0.05 - 0.10 : strong, top-decile fund territory
- > 0.10 : implausibly good (likely lookahead)

Our XGB at 0.0372 is in the lower-end-tradable band. That's the
reasonable target for the Rust transformer to match before promotion.

## Synthetic vs real setup

Same shape:
- Synthetic: 1000 dates × 99 tickers × 41 features (CSV)
- Real: 1256 dates × ~99 tickers × 41 features (the actual panel)

Synthetic deliberately uses a structure that linear regression can crush
(2-feature signal + Gaussian noise). Real is harder — XGBoost only gets
0.037 on it.

## Hyperparameters tested

The 0.068 run used:
```
--epochs 50 --batch 32 --lr 0.0005
--d-model 48 --n-heads 6 --n-layers 2 --ff-dim 96 --dropout 0.3
--val-frac 0.2 --patience 8
```

Reasonable for "small panel" per the regularization research papers
cited in our prior session (dropout 0.3-0.5, weight_decay 0.05-0.1
recommended by ApxML's transformer-regularization survey).

To push IC higher we'd want to A/B:
- weight_decay 1e-4 → 1e-3 → 1e-2
- d_model 48 → 32 → 16 (smaller is better on small panels)
- n_layers 2 → 1 (overparameterised at depth)
- LambdaRank loss vs ListNet
- Longer training + better schedule

## Reference papers (verified earlier session)

- **Poh-Lim-Zohren-Roberts 2020** ([arXiv 2012.07149](https://arxiv.org/abs/2012.07149)):
  cross-sectional learning-to-rank, listwise > pairwise > pointwise on
  S&P 500.
- **"On Evaluating Loss Functions for Stock Ranking" (CIKM 2025,
  [arXiv 2510.14156](https://arxiv.org/abs/2510.14156))**: comprehensive
  benchmark; informs loss-function choice.
- **ApxML transformer regularization survey**: dropout 0.3-0.5,
  weight_decay 0.05-0.1, layer freezing, aggressive early stopping
  for small data.

## Honest assessment as of this commit

- ✅ Rust training pipeline works end-to-end
- ✅ 31 Rust tests passing (loss, trainer, CV, dataset, metrics)
- ✅ Multi-core CPU utilization verified
- ⚠️  IC on real production data: NOT YET MEASURED
- ⚠️  Hyperparameters not yet tuned for the actual data SNR
- ❌ Has not beaten Python XGB on any real panel
