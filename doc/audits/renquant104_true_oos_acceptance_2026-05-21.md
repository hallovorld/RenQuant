# renquant_104 True-OOS Acceptance Run — 2026-05-21

## Command

```bash
.venv/bin/python scripts/research_acceptance_104.py --target true-oos --workers 1
```

Pipeline path:

1. `scripts/retrain_prod_truly_oos.py --train-cutoff 2024-07-01`
2. `scripts/eval_truly_oos.py --artifact-dir backtesting/renquant_104/artifacts/walkforward_truly_oos_2024-07-01`
3. `scripts/dsr_pbo_truly_oos.py --eval-json backtesting/renquant_104/artifacts/prod/truly_oos_eval/eval_truly_oos.json`

## Training Contract

- Train rows: 598,347
- Tickers: 292
- Train dates: 2016-01-04 through 2024-06-28
- Label: `fwd_60d_excess`
- Cutoff: 2024-07-01
- Artifact path: `backtesting/renquant_104/artifacts/walkforward_truly_oos_2024-07-01/panel-ltr.json`
- Feature columns: 172

Walk-forward-style CV inside the cutoff training run:

| Fold | IC |
| --- | ---: |
| 1 | +0.0464 |
| 2 | +0.0101 |
| 3 | +0.0660 |

Mean CV IC: **+0.0408**.

## Strict Post-Cutoff Evaluation

- Eval dates: 2024-07-02 through 2026-02-10
- Trading days: 404
- Rows scored: 117,968
- Mean IC: **+0.0529**
- Median IC: **+0.0450**
- IC std: **0.1105**
- Positive IC days: **276 / 404 = 68%**
- Top-10 alpha: **+0.0792**
- Bottom-10 alpha: **-0.0910**
- Long-short: **+0.1702**

## Regime Breakdown

| Regime | Days | IC | Top-10 Alpha | Long-Short |
| --- | ---: | ---: | ---: | ---: |
| BEAR | 17 | +0.3453 | +0.6955 | +1.3427 |
| BULL_CALM | 234 | +0.0053 | -0.0448 | -0.0704 |
| BULL_STRONG | 38 | +0.0597 | +0.2455 | +0.3486 |
| BULL_VOLATILE | 65 | +0.1052 | +0.1287 | +0.4356 |
| CHOPPY | 50 | +0.1028 | +0.2594 | +0.4168 |

## DSR / PBO

- Annualized Sharpe of daily IC series: **+7.5975**
- DSR: **1.0000** with `n_trials=5`
- PBO: **0.5907**

## Verdict

The true-OOS signal is positive in aggregate and passes the DSR gate, but
the regime detail still says not to trust pooled performance blindly.
`BULL_CALM` remains the structural weak point: it dominates the sample by
day count while its top-10 alpha is negative. PBO above 0.5 also says the
post-hoc regime selection risk is not clean.

Operational implication: do not disable BULL_CALM buys with a hard kill
switch, but do not treat this run as a clean global promotion either. The
next design step should be graduated BULL_CALM de-risking: lower sizing,
stricter panel floor, or shadow-confirmation gating, measured against SPY
and XGB baseline per regime.
