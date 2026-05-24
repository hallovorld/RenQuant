# WF Sanity Placebo Diagnostic

- Artifact: `/Users/renhao/git/github/RenQuant/backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.codex_featspace_20260523-211211.staging.json`
- Manifest: `/Users/renhao/git/github/RenQuant/backtesting/renquant_104/artifacts/sim/walkforward_manifest_172_featspace_20260523.scopefixed.covered.json`
- Label: `fwd_60d_excess_raw`
- Validation: 2024-02-01 to 2026-02-10 (508 dates, 71840 rows)
- Promotion evidence: `False`

## Summary

| Metric | Value |
|---|---:|
| Real mean IC | +0.0385 |
| 60d model-placebo IC | +0.0460 |
| 60d label autocorr IC | -0.0008 |
| Warning | 60-day placebo is too large relative to real IC |

## Shift Profile

| Shift | Model-placebo IC | Label autocorr IC | Rows | Dates |
|---:|---:|---:|---:|---:|
| 5 | +0.0378 | +0.9035 | 71130 | 503 |
| 10 | +0.0372 | +0.8142 | 70420 | 498 |
| 20 | +0.0395 | +0.6474 | 69000 | 488 |
| 40 | +0.0415 | +0.3173 | 66160 | 468 |
| 60 | +0.0460 | -0.0008 | 63320 | 448 |
| 80 | +0.0463 | -0.0103 | 60480 | 428 |
| 120 | +0.0423 | +0.0368 | 54800 | 388 |
| 180 | +0.0185 | +0.0047 | 46280 | 328 |
| 252 | +0.0634 | -0.0235 | 36056 | 256 |

## By Regime

| Regime | Mean IC | Hit Rate | Dates | Rows | Mean Conf |
|---|---:|---:|---:|---:|---:|
| BEAR | +0.2565 | +0.9600 | 50 | 7100 | +0.8294 |
| BULL_CALM | +0.0152 | +0.5200 | 400 | 56635 | +0.6376 |
| BULL_VOLATILE | -0.0296 | +0.3158 | 19 | 2698 | +0.6439 |
| CHOPPY | +0.0315 | +0.6154 | 39 | 5407 | +0.3378 |

## 60d Placebo By Regime

| Regime | Model-placebo IC | Label autocorr IC | Rows | Dates |
|---|---:|---:|---:|---:|
| BEAR | +0.1828 | +0.2229 | 7100 | 50 |
| BULL_CALM | +0.0312 | -0.0327 | 49721 | 352 |
| BULL_VOLATILE | +0.0426 | +0.0231 | 2239 | 16 |
| CHOPPY | -0.0055 | -0.0119 | 4260 | 30 |
