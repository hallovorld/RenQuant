# WF Sanity Placebo Diagnostic

- Artifact: `/Users/renhao/git/github/RenQuant/backtesting/renquant_104/artifacts/walkforward_gbdt_prod_recipe_v2/2026-03-02/panel-ltr.json`
- Manifest: `/Users/renhao/git/github/RenQuant/backtesting/renquant_104/artifacts/sim/walkforward_manifest_gbdt_prod_recipe_v2.json`
- Label: `fwd_60d_excess`
- Validation: 2024-02-02 to 2026-02-11 (508 dates, 147066 rows)
- Promotion evidence: `True`

## Summary

| Metric | Value |
|---|---:|
| Real mean IC | +0.0543 |
| 60d aligned real IC | +0.0760 |
| 60d model-placebo IC | +0.0343 |
| 60d label autocorr IC | -0.0198 |
| Warning | none |

## Shift Profile

| Shift | Aligned real IC | Model-placebo IC | Label autocorr IC | Rows | Dates |
|---:|---:|---:|---:|---:|---:|
| 5 | +0.0535 | +0.0550 | +0.8802 | 145606 | 503 |
| 10 | +0.0549 | +0.0538 | +0.7889 | 144146 | 498 |
| 20 | +0.0587 | +0.0549 | +0.6167 | 141226 | 488 |
| 40 | +0.0681 | +0.0430 | +0.2939 | 135386 | 468 |
| 60 | +0.0760 | +0.0343 | -0.0198 | 129546 | 448 |
| 80 | +0.0803 | +0.0334 | -0.0230 | 123706 | 428 |
| 120 | +0.0866 | +0.0529 | +0.0398 | 112026 | 388 |
| 180 | +0.0691 | +0.0244 | +0.0001 | 94506 | 328 |
| 252 | +0.0040 | +0.0463 | -0.0181 | 73482 | 256 |

## By Regime

| Regime | Mean IC | Hit Rate | Dates | Rows | Mean Conf |
|---|---:|---:|---:|---:|---:|
| BEAR | +0.3349 | +0.9800 | 50 | 14600 | +0.8294 |
| BULL_CALM | +0.0234 | +0.5338 | 399 | 115968 | +0.6377 |
| BULL_VOLATILE | +0.0254 | +0.4737 | 19 | 5548 | +0.6439 |
| CHOPPY | +0.0256 | +0.7000 | 40 | 10950 | +0.3303 |

## 60d Placebo By Regime

| Regime | Model-placebo IC | Label autocorr IC | Rows | Dates |
|---|---:|---:|---:|---:|
| BEAR | +0.0993 | +0.1728 | 14600 | 50 |
| BULL_CALM | +0.0266 | -0.0515 | 101513 | 351 |
| BULL_VOLATILE | +0.0170 | +0.0365 | 4564 | 16 |
| CHOPPY | +0.0268 | -0.0010 | 8869 | 31 |
