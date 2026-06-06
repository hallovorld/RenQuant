# WF Sanity Placebo Diagnostic

- Artifact: `/Users/renhao/git/github/RenQuant/artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt`
- Manifest: `/Users/renhao/git/github/RenQuant/artifacts/walkforward_manifest_patchtst_seed44_pt07.json`
- Label: `fwd_60d_excess`
- Validation: 2025-02-06 to 2026-02-10 (254 dates, 73081 rows)
- Promotion evidence: `True`

## Summary

| Metric | Value |
|---|---:|
| Real mean IC | +0.0246 |
| 60d aligned real IC | +0.0566 |
| 60d model-placebo IC | -0.0012 |
| 60d label autocorr IC | +0.0869 |
| Warning | none |

## Shift Profile

| Shift | Aligned real IC | Model-placebo IC | Label autocorr IC | Rows | Dates |
|---:|---:|---:|---:|---:|---:|
| 5 | +0.0208 | +0.0277 | +0.8843 | 71621 | 249 |
| 10 | +0.0228 | +0.0321 | +0.7996 | 70161 | 244 |
| 20 | +0.0265 | +0.0333 | +0.6396 | 67241 | 234 |
| 40 | +0.0396 | +0.0120 | +0.3511 | 61401 | 214 |
| 60 | +0.0566 | -0.0012 | +0.0869 | 55561 | 194 |
| 80 | +0.0701 | -0.0014 | +0.0853 | 49721 | 174 |
| 120 | +0.0781 | -0.0005 | +0.0536 | 38041 | 134 |
| 180 | +0.0867 | -0.0338 | -0.0046 | 20521 | 74 |
| 252 | -0.2700 | +0.1824 | -0.3173 | 218 | 2 |

## By Regime

| Regime | Mean IC | Hit Rate | Dates | Rows | Mean Conf |
|---|---:|---:|---:|---:|---:|
| BEAR | +0.1215 | +0.9535 | 43 | 12556 | +0.8597 |
| BULL_CALM | +0.0064 | +0.5882 | 187 | 54064 | +0.6046 |
| BULL_VOLATILE | -0.0863 | +0.1111 | 9 | 2628 | +0.6023 |
| CHOPPY | +0.0408 | +0.6000 | 15 | 3833 | +0.3857 |

## 60d Placebo By Regime

| Regime | Model-placebo IC | Label autocorr IC | Rows | Dates |
|---|---:|---:|---:|---:|
| BEAR | +0.0616 | +0.1662 | 12556 | 43 |
| BULL_CALM | -0.0152 | +0.0631 | 39609 | 139 |
| BULL_VOLATILE | -0.0515 | +0.1352 | 1644 | 6 |
| CHOPPY | -0.0763 | +0.0216 | 1752 | 6 |
