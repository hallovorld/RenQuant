# WF Sanity Placebo Diagnostic

- Artifact: `/Users/renhao/git/github/RenQuant/backtesting/renquant_104/artifacts/walkforward_patchtst_pilot_20260524/2025-01-23/hf_patchtst_all_seed44_model.pt`
- Manifest: `/Users/renhao/git/github/RenQuant/backtesting/renquant_104/artifacts/walkforward_patchtst_pilot_20260524.json`
- Label: `fwd_60d_excess_raw`
- Validation: 2025-01-02 to 2026-02-10 (277 dates, 39038 rows)
- Promotion evidence: `False`

## Summary

| Metric | Value |
|---|---:|
| Real mean IC | +0.0049 |
| 60d model-placebo IC | +0.0240 |
| 60d label autocorr IC | +0.1110 |
| Warning | 60-day placebo is too large relative to real IC |

## Shift Profile

| Shift | Model-placebo IC | Label autocorr IC | Rows | Dates |
|---:|---:|---:|---:|---:|
| 5 | +0.0126 | +0.9123 | 38328 | 272 |
| 10 | +0.0102 | +0.8335 | 37618 | 267 |
| 20 | +0.0201 | +0.6791 | 36198 | 257 |
| 40 | +0.0272 | +0.3684 | 33358 | 237 |
| 60 | +0.0240 | +0.1110 | 30518 | 217 |
| 80 | +0.0361 | +0.0936 | 27678 | 197 |
| 120 | +0.0246 | +0.0249 | 21998 | 157 |
| 180 | -0.0242 | -0.0148 | 13478 | 97 |
| 252 | -0.0216 | -0.1729 | 3254 | 25 |

## By Regime

| Regime | Mean IC | Hit Rate | Dates | Rows | Mean Conf |
|---|---:|---:|---:|---:|---:|
| BEAR | +0.0367 | +0.6279 | 43 | 6106 | +0.8597 |
| BULL_CALM | +0.0030 | +0.5286 | 210 | 29655 | +0.6029 |
| BULL_VOLATILE | -0.1164 | +0.2222 | 9 | 1278 | +0.6023 |
| CHOPPY | +0.0126 | +0.5333 | 15 | 1999 | +0.3934 |

## 60d Placebo By Regime

| Regime | Model-placebo IC | Label autocorr IC | Rows | Dates |
|---|---:|---:|---:|---:|
| BEAR | +0.0633 | +0.2261 | 6106 | 43 |
| BULL_CALM | +0.0175 | +0.0742 | 22741 | 162 |
| BULL_VOLATILE | -0.0304 | +0.2456 | 819 | 6 |
| CHOPPY | -0.0263 | +0.1452 | 852 | 6 |
