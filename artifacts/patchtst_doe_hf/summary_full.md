# HF PatchTST DOE — §5.14 FULL Verdict (post-hoc)
**Source**: scripts/patchtst_doe_hf.py + scripts/postprocess_doe_hf.py
**N design points**: 9
**Objective**: bull_regime_IC (HMM {BULL_CALM, BULL_VOLATILE})

## PBO (Bailey-Borwein-LdP-Zhu 2015): **nan**
PBO > 0.5 → overfit; PBO < 0.5 → robust.

## Per-Point: bull_ic + DSR
| Point | lr | wd | warmup | seq | bull_ic_mean | bull_ic_std | DSR | n_cuts |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.0e-04 | 1.0e-02 | 4 | 24 | +0.1034 | 0.0044 | +21.881 | 2 |
| 0 | 1.0e-05 | 1.0e-02 | 4 | 8 | +0.0136 | 0.0250 | -0.983 | 3 |

## Main Effects (sorted by |β|)
| Knob | β |
|---|---|

## 2-Way Interactions (sorted by |β|)
| A | B | β |
|---|---|---|

## §5.14 Pass-Gate Check
- ❌ PBO < 0.5 (PBO=nan)
- ✅ Best DSR > 0 (DSR=+21.881)
- ✅ Best bull_ic > 0 (bull_ic=+0.1034)
