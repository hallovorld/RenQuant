# HF PatchTST DOE — §5.14 FULL Verdict (post-hoc)
**Source**: scripts/patchtst_doe_hf.py + scripts/postprocess_doe_hf.py
**N design points**: 9
**Objective**: bull_regime_IC (HMM {BULL_CALM, BULL_VOLATILE})

## PBO (Bailey-Borwein-LdP-Zhu 2015): **1.00**
PBO > 0.5 → overfit; PBO < 0.5 → robust.

## Per-Point: bull_ic + DSR
| Point | lr | wd | warmup | seq | bull_ic_mean | bull_ic_std | DSR | n_cuts |
|---|---|---|---|---|---|---|---|---|
| 4 | 1.0e-05 | 1.0e-02 | 10 | 24 | +0.0834 | 0.0624 | -0.185 | 2 |
| 1 | 1.0e-04 | 1.0e-02 | 4 | 24 | +0.0580 | 0.0787 | -0.777 | 3 |
| 2 | 1.0e-05 | 3.0e-01 | 4 | 24 | +0.0460 | 0.0807 | -0.960 | 3 |
| 0 | 1.0e-05 | 1.0e-02 | 4 | 8 | +0.0136 | 0.0250 | -0.983 | 3 |
| 3 | 1.0e-04 | 3.0e-01 | 4 | 8 | +0.0075 | 0.0221 | -1.206 | 3 |

## Main Effects (sorted by |β|)
| Knob | β |
|---|---|
| `seq_len` | +0.0136 |
| `warmup_epochs` | +0.0100 |
| `weight_decay` | -0.0082 |
| `lr` | -0.0062 |

## 2-Way Interactions (sorted by |β|)
| A | B | β |
|---|---|---|
| `lr` | `warmup_epochs` | -0.0038 |
| `weight_decay` | `seq_len` | -0.0038 |
| `lr` | `weight_decay` | -0.0036 |
| `warmup_epochs` | `seq_len` | -0.0036 |
| `lr` | `seq_len` | -0.0018 |
| `weight_decay` | `warmup_epochs` | -0.0018 |

## §5.14 Pass-Gate Check
- ❌ PBO < 0.5 (PBO=1.00)
- ❌ Best DSR > 0 (DSR=-0.185)
- ✅ Best bull_ic > 0 (bull_ic=+0.0834)
