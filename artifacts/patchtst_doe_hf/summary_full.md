# HF PatchTST DOE — §5.14 FULL Verdict (post-hoc)
**Source**: scripts/patchtst_doe_hf.py + scripts/postprocess_doe_hf.py
**N design points**: 9
**Objective**: bull_regime_IC (HMM {BULL_CALM, BULL_VOLATILE})

## PBO (Bailey-Borwein-LdP-Zhu 2015): **0.33**
PBO > 0.5 → overfit; PBO < 0.5 → robust.

## Per-Point: bull_ic + DSR
| Point | lr | wd | warmup | seq | bull_ic_mean | bull_ic_std | DSR | n_cuts |
|---|---|---|---|---|---|---|---|---|
| 7 | 1.0e-04 | 3.0e-01 | 10 | 24 | +0.1021 | nan | — | 1 |
| 1 | 1.0e-04 | 1.0e-02 | 4 | 24 | +0.0580 | 0.0787 | -0.777 | 3 |
| 2 | 1.0e-05 | 3.0e-01 | 4 | 24 | +0.0460 | 0.0807 | -0.960 | 3 |
| 4 | 1.0e-05 | 1.0e-02 | 10 | 24 | +0.0458 | 0.0810 | -0.955 | 3 |
| 0 | 1.0e-05 | 1.0e-02 | 4 | 8 | +0.0136 | 0.0250 | -0.983 | 3 |
| 6 | 1.0e-05 | 3.0e-01 | 10 | 8 | +0.0120 | 0.0236 | -1.017 | 3 |
| 5 | 1.0e-04 | 1.0e-02 | 10 | 8 | +0.0101 | 0.0255 | -1.141 | 3 |
| 3 | 1.0e-04 | 3.0e-01 | 4 | 8 | +0.0075 | 0.0221 | -1.206 | 3 |

## Main Effects (sorted by |β|)
| Knob | β |
|---|---|
| `seq_len` | +0.0261 |
| `lr` | +0.0075 |
| `warmup_epochs` | +0.0056 |
| `weight_decay` | +0.0050 |

## 2-Way Interactions (sorted by |β|)
| A | B | β |
|---|---|---|
| `weight_decay` | `warmup_epochs` | +0.0048 |
| `lr` | `seq_len` | +0.0048 |
| `lr` | `warmup_epochs` | +0.0030 |
| `weight_decay` | `seq_len` | +0.0030 |
| `warmup_epochs` | `seq_len` | +0.0027 |
| `lr` | `weight_decay` | +0.0027 |

## §5.14 Pass-Gate Check
- ✅ PBO < 0.5 (PBO=0.33)
- ❌ Best DSR > 0 (DSR=+nan)
- ✅ Best bull_ic > 0 (bull_ic=+0.1021)
