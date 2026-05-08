# IC Evaluation Methodology — Rigorous Standards

**Status**: Mandatory protocol for all IC claims. Adopted 2026-05-08 after a session of incorrect conclusions from non-rigorous tests.

This document defines how to measure cross-sectional rank IC in a way that's reproducible, theoretically grounded, and free from common pitfalls. **Read before claiming any IC number.**

---

## TL;DR — Decision Rules

| Question | Required test | Reject if |
|---|---|---|
| "Is this signal real OOS?" | Walk-forward, ≥5 cuts, report mean ± std | std > 1.5× mean |
| "Does model X beat baseline?" | Same WF cuts as baseline, paired comparison | Difference < 1 SE of cut-to-cut variation |
| "Is it regime shift or overfit?" | WF cut breakdown + Linear baseline | Linear fluctuates < complex model fluctuates → complex is overfit |
| "Can transformer beat linear here?" | Param count / training samples ratio < 1/100 → likely overfit | Param/sample > 1/300 in our regime → reject before training |

---

## 1. The Standard Test: Walk-Forward Validation

Reference: López de Prado, *Advances in Financial Machine Learning* (2018), Ch. 7 (Cross-Validation in Finance).

### The protocol

1. **Rolling window cuts** (NOT a single train/val/test split):
   - Train: years [Y−N, Y−1] (e.g., N=3)
   - Embargo: ≥21 trading days after train_end
   - Test: year [Y]
   - Repeat for Y in 5+ consecutive years

2. **Per-cut measurements**:
   - Re-normalize features per cut using train-only stats (no leak)
   - Train fresh model from scratch (no warm start)
   - Compute test IC on that cut's test period

3. **Aggregate**:
   - Report mean ± std of test IC across cuts
   - Report min and max
   - Report per-cut breakdown

### Why a single split is wrong

A single train/val/test split conflates three different problems:
- **Genuine OOS predictive power** (what we want)
- **Specific test period regime** (what year is in test)
- **Model selection bias** (val is used to pick the model)

Walk-forward separates them: the cut-to-cut std measures regime dependence, the mean measures stable predictive power, and there's no model selection bias if hyperparameters are fixed before WF starts.

### Today's example — what went wrong

We measured iTransformer val IC = +0.041 on 2023 (train 2016-2022). This looked great. But:

| Test | IC on 2023 |
|---|---|
| iTransformer (train 2016-2022) | **+0.041** |
| Linear OLS walk-forward Cut 5 (train 2020-2022) | **−0.001** |
| iTransformer with swap (train 2024-2026) | **+0.004** |

The iTransformer's +0.041 was overfitting noise specific to the 2016-2022 → 2023 mapping. Walk-forward and swap both confirmed 2023 itself doesn't have +0.041 of harvestable signal.

---

## 2. Single-Split IC Numbers — How to Read Them

If you only have a single train/val/test result:

**Be skeptical when:**
- val IC >> test IC (sign of val-specific overfitting via early stopping or epoch selection)
- IC reported is val IC, not test IC
- Test period is short (< 6 months)
- Model has > 1 parameter per 1000 training samples (transformers, deep nets)

**Trust more when:**
- val IC ≈ test IC (similar magnitude, similar sign)
- Test period spans ≥1 full market cycle (~2 years)
- Reported test IC is consistent with prior WF measurements
- Comes from Linear/Ridge with closed-form solution (cannot overfit by training time)

---

## 3. Comparing Models — Paired vs Pooled

To claim "model A beats model B":

**Required**: Run both models on the **same WF cuts**, with **same train/test boundaries**, **same features**, **same labels**.

**Sufficient**: 
- Per-cut win rate > 4/5 cuts AND
- Mean IC difference > 0.01 AND
- Difference > 1 standard error of the cut-to-cut variation

**Not sufficient**:
- Single-split val IC of A > single-split val IC of B
- Different feature sets, different cuts, different labels

**Today's example — wrong claim**: "PatchTST val IC +0.038 > iTransformer val IC +0.041 → PatchTST is more stable" — wrong because both were single-split measurements with no WF. The proper comparison would be both on the same 7-cut WF.

---

## 4. Distinguishing Overfit from Regime Shift

**Both can produce**: high val IC, low test IC.

**Only overfitting produces**:
- Train IC >> val IC during training (typical sign: train_ic = 0.10, val_ic = 0.02)
- Linear baseline gives different (usually more honest) IC than complex model
- Single-split numbers don't reproduce in WF

**Only regime shift produces**:
- Linear baseline ALSO has high IC variance across regimes
- IC is high in similar regimes, low in different regimes
- Stable Linear IC within each regime

**Today's example**: We saw iTransformer val=+0.041 / test=−0.001. To distinguish:
- Walk-forward Linear OLS: mean +0.029, std 0.038 → real signal exists, IS regime-dependent
- Swap test: train 2024-2026 → 2023 val gave +0.004 → iTransformer didn't have a regime-stable +0.04 on 2023
- Conclusion: Linear's regime-dependent +0.029 is the real signal; iTransformer's +0.041 was overfit on top of that

---

## 5. Parameter / Sample Count Sanity

Before training a complex model, check:

- **Linear model**: 1 parameter per feature. ~150 parameters for alpha158. No constraint on sample count.
- **Tree (XGBoost)**: ~depth² × n_trees parameters. ~1000-10000 effective. Need ≥10× as many samples per cut.
- **Transformer**: 100k–10M parameters. Need ≥100× as many samples per parameter to avoid pure memorization.

**Our scale**: 291 stocks × 1500 train dates ≈ 440k samples. 
- Linear (158 params): comfortable
- XGB (depth=5, 100 trees, ~5k effective): comfortable  
- Transformer (350k params): **1.3 samples per parameter** — guaranteed to overfit

**Decision rule**: if param/sample ratio > 1/100, do not train — the complex model will overfit even with strong regularization. Increase data first.

Reference: Hastie, Tibshirani, Friedman (ESL Ch. 7) on bias-variance tradeoff and effective sample complexity.

---

## 6. Three-Test Sanity Suite (per CLAUDE.md §5.2)

Mandatory before declaring any IC > 0.02:

1. **A/A test**: same data, run twice with different seeds. IC difference should be < 1 SE.
2. **Label shuffle**: shuffle labels, retrain. IC should be ≈ 0.
3. **Time-shift placebo**: shift labels forward by 60 days, retrain. IC should be ≈ 0 (any signal is regime persistence, not real causal alpha).

If any of these fail, the original IC measurement is not valid.

---

## 7. Common Pitfalls Checklist

Before publishing any IC number, confirm:

- [ ] Used WF, not single split
- [ ] Reported mean ± std across cuts, not just mean
- [ ] Linear/Ridge baseline measured on the same cuts
- [ ] Param/sample ratio < 1/100 for the model
- [ ] Same features and label across all comparisons
- [ ] No data leakage in feature normalization (train-only stats per cut)
- [ ] Test period is OOS, not validation set
- [ ] Three sanity tests passed (A/A, label shuffle, time shift)

---

## 8. References

- López de Prado (2018), *Advances in Financial Machine Learning*, Ch. 7 (CV), Ch. 8 (Feature Importance)
- Gu, Kelly, Xiu (2020 RFS), *Empirical Asset Pricing via Machine Learning* — uses 30+ year monthly WF, R² of 0.35% corresponds to IC ≈ 0.025-0.035
- Cakici et al. (2023), *Machine Learning Goes Global* — OLS beats 8 ML methods on US large-cap, ML advantage concentrates in small-caps where param/sample is more favorable
- Hastie, Tibshirani, Friedman (2009), *Elements of Statistical Learning*, Ch. 7 — bias-variance, effective sample complexity
- Internal: this project's `walk_forward_panel.py` is the reference implementation
- Internal: `failed-experiments-log.md` E27, E33, E34 — examples of what NOT to claim from single-split tests

---

## 9. Today's Walk-Forward Result (2026-05-08)

7-cut WF on 291-ticker alpha158 + SEC fundamentals, fwd_20d label:

```
       mean      std    min      max     per-cut
OLS    +0.0292  0.038  -0.018  +0.087  [+0.041, +0.087, -0.018, +0.005, -0.001, +0.011, +0.079]
RIDGE  +0.0303  0.038  -0.018  +0.090  [+0.044, +0.090, -0.018, +0.005, +0.000, +0.015, +0.078]
XGB    +0.0390  0.046  -0.016  +0.106  [+0.062, +0.106, -0.016, -0.016, +0.032, +0.014, +0.092]
```

This is the durable, honest baseline. All future model claims must beat this on the same 7 cuts to count.
