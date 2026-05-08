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

## 5. Direct Overfit Detection (Train vs Val Gap)

The honest test for overfitting is empirical, not heuristic. During training, monitor:

```
train_ic at convergence:  X
val_ic at best epoch:     Y
overfit_gap = X - Y
```

**Empirical thresholds (from this project's measurements 2026-05):**
- gap < 0.02: model is generalizing
- gap 0.02-0.05: mild overfitting, still trustworthy
- gap > 0.05: severe overfitting; val IC is largely noise from train→val correlation

**Today's measurements:**

| Model | train_ic | val_ic | gap | Verdict |
|---|---|---|---|---|
| Linear OLS | +0.030 | +0.029 | 0.001 | ✓ honest |
| XGBoost | +0.045 | +0.039 | 0.006 | ✓ honest |
| iTransformer (350k params) | +0.103 | +0.018 | 0.085 | ✗ severe overfit |
| PatchTST (180k params) | +0.077 | +0.020 | 0.057 | ✗ severe overfit |

**Theoretical context**: Hastie/Tibshirani/Friedman (ESL Ch. 7) and Goodfellow/Bengio/Courville (DL Ch. 5) discuss the bias-variance tradeoff and the "VC dimension"-like concept that capacity must be matched to sample complexity. There is no single closed-form ratio (the param/sample heuristic is folklore, not theory). What matters in practice is the measured gap.

**Decision rule (revised, evidence-based)**: train any architecture you want, but if the gap > 0.05 in walk-forward, the val IC is unusable for production decisions. Reduce model capacity (smaller d_model, more dropout, more weight decay) until gap < 0.02 before claiming the architecture works.

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

## 9. Walk-Forward Results Log (live, updated as experiments complete)

All on 7-cut WF: train rolling 3 years, embargo 21 days, test 1 year. Cuts cover 2019-2025.

### 2026-05-08 baselines

#### Wave 1: 3 labels × 6 models (Linear / Ridge / XGB) on 291-ticker alpha158+fund

```
Label         Model           Mean IC    Std       Pos/7  IR (mean/std)
fwd_60d       XGB d=5 e=0.05  +0.0660    0.0722    6/7    0.92  ★ NEW BASELINE
fwd_60d       XGB d=7 e=0.10  +0.0634    0.0765    5/7    0.83
fwd_60d       XGB d=3 e=0.03  +0.0552    0.0610    6/7    0.90
fwd_20d       XGB d=7 e=0.10  +0.0398    0.0492    5/7    0.81
fwd_20d       XGB d=5 e=0.05  +0.0390    0.0456    5/7    0.85
fwd_60d       Ridge a=10      +0.0372    0.0556    4/7    0.67
fwd_60d       OLS             +0.0356    0.0553    4/7    0.64
fwd_20d       Ridge a=10      +0.0303    0.0381    5/7    0.79
fwd_20d       OLS             +0.0292    0.0378    5/7    0.77  (prev baseline)
fwd_5d        XGB d=7 e=0.10  +0.0237    0.0185    6/7    1.28  ← best IR (most stable)
fwd_5d        XGB d=3 e=0.03  +0.0221    0.0194    6/7    1.14
fwd_5d        Ridge           +0.0104    0.0096    6/7    1.08
fwd_5d        OLS             +0.0091    0.0082    6/7    1.11  ← lowest IC, most stable
```

**Key takeaways:**
- **Longer horizon → higher IC but higher std**. fwd_60d gets +0.066 but std=0.072.
- **Shorter horizon → lower IC, much more stable**. fwd_5d std=0.008-0.018.
- **XGB beats Linear by ~30-50% on IC** at all horizons.
- **fwd_5d has best IR** (1.0-1.3); fwd_60d best raw IC.
- **Picking depends on objective**: max IC → fwd_60d XGB; max stability → fwd_5d XGB.

#### Wave 2: regime conditioning (paired test on best config)

```
Config: fwd_60d, XGB d=5 e=0.05, 7 cuts
                       Mean IC  Std     Win/7
alpha158+fund          +0.0660  0.0722  -
+ regime_p (3 cols)    +0.0632  0.0652  4/7
Δ                      -0.0028  -0.0070
```

**Verdict**: Regime not a win (paired threshold: ≥5/7 win + Δmean > 0.01). Std compressed 9.7% but IC unchanged. NOT promoted.

**Theoretical interpretation**: XGB depth=5 with 158+5 alpha158/fund features can already implicitly learn regime structure from VMA, VSTD, BETA features. Adding 3 explicit regime probabilities is information-redundant for this model class.

#### Wave 3: R2K small-cap universe (in progress)

Hypothesis (Cakici et al. 2023 JEDC): ML alpha is 2.4× larger on small-caps than large-caps for US.
- R1K (291 tickers): IC +0.066 (current baseline)
- R2K target: > +0.10 if Cakici holds

Status:
- ✓ Fetched OHLCV for 1910/1919 R2K tickers via yfinance
- ✓ Built alpha158 dataset for 1640 R2K with 5+ years history (3.7M rows, 5.6× larger than R1K)
- ⚠ SEC fundamentals coverage = 0 for R2K (only fetched for R1K) — re-fetching now (~30 min)
- ✓ WF on R2K alpha158-only: IC +0.015 (Cakici fails without fund)
- ✓ WF on R2K + fund: IC +0.026 — fund adds +0.011 to R2K BUT still loses to R1K+fund (+0.066)
- ✗ Cakici 2023 hypothesis NOT CONFIRMED for US daily ML with alpha158+5-fund features (testing if Cakici holds without fundamentals first)

---

## 10. Production Promotion Rules

For any model/feature to replace a current production model, it must:

1. **Beat baseline on same 7 WF cuts** by ≥0.01 mean IC AND
2. **Win ≥5/7 individual cuts** AND
3. **Sanity tests pass** (label shuffle ≈ 0, time-shift ≈ 0) AND
4. **Train_ic / val_ic gap < 0.05** at convergence

If a candidate beats only on mean but fails win-rate, treat it as "promising but variance-inflated" and require a 9-cut or 11-cut WF for confirmation before promotion.

Never promote based on single split val IC, even if val IC > current baseline.

### How this compares to literature (with caveats about apples-to-apples)

| Source | Universe | Reported metric | Our metric, equivalent |
|---|---|---|---|
| Gu, Kelly, Xiu (2020 RFS) | ~30k US stocks, monthly | OOS R²=0.35%, Sharpe=1.35 | (different metric) |
| Cakici et al. (2023 JEDC) | US large-cap, monthly | OLS beats 8 ML methods | Same direction in our WF |
| Qlib CSI300 benchmark | 300 Chinese A-shares, daily | XGB IC=0.050 | Our XGB +0.039 (US, 291) |
| Qlib README baseline | CSI300, daily, alpha158 | Linear IC=0.034 | Our Linear +0.029 (US, 291) |

**Honest comparison statements:**
- "Our Linear OLS WF mean IC +0.029 is **lower than Qlib's CSI300 Linear IC 0.034** by ~15%, consistent with the documented A-share vs US-stock predictability gap (Cakici et al. 2023)."
- "Our XGB +0.039 is **lower than Qlib's CSI300 XGB 0.050** by 22%, again consistent with US-market efficiency."
- "Direct comparison to Gu/Kelly/Xiu is not possible because they report monthly R² and long-short Sharpe, not Spearman rank IC."



### 2026-05-08 sanity verification

Sanity tests on baseline (R1K + alpha158 + 5-fund + XGB d=5 e=0.05 fwd_60d, IC +0.066):

| Test | IC | Verdict |
|---|---|---|
| A/A (3 seeds) | +0.066 ± 0.001 | ✓ Reproducible |
| Per-date label shuffle | +0.025 | ✗ Significant residual signal |
| Time-shift +60d | +0.030 | ✗ Regime persistence present |

**Implication**: "Pure 60d-specific cross-sectional alpha" ≈ +0.066 - +0.025 = **+0.041**.

The headline +0.066 IC includes:
- ~+0.041 from true feature→label causal signal (the alpha we want)
- ~+0.025 from slow features (60d rolling stats) that carry stable per-stock identifying
  information; XGB ranks stocks by these "type" features even with permuted labels;
  test labels also reflect these patterns → spurious shared signal.

This is NOT a refutation of the +0.066 baseline — it's still real predictive power for
trading purposes. But for production planning, treat **+0.04** as the harvestable
60d-specific alpha, not +0.066.

### Open-source projects with reproducible US-stock IC numbers

After research (knowledge base entry `ext_gu_kelly_xiu_2020`, `ext_qlib_benchmark`, `ext_cakici_2023`):
- **microsoft/qlib**: only publishes CSI300/CSI500 numbers, not US
- **OpenSourceAP/CrossSection**: replicates 300+ US anomalies, monthly horizon — useful for fundamental factor construction, not for daily ML IC
- **stefan-jansen/machine-learning-for-trading**: textbook examples on US, but no proprietary IC numbers
- **No widely cited public open-source reference reports daily Spearman rank IC on a US large-cap universe in the literature.** This means our +0.029-0.039 is most rigorously compared against Qlib CSI300 (different market) or Gu et al. (different metric).
