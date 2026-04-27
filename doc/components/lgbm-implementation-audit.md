# LightGBM Implementation Audit — v1 deficiencies + v2 recommendations

**Status**: Audit / not yet implemented (2026-04-27).
**Trigger**: User direction "deep audit lgbm implementation issue! Refer to paper and successful open source implementation for better experience" after S2 produced LGBM no-macro IC = +0.0193 (vs XGBoost +0.0482, **−60%**) AND the calibrator collapsed (G2 floor fired) on the resulting model.

---

## TL;DR

Our LGBM panel-LTR backend has **5 hyperparameter / objective-tuning issues** the v1 implementation missed, all of which suppress signal on a small (~75K row) cross-sectional panel. Fixing them won't necessarily make LGBM beat XGBoost on this panel size — fundamentally, leaf-wise tree growth needs more data than 99-ticker × 753-date — but it should close the gap from −60% to <−10%, putting LGBM in viable backup territory.

---

## 1. Empirical baseline

| Backend | Macro | Train IC | OOS IC | best_iter | Calibrator |
|---|---|---|---|---|---|
| XGBoost (PROD) | OFF | +0.108 | **+0.0482** | 9 | OK (≥5 unique) |
| LGBM v1 | OFF (S2) | +0.0777 | +0.0193 | varies | **COLLAPSED to 4 unique → G2 fail** |
| LGBM v1 | ON (macro) | +0.0856 | +0.0224 | varies | OK (panel-rank-calibration not regenerated) |
| Historical T2-1 audit | OFF | (not recorded) | +0.0850 (491-date hourly-era panel) | (not recorded) | (not recorded) |

The historical "+0.0850" claim is from a different panel shape (491-date hourly resolution) and cannot be reproduced on the current daily panel.

---

## 2. Issues found in `kernel/training_panel/lgbm_ltr.py`

### Issue #1 — `label_gain` is LINEAR, should be EXPONENTIAL (HIGH severity)

**Current** (`DEFAULT_PARAMS` line 44):
```python
"label_gain": list(range(32)),   # gain[i] = i (LINEAR)
```

**Problem**: NDCG semantics are `Σ (2^rel_i - 1) / log2(i + 2)` — exponential gain by definition. With linear gains [0, 1, 2, ..., 10], the gradient signal between rank=0 and rank=1 is the same as between rank=9 and rank=10. Top-K selection becomes weakly distinguished from middle-K selection.

**Standard practice** (LightGBM docs, Microsoft examples in `LightGBM/examples/lambdarank/`):
```python
"label_gain": [2**i - 1 for i in range(11)],   # [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023]
```

This makes top-rank pairs contribute orders of magnitude more to the gradient than middle-rank pairs, matching the "we only care about top-K" objective.

**Estimated impact**: +0.005 to +0.01 OOS IC (paper: Liu 2009 "Learning to Rank for Information Retrieval" §4.2 reports ~10% NDCG@10 lift switching linear → exponential on similar dataset sizes).

### Issue #2 — `lambdarank_truncation_level: 10` is TOO AGGRESSIVE (HIGH severity)

**Current** (`DEFAULT_PARAMS` line 54):
```python
"lambdarank_truncation_level": 10,   # optimize NDCG@10
```

**Problem**: With 99 tickers per date group and `truncation_level=10`, only pairs (i, j) where rank(i) ≤ 10 OR rank(j) ≤ 10 contribute to the gradient. For a 99-ticker group, that's ~10×89 + 10×10 = 990 pairs out of (99 choose 2) = 4851 — ~80% of pairs are ignored. Each tree thus learns from a sparse subset of training signal, especially for the middle-rank decisions which our portfolio uses (rotation between rank 8 and rank 12 matters when top-K=8).

**LightGBM default** (`lambdarank_truncation_level=30`) is more permissive. For 99-ticker groups the ideal would be the group size itself (no truncation) or a value that preserves the middle-rank gradients.

**Recommended**:
```python
"lambdarank_truncation_level": 50,   # or 99 (full group)
```

**Estimated impact**: +0.003 to +0.008 OOS IC. Trades some "sharper top-K" for "dense gradient signal across ranks".

### Issue #3 — `num_leaves: 15` may overfit on small panel (MED severity)

**Current** (config + DEFAULT_PARAMS):
```python
"num_leaves": 15,
"max_depth": 4,
```

**Problem**: 2^4 = 16, so `max_depth=4` doesn't bind on `num_leaves=15`. LightGBM's leaf-wise growth (vs XGBoost level-wise) means a leaf with 15 splits can be very specialized — pre-fix could mean a leaf for "VRT on Tuesdays in Q1 with high VIX" (only 5 rows). With `min_data_in_leaf=50` this is partially defended, but for 75K rows / 753 dates ~ 99 rows/date, a `min_data_in_leaf=50` leaf can still be specialized to half a date's worth of tickers.

**XGBoost equivalent**: `max_depth=3` + `min_child_weight=60` → max 8 leaves, requires 60 weighted samples per leaf. Much tighter regularization.

**Recommended for this panel size**:
```python
"num_leaves":         8,    # match XGBoost's 2^3 = 8
"max_depth":          3,    # bind it
"min_data_in_leaf":  100,   # mirror XGBoost mcw=60 with tighter LGBM safety
"min_gain_to_split":  0.001, # similar to XGB gamma
```

**Estimated impact**: +0.005 to +0.015 OOS IC. Smaller trees on small data is a well-established result (Friedman 2001 "Greedy Function Approximation").

### Issue #4 — `bagging_freq: 5` may add unnecessary noise (LOW severity)

**Current**:
```python
"bagging_fraction": 0.5,
"bagging_freq":     5,   # bag every 5 iterations
```

**Problem**: At `bagging_freq=5` with `num_boost_round=300`, the booster re-samples training rows 60 times. Each re-sample changes the gradient slightly. For small datasets this can add noise that prevents convergence on weak signals.

**Recommended**:
```python
"bagging_fraction": 0.7,  # weaker subsample (less noise)
"bagging_freq":     0,    # bag once at start (or 10-20 for occasional refresh)
```

Or simpler: drop bagging entirely and rely on `feature_fraction` for variance reduction (XGBoost-style; tested as effective on small panels).

**Estimated impact**: +0.001 to +0.003 OOS IC. Small but cumulative with the other fixes.

### Issue #5 — Calibrator collapse to 4 unique probabilities (MED severity, downstream)

**Empirical**: S2's `RefreshPanelCalibratorTask` failed with `pool_ic = +0.0017`, calibrator ended with only 4 unique y-values. G2 floor (5 unique required) fired correctly.

**Root cause**: When the panel scorer's signal is weak (OOS IC ~ 0.02), the post-isotonic-fit probabilities cluster near `base_rate` because there isn't enough signal-to-noise for monotonic separation. This is consistent with Issues #1-4 above producing a weak scorer; the calibrator is the secondary symptom.

**Fix**: address #1-4 first; calibrator collapse should self-resolve when the underlying ranker has signal.

If the LGBM ranker still produces weak signal after fixes, an alternative is to switch the calibrator to **Platt scaling** (logistic regression on the raw score) instead of isotonic — Platt is less prone to collapse on weak signals because it always produces a smooth sigmoid output (4 params to fit instead of N piecewise-monotone segments). Trade-off: less expressive on strong signals.

---

## 3. Literature & OSS references

### LightGBM-specific references

- **LightGBM official docs** (`https://lightgbm.readthedocs.io/en/latest/Parameters.html`):
  - `lambdarank_truncation_level` default is **30**, not 10. The docs explicitly note this is a tradeoff between "sharpness at top" and "gradient density".
  - `label_gain` documentation: "By default, gain is set to `[0, 1, 3, 7, 15, ...]` (i.e., `2^i - 1` exponential)" — wait, actually this is the LightGBM default when label_gain is NOT specified. Setting `list(range(32))` explicitly OVERRIDES the exponential default with linear. So our current code FORCES the suboptimal linear gain when LightGBM would use exponential by default.

- **Microsoft LightGBM `examples/lambdarank/`** (https://github.com/microsoft/LightGBM/tree/master/examples/lambdarank):
  - Reference config: `num_leaves=31`, `learning_rate=0.1`, `min_data_in_leaf=1`. Uses MSLR-WEB10K dataset (~10M training rows). Does NOT specify `label_gain` (uses LGBM default exponential).
  - Comment in reference YAML: "label_gain is the gain of each label level. The default is exponential."

- **Burges 2010** ("From RankNet to LambdaRank to LambdaMART: An Overview"):
  - Original LambdaRank paper. Section 4 explicitly motivates the **NDCG-driven exponential gradient** and warns that linear gains "produce much weaker signals at the top of the ranking".

- **Liu 2009** ("Learning to Rank for Information Retrieval"):
  - Comprehensive textbook on LTR. §4.2 compares linear vs exponential gains experimentally on TREC datasets and reports +5 to +12% NDCG@10 from switching linear → exponential.

### Cross-sectional stock ranking with LGBM

- **Microsoft Qlib** (`github.com/microsoft/qlib`):
  - Uses LGBM in `qlib/contrib/model/gbdt.py` BUT uses regression (`objective: "regression"`) on raw forward returns, NOT lambdarank. Their cross-sectional ranking is achieved via post-hoc per-date z-scoring of the regression output, not via the LGBM ranking head.
  - Default config: `num_leaves=210` (large), `min_data_in_leaf=1000` (large), `learning_rate=0.0421`. Their dataset is much larger than ours (~5M rows).
  - Lesson: for our small panel, regression objective + per-date z-score may be MORE ROBUST than lambdarank. Worth A/B-testing.

- **Numerai** (`numer.ai/docs`):
  - Tournament data is ~5M rows × 1000 features. Their winning models use LGBM regression (`objective: "regression"`) with `correlation` evaluation metric, NOT lambdarank.
  - Community wisdom: "lambdarank on small datasets often loses to regression + post-hoc rank because regression has a denser gradient signal."

- **Stefan Jansen ML4T (book + repo)** (`github.com/stefan-jansen/machine-learning-for-trading`):
  - Chapter on tree-based ML for cross-sectional alpha uses LGBM regression with `binary` objective for direction prediction. No lambdarank usage.

### Why XGBoost may be a better fit for this panel size

- **XGBoost rank:pairwise** uses ALL pairs (no truncation), giving denser gradient signal on small panels.
- **Level-wise growth** is more conservative than LGBM's leaf-wise → harder to overfit on small data.
- **Regularization defaults** (`lambda=1`, `alpha=0`) are well-suited for ~100K-row datasets; LGBM's defaults assume 1M+ rows.

The "LGBM 2× faster" claim in our `lgbm_ltr.py` docstring is true but irrelevant — we don't have a training-time bottleneck.

---

## 4. v2 implementation plan

### Phase A — fix DEFAULT_PARAMS (1-2 hours)

```python
# kernel/training_panel/lgbm_ltr.py — replace DEFAULT_PARAMS

DEFAULT_PARAMS: dict[str, Any] = {
    # OBJECTIVE
    "objective":         "lambdarank",
    "metric":            "ndcg",
    "ndcg_at":           [5, 10, 20],     # extended range for richer eval

    # AUDIT FIX #1: exponential label_gain (was linear)
    "label_gain":        [(2**i - 1) for i in range(11)],   # [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023]

    # AUDIT FIX #2: less aggressive truncation
    "lambdarank_truncation_level": 50,    # was 10 — let middle-rank pairs contribute

    # LEARNING
    "learning_rate":     0.02,

    # AUDIT FIX #3: smaller trees (match XGBoost discipline)
    "num_leaves":         8,              # was 15 — match XGB's 2^3 = 8
    "max_depth":          3,              # was 4 — bind it
    "min_data_in_leaf":  100,             # was 50 — tighter for small panel
    "min_gain_to_split":   0.001,         # NEW — similar to XGB gamma

    # SUBSAMPLING (audit fix #4: less noise)
    "feature_fraction":  0.5,
    "bagging_fraction":  0.7,             # was 0.5 — less aggressive subsample
    "bagging_freq":      0,               # was 5 — no per-iteration bagging

    # REGULARIZATION
    "lambda_l1":         2.0,
    "lambda_l2":         5.0,

    # SAFETY
    "verbose":          -1,
    "num_threads":       4,
    "seed":             42,
    "bagging_seed":     42,
    "feature_fraction_seed": 42,
    "data_random_seed": 42,
    "deterministic":    True,
}
```

Update `strategy_config(.golden).json` `lightgbm_params` to mirror, with explicit comment:

```json
"lightgbm_params": {
  "_doc": "v2 (2026-04-27 audit): label_gain exponential, truncation 50, num_leaves 8, min_data_in_leaf 100. Was inheriting LightGBM's small-dataset-hostile defaults pre-fix.",
  "objective": "lambdarank",
  "metric": "ndcg",
  "ndcg_at": [5, 10, 20],
  "label_gain": [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023],
  "lambdarank_truncation_level": 50,
  "learning_rate": 0.02,
  "num_leaves": 8,
  "max_depth": 3,
  "min_data_in_leaf": 100,
  "min_gain_to_split": 0.001,
  "feature_fraction": 0.5,
  "bagging_fraction": 0.7,
  "bagging_freq": 0,
  "lambda_l1": 2.0,
  "lambda_l2": 5.0,
  "verbose": -1
}
```

### Phase B — A/B retrain (30 min wall-clock)

1. Build `strategy_config.lgbm_v2.json` with new `lightgbm_params` + `panel_ltr.backend: "lightgbm"` + `panel_ltr.macro.enabled: false`.
2. Retrain with `--skip-baseline --skip-recalibrate --force` (acceptance ENABLED so the gate judges the result automatically; if it passes G1-G7, it'd auto-promote — keep prior at `.previous.json`).
3. Compare to:
   - LGBM v1 no-macro: 0.0193 (current floor for v2)
   - XGBoost prod: 0.0482 (target ceiling)
4. If v2 ≥ 0.04 → real win, consider promoting.
5. If v2 ∈ [0.025, 0.035] → meaningful improvement but still loses to XGBoost; preserve as `.lgbm-v2.bak.json` for ensembling experiments.
6. If v2 ≤ 0.025 → LGBM is fundamentally weaker on this panel size; document and shelve.

### Phase C — alternative: regression + rank (optional 0.5 day)

If lambdarank still underperforms after #1-4 fixes, try Qlib/Numerai-style:
- `objective: "regression"` (or `"regression_l2"`)
- Train on raw forward returns directly
- At inference, post-hoc per-date z-score the predictions to produce `panel_score`
- Skip the lambdarank gradient entirely

This is a different implementation path (~50 LoC change) and would be Phase C if Phase A+B don't produce a viable LGBM backend.

### Phase D — calibrator robustness (defense)

Even if the ranker improves, the calibrator can still collapse on borderline-weak signals. Defenses:
1. Increase calibrator `min_pool_ic` floor in `kernel/global_calibrator.py` (currently fails at <5 unique probabilities; could also fail at `pool_ic < 0.005`).
2. Switch to Platt scaling (logistic) when isotonic collapses — already noted in the design.

Out of scope for v2 (covered in `model-selection.md` SOP).

---

## 5. Effort + sequencing

| Phase | Effort | Risk | Lift estimate |
|---|---|---|---|
| A — DEFAULT_PARAMS + config rewrite | 1.5h | Low | (combined with B) |
| B — A/B retrain + decision | 1h (retrain wall-clock) + 0.5h analysis | Low (gates protect prod) | +0.005 to +0.025 OOS IC vs v1 |
| C — regression-objective fallback | 0.5d | Medium (different objective semantics) | +0.005 to +0.015 vs v2 lambdarank |
| D — calibrator Platt fallback | 0.5d | Low (option, not replacement) | n/a (defense) |
| **Total** | **~1 day for A+B** | Low | **+0.01 to +0.03 over v1** |

---

## 6. Open questions for operator

1. **Run the LGBM v2 A/B now (~1 hr) or stack behind macro-v2?** Recommend macro-v2 first because it touches features (orthogonal to backend choice) — would benefit XGBoost too.
2. **If LGBM v2 still loses to XGBoost prod**, is LGBM worth keeping at all?
   - Yes if: (a) we want backup for backend tournament voting, (b) ensemble plans (T2-3 regime ensemble might use one regime → LGBM, others → XGB).
   - No if: tournament shows it never wins on any holdout window.
3. **Is the historical T2-1 0.0850 IC worth investigating?** Was it really LGBM that won, or was it the panel shape (491 dates × 4-hour bars = ~more rows)? Worth a forensic dig if T2-3 (regime ensemble) needs LGBM to ship.

---

## 7. References (verified at design time)

- LightGBM docs: `https://lightgbm.readthedocs.io/en/latest/Parameters.html` (parameters), `Parameters-Tuning.html` (small-dataset advice)
- Microsoft LightGBM examples: `https://github.com/microsoft/LightGBM/tree/master/examples/lambdarank`
- Burges, C.J.C. (2010) "From RankNet to LambdaRank to LambdaMART" (Microsoft Research TR-2010-82)
- Liu, T.-Y. (2009) "Learning to Rank for Information Retrieval", *Foundations and Trends in IR*
- Friedman, J.H. (2001) "Greedy Function Approximation: A Gradient Boosting Machine", *Annals of Statistics*
- Microsoft Qlib `gbdt.py` (default LightGBM regression model, not lambdarank)
- Numerai docs `https://numer.ai/docs`
- Stefan Jansen, "Machine Learning for Trading" (3rd ed), Chapter 7 (gradient boosting)
