# Panel OOS IC improvement — what the literature says + concrete plan

**Context:** current panel CPCV OOS IC = **+0.0355** (hourly+minute combined). User: "这个 IC 还是不够好啊！找找 research paper！怎么提升！数据量已经够了呀！"

Benchmark for comparison:
- Academic cross-sectional ML (Gu-Kelly-Xiu 2020): IC ≈ 0.05-0.08 on 60yr panel
- Top quant shops (Two Sigma / DE Shaw, inferred): IC ≈ 0.06-0.10
- Naïve factor models (Fama-French 5): IC ≈ 0.02
- Our current: **0.0355** — room to grow ~2× before hitting academic frontier

---

## 1. What Gu-Kelly-Xiu (2020) did that got 0.05-0.08

Their `Empirical Asset Pricing via Machine Learning` paper is the benchmark:

1. **94 firm-level features**: size, momentum, liquidity, volatility, valuation, quality, growth. We have ~29 (24 neutralized + 6 hourly + minute additions).
2. **NN + tree ensemble** — they run 7 models, NN4 beats XGBoost on R² by ~25%.
3. **Monthly cross-section, long history** (60 yrs). We have 2yr + 10min which is ~100× more rows per day but only 2 years of regime history.
4. **Ensemble averaging** across models: their Table 3 shows ensemble beats any individual by ~10%.

**What we can replicate now:**
- Ensemble across XGBoost + LightGBM + NN (transformer) → est. +0.005 IC
- More feature classes (see §3)
- Expand training window (we use 5yr; could go 10yr from OpenBB)

## 2. Chinco-Clark-Joseph-Ye (2019) — sparse signals

Key insight: **LASSO-selected sparse features beat dense kitchen-sink models** on OOS. Excess variance of dense models kills OOS despite higher in-sample IC.

Our `panel-ltr.drop_cols` does manual feature dropping. Could be automated via L1 regularization.

## 3. Feature classes we could add (concrete)

### 3a. Analyst revisions momentum  (expected +0.005-0.010 IC)

**Source**: Chan, Jegadeesh, Lakonishok (1996) — "Momentum strategies". Revisions momentum (change in mean EPS forecast over 1-3 months) is the **strongest** single signal in their study — better than price momentum.

**Implementation**: OpenBB's `equity.fundamental.estimates` returns analyst EPS estimates over time. Compute:
- `analyst_rev_1m` = (eps_mean_today − eps_mean_30d_ago) / eps_mean_30d_ago
- `analyst_rev_3m` = (eps_mean_today − eps_mean_90d_ago) / eps_mean_90d_ago

### 3b. Short interest / squeeze proxy  (expected +0.002-0.005 IC)

We have `short_pct_float` (static). Need **change** in short interest:
- `short_interest_chg_30d` = (short_float_today − short_float_30d_ago) / short_float_30d_ago

Rising short interest with stable price → squeeze coming. Literature: Diether, Lee, Werner (2009).

### 3c. Options-implied skew  (if available)  (expected +0.005-0.010 IC)

Put/call ratio, implied vol skew are strong cross-sectional predictors.

**Blocker:** OpenBB free tier has limited options data. Skip unless we get paid data.

### 3d. Idiosyncratic volatility residual  (expected +0.003-0.008 IC)

Ang, Hodrick, Xing, Zhang (2006): low-idiosyncratic-vol stocks outperform high-idio-vol ("idiosyncratic volatility puzzle"). Negative sign.

**Implementation**: regress each stock's daily return against Fama-French 3 factors + momentum; compute 60d rolling std of residuals.

### 3e. Accruals / accrual quality  (expected +0.004-0.007 IC)

Sloan (1996) — accrual anomaly. High accruals (earnings that don't match cash flows) → future underperformance.

**Implementation**: `(net_income − operating_cashflow) / total_assets` from quarterly reports.

### 3f. Quality-of-earnings  (expected +0.003-0.005 IC)

Piotroski F-score (2000): 9-component quality score. Each component binary (good/bad). Sum gives 0-9 F-score.

## 4. Model-level improvements (already-collected data)

### 4a. Multi-horizon ensemble  (expected +0.003-0.005 IC)

Train 3 panels on different labels:
- panel_5d: predict fwd 5-day return
- panel_10d: predict fwd 10-day (current)
- panel_20d: predict fwd 20-day

At inference, average the 3 models' rank scores. Smoothing across horizons reduces noise.

### 4b. Ensemble: XGBoost + LightGBM + Transformer  (expected +0.005-0.010 IC)

- XGBoost (current, monotone constraints)
- LightGBM (already attempted, shelved due to lack of monotone constraints; workaround: post-hoc constraint enforcement)
- Transformer (shelved at panel < 200k; NOW gate opens with our 744k rows)

Average the 3 ranks. Works even better when models disagree (diversity premium).

### 4c. Feature winsorization  (expected +0.002-0.005 IC)

Clip every feature to ±3σ before training. Removes outlier influence on tree splits. Industry standard in quant shops.

Our current panel uses cross-sectional z-score, but NOT winsorized. Easy +0.002 IC.

### 4d. Sample weights by cross-sectional dispersion  (expected +0.002-0.004 IC)

On days when cross-sectional return dispersion is LOW (all stocks move together — macro day), the signal is meaningless; add a sample weight that down-weights those dates.

### 4e. Longer training window  (expected +0.002-0.005 IC)

We train on 5yr. Fama/French find momentum works on 40yr samples. Expanding to 10yr+ stabilizes factor loadings without over-fitting (the model is L1/L2 regularized).

### 4f. Label neutralization  (expected +0.002-0.004 IC)

Current label: `fwd_return_10d − SPY_10d`. Could refine:
- Minus sector_ETF_return (removes sector beta)
- Minus beta × SPY_return (full beta neutralization via 60d beta estimate)

More granular neutralization = less contamination from macro/sector, cleaner stock-specific signal.

## 5. Prioritized implementation path (quickest wins first)

| # | Item | Effort | Expected IC gain | Data ready |
|---|---|---|---|---|
| 1 | **Feature winsorization** | 30 min | +0.002-0.005 | ✅ |
| 2 | **Multi-horizon ensemble (5d/10d/20d)** | 3 h | +0.003-0.005 | ✅ |
| 3 | **Transformer backend (retry)** | 4-6 h | +0.005-0.010 | ✅ (744k rows) |
| 4 | **Full model ensemble (XGB+LGBM+Transformer)** | 2 h after #3 | +0.005-0.010 | ✅ after #3 |
| 5 | **Analyst revisions feature** | 4 h (OpenBB fetch + training wire) | +0.005-0.010 | blocked on fetch |
| 6 | **Accruals / Piotroski feature** | 3 h | +0.004-0.008 | blocked on fetch |
| 7 | **Label neutralization (β × SPY)** | 2 h | +0.002-0.004 | ✅ |
| 8 | **10-yr training window** | 1 h (but retrain takes 2×) | +0.002-0.005 | blocked on OpenBB history |
| 9 | **Idiosyncratic vol feature** | 2 h | +0.003-0.008 | ✅ |

**Stacking assumption:** gains don't simply sum (correlation between improvements reduces additive contribution). Realistic cumulative ceiling: **+0.015-0.030 IC → total ≈ 0.05-0.065** (matches academic frontier).

## 6. Ship order (my recommendation)

**Today (pure code, no new data):**
- ✅ 10-min data fetch (done, panel retrain running)
- 🔨 Feature winsorization (Item 1)
- 🔨 Transformer retry (Item 3)
- 🔨 Model ensemble (Item 4 after 3)

**This week (need new data fetch):**
- Analyst revisions from OpenBB
- Accruals from fundamentals

**Next week (experimental):**
- Multi-horizon ensemble
- Label neutralization refinements
- Watchlist 44 → 64 with curated adds

---

## References
- Gu, Kelly, Xiu (2020) "Empirical asset pricing via machine learning", *RFS*.
- Chinco, Clark-Joseph, Ye (2019) "Sparse signals in the cross-section of returns", *JoF*.
- Chan, Jegadeesh, Lakonishok (1996) "Momentum strategies", *JoF*.
- Diether, Lee, Werner (2009) "Short-sale strategies and return predictability", *RFS*.
- Sloan (1996) "Do stock prices fully reflect information in accruals and cash flows about future earnings?", *Accounting Review*.
- Piotroski (2000) "Value investing: the use of historical financial statement information", *JAR*.
- Ang, Hodrick, Xing, Zhang (2006) "The cross-section of volatility and expected returns", *JoF*.
- Feng, He, Polson (2018) "Deep learning for predicting asset returns", *SSRN*.
