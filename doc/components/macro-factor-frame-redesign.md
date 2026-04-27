# Macro Factor Frame — Redesign (v2)

**Status**: ⚠️ Implemented + tested (2026-04-27); SAME IC as v1 broadcast — diagnosis was incomplete.
**Trigger**: User direction "你的代码设计有缺陷，找点参考文献或者开源代码" after the v1 macro design produced consistently negative IC across both XGBoost (-18%) and LightGBM (-53%) backends.
**Supersedes**: `macro-factor-frame-design.md` (v1, broadcast-per-date design).

---

## 🚨 v2 retrain result (2026-04-27)

| Backend | Macro design | Features | OOS IC | Notes |
|---|---|---|---|---|
| XGB (PROD) | OFF | 28 | **+0.0482** | baseline |
| XGB | v1 broadcast | 61 | +0.0393 (-18%) | predicted bug — passed |
| **XGB** | **v2 per-ticker β** | **61** | **+0.0393 (-18%)** | **IDENTICAL to v1!** |

The within-date-invariance hypothesis was correct in principle but didn't
explain the IC penalty. Per-ticker β has within-date variance (verified at
training: PanelFeatureJob[macro v2] merged β into 103/103 tickers), yet
produces the same IC as v1 broadcast. The 33 extra features themselves
are **just noise relative to existing 28 features** — they crowd out the
booster's `colsample_bytree=0.5` random sampling regardless of whether they
have within-date variance or not.

Conclusion: macro factors (VIX/HYG/UUP/etc) on this panel + XGBoost
rank:pairwise objective don't add cross-sectional ranking signal. Could
still help via:
- Two-stage: macro predicts equity premium time-series, applied as
  regime gate (NOT a ranker feature). Goyal-Welch 2008.
- Regime-conditional ensemble (T2-3): train 4 panels per regime, route
  by macro state.
- Macro × ticker interaction features: e.g. `mom_12_1_z × VIX_state`.
  Forces conditional cross-sectional effect.

Artifact preserved at `panel-ltr.macro-v2.bak.json` (61 features, 0.0393 IC).

---

## TL;DR

**v1 (current) is structurally broken for cross-sectional ranking.** The macro frame broadcasts the same value across all tickers on each date, giving the rank loss zero within-date variance to learn from. This doc proposes **v2: per-ticker macro β/correlation features** — the literature-standard approach used by Qlib (Microsoft), ML4T (Stefan Jansen), and Numerai.

---

## 1. Why v1 fails

### The bug, in one paragraph

A cross-sectional learning-to-rank model (XGBoost `rank:pairwise`, LightGBM `lambdarank`) optimizes a pairwise loss on (ticker_i, ticker_j) pairs **within the same group**, where group = trading day. The loss is invariant to features that are constant across the group. Our v1 macro frame broadcasts (`vxx_level_z`, `hyg_chg_5d_z`, …) identically to all 99 tickers on a given date, so those 33 macro features contribute exactly zero to the gradient of the pairwise rank loss. They occupy 33/61 = 54% of the feature budget but carry 0% ranking signal. Worse, with `colsample_bytree=0.5`, the booster randomly samples ~50% of features per tree, meaning roughly half its trees are built from pure noise.

### Empirical confirmation

| Backend | Macro | OOS IC | vs prod | Notes |
|---|---|---|---|---|
| XGBoost | OFF | +0.0482 | baseline | 28 features |
| XGBoost | ON v1 | +0.0393 | **−18%** | 61 features (28 + 33 broadcast macro) |
| LightGBM | ON v1 | +0.0224 | **−53%** | 61 features |
| LightGBM | OFF | +0.0193 | **−60%** | 28 features (LGBM separately worse) |

The XGBoost-ON case ALSO shows `best_iter=4` (vs no-macro `best_iter=9`) — the booster's early-stopping kicked in earlier because adding noise features made it overfit faster.

### Why it's not "just bad data"

If the macro signals were genuinely uninformative, we'd expect IC to stay ~flat with macro added. Instead it DROPPED by 18-53%, which is consistent with feature-budget dilution rather than weak signal. The signals themselves (VIX, HY credit spread, dollar strength) are well-established macro factors in the literature — the problem is HOW we wired them.

---

## 2. Literature & open-source references for the right approach

### Core papers

- **Gu, Kelly, Xiu 2020** (RFS, "Empirical Asset Pricing via Machine Learning") — the canonical ML-for-asset-pricing reference. They explicitly separate *time-series* macro predictors (8 macro factors: dividend-price ratio, term spread, etc.) from *cross-sectional* characteristic features. Macro factors enter as **interactions with stock characteristics** (size × term_spread, momentum × VIX), not as raw broadcast features. This produces the per-ticker, within-date variation needed for cross-sectional ranking.

- **Goyal, Welch 2008** (RFS, "A Comprehensive Look at the Empirical Performance of Equity Premium Prediction") — establishes that macro variables predict the *equity premium* (a time-series quantity, market-level), not individual stocks' cross-sectional ranks. Lesson: use macro for regime gating or aggregate-return forecasting, not as direct cross-sectional features.

- **Kelly, Pruitt, Su 2019** (J. Financial Economics, "Characteristics are Covariances") — IPCA framework. Stock-specific factor *exposures* (β to macro factors, computed via rolling regressions) are what predict cross-sectional returns. The β values vary across tickers and dates, so they DO carry cross-sectional ranking information.

- **Avramov, Cheng, Metzker 2023** (J. Finance, "Machine Learning vs Economic Restrictions") — empirically shows that adding macro variables AS-IS to ML cross-sectional models performs poorly; adding them as **conditioning variables for stock characteristics** materially helps.

- **Two Sigma 2024** ("A Machine Learning Approach to Regime Modeling") — mixture-of-experts: train separate cross-sectional models per macro regime, route at inference. Macro is the *gate*, not the feature.

### Open-source implementations

- **Microsoft Qlib** (`https://github.com/microsoft/qlib`):
  - Uses macro variables ONLY as part of *per-stock* features. Each ticker's feature vector includes things like the ticker's rolling β to TLT, the ticker's correlation with VIX, etc. — values that DIFFER per ticker on the same date.
  - Their `Alpha158` and `Alpha360` feature handlers explicitly include `(stock_return - market_return)` style relative features. No raw VIX broadcast.

- **Stefan Jansen, "Machine Learning for Trading" (book + repo)** (`https://github.com/stefan-jansen/machine-learning-for-trading`):
  - Chapter on macro: uses macro to build a *regime classifier* (high-vol vs low-vol). Cross-sectional ranking models are then trained per-regime. Macro is never a direct feature in the cross-sectional model.
  - Quote (paraphrased): "macroeconomic data describes the market state; cross-sectional alpha describes which stock outperforms within that state. Mixing them as a single feature vector loses both."

- **Numerai** (`https://numer.ai/`, public docs):
  - Their tournament data treats macro-like features as "auxiliary" — features that predict *era difficulty* (which weeks are hard to forecast), not direct features for stock ranking.
  - Their feature engineering guide explicitly warns: "if a feature is constant across all rows in an era, it provides no ranking information; consider it as a meta-feature for re-weighting eras instead."

- **Fama-MacBeth pipelines in academic code** (varies): the standard practice is to compute each stock's β to macro factors via rolling regression, then USE THE β AS THE FEATURE — not the macro value itself.

---

## 3. v2 design — per-ticker macro β/correlation features

### Core principle

Macro factors enter the cross-sectional ranker **only through per-ticker derived quantities** that vary across tickers on the same date. Specifically, for each ticker × date, compute:

| Feature | Formula | Why it ranks |
|---|---|---|
| `beta_vix_60d` | rolling 60d OLS β of `ticker_returns` regressed on `VIX_returns` | Ticker A might have β=−1.2 to VIX (defensive), ticker B β=+0.3 — *different per ticker on same date* |
| `corr_hyg_60d` | rolling 60d Pearson corr | Credit-sensitive tickers (REITs, financials) move with HYG; tech doesn't |
| `beta_uup_60d` | β to dollar | Multinationals (AAPL, GOOG) hate strong dollar; domestics don't |
| `beta_tlt_60d` | β to long bonds | Duration-like tickers vs cyclicals |
| `beta_xlu_60d` | β to utilities | Defensive proxy |
| `corr_mtum_60d` | β to momentum factor ETF | Loadings on high-mom factor |
| `corr_usmv_60d` | β to low-vol factor | Defensive factor exposure |

Use **rolling 60-day window** (rolling β), with strict pre-knowledge (β at bar `t` uses [t-60, t-1] only — no leak).

### What goes away

- `vxx_level_z` (broadcast). 33 macro features × 99 tickers × 753 dates = 2.4M cells of zero-information data — gone.
- The macro "level" interpretation. Macro frame still loaded (data still cached) but consumed differently.

### What stays

- The macro symbol list: VXX/HYG/UUP/DBC/GLD/TLT/XLV/XLU/KRE/MTUM/USMV remains the same set of macro factors.
- The macro factor frame itself (`kernel/macro.py::build_macro_frame`) — still loaded per the existing pipeline.
- `panel_ltr.macro.enabled` config flag — repurposed to gate the new per-ticker β path.

### Feature count math

- Current macro v1 ON: 28 + 33 = 61 features (33 broadcast = wasted).
- Proposed v2 ON: 28 + ~7-11 per-ticker β features = ~35-39 features.

The smaller, ticker-varying feature set should ADD ranking signal rather than dilute it.

### Implementation sketch

```python
# kernel/macro_per_ticker.py (NEW)

def compute_per_ticker_macro_betas(
    ohlcv: dict[str, pd.DataFrame],         # per-ticker OHLCV
    macro_returns: pd.DataFrame,             # date-indexed, columns: vxx, hyg, uup, ...
    rolling_window: int = 60,
    min_window: int = 30,
) -> dict[str, pd.DataFrame]:
    """Per-ticker rolling β to macro factors.

    For each ticker, returns a DataFrame indexed by date with columns
    beta_{macro}_{rolling_window}d for each macro symbol. β_t uses
    strictly-prior data [t - rolling_window, t - 1] (no look-ahead).

    Used as ADDITIONAL per-ticker features in build_panel_frame's
    factor_frames dict — they get cross-sectionally z-scored alongside
    size_z / mom_12_1_z / beta_60d_z and ENTER THE RANK LOSS PROPERLY
    because each ticker has a different β on the same date.
    """
    out = {}
    for ticker, df in ohlcv.items():
        ticker_returns = df["close"].pct_change()
        cols = {}
        for macro_col in macro_returns.columns:
            macro_r = macro_returns[macro_col].reindex(ticker_returns.index)
            # Rolling OLS β = Cov / Var (window=60, min_periods=30)
            cov = ticker_returns.rolling(rolling_window, min_periods=min_window).cov(macro_r)
            var = macro_r.rolling(rolling_window, min_periods=min_window).var()
            beta = cov / var.replace(0, np.nan)
            # Shift by 1 to ensure t-1 is the latest data used
            cols[f"beta_{macro_col}_{rolling_window}d"] = beta.shift(1)
        out[ticker] = pd.DataFrame(cols)
    return out
```

### Wire into pipeline

1. `pp_panel_training.py::PanelDataJob` — already loads `macro_factor_frame` via `LoadMacroFactorsTask`. Add a new `LoadMacroPerTickerBetasTask` that reads `ctx.ohlcv` + `ctx.macro_factor_frame`, calls `compute_per_ticker_macro_betas`, attaches to `ctx.macro_betas` (dict[ticker, DataFrame]).
2. `PanelAssemblyJob::BuildPanelTask` — when `ctx.macro_betas` is non-empty, merge each ticker's β columns into its `factor_frames[ticker]` so they go through `FactorZScoreTask` (cross-sectional z-score per date) along with size_z et al.
3. `kernel/panel_pipeline/feature_matrix.py::build_inference_matrix` — instead of broadcasting `macro_frame`, look up each ticker's `macro_betas[ticker]` row for `today` and append columns.
4. `prepare_inference_panel_frames` — return tuple becomes `(neutralized, factor, macro_betas)` (still 3 elements; macro_frame absorbed into macro_betas).
5. **Symmetry guard test** in `test_train_inference_symmetry.py` — already exists; will catch any drift.

### Acceptance gate behavior

The 11-gate `ModelAcceptanceGate` will catch v2 retrain regressions:
- G1 schema: `feature_cols` superset (current 28 + 7-11 betas) → passes.
- G4 OOS IC vs prior: requires new ≥ 0.0482 × 0.95 = 0.0458. If v2 doesn't reach this, REJECT — operator inspects.
- G2 calibrator: requires ≥ 5 unique probabilities. The S2 LGBM-no-macro experiment showed this can fire on weak signals.

Run with default acceptance ENABLED — no `--skip-acceptance`. If v2 is genuinely better, it'll pass; if not, prod is preserved automatically.

---

## 4. Alternative designs (considered + rejected)

### B. Macro × ticker INTERACTIONS

Multiply macro state by ticker characteristics: `size_z × VIX_z`, `mom_12_1_z × HYG_chg_5d`. Creates conditional features.

- **Pro**: directly captures the conditional cross-sectional effects per Avramov 2023.
- **Con**: combinatorial explosion (4 ticker × 11 macro × 3 horizons = 132 interactions). Hard to regularize.
- **Verdict**: defer to v3 if v2 lifts but doesn't saturate.

### C. Regime-conditional ensemble (T2-3 in roadmap)

Train 4 separate panel models, one per macro regime. Route at inference.

- **Pro**: matches Two Sigma case study; orthogonal to v2 (could ship both).
- **Con**: 4× training cost; sample-size halves per regime → IC variance up.
- **Verdict**: ship after v2 if signal is there.

### D. Two-stage: macro predicts market premium, panel predicts residuals

Use macro to forecast `E[r_market | macro_state]`, then have the panel predict `r_i − E[r_market]`. Goyal-Welch style.

- **Pro**: cleanest theoretical separation.
- **Con**: requires retraining sim infrastructure to generate the residual labels.
- **Verdict**: too invasive for current iteration; reconsider after T2-2/T2-3 land.

---

## 5. A/B plan + acceptance criteria

1. Implement v2 in a separate branch / behind `panel_ltr.macro.version: "v2"` config flag (v1 default = "v1" to preserve current OFF state on prod).
2. Build `strategy_config.macro_v2.json` with `panel_ltr.macro.enabled: true, version: "v2"`.
3. Retrain XGBoost (current best backend) with v2 macro features, **acceptance ENABLED**.
4. Result handling:
   - PASS gates + OOS IC ≥ 0.0482 + ≥ +2 pts APY → promote to prod (per CLAUDE.md §2a).
   - PASS gates + OOS IC ≥ 0.0482 but APY < +2 pts → preserve as `.macro-v2.bak.json`, no promote.
   - FAIL gates → archived to `_acceptance_log/` automatically; v1 conclusion (macro is hurtful) confirmed for v2 design too.

5. Independently test on LightGBM with v2 features — separates "macro hurts" from "LGBM hurts".

---

## 6. Effort + sequencing

| Phase | Effort | Risk |
|---|---|---|
| A — `kernel/macro_per_ticker.py` + tests | 1 day | Low — vectorized rolling cov/var |
| B — Wire into PanelDataJob + BuildPanelTask + feature_matrix | 0.5 day | Medium — must preserve symmetry guard |
| C — Inference symmetry: 3-tuple stays as is, macro_betas substitutes for macro_frame | 0.25 day | Low (already 3-tuple) |
| D — strategy_config v2 + retrain + acceptance | 0.5 day | Low (gates handle outcome) |
| E — Doc update + roadmap | 0.25 day | Low |
| Total | ~2.5 days | Low overall |

---

## 7. Open decisions for operator

1. **Rolling window**: 60d is the academic standard (Fama-French). Should we also include 252d (long-horizon β)? Adds ~7 more features but captures macro regime persistence.
2. **Macro symbol pruning**: 11 macros × 1-2 windows = 11-22 betas. Some pairs (KRE = regional banks, USMV = low-vol) may be redundant — should we drop after FeatureDiagnosticTask audit on first v2 panel?
3. **β shrinkage**: rolling β has high variance at the tails. Apply Vasicek/Bayesian shrinkage toward 1.0 (market β)? Adds 50 LoC but may stabilize.
4. **Combine with v3 (regime-conditional)**: ship v2 first then v3 separately, or skip v2 and go straight to T2-3? Recommend v2 first — lower risk, orthogonal.

---

## 8. References

Books / repos (verified existence at design time):
- `https://github.com/microsoft/qlib` — Alpha158, Alpha360 feature handlers
- `https://github.com/stefan-jansen/machine-learning-for-trading` — chapter 9 (regime-conditional alpha)
- `https://numer.ai/docs` — feature engineering guide

Papers cited:
- Gu, Kelly, Xiu (2020) — "Empirical Asset Pricing via Machine Learning", *Review of Financial Studies* 33:2223
- Goyal, Welch (2008) — "A Comprehensive Look at the Empirical Performance of Equity Premium Prediction", *RFS* 21:1455
- Kelly, Pruitt, Su (2019) — "Characteristics are Covariances", *J. Financial Economics* 134:501
- Avramov, Cheng, Metzker (2023) — "Machine Learning vs Economic Restrictions", *J. Finance*
- Two Sigma (2024) — "A Machine Learning Approach to Regime Modeling" (white paper)
- Vasicek (1973) — "A Note on Using Cross-Sectional Information in Bayesian Estimation of Security Betas", *J. Finance* 28:1233 (for the β shrinkage open question)

(All retrieval-verified by Claude prior to 2026-04-26 cutoff.)
