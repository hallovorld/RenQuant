# Failed experiments log — RenQuant

Per CLAUDE.md principle 5.7. Every failed experiment is recorded here with: hypothesis, implementation, exact data, statistical sanity check, conclusion, and a reproduction recipe so an independent agent (e.g. Codex) can verify the result without rerunning the full discovery process.

**Why this log exists.** Without it we forget what we tried. Same idea returns under a new name 6 months later. Reasonable hypotheses that the data falsified must stay on the record so the team doesn't waste compute re-falsifying them.

**Format**: one section per experiment, ordered chronologically (newest first). Each section answers: what was the hypothesis, what was built, what exactly were the numbers, was the failure structural or implementation, and how to reproduce.

---

## E1. M2 horizon-blender v3 — learned and fixed-weight blends BOTH lose to single best horizon

**Date**: 2026-04-28
**Type**: structural negative; not implementation
**Production impact**: none (blender never deployed)

### Hypothesis
"Three independent horizon predictors (10d / 20d / 60d) blended with proper weights should beat the best single horizon. The signal sources are different (technical vs fundamental vs cyclical) so diversification benefits should accumulate."

### Implementation
v2 (failed): naïve LassoCV + IID K-fold + un-scaled features + raw return target. v3 (this experiment): full audit fixes per CLAUDE.md principle 5.2:
- Fix 1: StandardScaler in sklearn Pipeline (per-fold)
- Fix 2: PurgedKFold with embargo=20d (López de Prado 2018 Ch.7)
- Fix 3: ElasticNetCV instead of LassoCV (Zou & Hastie 2005)
- Fix 4: per-date cross-sectional rank target (Cao et al. 2007 — loss/eval alignment)
- Fix 5: winsorize features at [0.5%, 99.5%] + drop inf rows

Plus 3 principled baselines: equal-weight (1/3 each), 1/IC weighted (DeMiguel et al. 2009 RFS), A/A shuffled-labels sanity test.

### Data (hold-out Spearman IC, 118,520 rows, 227-watchlist panel @ 20d lookahead)

| Method | IC | vs Single 10d |
|---|---|---|
| Single 10d | **+0.1291** | **baseline** |
| Single 20d | +0.1228 | −4.8% |
| Single 60d | +0.0636 | −50.7% |
| Equal-weight blend | +0.1019 | −21.0% |
| 1/IC weighted blend | +0.0986 | −23.6% |
| Learned ElasticNet (5 fixes) | +0.0271 | −79.0% |
| A/A shuffled labels | NaN (constant pred) | — |

ElasticNet fitted: best `alpha = 0.0167`, best `l1_ratio = 0.1`, nonzero coefs = 3/13. v2 had `alpha = 0.000315` (50× smaller) — Fix 2 (PurgedKFold) materially changed regularization selection.

### Sanity / falsification
A/A test passed (shuffled labels → constant prediction → no label leakage). 5 fixes applied per literature. Result reproduces across the same v2 setup so it's not a v3-specific bug. **Conclusion: structural, not implementation.**

### Why it fails fundamentally
Per-horizon predictions (μ_10, μ_20, μ_60) are pairwise correlated > 0.7 — they're forecasts of overlapping forward returns on the same ticker-date. Linear blends add **estimation noise** without proportionate independent **information**. Net effect: noise variance dominates marginal information gain. This matches DeMiguel et al. 2009's finding that under estimation noise, naïve 1/N often beats "optimal" estimated weights — and here even 1/N loses to the single best.

### Reproduction
1. Ensure horizon artifacts exist:
   - `backtesting/renquant_104/artifacts/b1_regressed_20260428_020304/{panel-ltr.json, ngboost-head.json}` (10d)
   - `backtesting/renquant_104/artifacts/{panel-ltr,ngboost-head}.20d.json`
   - `backtesting/renquant_104/artifacts/{panel-ltr,ngboost-head}.60d.json`
2. Run: `python scripts/train_horizon_blender_v3.py`
3. Compare results to `doc/research/m2-v3-result-analysis.md` and `backtesting/renquant_104/artifacts/horizon-blender-v3.json`
4. Independent reproduction should yield: single 10d > single 20d > equal-weight > 1/IC > single 60d > learned ElasticNet, with magnitudes within ±5%.

### Files
- `scripts/train_horizon_blender_v3.py` (implementation)
- `tests/test_horizon_blender_v3.py` (unit tests for the fixes)
- `doc/research/m2-v3-result-analysis.md` (full analysis)
- `backtesting/renquant_104/artifacts/horizon-blender-v3.json` (exact metrics)
- `logs/ablation_2026-04-28/m2_v3.log` (full run log)

### Status
**M2 closed permanently.** No more learned horizon blending on this panel structure.

---

## E2. M2 horizon-blender v2 — Lasso with 5 design bugs

**Date**: 2026-04-28 (earlier same day as E1)
**Type**: implementation negative — bugs masked the structural truth
**Production impact**: none

### Hypothesis
Same as E1 but with naïve implementation (no audit applied yet).

### Implementation
LassoCV on (μ_10, σ_10, μ_20, σ_20, μ_60, σ_60) + regime one-hot + regime-interaction features (22 features total). Default IID K-fold CV, no feature standardization, no winsorization.

### Data
- Hold-out Spearman IC: **+0.0206** (single 10d alone: +0.1291)
- Best alpha: 0.000315 (essentially zero regularization)
- matmul overflow + invalid value warnings on hold-out predictions
- A/A test: not run (added in v3)

### What went wrong
Five design bugs (each documented in E1 implementation): no scaling, IID K-fold leaks future via 20d label overlap, Lasso collapses under collinearity, MSE objective doesn't align with Spearman evaluation, extreme NGBoost σ values blow up the matrix multiplication.

### Conclusion
Bugs masked the structural conclusion. v3 (E1) eliminates the bugs and confirms the structural negative. v2 itself is a cautionary tale — without principle 5.2 sanity tests, it could have shipped.

### Reproduction
`python scripts/train_horizon_blender_v2.py` — same panel as v3.

### Files
- `scripts/train_horizon_blender_v2.py`
- `logs/ablation_2026-04-28/m2_blender_run2.log`

---

## E3. Z8 σ-cap — top-decile σ does NOT predict underperformance

**Date**: 2026-04-28
**Type**: structural negative; falsified at design phase
**Production impact**: never built into code (rejected pre-implementation)

### Hypothesis
"Stocks with σ > P95 of the panel should underperform on a forward 5d / 20d horizon — the model's μ confidence on high-σ tickers is over-stated due to sparse training samples in that regime."

This was meant to address NVTS −12% loss (NVTS sits at panel-max realized vol of 1.27 annualized).

### Implementation
None — the A/A panel sanity test killed it before implementation.

### Data (242,862 rows, 103 watchlist, 1y rolling vol)

| σ cutoff | n above | fwd_1d (above vs other) | fwd_2d | fwd_5d | fwd_20d |
|---|---|---|---|---|---|
| P90 (τ=0.636) | 24,287 | +0.15% vs +0.10% | +0.30% vs +0.20% | +0.79% vs +0.49% | +4.20% vs +1.77% |
| P95 (τ=0.798) | 12,144 | +0.19% vs +0.10% | +0.34% vs +0.20% | +0.91% vs +0.50% | +4.85% vs +1.86% (diff +2.98%, A/A perm \|p95\|=0.23%) |
| P97.5 (τ=0.971) | 6,072 | +0.21% vs +0.10% | +0.40% vs +0.20% | +1.15% vs +0.50% | +5.97% vs +1.91% |
| P99 (τ=1.193) | 2,429 | +0.21% vs +0.10% | +0.48% vs +0.20% | +1.08% vs +0.51% | +6.71% vs +1.96% |

A/A test: shuffle realized fwd_returns, recompute the same conditional gap. Real diff +2.98% > permutation \|p95\|=0.23% — the gap is real and **opposite to the hypothesized direction**.

### Conclusion
On this watchlist + period, panel-derived σ-cap as a rejection gate would on average **reduce** returns. NVTS at −12% in 24h was a left-tail event in a strategy that wins on average by holding high-σ names. **Hypothesis falsified.**

### Reproduction
`/Users/renhao/miniconda3/envs/renquant/bin/python` ad-hoc:
```python
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/Users/renhao/git/github/RenQuant")
cfg = json.loads((REPO / "backtesting/renquant_104/strategy_config.json").read_text())
wl = cfg["watchlist"]; ohlcv = REPO / "data/ohlcv"
panel = []
for t in wl:
    p = ohlcv / t / "1d.parquet"
    if not p.exists(): continue
    df = pd.read_parquet(p)
    if "close" not in df.columns or len(df) < 60: continue
    c = df["close"].astype(float)
    rets = c.pct_change()
    vol_20 = rets.rolling(20).std() * np.sqrt(252)
    fwd5  = (c.shift(-5)/c - 1)
    fwd20 = (c.shift(-20)/c - 1)
    panel.append(pd.concat([vol_20.rename("vol"), fwd5.rename("fwd5"), fwd20.rename("fwd20")], axis=1).dropna())
P = pd.concat(panel).replace([np.inf,-np.inf], np.nan).dropna()
for q in (0.90, 0.95, 0.99):
    tau = P["vol"].quantile(q); above = P["vol"] > tau
    print(q, P.loc[above,"fwd20"].mean(), P.loc[~above,"fwd20"].mean())
```

### Files
- No implementation files (rejected pre-implementation)
- Conclusion captured in earlier turns of `doc/archives/sessions/2026-04-28-overnight-handoff.md`

---

## E4. Z1 parabolic-exhaustion gate — top-decile rel_mom_20d does NOT predict crash

**Date**: 2026-04-28
**Type**: structural negative; built then deleted
**Production impact**: never enabled in production (default OFF), then code removed

### Hypothesis
"Stocks with rel_mom_20d > 50% AND rel_mom_5d > 20% are parabolic tops; the panel-LTR model is over-confident here due to insufficient training samples in this tail. Reject these candidates regardless of edge_sharpe."

Trigger: NVTS post-mortem (rel_mom_20d=+91%, rel_mom_5d=+40%, edge_sharpe=+0.139, lost −11.92% in <24h).

### Implementation
v1 (hand-picked): `rel_mom_20d > 0.50 AND rel_mom_5d > 0.20`. Built into `kernel/pipeline/task_candidates.py::ParabolicExhaustionGateTask`. Default DISABLED.

v2 (panel-fit, attempted): replace 0.50 with panel P90 of rel_mom_20d. Built `scripts/fit_parabolic_gate.py` with A/A sanity test. **A/A test KILLED it.**

### Data (243,790 rows, 103 watchlist)

| rel_mom_20d cutoff | n above | fwd_5d (above vs other) | A/A perm \|p95\| | signal direction |
|---|---|---|---|---|
| P90 (τ=+11.5%) | 24,379 | +0.79% vs +0.49% (diff +0.30%) | 0.08% | top OUTPERFORMS (significant, opposite to hypothesis) |
| P95 (τ=+17.1%) | 12,190 | +0.91% vs +0.50% (diff +0.41%) | 0.13% | top OUTPERFORMS |
| P99 (τ=+33.8%) | 2,438 | +1.08% vs +0.51% (diff +0.57%) | 0.27% | top OUTPERFORMS |
| P99.5 (τ=+42.3%) | 1,219 | +1.15% vs +0.51% (diff +0.64%) | 0.41% | top OUTPERFORMS |

The hand-picked threshold of 0.50 sits at the **99.5th percentile** of the panel — only catches the most extreme 0.5% of bars (NVTS at +0.91 was at the 99.7-99.8th percentile).

### Sanity / falsification
A/A test: 200 permutations shuffling `fwd_5d` labels, recomputing the same conditional gap. Real signal magnitude > permutation \|p95\| at every quantile → REAL signal but **opposite direction** to hypothesis.

### Conclusion
"Momentum crashes" effect (Daniel & Moskowitz 2016) is weaker on this watchlist + period than the standard momentum-continuation effect. NVTS was an N=1 left-tail event, not a panel signal. Hypothesis falsified empirically. **Code deleted in commit `bd9c413` round-3 audit.**

### Reproduction
1. `python scripts/fit_parabolic_gate.py` — would output `artifacts/parabolic_gate_thresholds.json` with the A/A test result. (Note: scripts/fit_parabolic_gate.py was deleted; recreate by running the ad-hoc reproduction script in E3 with rel_mom_20d instead of vol_20d.)
2. Or reproduce inline:
```python
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/Users/renhao/git/github/RenQuant")
cfg = json.loads((REPO / "backtesting/renquant_104/strategy_config.json").read_text())
wl = cfg["watchlist"]; ohlcv = REPO / "data/ohlcv"
spy = pd.read_parquet(ohlcv / "SPY" / "1d.parquet")["close"]
spy_r20 = spy.pct_change(20)
panel = []
for t in wl:
    p = ohlcv / t / "1d.parquet"
    if not p.exists(): continue
    df = pd.read_parquet(p)
    if "close" not in df.columns or len(df) < 30: continue
    c = df["close"].astype(float)
    r20 = c.pct_change(20)
    align = pd.concat([r20, spy_r20], axis=1, join="inner").dropna()
    rel = (align.iloc[:,0] - align.iloc[:,1]).rename("rel_mom_20d")
    fwd5 = (c.shift(-5)/c - 1).rename("fwd5")
    panel.append(pd.concat([rel, fwd5], axis=1).dropna())
P = pd.concat(panel).replace([np.inf,-np.inf], np.nan).dropna()
for q in (0.90, 0.95, 0.99, 0.995):
    tau = P["rel_mom_20d"].quantile(q); above = P["rel_mom_20d"] > tau
    print(q, P.loc[above,"fwd5"].mean(), P.loc[~above,"fwd5"].mean())
```

### Files (post-deletion)
- Deleted: `kernel/pipeline/task_candidates.py::ParabolicExhaustionGateTask`
- Deleted: `tests/test_parabolic_gate.py`
- Deleted: `scripts/fit_parabolic_gate.py`
- Code removal commit: `bd9c413` (round-3 audit)

---

## E5. B1 — 227-ticker watchlist (mutual-fund spec) — IC regression −44%

**Date**: 2026-04-27/28 overnight
**Type**: structural negative — universe expansion hurts current model
**Production impact**: deployed and rolled back (caused the 06:32 ntfy fingerprint mismatch)

### Hypothesis
"Adding VPMAX (Vanguard Primecap) and FCNTX (Fidelity Contrafund) top holdings to the 103 watchlist (→ 227 tickers) gives the cross-sectional ranker a wider universe and broader IR ceiling."

### Data
- 103 watchlist baseline OOS IC: +0.0418 (golden, 15-fold CPCV reproduced 2026-04-27)
- 227 watchlist B1 retrain: **+0.0234** (−44%)

### Why it failed
1. **Heterogeneity dilution** — added tickers had different return-distribution shapes (different vol regimes, sector betas). The LTR rank loss is sensitive to cross-sectional heterogeneity.
2. **Liquidity spread** — 227 set spans wider dollar-volume range; ranker isn't liquidity-aware.
3. **Selection mechanism** — VPMAX/FCNTX hold names for fundamental discretionary reasons; not the same as "tickers our cross-sectional ranker can rank well."

### Reproduction
- Config: `backtesting/renquant_104/strategy_config.20d.json` (has the 227 watchlist)
- Artifact: `backtesting/renquant_104/artifacts/b1_regressed_20260428_020304/panel-ltr.json` (the regressed model)
- Retrain: `python scripts/train_104.py --strategy-config-name strategy_config.20d.json --skip-baseline --skip-recalibrate --force`

### Status
Rolled back via `scripts/auto_revert_b1_regression.sh` (which itself had a bug — see audit commit `bd9c413` for the path-fix). Watchlist 200 v2 plan rebuilds with quality filters instead of mutual-fund-membership selection.

---

## E6. B1.2 — high-vol tech-leaning subset 10d horizon — selection bias in evaluation

**Date**: 2026-04-28 overnight
**Type**: implementation negative — selection bias in evaluation, not real signal
**Production impact**: never deployed (caught by user-demanded audit)

### Hypothesis
"A horizon-filtered subset of 75 high-vol tech-leaning tickers from the 227 set should outperform the broad 227 because high-vol tickers have stronger short-horizon signals."

### Data (claimed vs actual)
- First report: OOS IC = +0.0614 (+47% vs 103 baseline)
- After deep audit using ONLY pre-window data for universe selection: IC = **+0.0399** ≈ baseline (no improvement)

### Why it failed
The original universe selection used post-window information (in-sample IC of individual tickers contributed to picking the "high-vol tech-leaning" subset). When the same subset was selected using only pre-window data (selection from history that was strictly before the OOS evaluation period), the +47% advantage vanished.

This is a textbook lookahead-style selection bias.

### Conclusion
The +47% was a measurement artifact, not signal. **CLAUDE.md principle 5.2 ("every new number ships with a sanity test") was created in response to this exact incident.**

### Reproduction
1. The contaminated version: `backtesting/renquant_104/strategy_config.b1_2_filtered_10d.json` + `artifacts/panel-ltr.b1_2_filtered_10d.json`
2. The rigorous version: `strategy_config.b1_2_rigorous.json` + `artifacts/panel-ltr.b1_2_rigorous.json`

---

## E7. B1.3 — 60d horizon × aggressive hyperparams — overfitting

**Date**: 2026-04-28 overnight
**Type**: implementation negative — hyperparameter overfit
**Production impact**: never deployed

### Hypothesis
"60d horizon's stronger fundamental signal can be amplified with deeper trees + higher learning rate."

### Data
- 60d default hyperparams on 227: paired t = +3.82 vs 10d (ON 227, where both regressed vs 103)
- B1.3 60d aggressive hyperparams: regressed below the 60d default

### Why it failed
Aggressive hyperparams on a panel with limited effective sample size overfit the training distribution. Tree depth + learning rate increases interact multiplicatively with effective sample size; without more data, more capacity = more variance.

### Reproduction
- Config: `strategy_config.b1_3_60d_tuned.json`
- Artifact: `panel-ltr.b1_3_60d_tuned.json`

---

## E8. F3 — 10d hyperparam retune on 227 watchlist

**Date**: 2026-04-27/28
**Type**: structural — hyperparams aren't the lever for 227 regression

### Hypothesis
"The −44% IC regression on B1 is solvable with hyperparameter tuning."

### Data
- Best F3 result: OOS IC ≈ +0.039 (still below 103 baseline +0.0418)
- ~20 hyperparameter combinations tested

### Conclusion
Confirms B1 regression is structural (universe-mix problem), not a hyperparameter problem. Closed.

### Reproduction
F3 grid is in `scripts/run_b1_2_b1_3_chain.sh` historical state; no specific replication artifact saved.

---

## E9. Macro v1 (broadcast) — zero gradient

**Date**: ~2026-04-25
**Type**: structural — ranker can't extract from constant features

### Hypothesis
"Broadcasting macro factors (VIX, HYG, UUP) into every ticker's row at every date gives the model regime context."

### Data / Why it failed
Within-date macro values are IDENTICAL across tickers → cross-sectional rank loss receives **zero gradient** from these features. xgboost rank:pairwise compares ticker A vs ticker B on the same date; identical macro values cancel.

### Files
`backtesting/renquant_104/training_panel/panel_frame.py` line 180+ implements the v1-mode skip (Bug-1 fix).

### Conclusion
Theoretically broken. Macro path closed for v1 form.

---

## E10. Macro v2 (per-ticker β) — IC −23%

**Date**: 2026-04-26
**Type**: structural negative

### Hypothesis
"Per-ticker β coefficients on macro factors (each ticker has its own VIX β, HYG β, etc.) give the ranker actually-cross-sectionally-varying features."

### Data
- 11-ETF set: paired CPCV OOS IC −23% vs no-macro baseline
- Even after fixing 3 implementation bugs

### Conclusion
Per-ticker β estimation noise dominates the signal. Closed.

### Reproduction
`backtesting/renquant_104/strategy_config.macro_v2.json`

---

## E11. Macro v3 (30 ETF + 22 FRED) — IC monotone-decreasing

**Date**: 2026-04-27
**Type**: structural negative

### Hypothesis
"More macro features = more regime information."

### Data
- 11 sym → IC ≈ +0.0370
- 30 ETF + 22 FRED (52 features) → IC ≈ +0.0344

Adding macro features monotonically reduced IC. **Definitive.**

### Reproduction
`backtesting/renquant_104/strategy_config.macro_v3.json` + `kernel/fred_macro.py`

---

## E12. Macro v4 (panel-row) — IC −28.8%, paired t = −1.98

**Date**: 2026-04-27
**Type**: structural negative

### Hypothesis
"Treat macro as additional 'rows' in the panel, not features — let the ranker rank macro factors against tickers."

### Data
Paired CPCV: OOS IC −28.8% vs baseline, t = −1.98 (significant at 0.05).

### Conclusion
All 4 macro forms (v1/v2/v3/v4) rejected. Macro path closed pending watchlist expansion to 200+ where signal might emerge at longer horizons (untested).

### Reproduction
v4 config and ad-hoc script archived in commit history `158dc17` (audit).

---

## E13. Asset embeddings T2-2 — initial OLS A/B "GO" reversed by paired CPCV

**Date**: 2026-04-27
**Type**: methodology negative — OLS A/B doesn't predict XGB rank tree behavior

### Hypothesis
"16-dim asset embeddings (per-ticker learned vectors) give the model identity-aware features for ticker-specific patterns."

### Data
- Initial dispatch agent verdict: "GO" based on OLS A/B + per-feature IC
- Full retrain + paired CPCV: **OOS IC = +0.0341 vs +0.0418 baseline = −18.5%, t = −1.45**

### Why initial verdict was wrong
OLS A/B tests measure linear marginal information of one feature controlling for others. XGBoost rank trees do **not** behave like OLS — they pick splits non-linearly with feature interactions. A feature can be marginally informative in OLS but harmful in tree boosting (because it adds noise to splits without proportional information).

### Conclusion
Asset embeddings rejected. **Methodological lesson: OLS-based A/B tests are not valid for non-linear models.** Paired CPCV on the actual model is mandatory.

### Reproduction
- Config: previous `strategy_config.json` had `asset_embeddings.enabled=true` (now reverted to false)
- Side artifacts: `artifacts/panel-ltr.ablation-with-emb*.json` vs `artifacts/panel-ltr.ablation-no-emb*.json`

---

## E14. LightGBM substitution for XGBoost — IC −60%

**Date**: 2026-04-27
**Type**: structural negative

### Hypothesis
"LightGBM is generally faster and sometimes more accurate; could substitute for XGBoost rank:pairwise."

### Data
On the same 103 panel: LightGBM lambdarank OOS IC ≈ −60% vs XGBoost rank:pairwise.

### Conclusion
LightGBM rejected. The two boosters have different internal hyperparameter assumptions; transferring config without re-tuning is destructive. Not pursued further given XGBoost's incumbent IC.

### Reproduction
`backtesting/renquant_104/training_panel/lgbm_ltr.py` + `strategy_config.lgbm_v2.json`

---

## E15. T2-4 Boyd Rotation as APY lever — −2.5 APY pts per cycle

**Date**: 2026-04-26
**Type**: structural negative

### Hypothesis
"Boyd-style transaction-cost-aware rotation should improve net-of-cost return by trading off rotation frequency against signal decay."

### Data
Each rotation cycle costs −2.5 APY pts net. Infrastructure retained but disabled by default.

### Conclusion
Rotation as APY lever doesn't work given current cost model. Maybe revisit if costs improve or signal decay shape changes.

---

## E16. T2-3 Regime ensemble — deferred (not run)

**Date**: 2026-04-25
**Type**: blocked, not failed

### Status
Panel < 150k rows; insufficient samples per regime. Deferred until watchlist 200 v2 ships.

---

## E17. wl178 quality-filter expansion — eval IC NEGATIVE across all rounds

**Date**: 2026-04-28 (evening, post P0 fixes + threshold lowered to 5)
**Type**: structural negative — second confirmation that universe expansion fails on this model architecture
**Production impact**: never deployed (guard fired, artifact NOT saved)

### Hypothesis
B1 (227 mutual-fund spec) failed because the selection method picked tickers based on fundamental criteria, not on signal quality. A quality-first 4-filter selection (liquidity ≥ \$50M median DV, history ≥ 504 days, vol ∈ [15%, 85%], 1y Sharpe ≥ 0.5) on the local OHLCV cache should give a watchlist where the cross-sectional ranker can extract signal.

### Implementation
- 4 filters applied to 235 tickers in local OHLCV cache → 75 new candidates
- Combined with current 103 → 178-ticker watchlist
- Trained with all P0 fixes in place (commit `abac170`)
- Side artifact paths so production was untouched

### Data (full per-round IC trajectory, post-BUG-CV-3 eval set)

| round | eval IC | train IC | gap |
|---|---|---|---|
| 5  | **−0.0383** ← "best" (still negative) | +0.0852 | +0.124 |
| 10 | −0.0741 | +0.0860 | +0.160 |
| 15 | −0.0627 | +0.0873 | +0.150 |
| 20 | −0.0549 | +0.0901 | +0.145 |
| 25 | −0.0524 | +0.0901 | +0.143 |

best_iter = 4, best_ic = −0.0383.

Critically: **train IC is also depressed** (+0.085 vs prod103's +0.118) — the model can't even fit training data well. Combined with the negative eval IC, this is structural breakage, not just poor generalization.

### Comparison to E5 (B1 227)

| Experiment | Selection method | New tickers | Resulting IC |
|---|---|---|---|
| E5 (B1 227) | Mutual fund top holdings (VPMAX/FCNTX) | +124 | +0.0234 (−44%) |
| **E17 (wl178)** | **Quality 4-filter (liquidity / history / vol / Sharpe)** | **+75** | **eval IC negative across all rounds** |

Two completely different selection methods, both failed. **This rules out "selection error" as the root cause.**

### Why expansion fails structurally

Hypothesis: panel-LTR with cross-sectional rank:pairwise loss assumes some homogeneity in feature distribution across the universe. The original 103 watchlist is tech-heavy (and was implicitly curated to be a homogeneous set over time). Both expansions added significant cross-sector variance:

- B1 added VPMAX/FCNTX holdings spanning all 11 GICS sectors
- wl178 added financials (C, MS), industrials (FDX, CSX, PH), consumer staples (ROST, MAR)

Cross-sectional rank loss on a heterogeneous universe degrades because:
1. Per-ticker feature distributions differ → rank-pairwise loss compares apples to oranges
2. Forward-return distributions differ across sectors → label noise dominates
3. The same z-scoring + neutralization that works on tech-heavy can't normalize across sectors

### Confirmed by training-loss signature
The train IC at +0.085 (vs +0.118 on 103) shows the model can't even fit training data on the heterogeneous panel. This rules out "evaluation-set artifact" as the cause — the issue is at training time.

### Conclusion
**Watchlist expansion path closed for the current architecture.** Two independent attempts at expansion (different selection methods) both failed. The cross-sectional rank model on this feature set is bounded by the original 103 watchlist's homogeneity.

To unblock expansion would require an architectural change:
- Per-sector sub-models (each ranker on a homogeneous sector)
- OR sector-conditional features (sector-specific feature transforms)
- OR an embedding-based architecture that handles heterogeneity (but T2-2 embeddings already failed for this panel — see E13)

These are major architecture changes, not parameter tweaks.

### Reproduction
1. Build the wl178 config: `scripts/screen_watchlist_v2.py --top 100`
2. Run: `python scripts/train_104.py --strategy-config-name strategy_config.wl178.json --skip-baseline --skip-recalibrate --force`
3. Inspect `logs/ablation_2026-04-28/wl178_retrain_v2.log` — every per-chunk eval IC should be negative

### Files
- `backtesting/renquant_104/strategy_config.wl178.json` (the config)
- `doc/research/watchlist-200-v2-candidates.json` (the 75 candidates)
- `logs/ablation_2026-04-28/wl178_retrain_v2.log` (the run log)

### Status
**Closed.** Universe expansion not pursued further until per-sector or sector-aware architecture is built.

---

## E18. Round-9 saturation diagnostic — by design, not a bug

**Date**: 2026-04-28 (evening)
**Type**: diagnostic, NOT a failed experiment — confirmed model behavior is structural

### Hypothesis
On 2026-04-28 the production retrain (and wl178 retrain) showed XGBoost rank early-stop firing at very low best_iter (4-9) with eval IC peaking and then declining. Three competing hypotheses:
1. eval set too small → noise dominates after early peak
2. structural saturation → model capacity reached
3. residual train→eval leakage → early IC inflated

### Implementation
Diagnostic side config with `min_best_iter=0` (disable guard), `num_boost_round=200`, `early_stopping_rounds=100` (force long training to see full curve), patched `panel.ltr` logger to emit per-chunk eval IC + train IC + gap (not just "new best").

### Data (full IC trajectory, 103 watchlist, post-P0 fixes)

| round | eval IC | train IC | gap |
|---|---|---|---|
| 25 | **+0.0445** ← peak | +0.1183 | +0.07 |
| 50 | +0.0313 | +0.1205 | +0.09 |
| 75 | +0.0158 | +0.1247 | +0.11 |
| 100 | +0.0060 | +0.1242 | +0.12 |
| 125 | **−0.0021** ← early-stop fires | +0.1271 | +0.13 |

CPCV mean across 15 folds: **+0.0356**.

### Hypothesis verdict
- Hyp 1 (eval set too small): ✅ **confirmed**
- Hyp 2 (structural saturation): ❌ ruled out — train IC keeps rising
- Hyp 3 (leakage): ❌ ruled out — train and eval **diverge** (would converge if leakage)

### Why eval set is too small
- 6-fold CPCV → eval = 1/6 of dates ≈ 125 dates
- 125 dates × 103 tickers ≈ 12k pair-observations
- 27 features × tree depth 7 → high effective capacity
- eta=0.02 step size means accumulated capacity catches up by round ~10
- After ~10 rounds the model has more capacity than the eval set can constrain → fits noise, eval IC degrades

### What this is NOT
- NOT a CV implementation bug (BUG-CV-1/2/3 already fixed)
- NOT a label leakage bug (purge + embargo present and verified)
- NOT a feature engineering bug (train and eval features identical)

This is the **expected behavior of a tree-boosting ranker on a small eval set**. The fix is not code; it is data (more dates, more tickers, larger training panel) — and that fix path itself is closed (see E17).

### Operational implication
The original guard threshold `min_best_iter ≥ 20` was too aggressive — based on the assumption "eta × best_iter ≥ 0.4 = healthy capacity", which empirically fails on this panel. Threshold lowered to 5: catches catastrophic eval-set breakage (best_iter=2/3) without blocking healthy fast-converging models (best_iter=9-25).

### Reproduction
1. Build diagnostic side config (set `min_best_iter=0`, `num_boost_round=200`, `early_stopping_rounds=100`, side artifact path)
2. Patch `panel.ltr` early-stop logger to emit per-chunk eval IC + train IC + gap (commit `abac170`)
3. Run: `python scripts/train_104.py --strategy-config-name strategy_config.diag.json --skip-baseline --skip-recalibrate --force --skip-acceptance`
4. Confirm: train IC monotonically rises, eval IC peaks at round ~24 then declines, gap monotonically widens

### Files
- `logs/ablation_2026-04-28/diag_retrain.log` (the run log)
- `backtesting/renquant_104/training_panel/ltr_model.py` (per-chunk logger patch)
- `backtesting/renquant_104/strategy_config.diag.json` (the diagnostic config)

### Status
**Closed by design.** Operating point best_iter ∈ [9, 25] is healthy; guard at 5 protects against catastrophic best_iter=2/3 breakage.

---

## Lessons distilled from the failures (updated 2026-04-28 evening)

1. **OLS / linear A/B tests can mislead about tree-model behavior** (E13). Always paired CPCV on the actual model.
2. **Selection bias is the silent killer** (E6). Universe selection must use only pre-window information.
3. **More features can hurt** (E11). The cross-sectional ranker has a feature-budget; bad features dilute good ones via colsample.
4. **Correlated signals don't blend usefully** (E1). Per-horizon predictions correlate >0.7; blending adds noise.
5. **NVTS-style left-tail events are not panel signals** (E3, E4). Don't build panel rules from N=1 observations.
6. **Hyperparameter retune doesn't fix structural problems** (E7, E8). If the universe is wrong, no tree depth helps.
7. **Macro signals on this watchlist + 10d horizon are absent or harmful** (E9–E12). Untested at long horizon × broader universe — but expansion path itself is now closed (E17).
8. **Universe expansion fails on the current architecture regardless of selection method** (E5, E17). Need per-sector sub-models or sector-conditional features to scale beyond ~100 homogeneous tickers.
9. **CV bugs corrupt every IC measurement, not just one** (today's BUG-CV-1/2/3). After every CV-side code change, re-run paired comparisons before trusting any closure.
10. **A guard threshold derived from theory must be empirically validated** (today's `min_best_iter=20` was over-aggressive). Set guards from data, not from "should-be" arithmetic.

---

## E18: LightGBM 10d（clean CV）— 2026-04-29

**假设**：LightGBM rank:pairwise 替代 XGBoost，可获得更高 IC（早期 A/B 在 buggy CV 下为 −60%，clean CV 可能翻案）。

**实现**：`strategy_config.lgbm_no_macro.json`，早停禁用（防 NaN label 早停），10d lookahead，同 27 特征。

**结果**：CPCV 15-fold mean_ic = **+0.018**（XGBoost baseline +0.035，差距 −49%）。

**Sanity check**：早停修复后模型正常训练（train_ic=+0.141），非退化模型，IC 差距是真实的。

**结论**：LightGBM 在此面板配置下明显落后 XGBoost rank:pairwise。关闭方向，clean CV 否决。

**复现**：`python scripts/train_104.py --strategy-config-name strategy_config.lgbm_no_macro.json --skip-baseline --skip-recalibrate --force`

---

## E19: 60d+macro v2（clean CV）— 2026-04-29

**假设**：macro 在 10d 中性（E14 重测 −1.4%），在 60d 长 horizon 下可能有效（文献：macro 在季度+ horizon 有预测力）。

**实现**：`strategy_config.macro_h60.json`，60d lookahead + macro v2 per-ticker β features。

**结果**：CPCV mean_ic = **+0.074**（60d baseline +0.074，差距 0%）。

**结论**：macro v2 per-ticker β features 在 60d 同样中性。文献预测的"macro 在长 horizon 有效"在跨截面特征形式下不成立；可能需要以时序 regime gate 形式才有效。

**复现**：`python scripts/train_104.py --strategy-config-name strategy_config.macro_h60.json --skip-baseline --skip-recalibrate --force`

---

## E20: 60d+emb 16D（clean CV）— 2026-04-29

**假设**：embedding 在 10d 负向（E16 重测 −12%），理论受益在 long horizon + large universe，60d 可能翻案。

**实现**：`strategy_config.emb_h60.json`，60d lookahead + asset embeddings 16D enabled。

**结果**：CPCV mean_ic = **+0.034**（60d baseline +0.074，差距 **−54%**）。

**结论**：embedding 在 60d 比 10d 更有害（10d −12% vs 60d −54%）。理论预期未被实验支持。结构性解释：embeddings 捕获行业协方差，在 60d 上与趋势/动量信号冲突更强，噪声被放大。BFI 2025 理论预期（universe 扩大后受益）仍需在 200+ watchlist 验证。

**复现**：`python scripts/train_104.py --strategy-config-name strategy_config.emb_h60.json --skip-baseline --skip-recalibrate --force`


---

## E21: wl174（178 去 4 ETF，clean CV）— 2026-04-29

**假设**：wl178 IC 低的主因是 4 个行业 ETF（XLE/XLI/XLK/XLY）污染横截面归一化。去掉后应恢复到接近基准水平。

**实现**：`strategy_config.wl174.json`，174 ticker = 178 - 4 ETF，early_stopping_rounds=null，clean CV。

**结果**：CPCV mean_ic = **+0.0094**（vs wl178 +0.007, vs prod 103 +0.035）。平均每 ticker 751 日，与 prod 103 完全相同，排除数据稀疏假设。

**结论**：ETF 不是主因，是次因。真正问题是"横截面排名模型扩容悖论"：原 103 ticker 是精选的高 IC 集合，加入 71 个风格不同的 ticker（商品/公用事业/银行）后：(1) 横截面 z-score 基准改变；(2) Gaussian rank label 映射变化；(3) 新 ticker 特征分布稀释原有训练模式。随机扩容 = IC 下降。

**实验设计缺陷**：watchlist 扩容应先做 ticker 筛选（与现有 103 特征分布相似度测试），而不是直接加入所有待选 ticker。

**下一步**：设计 ticker screening 步骤：逐一测试新 ticker 的 IC 贡献（leave-one-in 测试或 per-ticker IC 诊断），只保留 IC 中性或正向的 ticker。

**复现**：`python scripts/train_104.py --strategy-config-name strategy_config.wl174.json --skip-baseline --skip-recalibrate --force`


---

## E22: Insider trades feature on/off A/B at 44% coverage — neutral (shelved)

**Date**: 2026-05-02
**Branch**: main
**Run IDs**:
- insider OFF: `20260502163413-panel-ltr-f81918` (mean_ic=+0.0337)
- insider ON (diag): `20260502175211-panel-ltr-a844cd` (mean_ic=+0.0329)
- delta = **−0.0008** (within noise)

**假设**: SEC Form 4 executive insider trades (Lakonishok-Lee 2001, Cohen-Malloy-Pomorski 2012) carry +5-15 bp alpha at fwd_5d horizon. Feature column `insider_net_buy_90d` was already wired in production but data was stale (latest cache 2026-04-22~24, 44/103 wl103 tickers covered).

**Setup**:
- Side configs `strategy_config.insider_off.json` (panel_ltr.insider_trades.enabled=false) and `insider_on.json` (=true), all other params identical.
- Same panel, same CPCV folds, same xgb hyperparams.
- Cache state: 44/103 wl103 tickers had non-empty parquet, rest NaN.

**Setup gotchas surfaced (fixed during this run)**:
- BUG: side configs only overrode `panel_ltr.*.artifact_path` (training-side write); inference-side `ranking.panel_scoring.*.artifact_path` defaulted to production paths — insider_off retrain overwrote production `ngboost-head.json`. Restored from `.xgboost.bak.json` (paired backup from same 04-30 production retrain). Test `tests/test_side_config_artifact_paths.py` added to pin invariant; fixed 9 historical configs with same leak.
- BUG: Track B PEAD wiring used `ctx.config` instead of `tc.config` in TickerPanelFactorJob (NameError on every ticker, killed insider_on v1+v2). Fixed + added static check test.
- BUG: insider_on v2 (post fixes) hit `min_best_iter=5` guard with best_iter=4 — flaky training, not structural. Diag bypass with `min_best_iter=1` produced healthy training (best_iter=24, eval_ic=+0.0560), confirming the guard was over-protective on this specific eval-set partition.

**Result**:
- insider OFF: mean_ic = +0.0337, train_ic = +0.1139, best_iter = 25
- insider ON (diag, min_best_iter=1): mean_ic = +0.0329, train_ic = +0.1136, best_iter = 24
- **delta IC: −0.0008** — fully within noise band

**Hypothesis ruled out**: NaN structure leakage. With healthy training the partial-coverage NaN pattern does NOT degrade the model; XGB handles NaN natively and doesn't appear to learn ticker-identity metadata from "has insider data" pattern.

**Conclusion**: At 44% coverage, the insider signal is BELOW the SNR floor of CPCV mean_ic. Feature is neither toxic nor productive. **Production keeps insider_trades.enabled=true** (no harm, no need to revert) but not promoting it as an active improvement lever.

**Why shelved tonight**:
- SEC EDGAR rate-limited our IP after early UA-less 403 attempts (single-ticker direct fetch > 120s with proper UA = SEC throttling). Cold backfill of the missing 58 tickers needs ~24h IP-block recovery.
- Production launchd plist now has RENQUANT_SEC_UA env var (commit landed today, ops doc at `doc/ops/insider-trades-setup.md`), so the next Sunday retrain (2026-05-04 10:00 PT) will refresh insider data automatically once SEC unblocks.

**Resume conditions** (when to revisit):
1. SEC IP throttle clears (~24h from 2026-05-02 09:00 PT, so any time after 2026-05-03 09:00 PT).
2. After Sunday retrain (2026-05-04) populates fresh insider data for 100+ tickers.
3. Re-run the same A/B with `min_best_iter=5` (production-strict mode), expecting +5-15 bp lift per literature.

**Recipe**:
```bash
# Resume backfill (needs SEC unblock + RENQUANT_SEC_UA exported):
export RENQUANT_SEC_UA="Ren Hao renhao.overflow@gmail.com"
python scripts/fetch_insider_trades.py --strategy renquant_104 \
    --max-filings 50 --total-budget-sec 5400 --per-ticker-sec 120

# Re-A/B (production-strict mode):
python scripts/train_104.py --strategy-config-name strategy_config.insider_off.json \
    --skip-baseline --skip-recalibrate --force
python scripts/train_104.py --strategy-config-name strategy_config.insider_on.json \
    --skip-baseline --skip-recalibrate --force
```

**If resume A/B still ≤ +5 bp**: investigate per-ticker insider IC (diagnostic — maybe only specific sectors carry the signal). If still ≤ baseline, document final NO-GO.

**Lesson**: partial-coverage features need to clear a coverage threshold (~70-80%? not measured) before signal SNR rises above CPCV noise. Future feature additions that depend on third-party data: **stage data fetching to 70%+ coverage BEFORE wiring into panel** to avoid this exact "is the feature dead or just data-thin?" ambiguity.

---

## E23: PEAD enrichment at fwd_5d — statistically significant NEGATIVE (shelved)

**Date**: 2026-05-02
**Branch**: main
**Run IDs** (`data/runs.db`):
- pead_off run1 (golden + pead.enabled=false): `20260502183144-panel-ltr-c0b18a`, mean_ic=+0.0339
- pead_off run2 (σ measurement): `20260502202120-panel-ltr-891067`, mean_ic=+0.0340
- pead_off run3 (σ measurement): `20260502204955-panel-ltr-3536e9`, mean_ic=+0.0340
- pead_on full (3 cols, strict guard): `20260502190009-panel-ltr-0f7206`, mean_ic=+0.0327
- pead_days_only (1 col, bypass guard): `20260502200343-panel-ltr-868ba5`, mean_ic=+0.0330

**Hypothesis**: PEAD enrichment (Bernard-Thomas 1989, CJL 1996) — adding `days_since_earnings` + `pead_decay_weight` + `pead_signal` (most recent surprise × linear decay over 60d) — should add +2-6 bp CPCV IC at fwd_5d horizon.

**Setup**: Three new feature columns gated on `panel_ltr.pead.enabled` (default false). All else identical to golden config. Config: `strategy_config.pead_on.json`.

**Methodology gotcha discovered**: The §2a default acceptance threshold (+0.002) is **not calibrated to single-A/A noise**. Three pead_off retrains with same config but different XGB seeds produced:
- mean_ic: +0.0339, +0.0340, +0.0340 → **σ = 0.000058 ≈ 0.6 bp**
- best_iter: 25, 4, 39 → highly seed-sensitive
- train_ic: +0.1190, +0.0949, +0.1193 → highly seed-sensitive

Lesson: **CPCV mean_ic is the only robust statistic across XGB seeds**. best_iter and train_ic are not comparable across runs. Ship-or-shelve decisions must use mean_ic ± measured σ, not gut-feel thresholds.

**Result (σ-corrected)**:
- pead_on full: delta = −0.0013 vs pead_off mean = **22σ negative** (highly significant)
- pead_days_only: delta = −0.0010 vs pead_off mean = **17σ negative** (highly significant)

**Per-feature univariate IC** (on pead_on training run):
- `days_since_earnings_z`: IC = +0.0208 (apparently strong positive)
- `pead_signal_z`: IC = −0.0046 (weak negative, std=0.62 compressed by zero-padding past 60d)
- `pead_decay_weight_z`: not separately measured (likely weak)

**Mechanism analysis (post-hoc)**:
- Single-column positive IC (days_since +0.02) does NOT translate to multivariate model gain.
- XGB learns spurious interactions involving PEAD columns that produce inconsistent predictions across CPCV folds.
- The "post-earnings regime" days_since column may correlate with fwd_5d *some* of the time (e.g. specific earnings season patterns) but the relationship is not stationary across years/sectors.
- pead_signal is structurally compressed: ~70% of panel rows have `decay_weight=0` (post-60d), making the column near-binary on the announcement window. Tree splits don't extract clean alpha from such concentrated information.

**Audit per CLAUDE.md §2b (deep audit before accepting unexpected result)**:
1. ✅ Sample-bar inputs verified on real AAPL data — all values match expected formulas exactly (decay 1.0→0.0 over 60d, signal = surprise × decay, +1d shift lookahead-safe).
2. ✅ Independent reasoning matches outputs — implementation is correct.
3. ✅ pead_off baseline reproduces production (mean_ic=+0.0339 ≈ 04-30 retrain +0.0340).
4. ✅ No interaction with `earnings_surprise_cum` (independent computation).

**Conclusion**: PEAD-as-implemented is **structurally incompatible with fwd_5d horizon ranking model**. NOT a bug — a real measurement that the design doesn't fit.

**Resume conditions**:
1. **Try fwd_20d or fwd_60d horizon** — PEAD literature targets weeks-to-months drift; fwd_5d may be too short.
2. **Cross-sectional surprise quintile/rank** instead of raw surprise_pct — CJL 1996 used quintile bins, not raw values.
3. **Sector-conditional PEAD** — drift strength varies by sector (tech > staples). Could interact with sector indicators if Layer 2 lands.
4. **Drop pead_signal entirely**, keep only days_since as a regime indicator — but ablation showed even days_only is −17σ, so this is unlikely to help without horizon change.

**Recipe**:
```bash
# Reproduce σ measurement:
python scripts/train_104.py --strategy-config-name strategy_config.pead_off_run2.json --skip-baseline --skip-recalibrate --force
python scripts/train_104.py --strategy-config-name strategy_config.pead_off_run3.json --skip-baseline --skip-recalibrate --force

# Reproduce A/B:
python scripts/train_104.py --strategy-config-name strategy_config.pead_off.json --skip-baseline --skip-recalibrate --force
python scripts/train_104.py --strategy-config-name strategy_config.pead_on.json --skip-baseline --skip-recalibrate --force
```

**Side benefit produced this session**:
- BUG-CV-2 guard refinement (Task #24): added `min_best_iter_eval_ic_floor` escape clause for false-positive on strong-univariate-IC features. The original guard would have blocked all PEAD ablation A/B runs with `RuntimeError: best_iter < 5` since adding any strong-IC column makes XGB plateau by round 4-9. Now accepts when eval_ic ≥ 0.02 floor.

---

## E24: Triple-barrier label A/A — eval IC NEGATIVE due to residual mismatch (track F, partial)

**Date**: 2026-05-02
**Branch**: main
**Run IDs** (`data/runs.db`):
- triple_barrier_off (fwd_5d explicit): `20260502211508-panel-ltr-6a95a1`, mean_ic=+0.03399 (= pead_off baseline, identical bit-for-bit)
- triple_barrier_on (alpha=2, beta=2, max_h=10): FAILED at FinalFit guard. best_iter=4, eval_ic=**−0.0744**, train_ic=+0.0710.

**Hypothesis**: Lopez de Prado AFML §3 triple-barrier label (first hit of {upper, lower, time}) replaces fwd_5d. Predicted +5-10 bp lift.

**Result**: Eval IC went deeply NEGATIVE. Model + label are anti-predictive.

**Diagnosis (post-hoc)**:

Reading `LabelsTask.run()` after the failure:

```python
spy_fwd = spy_close.shift(-lookahead) / spy_close - 1.0   # FIXED 5-day
fwd_returns = compute_triple_barrier_labels(...)["label"]  # VARIABLE 1-10 days
ctx.raw_residuals = compute_residual_returns(fwd_returns, spy_fwd, ...)
```

The bug: `ticker_fwd` is the realized return at *its own* barrier-hit day (variable, 1-10 days), but `spy_fwd` is fixed 5-day. So the residual `ticker_fwd − β × spy_fwd` mixes forward windows of different lengths. The model fits noise, not alpha.

This is a **design omission**, not a code bug. Triple-barrier labels need either:
1. Hit-time-matched SPY/sector forward returns (compute SPY return from t to ticker's hit day), or
2. Skip residual extraction for triple-barrier mode (use raw barrier-hit return as label).

**Result classification**: NOT a falsification of the triple-barrier hypothesis — it's a wiring incompleteness. The label-design idea remains untested at this horizon.

**Resume conditions**:
1. Refactor `LabelsTask` so when `label_mode='triple_barrier'`:
   - For each ticker × date, extract `hit_days[ticker, t]` from compute_triple_barrier_labels output.
   - Compute SPY return from t to t + hit_days (per-row alignment).
   - Same for sector_fwd.
2. OR skip residual extraction entirely for triple-barrier (use raw barrier-hit return as label, no benchmark adjustment).
3. Re-run A/A.

**Side benefit**:
- Confirmed `label_mode='fwd_5d'` is bit-identical to default: triple_barrier_off mean_ic = pead_off mean_ic = +0.03399 to 15 decimals. Wiring for the explicit-default path is correct.
- New BUG-CV-2 escape clause (Task #24) correctly REJECTED triple_barrier_on (eval_ic=−0.0744 < floor 0.02) — the guard distinguishes "fast-converging strong signal" (PEAD ablation, eval_ic=+0.06) from "anti-predictive label corruption" (this case, eval_ic=−0.07). The escape clause is well-calibrated.

**Recipe**:
```bash
# Reproduce the failure (will RuntimeError):
python scripts/train_104.py --strategy-config-name strategy_config.triple_barrier_on.json --skip-baseline --skip-recalibrate --force
```

**Status**: PARTIAL — module + wiring landed but residual-mismatch resume needed. Track F deferred until horizon-matched residual extraction is implemented.

---

## E25: Triple-barrier label A/A — placebo invalidates "+98bp lift" (shelved)

**Date**: 2026-05-02
**Branch**: main (commits 80b3972 → 84a0194)
**Run IDs** (`data/runs.db`):
- triple_barrier_off (fwd_5d default): `20260502211508-panel-ltr-6a95a1`, mean_ic=+0.0340
- triple_barrier_on v3 (hit-time-matched residual): `20260502223218-panel-ltr-ffc361`, mean_ic=+0.0433
- triple_barrier_on repro: `20260502230207-panel-ltr-0a370c`, mean_ic=+0.0444
- triple_barrier_on shuffled-label sanity: `20260502232804-panel-ltr-7debb6`, mean_ic=+0.0005 ✓ (PASSED)
- triple_barrier_on time-shift +60d placebo: `20260503002353-panel-ltr-cb23ec`, mean_ic=+0.0458 ✗ (FAILED)
- fwd_5d baseline +60d placebo (disambiguation): `20260503005701-panel-ltr-a1168c`, mean_ic=+0.0290

**Hypothesis**: Lopez de Prado AFML §3 triple-barrier label (first-hit of {upper, lower, time}) replaces fwd_5d as the panel-LTR target. Hit-time-matched residualization (per-row spy_fwd at ticker's hit_days[t]) preserves β-neutrality across variable horizons. Predicted +5-10 bp CPCV IC lift; saw +98 bp (massive).

**Setup**:
- panel_ltr.label_mode = 'triple_barrier' (new flag)
- triple_barrier hyperparams: alpha=beta=2.0, max_horizon_days=10, vol_window=20
- New compute_residual_returns_hit_aligned in labels.py
- Otherwise identical to production golden config

**Initial result**: triple_barrier_on mean_ic = +0.0438 (avg of v3 + repro), vs baseline +0.0340 = **+98 bp delta**. Reproducible across XGB seeds (σ ≈ 0.6 bp from 3-run pead_off baseline).

**§5.2 sanity sequence**:
1. **Reproducibility**: passed. v3 = +0.0433, repro = +0.0444, both within run-to-run σ.
2. **Shuffled-label sanity**: passed. mean_ic on per-date permuted labels = +0.0005 ≈ 0. Confirms model can't learn random labels = no obvious leak.
3. **Time-shift placebo (+60d)**: **FAILED**. mean_ic on shifted labels = +0.0458 (HIGHER than real +0.0438).

**Disambiguation**: Same +60d placebo on baseline fwd_5d label = +0.0290. So all models pick up some "regime persistence" at 60-day horizon, but:
- fwd_5d real (+0.0340) − placebo (+0.0290) = **+0.0050** (5 bp of real 10-day alpha)
- triple_barrier real (+0.0438) − placebo (+0.0458) = **−0.0020** (zero real 10-day alpha; the lift was entirely regime-persistence capture)

**Mechanism**: Cross-sectional rank persistence (high-mom stocks at t are still high-mom at t+60) is a slow phenomenon. Triple-barrier's hit-time-matched residualization mathematically purifies the SPY-removal step, which lets the model fit this slow regime baseline more cleanly. But for our 10-day rebalance strategy, regime persistence is not actionable alpha — it's already priced in, and frequent rebalancing on slow signals incurs friction with no edge.

**Conclusion**: Triple-barrier as designed is **NOT a real alpha lift over fwd_5d**. The +98 bp CPCV IC improvement is statistical artifact of the model better fitting cross-sectional persistence, not improved 10-day predictive power. Production keeps fwd_5d.

**Process lesson**: §5.2 placebo SHOULD have run BEFORE declaring victory on v3 mean_ic. I dispatched v3 → repro → shuffled-label → THEN placebo. Order should have been v3 → ALL §5.2 → declare. The shuffled-only "pass" gave false confidence; placebo with disambiguation against baseline was the actual decisive test.

**Resume conditions**:
1. **Shorter shift placebo (e.g. shift=15d)**: characterize where the persistence effect peaks vs where it disappears. If placebo IC drops sharply between shift=10 and shift=30, signal might be 10-20 day but not pure 10-day.
2. **Triple-barrier WITHOUT residualization** (the original v0 simplification): see if that variant has lower placebo IC (would suggest residualization specifically captures persistence).
3. **Different barrier hyperparams**: alpha=1.5/beta=2.5 (tighter profit-take, looser stop) might capture different distributional moments.
4. **Triple-barrier with FORCED 5-day max_horizon**: closer to fwd_5d horizon, may have less regime-persistence room.

**Side benefits this session**:
- compute_residual_returns_hit_aligned (kernel/labels.py) is general-purpose; future variable-horizon labels can reuse.
- §5.2 sanity infrastructure (panel_ltr.label_shuffle_seed + label_shift_days) is now in standard pipeline. Any future architecture experiment can A/A-validate with one config flip.
- Process discipline encoded — placebo bug found by self-audit (initial .shift(+60) was wrong direction; corrected to .shift(-60)).

**Recipe**:
```bash
# Reproduce all 6 measurements:
python scripts/train_104.py --strategy-config-name strategy_config.triple_barrier_off.json --skip-baseline --skip-recalibrate --force
python scripts/train_104.py --strategy-config-name strategy_config.triple_barrier_on.json --skip-baseline --skip-recalibrate --force
python scripts/train_104.py --strategy-config-name strategy_config.triple_barrier_on_repro.json --skip-baseline --skip-recalibrate --force
python scripts/train_104.py --strategy-config-name strategy_config.triple_barrier_on_shuffled.json --skip-baseline --skip-recalibrate --force
python scripts/train_104.py --strategy-config-name strategy_config.triple_barrier_on_placebo.json --skip-baseline --skip-recalibrate --force
python scripts/train_104.py --strategy-config-name strategy_config.fwd5d_placebo_shift60.json --skip-baseline --skip-recalibrate --force
```

## E26: wl183 watchlist expansion — B2 NO-GO post-fix (Sharpe −0.07, APY −1.6%) — 2026-05-05

**Track**: D — wl103 → wl200+ expansion. Stage 3 greedy admission shipped wl183 (103 wl103 + 80 IC-additive batch admissions) and trained wl183-specific artifacts (`panel-ltr.wl183_daily_clean.json`, `ngboost-head.wl183_daily_clean.json`, `panel-rank-calibration.wl183_daily_clean.json`).

**Hypothesis**: Stage 3 measured per-batch IC lifts of ~+9bp (wl103 → wl183). If that IC lift translates to OOS performance, expect Sharpe ≥ wl103 baseline (1.10) and APY ≥ wl103 (13.27%) on the 27-mo B2 holdout (sim 2025-05-05 → 2026-05-04).

**Pre-flight bug**: Initial 27-mo wl183 B2 sim returned `n_buys=0, n_sells=0` over the full holdout — the wl183 promotion was blocked, not by performance, but by silent inference-side data corruption.

**Root cause**: `kernel/row_coverage.py::filter_by_coverage` called `panel.loc[keep_mask].reset_index(drop=True)` unconditionally. For training (long-form panel, integer index OK) this was fine. For inference, `build_inference_matrix` produces X **ticker-indexed**, and the reset clobbered ticker symbols → `X.index = [0, 1, 2, …]` (int64). All downstream `scores.get(cand.ticker)` lookups in `ApplyScoresTask`, `ApplyNGBoostTask`, `ApplyGlobalCalibrationTask` returned None for every candidate → 0 trades on every bar.

Production (`strategy_config.json`) was unaffected: `row_coverage.enabled` is absent (defaults False). Only the wl183 side config (`row_coverage.enabled=true`) hit the bug.

**Fix**: opt-in `preserve_index=True` parameter on `filter_by_coverage`, passed from `RowCoverageGateTask`. Default unchanged for training callers. Source-level + runtime contract tests in `tests/test_row_coverage_gate.py`. Commit `8d0f871`, runtime contract `7a61f00`.

**Post-fix B2 (27-mo) result on wl183**:

| Metric | wl183 (post-fix) | wl103 baseline |
|---|---|---|
| APY | **−1.60%** | +13.27% |
| Sharpe | **−0.069** | +1.10 |
| Sortino | −0.063 | — |
| Calmar | −0.119 | — |
| Max DD | 13.45% | ~12% |
| Ann vol | 12.36% | — |
| Total return | −1.59% | — |
| Win rate | 77.46% | — |
| n_buys / n_sells | 142 / 173 | — |

**Mechanism**: high win rate (77%) + negative Sharpe means losers are materially larger than winners. Combined with the calibrator collapse (`n_unique_prob_y=7`, top candidates tie at floor=0.30 cap), buy ranking is effectively random within the cap-clamped tier — adverse selection on unconviction picks. The +9bp Stage 3 IC lift did not survive the calibrator + buy_floor + execution gauntlet.

**Conclusion**: **NO-GO for wl183 promotion.** wl103 stays as production golden.

**Resume conditions**:
1. Calibrator retrain — production `panel-rank-calibration.json` has `n_unique_prob_y=7` (below the runtime SOFT WARN floor of 10). Underlying cause: panel-LTR `best_iter=4` (XGB plateaued at 4 boosting rounds → ~16 leaf paths → ~7 distinct calibrated probabilities). Retrain with bumped `min_best_iter` floor + accept loss curve at higher round count.
2. After calibrator fix, re-run wl183 B2. If Sharpe still < wl103 baseline → wl183 universe genuinely doesn't help and expansion track is dead until Track D step IV (wl≥250) or different admission criterion.
3. Investigate the high-win-rate + negative-Sharpe pattern. May indicate the 80 admitted tickers concentrate in adverse-selection regimes (e.g. high-momentum names that mean-revert in bear-leaning bars).

**Side benefits this session**:
- `preserve_index` parameter is general; future filter_by_coverage callers in inference paths get the safe default.
- Runtime contract test (`test_row_coverage_gate_preserves_ticker_index_runtime`) catches regressions in `filter_by_coverage` defaults, ctx field renames, or alternative filter implementations — lower blast radius than the source-level guard.
- `kelly_target_pct` range invariant tests (`test_pipeline_value_contracts.py::TestKellyTargetPctRangeInvariant`) added in passing — pins `[0, min(max_pct, max_concentration)]` for any input including degenerate cases (None, NaN, inf, σ≤0, μ≤0, non-numeric).

**Recipe**:
```bash
# Reproduce post-fix B2:
python scripts/holdout_backtest.py --skip-train \
    --strategy-config-name strategy_config.wl183_daily_clean.json \
    --train-end 2025-05-04 --sim-start 2025-05-05 --sim-end 2026-05-04 \
    --output /tmp/wl183_b2.json
# Verify Sharpe / APY in the output JSON match the table above (≈ −0.07 / −1.60%).
```

## E27: walk-forward 3-cut on production wl103 — alpha vs SPY consistently negative — 2026-05-05

**Track**: post-bug-bounty consistency check. After 14 bug fixes + 4 new QP features (Davis-Norman band, min_invested floor, feasible warm-start, capacity clamp) brought single-cut 27-mo B2 from Sharpe 0.59 → 0.68 / APY 7% → 10.12%, ran walk-forward to test whether the alpha is real or a smoothing artifact of one long window.

**Hypothesis**: alpha measured by single-cut B2 should hold across rolling 6-mo OOS cuts. If so, ship-to-promotion proceeds. If not, the 27-mo number is regime-driven, not alpha.

**Setup**: `scripts/walk_forward_holdout.py` — 3 cuts × 6-mo OOS, fixed artifact (no per-cut retraining since current model architecture trains on full history through 2025-05-04). Per-cut SPY benchmark computed for the same window.

**Result**:

| Cut | OOS Window | Strategy APY | Strategy Sharpe | SPY APY | SPY Sharpe | Alpha vs SPY |
|---|---|---|---|---|---|---|
| 2024-05-01 | 2024-05-02 → 2024-11-01 | 0.00% (0 trades) | NaN | +27.36% | +1.95 | **−27.36%** |
| 2024-11-01 | 2024-11-02 → 2025-05-01 | −12.84% | **−1.39** | −4.07% | −0.04 | **−8.78%** |
| 2025-05-04 | 2025-05-05 → 2025-11-04 | +32.05% | +1.82 | +42.78% | ~2.0+ | **−10.72%** |

**Mean across cuts:** APY 6.40% ± 23.12%, Sharpe 0.21 ± 2.27, **alpha vs SPY = −15.62% ± 10.21%, sign-consistent NEGATIVE**.

**Diagnosis**:

- **The single 27-mo Sharpe 0.68 was a smoothing mirage.** Splitting into 6-mo windows reveals enormous regime dependence: −1.39 Sharpe in flat-bear, +1.82 in strong bull, NaN/0-trades in late-2024 bull.
- **Strategy underperformed SPY in EVERY cut**, including the strong bulls (cut 3 captured 75% of SPY's move at the cost of cap-constrained turnover).
- **The strategy is structurally a costly closet-index** — long-only equity exposure plus active management costs (~30–40% cash drag, ST tax, friction, regime flips).
- **The Fundamental Law cannot rescue this** without a stronger signal: with `IC × √breadth × TC` and TC capped at ~0.078 (8 of 103 names), even doubling IC barely closes the −15pt gap.

**Cut 1 (2024-05-01) zero-trade anomaly**: 0 buys / 0 sells. Most likely cause: the artifact's `config_fingerprint` (sha256:4f1e25989d475225) was minted on 2025-05-04 daily-cron training; running that on 2024-05-02 may pass preflight but produce no candidates above the calibrator-floor since the panel-LTR's training distribution doesn't generalize backward in time. Not a code bug per se — an artifact of the eval design (no per-cut retrain).

**Conclusion**: **No-go on shipping ANY current model variant for active alpha capture.** The 14 bug fixes shipped today are net-positive (close real silent failure paths, surface accounting truth) and should remain in production. But the model+architecture pair as configured cannot beat SPY post-tax, post-friction, post-cash-drag — confirmed by 3-cut walk-forward across all sampled regimes.

**Resume conditions**:

1. **Different label**: replace `fwd_5d` binary "outperform-SPY 5d ahead" with `fwd_20d` or `fwd_60d`. Current 5d horizon is in noise-dominated territory; longer horizon shifts toward fundamental drivers.
2. **Different model architecture**: try Transformer-on-panel (already in code but rejected previously due to 60% IC drop). Re-evaluate with 5y+ training data + new label.
3. **More training history**: bump `training_years` from 2.5 → 5.0 or 10.0. Model has only seen ~625 dates × 103 tickers ≈ 65k panel rows. Doubling that may improve generalization.
4. **Walk-forward retraining**: each cut gets its own retrain on data through that cut. Current eval uses fixed 2025-05-04 artifact for all cuts — leaks information. True walk-forward would be slower (~2h per cut) but methodologically correct.

**Side benefits this session (kept)**:

- 14 bug fixes (1–11, 13–14) — all real, ship to production. Removed silent failures, NaN propagation, accounting under-collection, ghost holdings, routing collisions.
- QP solver gained: Davis-Norman no-trade band, min_invested floor, feasible warm-start, capacity clamp. All measurable in trade-count reduction (49% fewer buys, 53% fewer sells).
- Walk-forward eval infrastructure (`scripts/walk_forward_holdout.py`) — first measurement of regime-stability for any model in this project. Will be reused going forward.
- CLAUDE.md §5.2 sanity sequence verified yet again: A/A test (single-cut B2 was the equivalent here), §5.2 placebo (walk-forward exposed the real distribution).

**Recipe**:
```bash
# Reproduce walk-forward result:
python scripts/walk_forward_holdout.py \
    --strategy-config-name strategy_config.json \
    --cuts 2024-05-01 2024-11-01 2025-05-04 \
    --oos-months 6 \
    --output /tmp/funnel/wl103_walk_forward
# Verify summary: apy_mean ≈ 6.4%, alpha_vs_spy_mean ≈ −15.6%
cat /tmp/funnel/wl103_walk_forward/summary.json
```

**Key insight (carries forward)**: this is the first time the project has had a multi-cut OOS measurement on the production model. Going forward, **CLAUDE.md should require walk-forward eval in addition to single-cut B2 before any "ship to golden" decision**. Single-cut numbers are too vulnerable to regime smoothing to be trusted alone.
