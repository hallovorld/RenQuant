# Failed experiments log — RenQuant

Per CLAUDE.md principle 5.7. Every failed experiment is recorded here with: hypothesis, implementation, exact data, statistical sanity check, conclusion, and a reproduction recipe so an independent agent (e.g. Codex) can verify the result without rerunning the full discovery process.

**Why this log exists.** Without it we forget what we tried. Same idea returns under a new name 6 months later. Reasonable hypotheses that the data falsified must stay on the record so the team doesn't waste compute re-falsifying them.

**Format**: one section per experiment, ordered chronologically (newest first). Each section answers: what was the hypothesis, what was built, what exactly were the numbers, was the failure structural or implementation, and how to reproduce.

---

## E_LAYER1_ALONE. Sector-rank-norm alone insufficient on wl178

**Date**: 2026-05-01
**Type**: partial structural negative — single-layer fix doesn't close gap
**Production impact**: none (experimental branch; main wl103 unchanged)
**Branch**: `exp/wl500-and-sector-arch`

### Hypothesis
After Phase 0 diagnostic confirmed wl178 sector heterogeneity (KS ≥ 0.30 across all 28 sector pairs), Layer-1 of the design-v2 architecture (per-`(date, sector)` percentile rank columns appended to the existing `_z` global z-score) should restore OOS panel IC to ≈ wl103 production baseline (+0.0418).

### Implementation
- `cross_sectional_rank_within_sector` helper in `training_panel/factors.py` — qlib-CSRankNorm-style per-`(date, sector)` percentile in [0, 1].
- `SectorRankNormalizeTask` in `training_panel/pp_panel_training.py` — appends `{col}_sr` columns to factor_frames after FactorZScoreTask, in BOTH `PanelAssemblyJob` (training) and `prepare_inference_panel_frames` (inference). Default OFF; flag `panel_ltr.sector_rank_norm.enabled`.
- `RAW_FACTOR_COLS_FOR_NORM` shared module constant — single source of truth for which raw columns get cross-sectional normalization. Both FactorZScoreTask and SectorRankNormalizeTask read from it (audit fix M1).
- Test coverage: 16 helper tests + 11 task tests + 1 inference plumbing test, all green.

### Data (CPCV 15-fold, wl178 panel, 178 tickers, daily resolution, 2024-2026 window)

Reference series for context (all panel-LTR, XGBoost rank:pairwise, same hyperparams):

| Experiment | Layer 1 | Layer 2 | train_ic | CPCV mean_ic | n_splits | Status |
|---|---|---|---:|---:|---:|---|
| Production wl103 | off | off | +0.118 | **+0.0418** | 15 | reference |
| A/A half A (wl178/2) | off | off | +0.116 | +0.0004 | 15 | completed |
| A/A half B (wl178/2) | off | off | +0.085 | +0.0136 | 15 | guard_fired |
| **wl178 Layer 1** | **10 cols** | off | +0.069 | **−0.0008** | 15 | **guard_fired** |

The Layer-1 retrain landed CPCV mean_ic = −0.0008 — within ±1.5σ of A/A baseline (−0.0008 vs +0.0004 / +0.0136 for the two A/A halves; pooled std ≈ 0.020). Layer 1 alone produced **no detectable lift** despite individual `_sr` features showing IC up to +0.0353 (resid_mom_sr) on the within-date diagnostic.

### Sanity / falsification
- ✅ Layer 1 fired: `SectorRankNormalizeTask: added 1780 (ticker × _sr column) entries across 178 tickers, 10 feature cols`. Tasks ran. Implementation verified by:
  - 16 unit tests on the helper (range invariant, sector relativity, NaN propagation, A/A test, determinism).
  - 11 Task tests (default-off no-op, sector relativity under 100× scale gap, defensive paths).
  - Per-feature IC diagnostic logged inside the retrain — `*_sr` cols carried real signal individually.
- ✅ Compared against A/A baseline (random splits of same wl178 universe), not against fundamentally different baseline.
- ✅ Same XGBoost hyperparams, CPCV folds, eval set.

### Why the individual `_sr` IC doesn't aggregate
Single-feature IC ranges +0.035 (resid_mom_sr) down to −0.008 (beta_60d_sr). Yet ensemble CPCV is essentially zero. Plausible mechanisms:

1. **Redundancy with `_z`**: `_z` and `_sr` carry overlapping information (same underlying raw value, different normalization). Tree splits on `_sr` are masked by earlier `_z` splits — added `_sr` is mostly noise to the ensemble.
2. **Heterogeneity persists in the LABEL**: forward-return distributions across sectors are still heterogeneous; cross-sectional rank label can't be made comparable by feature normalization alone.
3. **Ensemble overfit on small sample**: 178-ticker × 750-date panel produces enough rows for a tree to memorize without generalizing.

Not yet falsified — need Layer 1 + Layer 2 (sector identity) to test whether explicit sector anchoring helps the model carve per-sector decision regions. That experiment dispatched 23:13 PT 2026-05-01.

### Conclusion
**Layer 1 (sector rank-norm alongside global z-score) is INSUFFICIENT to close the wl178 OOS-IC gap to wl103 baseline.** Not a complete failure — the helper code is correct, the integration is clean, default-off compat preserved. But on its own this layer doesn't move the needle.

Next steps (gating Layer 1+2 result):
- If Layer 1+2 also fails (mean_ic ~0): escalate to NN backend (Phase C / Phase D, Feng 2019 / MIGA 2024).
- If Layer 1+2 succeeds (mean_ic > +0.020): ship; keep Layer 3+4 as future ceiling work.
- Either way, document in this log + mark merge criteria accordingly.

### Reproduction
```bash
# In the exp/wl500-and-sector-arch worktree:
python -c "
import json
cfg = json.load(open('backtesting/renquant_104/strategy_config.wl178.json'))
cfg.setdefault('panel_ltr', {}).setdefault('sector_rank_norm', {})['enabled'] = True
open('backtesting/renquant_104/strategy_config.wl178_layer1.json', 'w').write(json.dumps(cfg, indent=2))
"
python scripts/train_104.py \
  --strategy-config-name strategy_config.wl178_layer1.json \
  --skip-baseline --skip-recalibrate --force --skip-acceptance \
  > /tmp/wl178_layer1.log 2>&1

python scripts/compare_panel_experiments.py --logs /tmp/wl178_layer1.log
# Expect: cpcv_mean_ic ≈ -0.0008 ± 0.020
```

### Files
- `backtesting/renquant_104/training_panel/factors.py::cross_sectional_rank_within_sector`
- `backtesting/renquant_104/training_panel/pp_panel_training.py::SectorRankNormalizeTask`
- `backtesting/renquant_104/strategy_config.wl178_layer1.json`
- Log: `/tmp/wl178_layer1.log`

### Status
🟡 **Partial — superseded by Layer 1+2 experiment in flight.** Final classification (failed / superseded / inconclusive) blocks on Layer 1+2 result.

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

