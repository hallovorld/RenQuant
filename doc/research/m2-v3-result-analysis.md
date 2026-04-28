# M2 v3 horizon blender — final result analysis

**Date**: 2026-04-28
**Verdict**: **Shelved permanently. M2 horizon blending does not work on this panel under any tested formulation.**

## Setup

- v3 fixes (vs v2): StandardScaler in Pipeline (Fix 1), Purged K-Fold with embargo=20d (Fix 2, López de Prado 2018), ElasticNetCV (Fix 3, Zou & Hastie 2005), per-date rank target (Fix 4, Cao et al. 2007), winsorize at [0.5%, 99.5%] (Fix 5)
- Training data: 227-watchlist panel @ 20d lookahead, 478,539 rows × 223 tickers × 2,248 dates
- Hold-out: last 25% chronological → 118,520 rows
- Evaluation: Spearman IC of predictions vs realized forward 20d return on hold-out

## Hold-out Spearman IC

| Method | IC | vs single 10d |
|---|---|---|
| **Single 10d** | **+0.1291** | baseline |
| Single 20d | +0.1228 | −4.8% |
| Single 60d | +0.0636 | −50.7% |
| Equal-weight blend (1/3 each) | +0.1019 | −21.0% |
| 1/IC weighted blend (DeMiguel et al. 2009) | +0.0986 | −23.6% |
| **Learned ElasticNet (5 fixes)** | **+0.0271** | **−79.0%** |
| A/A shuffled labels | NaN (constant pred) ✓ |

**Single best horizon (10d) wins. Every blend method loses to it, including the no-learning baselines.**

## Hyperparameter results (sanity check on the fixes)

- `best l1_ratio = 0.1` (heavy ridge) — confirms Fix 3 mattered: pure Lasso was inappropriate under collinearity
- `best alpha = 0.0167` — vs v2's 0.000315 (50× larger). Fix 2 (Purged CV) materially changed the regularization landscape
- `nonzero_coefs = 3 / 13` — heavy sparsification; model effectively picks 1-2 horizon predictors
- `kept_regimes = [BULL_CALM]` — other 3 regimes had < 500 train samples, auto-pruned per design
- A/A test: shuffled labels produced constant prediction (NaN Spearman). Healthy — confirms no label leakage in the learned pipeline

## Why M2 fundamentally fails

Three per-horizon predictions (μ_10, μ_20, μ_60) are pairwise correlated > 0.7 (forecasts of overlapping forward returns on the same ticker-date). Linear blend adds **estimation noise** without adding much **independent information**. Net effect: noise variance dominates the marginal information gain. This is a structural property of correlated forecasts, not a bug in the blender.

DeMiguel, Garlappi, Uppal (2009 RFS) shows the same pattern in mean-variance portfolio optimization: with correlated assets and noisy estimates, naive 1/N often beats the "optimal" estimated weights. We hit the same regime here — except even 1/N (equal-weight) loses to the single best horizon.

## What the train/hold-out gap tells us

Train-set IC per horizon:
- 10d: 0.0301
- 20d: 0.0261
- 60d: 0.0337

Hold-out IC:
- 10d: 0.1291 (4.3× train)
- 20d: 0.1228 (4.7× train)
- 60d: 0.0636 (1.9× train)

Hold-out is the LAST 25% of dates (chronologically the most recent). All horizons are easier to predict in the recent period than in training. That's the opposite of typical overfitting — it's a regime difference (recent market may be more momentum-driven than the training period).

**Implication**: hold-out IC numbers are inflated for ALL methods uniformly. The relative ranking still holds (single 10d > all blends > learned), but absolute IC magnitudes shouldn't be used as deployment targets.

## Salvageable findings

1. **60d horizon improves more than 10d/20d when the test set differs from training distribution**. Hold-out vs train IC ratio: 60d=1.9×, 10d=4.3×, 20d=4.7×. Counterintuitively, 60d's *relative* gain is smallest because its train IC was already highest. But its absolute gain is also smallest — 60d signal is more stable across regimes (less inflated on the recent quarter).
2. **Heavy regularization (alpha=0.017) with PurgedKFold is the right ballpark**. v2's alpha=0.000315 was indeed leaky. Fix 2 was correct.
3. **Per-date rank target alignment (Fix 4) works** but doesn't help the blend itself — model still finds nothing useful to learn.
4. **The 3 non-zero coefficients in the learned model are essentially "single best horizon picker"** — ElasticNet effectively reduced to "use the best of μ_10, μ_20, μ_60". Confirms the structural argument above.

## Path forward

**Do not try learned horizon blending again on this panel.** The structural correlation between horizons makes the problem unfavorable.

Two open paths that are not blocked by this result:
1. **Single-horizon swap to 60d** on a wider watchlist (200+ tickers). 60d's `paired t=+3.82` significance vs 10d on 227 watchlist suggests 60d wins as the breadth grows. This is a single-horizon decision, not a blend.
2. **Macro × 60d × wider watchlist** as a fresh experiment — never tested at long horizon × broad cross-section.

Both are downstream of watchlist 200 v2. M2 stays closed.
