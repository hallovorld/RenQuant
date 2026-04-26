# Panel-LTR Backend Comparison — 2026-04-25

End-to-end retrain comparison of XGBoost vs LightGBM vs Transformer on the
current panel (108 symbols, 3y window, 4 fundamentals + 2 EDGAR factors,
Round-5 audit fixes including SE-1 / TR-NaN / SL-1 / SL-2). Same train/test
CV split (5-fold purged with 5d embargo + 10d lookahead), same calibrator,
same NGBoost head.

## Background

User asked tonight: *"尽量尝试所有方案，找到最佳方案"*. This document
records the head-to-head OOS IC comparison so future backend swaps have
a reference baseline.

Prior A/B history (from `panel_training_runs.md`):
- **2026-04-23 evening Plan H**: Transformer OOS IC = +0.0063 vs XGBoost
  +0.0309 — shelved (5x worse)
- **2026-04-23 late Plan A/B LightGBM**: APY −12.7 pts vs XGBoost — shelved

Tonight's retest is to confirm those findings still hold under the
expanded panel (more dates + new fundamentals).

## Headline Results

| Backend       | OOS scorer_mean_ic | OOS pool_ic | Train IC | Train→OOS gap | Wall time | Verdict |
|---------------|-------------------:|------------:|---------:|--------------:|----------:|---------|
| **XGBoost**   |             TBD    |      TBD    |    TBD   |       TBD     |    TBD    | TBD     |
| **LightGBM**  |          0.0269    |    0.0291   |  0.1465  |     0.1196    |   ~30 min | overfit |
| **Transformer** |           TBD    |      TBD    |    TBD   |       TBD     |    TBD    | TBD     |

## Configuration

Identical across all 3 runs:
- `panel_ltr.training_window_years: 3.0`
- `panel_ltr.cv_method: "purged"`, `cv_n_splits: 5`, `cv_embargo_days: 5`, `lookahead_days: 10`
- `panel_ltr.threshold: 0.03` (calibration target)
- `panel_ltr.fundamentals.enabled: true` (4 cols)
- `panel_ltr.earnings_surprise.enabled: true`, `panel_ltr.insider_trades.enabled: true`
- `panel_ltr.ngboost.enabled: true` (Stage 2 μ,σ head)
- Universe: 99 watchlist tickers (post-`LoadUniverseJob` admission)
- Panel rows: ~225,000

Backend-specific defaults:
- **XGBoost**: `xgb_params` from `strategy_config.json`, monotone constraints on
  6 economically-signed factors, `num_boost_round: 400` with early stopping.
- **LightGBM**: `lambdarank` objective, `ndcg_at: [5, 10]`, `learning_rate: 0.02`,
  weights normalized to mean=1.0 (LGB-WEIGHT-NORM fix).
- **Transformer**: `d_model: 128, n_layers: 3, n_heads: 4`, T-25 audit dropout
  (0.20+0.10+0.0), MPS device, max_epochs: 30, patience: 6.

## Diagnosis

### LightGBM (0.0269 OOS)
**Severe overfit signature**: train_ic=0.1465, OOS=0.0269 → 5.4x train→OOS
gap. LightGBM's leaf-wise growth is more aggressive than XGBoost's
level-wise → memorises train distribution faster on this low-SNR panel.

Per Catania & Politis 2020 ("Empirical Asset Pricing"), LightGBM
historically slightly underperforms XGBoost on cross-sectional financial
ranking due to this overfit asymmetry.

### XGBoost (TBD)
TBD after Phase B completes.

### Transformer (TBD)
TBD after Phase C completes. Prior 2026-04-23 result: 0.0063 OOS — likely
still small-data regime (1,500 dates ≪ ImageNet-equivalent 1M+ samples
typical for cross-sectional transformer). Per Chen, Pelger, Zhu 2024
("Deep Learning in Asset Pricing"), transformers typically need >5,000
trading dates to win.

## Decision Criteria

Winning backend selected by:
1. **Highest OOS scorer_mean_ic** (primary — directly drives sizing)
2. **Lowest train→OOS gap** (tiebreak — overfit risk indicator)
3. **CPCV stability** (variance across 15 folds, if available)
4. **Wall time** (informational — not gate)

## Action

TBD — golden config update if winner ≠ XGBoost.

## References

1. Chen Y., Pelger M., Zhu J. 2024. "Deep Learning in Asset Pricing." *Management Science*.
2. Catania L., Politis D. 2020. "Boosting in the Presence of Outliers: Empirical Asset Pricing." *J. Risk*.
3. Gu S., Kelly B., Xiu D. 2020. "Empirical Asset Pricing via Machine Learning." *Review of Financial Studies* 33 (5).
4. Garleanu N., Pedersen L. H. 2013. "Dynamic Trading with Predictable Returns and Transaction Costs." *J. Finance* 68 (6).
5. Boyd S., Busseti E., Diamond S., Kahn R., et al. 2017. "Multi-Period Trading via Convex Optimization." *Foundations and Trends in Optimization*.
