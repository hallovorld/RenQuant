# Sunday Panel-LTR Sweep — 2026-04-26

Strategy: `renquant_104`. All 3 backends trained on the same panel with identical CV split (CPCV, 15 folds), same NGBoost head, same calibrator. XGBoost remains the active production backend; LightGBM and Transformer trained for comparison only.

## Results

| backend | scorer_oos_mean_ic | pool_ic | train_ic | n_rows | elapsed | status |
|---|---:|---:|---:|---:|---:|---|
| xgboost | 0.0482 | 0.0011 | — | 225506 | 1954s | OK |
| lightgbm | 0.0291 | 0.0097 | — | 225506 | 1974s | OK |
| transformer | — | — | — | — | 1670s | FAILED |

## Winner — xgboost (OOS IC = 0.0482)

## Active artifacts

`backtesting/renquant_104/artifacts/panel-ltr.json` and friends are restored to the XGBoost run after this sweep. Per-backend backups live as `*.{backend}.bak.json` in the same directory.
