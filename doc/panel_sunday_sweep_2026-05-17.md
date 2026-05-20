# Sunday Panel-LTR Sweep — 2026-05-17


> **📅 Historical snapshot — content below reflects state at the date in filename/header.**
> Verify against current code per CLAUDE.md §1 "code is the source of truth" before acting on
> present-tense claims. For current state see `doc/roadmap.md` § "📍 Current state" +
> `CLAUDE.md` § "🗂 Current state".

Strategy: `renquant_104`. All 3 backends trained on the same panel with identical CV split (CPCV, 15 folds), same NGBoost head, same calibrator. XGBoost remains the active production backend; LightGBM and Transformer trained for comparison only.

## Results

| backend | scorer_oos_mean_ic | pool_ic | train_ic | n_rows | elapsed | status |
|---|---:|---:|---:|---:|---:|---|
| xgboost | — | — | — | — | 1856s | FAILED |
| lightgbm | — | — | — | — | 1871s | FAILED |
| transformer | — | — | — | — | 1956s | FAILED |

## Winner — none (all backends failed)

## Active artifacts

`backtesting/renquant_104/artifacts/panel-ltr.json` and friends are restored to the XGBoost run after this sweep. Per-backend backups live as `*.{backend}.bak.json` in the same directory.
