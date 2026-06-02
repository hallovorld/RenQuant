# Usage Guide

## Workflow modes (5)

| Mode | Tool | When to use |
|------|------|-------------|
| **Research** | Open notebook → tweak | Iterate on features / models without Docker |
| **Validation** | `lean backtest .` (after `export_lean_watchlist.py --strategy X`) | Final OOS event-driven check |
| **Analysis** | `python scripts/analyze_backtest.py --strategy X` | Render charts + metrics from a finished backtest |
| **Live** | `python scripts/live_multirepo.py --strategy X --broker {paper,alpaca-paper,alpaca,ibkr} --once` | One-shot trade through pinned subrepos |
| **Scheduled** | macOS launchd plists in `scripts/launchd/` | Daily cron-style |

The active strategy is `renquant_104`. The same pipeline classes (`InferencePipeline`, `FullTrainingPipeline`) drive every mode — adapters (`LeanAdapter`, `RunnerAdapter`, `SimAdapter`) translate context.

---

## Daily 104 commands (operator runbook)

```bash
# Inspect production state — read-only, no mutations
python scripts/model_dashboard.py
python scripts/model_dashboard.py --json   # machine-readable

# Full retrain (gates auto-validate, atomic-swap or rollback on fail)
python scripts/train_104.py

# Partial retrains
python scripts/train_104.py --skip-baseline                   # panel-LTR + recalibrate only
python scripts/train_104.py --skip-baseline --skip-panel      # recalibrate only
python scripts/train_104.py --skip-acceptance                 # bypass gates (DANGEROUS — use only for known-broken-but-recoverable cases)

# Backend tournament — pick winner among .bak.json artifacts
python scripts/select_best_model.py --strategy renquant_104
python scripts/select_best_model.py --strategy renquant_104 --weights "ic=0.7,sharpe=0.2,calmar=0.1"
python scripts/select_best_model.py --strategy renquant_104 --promote xgboost

# Shadow-mode challenger — when window closes
python scripts/finalize_challenger.py --strategy renquant_104    # auto-detect window
python scripts/finalize_challenger.py --strategy renquant_104 --challenger-name macro-enabled --start-date 2026-04-12 --end-date 2026-04-26
bash scripts/check_challenger_window.sh                          # daily cron poll

# Live trade through pinned subrepos
python scripts/live_multirepo.py --strategy renquant_104 --broker alpaca --once
python scripts/live_multirepo.py --strategy renquant_104 --broker alpaca --once --sell-only

# B2 Hold-out backtest (single-cut OOS sanity check on a saved artifact)
python scripts/holdout_backtest.py \
    --strategy renquant_104 \
    --strategy-config-name strategy_config.alpha158_linear.json \
    --train-end 2025-05-04 --sim-start 2025-05-05 --sim-end 2025-11-04 \
    --skip-train --out /tmp/v7.json    # use existing artifact (fast)

# Same, but inflating side-config from data/runs.db (Task #38 path)
python scripts/holdout_backtest.py \
    --experiment-label alpha158_linear \
    --train-end 2025-05-04 --sim-start 2025-05-05 --sim-end 2025-11-04 \
    --skip-train --out /tmp/v7.json

# alpha158_linear retrain wrapper (not yet scheduled; manual run for now)
bash scripts/retrain_alpha158_linear.sh                 # full rebuild
bash scripts/retrain_alpha158_linear.sh --skip-features # reuse existing parquet

# DB experiment_configs (side-config-as-DB-row) management
python scripts/migrate_experiment_configs_to_db.py init           # one-time
python scripts/migrate_experiment_configs_to_db.py import         # import all FS side configs
python scripts/migrate_experiment_configs_to_db.py list           # show DB rows
python scripts/migrate_experiment_configs_to_db.py inflate --label alpha158_linear
```

---

## Research mode

```bash
source .venv/bin/activate
jupyter lab
```

For 104, the panel-LTR training is **driven by `scripts/train_104.py`** (not the notebook directly). The 104 notebook (when present) is for ablations / experiments — production retrains go through `FullTrainingPipeline`. The 103 notebook (`backtesting/renquant_103/renquant_103.ipynb`) and 101/102 notebooks (`Notebooks/`) remain for their respective per-symbol architectures.

The pipeline at `kernel/pipeline/pp_training_full.py::FullTrainingPipeline` runs:

1. **`BaselineTournamentJob`** — per-ticker manual / classification / Q-learning bake-off (skippable with `--skip-baseline`)
2. **`PanelTrainingJob`** — invokes `PanelTrainingPipeline` (PanelDataJob → PanelFeatureJob → PanelAssemblyJob → PanelModelJob → PanelNGBoostJob → RefreshPanelCalibratorJob)
3. **`RecalibrationJob`** — refits the global score-DB calibrator (skippable with `--skip-recalibrate`)

After training succeeds, `ModelAcceptanceGate` (kernel/model_acceptance.py) runs 11 gates against the staging artifact + the prior production artifact:
- G1-G6 hard catastrophic block (schema, calibrator collapse, NaN, IC degradation > 5%, score range, smoke)
- G7 hard absolute floor (OOS IC ≥ 0.02)
- G8 soft per-bar variance
- G9-G10 hard sim-based (APY drop ≤ 1pp, Sharpe drop ≤ 0.1) — opt-in via `acceptance.run_sim_smoke=true`
- G11 soft turnover bloat (≤ 1.5×)

Pass → atomic-swap promote (active → `.previous.json`, staging → active). Fail → archive staging to `_acceptance_log/`, prior model untouched, `sys.exit(2)`. Detail: [`doc/components/model-selection.md`](../components/model-selection.md).

---

## Validation mode (LEAN)

```bash
# Always export OHLCV first — LEAN reads from backtesting/data/equity/usa/daily/, NOT data/ohlcv/
python scripts/export_lean_watchlist.py --strategy renquant_104

cd backtesting/renquant_104 && lean backtest .

# Or with auto-rendered charts + notification
python scripts/backtest_and_analyze.py --strategy renquant_104          # default ntfy topic = renquant
python scripts/backtest_and_analyze.py --strategy renquant_104 --silent # no notifications
python scripts/backtest_and_analyze.py --strategy renquant_104 --open   # macOS: open PNGs after
```

LEAN data isolation: edits to `data/ohlcv/{SYMBOL}/` do NOT propagate to LEAN. The export step copies cached parquet → LEAN's `backtesting/data/equity/usa/daily/{sym}.zip` format. Always re-export when changing watchlist or extending dates.

LEAN respects `wash_sale_days`, `min_hold_days`, `max_hold_days`, and `position_sizing.{max_position_pct, cash_reserve_pct}` from `strategy_config.json` (and broader 104 config like `panel_ltr.*`, `acceptance.*`, `regime.*`).

Results land in `backtesting/renquant_104/backtests/<timestamp>/`.

---

## Analysis mode

```bash
python scripts/analyze_backtest.py --strategy renquant_104   # most-recent run
```

The analysis script auto-loads the most recent backtest, writes charts to the run directory, and prints metric summaries:

| Panel | Content |
|-------|---------|
| Price + Trades | Close price with LEAN entry/exit overlays |
| Decision Telemetry | Model score, thresholds, final action (when LEAN emits it) |
| Equity Curve | Portfolio value over time with profit/loss shading |
| Drawdown | Rolling drawdown (%) with max drawdown labelled |
| Statistics | Win rate, Sharpe, Sortino, max drawdown, alpha, beta, fees |

When `tax` is configured in `strategy_config.json`, LEAN reports after-tax metrics via `SetRuntimeStatistic()` (total tax, ST/LT trade counts). The 104 strategy uses `common.add_tax_columns()` to enrich closed trades with per-trade holding days + tax amount + after-tax P&L.

---

## Live mode

> **⚠️ Broker mode (2026-05-17 e2e mandate)**:
> `--broker alpaca` = **LIVE Alpaca account, real money** (per `feedback_e2e_means_real_broker`)
> `--broker alpaca-paper` = paper API (no real money). Cron schedules use `alpaca-paper` per 2026-05-11 safety mandate.
> When user says "e2e" without paper qualifier, they mean `--broker alpaca` LIVE. Locked in via CLAUDE.md Environment §"e2e".

```bash
# Local sim (no broker; for development only)
python scripts/live_multirepo.py --strategy renquant_104 --broker paper --once

# Alpaca paper (real API, no real money — cron default)
python scripts/live_multirepo.py --strategy renquant_104 --broker alpaca-paper --once

# Alpaca LIVE (real money — e2e / user explicit mandate only)
nohup bash -c 'set -a; source .env; set +a; .venv/bin/python scripts/live_multirepo.py --strategy renquant_104 --broker alpaca --once' \
  > logs/live_e2e/e2e_alpaca_live_$(date +%Y%m%d-%H%M%S).log 2>&1 &

# IBKR (stub — see live/ibkr_broker.py)
python scripts/live_multirepo.py --strategy renquant_104 --broker ibkr --once
```

Broker options: `paper`, `alpaca-paper`, `alpaca`, `ibkr`. Set `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in `.env` (gitignored). `.env` only has LIVE credentials — paper-API calls 401.

The runner auto-detects strategy shape from `strategy_config.json`. For 104, `RunnerAdapter` builds an `InferenceContext`, runs `prepare_inference_panel_frames` to populate `_panel_feature_frames` / `_panel_factor_frames` / `_panel_macro_frame`, then dispatches `InferencePipeline().run(ctx)`. Trade summaries land in `live/logs/<strategy>/<date>.json`.

**Daily log** (104) includes: REGIME PARAMS block (regime name, confidence, stop/hold/reserve values), price source tag (`[Alpaca]` or `[OHLCV <date>]`), max_hold exits with realized P&L, per-buy position sizing math, panel scoring summary, dropped candidates from `VetoWeakBuysTask`, rotation reject reasons (incl. `panel_advantage`).

**Multirepo pin guard:** `scripts/live_multirepo.py` and `scripts/daily_multirepo.py` resolve sibling repos through `subrepos.lock.json` and warn when local checkouts drift from the pinned commit or remote. Set `RENQUANT_STRICT_SUBREPO_PATHS=1` to fail closed on missing path, commit, or remote drift (`RENQUANT_STRICT_SUBREPO_PINS=1` is accepted as an alias). Set `RENQUANT_OPS_FAIL_CLOSED=1` when scheduled ops should also fail closed if a Python or shell delegate cannot import the pinned subrepo module. Dirty worktrees are only checked when `RENQUANT_STRICT_SUBREPO_CLEAN=1`; use that for production cron after local development worktrees are clean or an isolated runtime root is in use.

For production isolation, run `make subrepo-runtime-root` after lock updates. It clones/fetches pinned repos under `.subrepo_runtime/repos` and writes `.subrepo_assembly/<timestamp>/env.sh` plus `.subrepo_assembly/current.env`. Source that env before running launchd-style daily/intraday commands; it exports `RENQUANT_SUBREPO_ROOT`, `RENQUANT_STRICT_SUBREPO_PATHS=1`, and `RENQUANT_OPS_FAIL_CLOSED=1`.

---

## Scheduled mode (macOS launchd)

**11 active plists** (cross-reference `doc/ops/schedule.md` for the authoritative table). Cron schedules use `--broker alpaca-paper` per 2026-05-11 safety mandate. All NYSE-holiday-aware.

Daily 104 cron family (highlights):

| Run | Time (PT) | Script | Behavior |
|-----|-----------|--------|----------|
| Market open | 6:32 AM | `live_only_104.sh --sell-only` (DISABLED `.disabled.20260513`) | Sell-side only at open |
| Pre-close | 12:44 PM | `live_only_104.sh --sell-only` (DISABLED `.disabled.20260513`) | Sell-side at pre-close |
| Intraday | 1:30 PM | `intraday104.sh` | Mid-day intraday check |
| After close | 1:55 PM | `daily_104.sh` | Daily ops through `daily_multirepo.py` → trade; no model promote |
| Conditional | 1:10 PM | `conditional_retrain_104.sh` | SPY/VIX anomaly → force retrain (Mon-Fri) |
| Weekly WF promote | Sat 04:00 AM | `weekly_wf_promote.sh` | Walk-forward gate + sanity → actual promote (post 2026-05-17 enforcement) |
| Monthly calibrator | 1st-of-month | `monthly_calibrator_refresh.sh` | Calibrator refit + H2a/H2b hard gates |
| Sun screen | 12:05 PM | `screen_watchlist.py` | Weekly DROP/ADD candidate report |
| Backup | 07:00 / 11:00 / 15:00 / 19:00 / 23:00 PT | `daily_backup.sh` | Live state + DB to cloud |
| Daily news sentiment | 06:00 AM PT | `daily_news_sentiment_refresh.sh` | 2026-05-18 shipped — refresh sentiment features pre-market |
| Weekly fundamentals | Sat 03:00 AM | `weekly_fundamental_refresh.sh` | SEC EDGAR fundamentals refresh |

Plists live in `scripts/launchd/`. One-shot install:

```bash
for p in /Users/renhao/git/github/RenQuant/scripts/launchd/*.plist; do
    cp "$p" ~/Library/LaunchAgents/
    launchctl load ~/Library/LaunchAgents/$(basename "$p")
done
```

To uninstall one:

```bash
launchctl unload ~/Library/LaunchAgents/com.renquant.daily104.plist
rm           ~/Library/LaunchAgents/com.renquant.daily104.plist
```

Notifications (macOS banner + iPhone via [ntfy.sh](https://ntfy.sh) topic `renquant`) fire on success/failure for retraining and live trading.

Logs:
- `logs/daily_104/{date}.log`
- `logs/live_104/{date}-open.log`, `{date}-preclose.log`
- `logs/conditional_retrain_104/{stdout,stderr}.log`
- `logs/watchlist_screen/{stdout,stderr}.log`

103 plists (`daily_103.sh`, `live_only_103.sh`) remain for rollback but are typically unloaded.

---

## Adding a new strategy

```bash
python scripts/new_strategy.py --name foo --symbol AAPL --type classification
cd backtesting/foo && lean backtest .
python scripts/live_multirepo.py --strategy foo --broker paper --once
```

This scaffolds a 101-style single-stock layout. For panel-LTR-style strategies, copy the `renquant_104/` directory shape (kernel/, training_panel/, adapters/) and adapt.

Available model types via `common.create_model()` (legacy 101/102 path): `manual`, `classification`, `qlearning`, `fqi`, `optimization`, `xgboost`. For renquant_104 panel-LTR: see model registry kinds (`xgb`, `hf_patchtst`, `patchtst`, `regime_router`) at `kernel/panel_pipeline/model_registry.py`. See [`../arch/models.md`](../arch/models.md).

---

## Operator scripts cheatsheet

```
scripts/
├── train_104.py                 ★ FullTrainingPipeline driver — STAGES only (2026-05-17 walk-forward gate enforcement)
├── weekly_wf_promote.sh         ★ Saturday 04:00 PT — runs full WF gate + sanity → promote
├── monthly_calibrator_refresh.sh ★ 1st-of-month — Platt refit + H2a/H2b hard gates
├── monthly_meta_label_retrain.sh Monthly meta-label retrain (currently retraining a disabled artifact)
├── model_dashboard.py           ★ Single-screen production state (read-only)
├── select_best_model.py         ★ Backend tournament + --promote
├── patchtst_hf.py               ★ HF Trainer-based PatchTST shadow training (multi-task head + FiLM opt)
├── eval_hf_trainer_5cut_5seed.py 5-cut × 5-seed PatchTST baseline eval driver
├── eval_hf_film_5cut_5seed.py   FiLM A/B eval driver
├── eval_dlinear_5cut_5seed.py   DLinear baseline eval driver (§5.12 must-have)
├── compare_arch_5cut_5seed.py   Architecture comparison aggregator + verdict
├── verify_sigma_calibration.py  σ-calibration test (Student-t head)
├── dlinear_baseline.py          cure-lab/LTSF-Linear DLinear for cross-sectional ranking
├── daily_news_sentiment_refresh.sh Daily sentiment feature refresh (2026-05-18 ship)
├── weekly_fundamental_refresh.sh Weekly SEC EDGAR refresh
├── daily_iv_snapshot.sh         Daily options-IV snapshot (accumulation phase)
├── analyze_backtest.py          Render charts from a LEAN run
├── backtest_and_analyze.py      Run LEAN + render + notify
├── compare_panel_backends.py    A/B between two panel backends (retrains both)
├── compare_panel_vs_baseline.py 103 baseline vs 104 panel sim comparison
├── audit_transformer_vs_lgbm_4way.py   4-way transformer audit
├── ab_harness.py                Generic A/B harness for sim comparisons
├── archive_runs.py              Compress + move old logs / artifacts
├── backfill_forward_returns.py  Populate ticker_forward_returns table
├── bench_python_vs_rust.py      Rust transformer scorer parity benchmark
├── check_config_drift.py        Diff strategy_config.json vs .golden.json (pre-commit hook)
├── check_retrain_triggers.py    Detect SPY/VIX anomalies → force retrain
├── compute_portfolio_metrics.py Per-ticker contribution analytics
├── conditional_retrain_104.sh   Cron-triggered SPY/VIX-aware retrain
├── daily_104.sh                 ★ ACTIVE — daily ops + trade through multirepo bridge
├── daily_103.sh                 Rollback strategy daily driver
├── enable_hourly_transformer.py Stage C-3 enable / smoke test
├── export_lean_data.py          Single-symbol parquet → LEAN format
├── export_lean_watchlist.py     ★ Batch export — RUN BEFORE BACKTEST
├── export_panel_to_csv.py       Dump trained panel matrix for inspection
├── export_transformer_to_safetensors.py   PyTorch → safetensors conversion
├── fetch_earnings_calendar.py   yfinance → earnings-calendar.json (weekly)
├── fetch_macro_factors.py       yfinance → data/macro/{SYMBOL}.parquet
└── ...
```

---

## File index (104)

```
backtesting/renquant_104/
├── strategy_config.json + strategy_config.golden.json   (drift-checked)
├── strategy_config.lgbm_macro.json                      (experimental)
├── kernel/
│   ├── pipeline/                  InferencePipeline + jobs
│   ├── panel_pipeline/            PanelScoringJob + scorer + feature_matrix
│   ├── model_acceptance.py        11-gate validator + promote/reject/rollback
│   ├── sim_smoke.py               Sim-based gate metric helpers
│   ├── challenger.py              Shadow-mode infrastructure
│   ├── persistence.py             runs.db schema + writers
│   ├── macro.py                   Cross-asset feature builder
│   ├── regime.py / exits.py / rotation.py / sizing.py / scoring.py / selection.py
│   └── …
├── training_panel/
│   ├── pp_panel_training.py       PanelDataJob → PanelFeatureJob → PanelAssemblyJob → …
│   ├── pipeline.py                prepare_inference_panel_frames (train/inference symmetry)
│   ├── ltr_model.py               XGBoost backend
│   ├── lgbm_ltr.py                LightGBM backend
│   ├── transformer_model.py       Stage C-3 hourly transformer
│   ├── ngboost_head.py            μ/σ residual head (promoted to prod 2026-05-17)
│   └── global_calibrator.py       Platt scaling (switched from isotonic 2026-05-18)
├── adapters/
│   ├── lean.py / runner.py / sim.py
└── artifacts/
    ├── prod/
    │   ├── panel-ltr.alpha158_fund.json       ★ active production (172 features)
    │   ├── ngboost-head.alpha158_fund.json    promoted 2026-05-17 (val_IC +0.0352)
    │   ├── panel-rank-calibration.json        Platt scaling, pool_IC +0.094
    │   └── *.bak_* / *.broken_*               rollback + incident archive
    └── sim/                                    sim-only artifacts (sim/prod isolated per §sim_prod_artifact_isolation memory)
```
