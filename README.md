# RenQuant

Personal quantitative trading workstation for Apple Silicon. Glass-box pipeline: data ingestion → cross-sectional ML signal generation → backtesting (LEAN) → live trading (Alpaca/IBKR). Statistically interpretable, strictly decoupled.

**Active strategy: `renquant_104`** — panel learning-to-rank (cross-sectional XGBoost ranker + NGBoost μ/σ residual head + isotonic calibrator + portfolio QP) with an 11-gate model-acceptance system, shadow-mode challenger infrastructure, and the Phase 4 SOP at [`doc/components/model-selection.md`](doc/components/model-selection.md).

`renquant_103` (single-stock per-symbol scanner) is retained as the rollback strategy. `renquant_101` / `renquant_102` are reference scaffolding.

---

## Architecture (one diagram)

```
                          ┌──────── runs.db (SQLite, single source of truth) ────────┐
                          │  pipeline_runs · candidate_scores · trades · rotations    │
                          │  training_runs · challenger_decisions · ticker_fwd_ret    │
                          └──────────────────────────────────────────────────────────┘
                                                ▲              ▲
                                                │              │
  ┌──────────┐    ┌──────────────────────────┐  │  ┌──────────────────────────┐
  │ Data     │───>│ Training pipelines       │──┘  │ Inference pipelines      │
  │  yfinance│    │  FullTrainingPipeline    │     │  InferencePipeline       │
  │  intraday│    │  ├ BaselineTournamentJob │     │  SellOnlyPipeline        │
  │  parquet │    │  ├ PanelTrainingJob      │     │  (kernel/pipeline/)      │
  │  macro   │    │  │  ├ PanelDataJob       │     │                          │
  │  cache   │    │  │  ├ PanelFeatureJob    │     │   regime → drawdown      │
  └──────────┘    │  │  ├ PanelAssemblyJob   │     │  → buy gates → sell      │
        ▲         │  │  ├ PanelModelJob      │     │  → candidates →ranking   │
        │         │  │  ├ PanelNGBoostJob    │     │  → rotation → selection  │
        │         │  │  └ RefreshPanelCalib  │     └──────────────────────────┘
        │         │  └ RecalibrationJob      │             ▲       ▲       ▲
        │         └──────────────────────────┘             │       │       │
        │                          │                  ┌────┴───┐ ┌─┴────┐ ┌┴──────┐
        │                          v                  │ LEAN   │ │ live │ │ sim   │
        │                  ┌──────────────┐           │adapter │ │runner│ │adapter│
        │                  │ ModelAccept  │           └────────┘ └──────┘ └───────┘
        │                  │ 11 gates     │              │          │       │
        │                  │ promote/     │              v          v       v
        │                  │ reject path  │         backtest    Alpaca  notebook
        │                  └──────────────┘
        │                          │
        │                          v
        │                  artifacts/panel-ltr.json (active)
        │                  artifacts/panel-ltr.previous.json (rollback)
        │                  artifacts/panel-ltr.{xgboost,lightgbm,...}.bak.json
        └─────────────────────────────────────────────────── refresh feeds
```

`InferencePipeline` is the single source of truth for ALL decision logic. LEAN, live runner, and sim adapter all go through it via `LeanAdapter` / `RunnerAdapter` / `SimAdapter`.

---

## Quick Start (104)

```bash
# Setup (one-time)
conda create -n renquant python=3.10
conda activate renquant
pip install pandas numpy matplotlib seaborn yfinance scikit-learn xgboost lightgbm \
            ngboost jupyterlab pyarrow pandas-market-calendars
pip install "openbb[all]" openbb-cli backtesting scipy lean alpaca-py
lean login

# Daily workflow (104 — the active strategy)
python scripts/train_104.py                       # full retrain (gates auto-validate)
python scripts/train_104.py --skip-baseline       # panel + recalibrate only
python scripts/model_dashboard.py                 # production state + tournament
python scripts/select_best_model.py --strategy renquant_104  # backend tournament
python scripts/finalize_challenger.py             # close shadow window (when applicable)

# Live trading
python -m live.runner --strategy renquant_104 --broker alpaca --once

# Backtesting via LEAN (Docker)
python scripts/export_lean_watchlist.py --strategy renquant_104
cd backtesting/renquant_104 && lean backtest .

# Scheduled (macOS launchd)
# Plists in scripts/launchd/; daily_104.sh fires post-close on weekdays
```

`.env` holds Alpaca / IBKR creds (gitignored). Detailed setup: [`doc/ops/setup.md`](doc/ops/setup.md).

---

## Repo layout

```
RenQuant/
├── backtesting/
│   ├── renquant_104/                ★ active strategy
│   │   ├── kernel/                  inference + training pipelines (Tasks/Jobs/Pipeline)
│   │   │   ├── pipeline/            InferencePipeline + jobs (regime, sells, candidates, …)
│   │   │   ├── panel_pipeline/      PanelScoringJob + feature_matrix + scorer
│   │   │   ├── model_acceptance.py  11-gate ModelAcceptanceGate + promote/reject/rollback
│   │   │   ├── sim_smoke.py         sim-output gate metrics (Phase 2)
│   │   │   ├── challenger.py        shadow-mode infra (Phase 4a)
│   │   │   ├── persistence.py       runs.db schema + writers
│   │   │   ├── rotation.py / sizing.py / regime.py / exits.py / …
│   │   │   └── macro.py             VIX/HYG/UUP/etc cross-asset features
│   │   ├── training_panel/          PanelTrainingPipeline + per-backend impls
│   │   │   ├── pp_panel_training.py PanelDataJob → PanelFeatureJob → PanelAssemblyJob → …
│   │   │   ├── pipeline.py          prepare_inference_panel_frames (train/inference symmetry)
│   │   │   ├── ltr_model.py         XGBoost backend
│   │   │   ├── lgbm_ltr.py          LightGBM backend
│   │   │   ├── transformer_model.py Hourly transformer backend (Stage C-3)
│   │   │   ├── ngboost_head.py      μ/σ residual head
│   │   │   └── global_calibrator.py isotonic regression
│   │   ├── adapters/                LeanAdapter · RunnerAdapter · SimAdapter
│   │   ├── artifacts/               panel-ltr.json + .{backend}.bak + .previous + sidecars
│   │   └── strategy_config.json     + strategy_config.golden.json (drift-checked)
│   ├── renquant_103/                rollback strategy — adaptive regime per-symbol scanner
│   ├── renquant_102/                pre-trained scanner (legacy)
│   └── renquant_101/                single-stock classification (reference)
├── common/                          shared library (data, indicators, models, plotting)
├── live/                            live trading runner + broker abstraction
├── scripts/                         CLI tools (training, analysis, ops)
│   ├── train_104.py                 ★ FullTrainingPipeline driver + acceptance gates
│   ├── select_best_model.py         backend tournament + --promote
│   ├── model_dashboard.py           operator UX (read-only)
│   ├── finalize_challenger.py       shadow-window verdict generator
│   ├── check_challenger_window.sh   daily cron poll for window closure
│   ├── daily_104.sh / daily_103.sh  launchd-fired daily drivers
│   └── …                            ~50 ops/analysis tools
├── tests/                           ★ 2418 tests; pytest tests/ -v
├── data/                            local parquet cache + runs.db (gitignored)
├── doc/                             documentation (themed subdirs — see below)
└── CLAUDE.md                        AI-collaboration ground rules
```

---

## Documentation index

The `doc/` tree is themed; start at [`doc/README.md`](doc/README.md) for a navigable index. Top entries:

- **Architecture**
  - [`doc/arch/strategy-104.md`](doc/arch/strategy-104.md) — current active strategy spec
  - [`doc/arch/overview.md`](doc/arch/overview.md) — pipeline + data flow
  - [`doc/arch/decision-graph-103.md`](doc/arch/decision-graph-103.md) — decision flowchart (shared trunk)
  - [`doc/arch/strategy-103.md`](doc/arch/strategy-103.md) — rollback strategy spec
- **Components**
  - [`doc/components/model-selection.md`](doc/components/model-selection.md) — 4-tier acceptance + tournament + shadow SOP
  - [`doc/components/panel-ltr.md`](doc/components/panel-ltr.md) — primer + glossary
  - [`doc/components/buy-logic.md`](doc/components/buy-logic.md) — quality gates + portfolio QP
  - [`doc/components/sell-logic.md`](doc/components/sell-logic.md) — SellGateB + LimitSellsPerBar
  - [`doc/components/calibration.md`](doc/components/calibration.md) — saturation + score-DB
  - [`doc/components/transformer.md`](doc/components/transformer.md) — daily + hourly + Bug #21/#23/#24
  - [`doc/components/macro-factor-frame-design.md`](doc/components/macro-factor-frame-design.md) — VIX/HYG/UUP cross-asset
  - [`doc/components/portfolio-qp.md`](doc/components/portfolio-qp.md), [`rotation.md`](doc/components/rotation.md), [`databases.md`](doc/components/databases.md), [`training-pipeline.md`](doc/components/training-pipeline.md), [`trade-evaluation.md`](doc/components/trade-evaluation.md)
- **Operations**
  - [`doc/ops/usage.md`](doc/ops/usage.md) — workflow modes (research / validation / analysis / live / scheduled)
  - [`doc/ops/golden-config.md`](doc/ops/golden-config.md) — current golden config snapshot
  - [`doc/ops/setup.md`](doc/ops/setup.md), [`environment.md`](doc/ops/environment.md), [`tech-stack.md`](doc/ops/tech-stack.md)
  - [`doc/ops/transformer-promotion.md`](doc/ops/transformer-promotion.md), [`maintenance-103.md`](doc/ops/maintenance-103.md)
- **Research / experiments**
  - [`doc/research/papers-implemented.md`](doc/research/papers-implemented.md), [`scoring-research.md`](doc/research/scoring-research.md), [`rotation-research.md`](doc/research/rotation-research.md), [`watchlist-100.md`](doc/research/watchlist-100.md)
  - [`doc/experiments/ab-journal.md`](doc/experiments/ab-journal.md), [`panel-training-runs.md`](doc/experiments/panel-training-runs.md), [`sim-ab-results.md`](doc/experiments/sim-ab-results.md)
- **Roadmap**
  - [`doc/roadmap.md`](doc/roadmap.md) — living plan + decisions

---

## Development rules

See [`CLAUDE.md`](CLAUDE.md) for the full set; key ones:

1. **Logic graph is the source of truth.** Notebook ↔ LEAN ↔ kernel must agree on every decision branch.
2. **Every logical unit is a Task / Job / Pipeline.** New decision logic goes into `kernel/pipeline/` or `kernel/panel_pipeline/` as a Task wired into the appropriate Job. No hand-written loops bypassing the orchestration layer.
3. **Tests for every feature — and every bug.** A bug without a regression test is a bug you'll see again. 2418 tests as of 2026-04-26 round-7.
4. **Promotion thresholds are not floors for theoretically-sound wins.** Default promotion: APY win ≥ +2 pts on 27-mo OOS. Exception: live/sim parity fixes, theory-aligned wins where the predicted magnitude matches, mechanism-clean changes ship at < +2 pt.
5. **Unexpected A/B results = audit before accepting.** When theory predicts X and the result is ¬X, the first hypothesis is "my implementation has a bug", not "the theory is wrong".

---

## Strategy summaries

### renquant_104 (active, panel-LTR)

Cross-sectional ranker:
- **Stage 1**: per-ticker indicators (RSI, MACD, ADX, BB%, …) + SPY-relative neutralization + cross-sectional z-score
- **Stage 2**: panel features (size, momentum 12-1, beta 60d, residual mom) + macro factor frame (default off)
- **Stage 3**: XGBoost rank:pairwise model trained CPCV (6 splits, 2 test groups, 10d embargo). OOS mean IC 0.0482 (28-feature prod as of 2026-04-26).
- **Stage 4**: NGBoost head produces μ/σ for top-K candidates → edge-Sharpe gate
- **Stage 5**: Isotonic global calibrator → tier thresholds (0.10 / 0.20 / 0.30)
- **Stage 6**: Portfolio QP solves for rotation under correlation + sector + concentration constraints
- **Stage 7**: 8 acceptance gates (G1-G6 hard catastrophic, G7 hard floor, G8 soft variance) + 3 sim-based gates (G9/G10 hard, G11 soft) gate every retrain. Tournament + shadow infra ready.

Full spec: [`doc/arch/strategy-104.md`](doc/arch/strategy-104.md). SOP: [`doc/components/model-selection.md`](doc/components/model-selection.md).

### renquant_103 (rollback)

Per-symbol classification + Q-learning with a 3-layer regime detector (Hurst / CUSUM / GMM). Watchlist of 24 symbols. Full spec: [`doc/arch/strategy-103.md`](doc/arch/strategy-103.md).

### renquant_102 / renquant_101 (legacy)

Pre-trained scanner (102) and single-stock classification (101). Kept for reference/scaffolding.

---

## Quick links

- Bug reports / questions: [`doc/roadmap.md`](doc/roadmap.md) §Open decisions
- AI assistant ground rules: [`CLAUDE.md`](CLAUDE.md)
- Acceptance-gate SOP: [`doc/components/model-selection.md`](doc/components/model-selection.md)
- Run the test suite: `python -m pytest tests/ -v`
