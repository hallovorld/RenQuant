# RenQuant — Status

**Last updated**: 2026-05-07 (post cvxpy QP refactor + alpha158_linear V7 holdout)

This is the **canonical status reference** — what's live, what just changed, what's open. CLAUDE.md's status block at the top of the repo is the same content, kept short for context loads.

---

## Right now

| | |
|---|---|
| Active strategy | `renquant_104` panel-LTR cross-sectional ranking |
| Active model | **alpha158_linear** (Qlib alpha158 features + sklearn LinearRegression on z-scored fwd_5d_excess) |
| Watchlist | 103 tickers |
| Portfolio QP | **cvxpy + CLARABEL** (Boyd/Stanford `cvxportfolio.SinglePeriodOpt` idiom, soft cash-drag penalty) |
| Backend switch | `qp_solver_backend` config = `cvxpy` (default) \| `cvxportfolio` |
| Broker | Alpaca paper |
| Test count | ~14,200 passing |

---

## Recent results

### V7 single-cut holdout — alpha158_linear + cvxpy backend (2026-05-07)

Train end 2025-05-04, sim 2025-05-05 → 2025-11-04 (6 mo, 128 trading days):

| Metric | Value |
|---|---|
| APY | **+39.22%** |
| Sharpe | **+2.009** |
| Sortino | +1.665 |
| Calmar | +5.673 |
| Max DD | 6.91% |
| Win rate | 73% |
| Buys / Sells | 73 / 77 |
| Longest no-trade streak | 4d (was 128d before the QP refactor) |

> **Caveat**: single-cut. Per CLAUDE.md §5.10 / E27, the previous production XGB had single-cut Sharpe 0.68 but walk-forward 3-cut showed **mean alpha vs SPY = −15.62% ± 10.21%**. alpha158_linear's walk-forward truth is not yet measured. Treat V7 as a strong lower bound but not as walk-forward-promotable until 3-cut runs land.

---

## What just changed (2026-05-06 → 2026-05-07)

### QP refactor — cvxpy + CLARABEL primary, drop SLSQP

The pre-2026-05-06 solver was 700 lines of `scipy.optimize.minimize(method="SLSQP")` with hand-coded gradients, hand-coded slack clamps, and a half-baked cvxpy fallback. V4/V5 alpha158_linear sims produced **0 trades over 128 days** because the constraint set `Σwp ≥ min_invested_pct` + `‖Δw‖₁ ≤ turnover_max` was mathematically infeasible from cash (need 70% turnover in one bar to satisfy a 70% floor with a 30% turnover cap).

**Fix shape (commits b0acf90 … f0e6deb)**:
- Drop SLSQP entirely (-296 lines).
- cvxpy + CLARABEL primary, OSQP/SCS fallback chain.
- `min_invested_pct` becomes a **soft cash-drag penalty** (`cash_drag_lambda · max(0, target − Σwp)`), not a hard floor — Boyd/cvxportfolio textbook pattern.
- All bespoke features (Almgren-Chriss impact, Brown-Smith tax, RU CVaR, Garlappi robust μ, Garleanu-Pedersen signal decay) re-expressed as cvxpy DCP terms.
- Drop `tanh` smoothed fixed cost (not DCP, not used in production).
- Multi-bar Garleanu-Pedersen ramp pinned by acceptance tests.
- Parallel `cvxportfolio.SinglePeriodOpt` backend (commit f0e6deb) using Boyd's actual policy + cost + constraint classes; opt-in via `qp_solver_backend = "cvxportfolio"`.

**Verified**: 161+ QP-related tests green; soft penalty formulation cannot be infeasible by constraint conflict; per-bar Garleanu-Pedersen ramp produces canonical 30%/60%/70% deployment from cash.

### Side-config DB

Added `data/runs.db::experiment_configs` SQLite table for side-config storage (`scripts/migrate_experiment_configs_to_db.py`). `holdout_backtest.py --experiment-label <name>` inflates the config from DB instead of reading a side `strategy_config.*.json` file. Replaces the 60+ side-config-file proliferation problem.

### Doc cleanup (2026-05-07)

- 40 closed-experiment design docs moved to `doc/archives/shelved/`.
- Doc tree: 103 → 63 markdown files. Live docs match CLAUDE.md's canonical pointer list.

---

## Closed (don't re-open)

Per CLAUDE.md status — these are decided. The shelved design docs live in `doc/archives/shelved/` for `git log --follow` history.

| Track | What | Verdict |
|---|---|---|
| Macro overlay (v1–v4) | All variants tested | Net negative IC at current panel size; revisit at 200+ tickers |
| Asset embeddings (T2-2) | 16D embeddings full retrain | +0.0001 IC delta = 0 lift |
| LightGBM panel | Replacement for XGBoost | -60% IC, REJECTED |
| Boyd rotation (T2-4) | Per-bar rotation policy | -2.5 APY pts; infra kept, default OFF |
| Insider feature (Track A) | SEC EDGAR enrichment | -0.0008 contribution at 44% coverage |
| PEAD enrichment (Track B) | days_since/decay/signal | 17-22σ negative on fwd_5d (too short for drift) |
| Watchlist 183 (Track D) | wl103 → wl183 expansion | TC collapse: Sharpe +0.55 vs wl103 +0.68 |
| Triple-barrier label (Track F) | Hit-time-matched label | +98bp IC was placebo (time-shift +60d also +) |
| Walk-forward XGB (E27) | 3-cut OOS test | Mean alpha vs SPY −15.62% ± 10.21% |

---

## Open priorities

| Priority | Item | Why |
|---|---|---|
| 🔴 **P0** | Walk-forward 3-cut on alpha158_linear | V7 single-cut Sharpe 2.01 is great but not walk-forward proven. Compare to E27 baseline. |
| 🟡 P1 | Calibrator retrain | Production `panel-rank-calibration.json` `n_unique_prob_y=7` < runtime floor 10. Refit after panel-LTR pin. |
| 🟡 P1 | Microstructure / hourly bar (Track C) | Alpaca hourly cache empty (0/178 tickers). Data fetch + feature build ~3-5 days. |
| 🟢 P2 | Regime ensemble (T2-3) | Wait for >150k panel rows. |

---

## Doc map (canonical pointers)

See [`README.md`](README.md). Quick start:

- **What runs**: [`arch/strategy-104.md`](arch/strategy-104.md), [`arch/overview.md`](arch/overview.md), [`arch/decision-graph-103.md`](arch/decision-graph-103.md)
- **How it runs**: [`ops/usage.md`](ops/usage.md), [`ops/golden-config.md`](ops/golden-config.md), [`ops/maintenance-103.md`](ops/maintenance-103.md)
- **Why it runs the way it does**: [`components/portfolio-qp.md`](components/portfolio-qp.md), [`components/panel-ltr.md`](components/panel-ltr.md), [`components/buy-logic.md`](components/buy-logic.md), [`components/sell-logic.md`](components/sell-logic.md)
- **Failed-experiment durable record** (CLAUDE.md §5.7): [`research/failed-experiments-log.md`](research/failed-experiments-log.md)

---

## Engineering principles (CLAUDE.md §5)

These are the rules. Read CLAUDE.md §5.1 through §5.12 before any non-trivial change.

- **§5.1** Run relevant tests before every commit/push.
- **§5.2** Every new number ships with at least one sanity check (A/A, shuffled-label, time-shift placebo).
- **§5.3** Every "fix" must name the invariant that prevents the entire class of bug.
- **§5.5** Production-touching changes must rehearse rollback.
- **§5.6** Definition of "fixed" = full 24h audit clean.
- **§5.7** Every failed experiment goes into `failed-experiments-log.md` same day.
- **§5.10** Saturate hardware (M2 Pro 10 cores, 32 GB).
- **§5.11** Optimize experiment design for time-to-answer (range-find first; greedy/sweep only after).
- **§5.12 / §5.12a** Every architectural decision backed by literature OR mature open-source — actually READ, not name-dropped.
