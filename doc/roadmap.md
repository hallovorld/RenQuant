# RenQuant — Roadmap

**Single source of truth for what's next.** Ordered by priority. Last reset 2026-05-07 (post cvxpy QP refactor + alpha158_linear V7 holdout).

---

## End goal

Long-running quant trading workstation that does the boring infra well so the user can iterate on signal research:

- **Live alpha** — alpha158_linear (or successor) producing trades that net of cost/tax beat SPY by ≥ 5 pts annualized.
- **Walk-forward defensible** — 3-cut OOS (each 6 mo) with mean Sharpe ≥ 1.0; no single-cut promotions.
- **Self-maintaining** — daily retrain cron keeps every active model fresh; rollback rehearsed for each promotion.
- **Reproducible** — every config + artifact + run lineage tracked in `data/runs.db`; no file-system clutter.

---

## Right now

| | |
|---|---|
| Production model | 27-feature XGBoost (panel-ltr.json) — single-cut Sharpe 0.68, walk-forward −15.62% mean alpha (E27) |
| Researched winner | alpha158_linear panel-LTR — V7 single-cut Sharpe 2.009, walk-forward NOT yet validated |
| Portfolio QP | cvxpy + CLARABEL primary; cvxportfolio.SinglePeriodOpt opt-in (commit `5a636bb` fixes) |
| Live cron | open + preclose + intraday + daily + Sunday retrain (5 plists); open/preclose currently OFF pending alpha158 walk-forward + daily-retrain wiring |

---

## P0 — Plan B: universe expansion + multi-horizon ensemble + Transformer (2026-05-07 EOD pivot)

Walk-forward 5-cut v2 (with FIXED artifact-path) confirmed **alpha158_linear NO-GO**: mean alpha vs SPY = −2.0 pts, only 1/5 cut beats SPY (defensive factor). Production 27-feat XGB walk-forward (E27) also −15.62%. Both backends fail to beat passive SPY at our 103-ticker scale — this is structural per Grinold's law (breadth too small).

**User-confirmed plan**: scale up universe → multi-horizon ensemble → Transformer long-term.

### Phase 1 — Universe 103 → 300+ (Week 1)

**BLOCKER**: Stage 1 mechanical screen on R1K rejects 752/1009 for `no_sector_mapping`. Run `scripts/build_sector_map.py` to expand sector coverage FIRST, then re-run Stage 1.

Steps:
1. Expand sector map → re-run `scripts/screen_stage1_mechanical.py`
2. Cluster-based admission (sector × cap × beta bucket; top-IC per cluster) — avoids Track D wl183 TC collapse (E26)
3. Build alpha158 dataset for 300+ universe
4. Sanity train Linear: OOS IC ≥ +0.04 on new universe (vs 0.022 on 103)

**PASS gate**: 280-320 ticker, OOS IC ≥ +0.04, no single new ticker −IC contribution.

### Phase 2 — Multi-Horizon Ensemble (Week 1-2)

3 alpha158 models × {fwd_5d, fwd_20d, fwd_60d} → 1/IC weighted ensemble (DeMiguel et al. 2009; NOT learned weights — E1 ElasticNet lost -79% IC).

ALSO ensemble Linear + XGB heterogeneously (Linear = regime-stable defensive factor, XGB = regime-adaptive non-linear; errors likely uncorrelated).

**PASS gate**: ensemble OOS IC > best single horizon AND walk-forward mean alpha vs SPY > 0.

### Phase 3 — Transformer Fine-Tune (Week 2-3)

E30 showed Transformer IC < Linear IC at 290-ticker / 8-yr. With Phase 1 expanded universe + Phase 2 multi-horizon labels, Transformer has more data to leverage.

Architecture priority (literature: see session handoff):
1. **iTransformer** (Liu 2024 ICLR) — invert attention to cross-asset axis
2. **MASTER** (Li 2024 AAAI) — market-guided cross-stock attention
3. **AlphaPortfolio** (Cong et al 2021/2024) — end-to-end Sharpe loss

`scripts/transformer_v4.py` is current Paradigm A. Phase 3 ports iTransformer or MASTER.

### Phase 4 — Promotion (Week 3-4)

Full cron wire-up + paper smoke + 7d/14d watch before live equity.

### If Phase 1 fails

Pivot to hourly bars (Track C) — same breadth-lift goal via 16x more data points per ticker.
- Try ensemble of XGBoost + alpha158_linear scores.

---

## P1 — engineering hygiene

4. **Migrate side configs out of file system** (Task #38). DB schema + `migrate_experiment_configs_to_db.py` exist; `holdout_backtest.py --experiment-label` flag works. Remaining: live runner + retrain cron read-from-DB; delete stale `strategy_config.*.json` files once everything reads DB.
5. **Calibrator runtime SOFT-WARN** (CLAUDE.md status §1). Production `panel-rank-calibration.json` has `n_unique_prob_y=7 < 10` floor; refit after panel-LTR pin or boost `min_best_iter` so XGB plateaus higher.
6. **Test suite hygiene**. Currently ~14k passing; trim slow tests, surface ones that exercise sim-level integration (V8 leverage bug shipped because no sim test pinned cvxportfolio backend constraints).

---

## P2 — research

These are deferred until P0 ships. Don't start without explicit user request.

7. **Microstructure / hourly bar (Track C)**. Alpaca hourly cache empty (0/178 tickers). Data fetch + feature build ~3-5 days.
8. **Regime ensemble (T2-3)**. Wait for >150k panel rows or watchlist expansion past 150 tickers.
9. **Walk-forward retraining** (sliding train window). Per-cut retrain on the cut-specific train period; tells us robustness vs a single fixed model. ~10× compute of a static walk-forward.

---

## Closed (per CLAUDE.md status — don't re-open)

Each of these has a shelved design doc in `doc/archives/shelved/` for git-log provenance.

| Track | Verdict |
|---|---|
| Macro overlay (v1–v4) | All variants net negative IC at panel size 103. Revisit at 200+ tickers. |
| Asset embeddings (T2-2) | +0.0001 IC delta = no lift. |
| LightGBM panel | -60% IC vs XGBoost; REJECTED. |
| Boyd rotation (T2-4) | -2.5 APY pts; default OFF. |
| Insider feature (Track A) | -0.0008 contribution at 44% coverage. |
| PEAD enrichment (Track B) | 17-22σ negative on fwd_5d (label too short for drift). |
| Watchlist 183 (Track D) | Sharpe collapse: wl183 +0.55 vs wl103 +0.68. |
| Triple-barrier label (Track F) | +98bp lift was placebo (time-shift +60d also +). |
| Walk-forward XGB (E27) | Mean alpha vs SPY −15.62% ± 10.21% across 3 cuts. |
| QP refactor — adopt cvxportfolio | DONE 2026-05-06: cvxpy + CLARABEL primary, SinglePeriodOpt opt-in backend. |
| Doc reorganization | DONE 2026-05-07: 103 docs → 63; archived/shelved 40 + deleted REORG_PLAN. |

---

## How to use this doc

- Pick the topmost unblocked item.
- Open a small branch, ship the smallest reversible step, commit, test vs golden.
- If APY +2 pts (or CLAUDE.md §2a exception applies) → promote golden in the same commit → mark item done here.
- Otherwise update this doc with what you learned and pick the next item.

Working rhythm: ship don't ponder; one task in flight; commit each meaningful chunk.

---

## Renquant_105 (future, planned)

The 30-min level model. Designed but not started.

- Pre-requisite: renquant_104 stable for ≥ 3 months on live (build a baseline track record first).
- Pre-requisite: minute-bar panel > 1M rows (103 tickers × 2 yrs × ~16 bars/day).
- Independent strategy dir at `backtesting/renquant_105/`; does NOT reuse 104's `strategy_config.json`.

105 design work resumes after 104 is stably live.
