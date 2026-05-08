# Session Handoff — 2026-05-07

**Author**: end-of-session checkpoint before user restart
**Status**: Production unchanged (27-feat XGB live). All today's work committed to main.

---

## 1. Production state (verify before any change)

```
strategy_config.json    → artifacts/panel-ltr.json (27-feat XGB)
strategy_config.golden  → artifacts/panel-ltr.json (matches live)

launchctl list | grep renquant.104:
  com.renquant.daily104        (active)
  com.renquant.intraday104     (active)
  com.renquant.retrain-panel104 (active, Sun)
  com.renquant.backup           (active)
  com.renquant.weekly-apy104    (active)

NOT loaded: open104, preclose104  (intentionally OFF since alpha158 revert)
```

**No alpha158 plist scheduled.** All today's experiments isolated to /tmp.

---

## 2. What shipped today (commits on main)

| Commit | Subject |
|---|---|
| `f46eaca` | refactor(qp): split EmitOrdersFromQPSolutionTask into helpers (§1c) |
| `db30874` | (superseded) cvxpy fallback capacity clamp |
| `d756be1` | feat(experiment-configs): SQLite table + tests |
| `9d1aa6f` | (superseded) turnover-reachable clamp |
| **`b0acf90`** | **refactor(qp): cvxpy + CLARABEL primary, drop SLSQP** |
| `f0e6deb` | feat(qp): cvxportfolio.SinglePeriodOpt opt-in backend |
| `5a636bb` | fix(qp): V8 cvxportfolio leverage bug — TurnoverLimit ÷2 + cash≥0 |
| `13b2cbc` | test(qp): pin qp_solver_backend config switch |
| Doc commits (~10) | README/STATUS/roadmap rewrite, 40 docs archived to shelved/, panel-ltr/portfolio-qp/buy-logic banners, E29 entry |
| `7712c76` | fix(rotation_convex): per-position cap in cvxpy path |
| `182e642` | feat(retrain): scripts/retrain_alpha158_linear.sh |
| `2386c69` | perf(transformer-dataset): vectorized rolling z-score |
| `821752a` | fix(walk-forward): set ranking.panel_scoring.artifact_path |
| `60d3a48` | feat(walk-forward): 5-cut × 60d driver + alpha158 daily plist (NOT loaded) |

Latest commit: `821752a` (artifact-path fix). All pushed.

---

## 3. Experimental data summary (persistent record)

### IC measurements

| Model | Features | OOS IC | Source |
|---|---|---|---|
| Production 27-feat XGB | 21 used | **+0.034** | `panel-ltr.json` `oos_mean_ic` (CPCV 6 splits, 15 folds) |
| alpha158_linear (sklearn OLS) | 158 | **+0.022** | E30 + 5-cut test mean today |
| alpha158 + XGBoost (rank:pairwise) | 158 | **+0.036** | 5-cut test mean today |
| Transformer (Qlib-faithful) | 158 | +0.026 | E30 (yesterday) |

### Walk-forward (5-cut × 60d, with FIXED artifact-path commit `821752a`)

**alpha158_linear daily-retrain v2** (each cut retrains on all data through train_end):

| Cut | Period | Sharpe | APY | 60d return | SPY 60d | **Alpha** |
|---|---|---|---|---|---|---|
| T1 | 2024-05→07 | +2.00 | +27.6% | +4.5% | +7.1% | −2.6 |
| T2 | 2024-08→10 | +0.75 | +14.7% | +2.4% | +11.1% | −8.7 |
| T3 | 2024-11→25-01 | +0.58 | +9.7% | +1.6% | +3.0% | −1.4 |
| T4 | 2025-02→04 | −4.24 | −36.6% | −6.0% | −15.8% | **+9.8** |
| T5 | 2025-05→07 | +1.85 | +24.5% | +4.0% | +11.3% | −7.3 |
| **mean** | | **+0.19** | **+8.0%** | | | **−2.0** |

**Verdict**: NO-GO. Mean alpha vs SPY = −2.0 pts. T4 (2025 Trump tariff crash) is the only positive alpha cut — alpha158_linear is a **defensive low-vol/stable-volume factor**, not an alpha generator.

**alpha158 + XGB v2**: 0 trades across all 5 sims (sim dispatch bug, not root-caused).

**Production 27-feat XGB walk-forward** (E27, earlier): mean alpha vs SPY = **−15.62%** (worse than alpha158_linear).

### Why both backends underperform SPY

Linear coef analysis: VMA + VSTD families dominate 80% of |coef|. The model is essentially a single low-volatility / stable-liquidity factor (Ang et al. 2006). Wins in defensive rotations (T4 +9.8 pts), loses 4-9 pts in momentum bull markets.

XGB has higher OOS IC (0.034) but worse walk-forward (-15.62) because tree splits overfit training-regime patterns.

**Both lose to passive SPY long-only at our 103-ticker scale.** Per Grinold's law expected at this breadth.

---

## 4. Open bugs / unfinished

1. **alpha158 + XGB walk-forward 0 trades** — sim dispatch bug. v2 grid script `scripts/walk_forward_60d_5cut_xgb.sh` ran but every sim emitted 0 BUY/SELL. Needs root-cause (probably `kind` field handling in PanelScorer dispatch).
2. **Stage 1 mechanical screen rejected 752 / 1009 R1K tickers for `no_sector_mapping`** — only 96 admitted (less than current 103 wl). `scripts/build_sector_map.py` needs to be run first to expand sector coverage before any universe expansion. **This is the blocker for Plan B Phase 1.**
3. **TaskList #36 "Phase 3 Live A/B alpha158_linear vs production XGB"** still in_progress — should be marked completed (decision: no promote) or deleted.
4. **scripts/train_panel_alpha158_xgb.py** committed (commit 8d787ae) but the model.score → model.predict fix didn't include adding `--train-end-date` to alpha158 retrain wrapper. Re-test before relying.

---

## 5. Plan B — Path forward (user-confirmed 2026-05-07)

User feedback: "当然是 B" (universe expansion + multi-horizon ensemble + Transformer long-term).

```
Week 1   Week 2   Week 3   Week 4
[P1]──→ [P2]──→ [P3]──→ [P4]
Universe Multi-h Trans-  Walk-fwd
expand   ensemble former  + promote
```

### Phase 1 — Universe 103 → 300+ (Week 1)

**Goal**: Grinold's √breadth lift IC from 0.022/0.034 → ~0.04/0.06.

**Avoid Track D wl183 TC collapse** (E26): use **cluster-based** admission (sector × market_cap × beta bucket — pick top-IC per cluster), not single greedy IC-additive.

**Steps**:
1. **Expand sector map**: `scripts/build_sector_map.py` to cover the 752 currently-unmapped R1K tickers. **THIS IS THE BLOCKER.**
2. Re-run `scripts/screen_stage1_mechanical.py` → expect ~400 admitted (vs 96 today)
3. Build alpha158 dataset for 300+ universe: `scripts/build_alpha158_qlib.py` w/ `--watchlist <new>`
4. Cluster admission (~300 final tickers)
5. Sanity train Linear: OOS IC should ≥ +0.04 on new universe (vs 0.022 on 103)

**PASS gate**: 280-320 ticker, OOS IC ≥ +0.04, no single new ticker −IC contribution.

**FAIL gate**: keep 103 watchlist; pivot Phase 2 onto hourly bars (Track C) instead.

### Phase 2 — Multi-Horizon Ensemble (Week 1-2)

**Goal**: fwd_5d + fwd_20d + fwd_60d 加权 → IC 0.04 → 0.05+

**Avoid E1 horizon-blender failure**: NO learned weights (ElasticNet -79% IC). Use **1/IC weighted** per DeMiguel et al. 2009 RFS.

**Steps**:
1. Train 3 alpha158_linear × {fwd_5d, fwd_20d, fwd_60d}
2. §5.2 sanity each (A/A + shuffled-label + time-shift placebo)
3. Ensemble: `score = Σ_h (IC_h / Σ IC) · score_h`
4. 5-cut × 60-day walk-forward (Linear, XGB, AND Linear+XGB ensemble)

**Linear vs XGB choice**:
- IC: XGB > Linear (0.036 vs 0.022)
- Walk-forward stability: Linear > XGB (Linear -2.0 vs XGB -15.62)
- **Use BOTH as ensemble** — their errors are likely uncorrelated (Linear regime-stable, XGB regime-adaptive). Heterogeneous ensembles often beat single best.

**PASS gate**: ensemble OOS IC > best single horizon AND walk-forward mean alpha vs SPY > 0.

### Phase 3 — Transformer Fine-Tune (Week 2-3)

**Goal**: Transformer beats Phase 2 ensemble.

**Avoid E30 Transformer < Linear**: with universe expanded to 300+ AND multi-horizon labels, Transformer has more data + more learning targets — should benefit more than Linear.

**Architecture priority** (from literature review committed with this doc):
1. **iTransformer** (Liu 2024 ICLR) — invert attention to cross-asset axis. Most aligned with cross-sectional ranking. Repo: `thuml/iTransformer`.
2. **MASTER** (Li 2024 AAAI) — market-guided cross-stock attention. Repo: `SJTU-Quant/MASTER`.
3. **AlphaPortfolio** (Cong et al 2021/2024) — end-to-end Sharpe loss. Most theory-clean.

`scripts/transformer_v4.py` is current Paradigm A (time-axis per-asset, Qlib-style). Phase 3 = port iTransformer or MASTER style.

**PASS gate**: Transformer OOS IC > ensemble IC by ≥ 0.005, walk-forward mean alpha vs SPY > +1 pt.

### Phase 4 — Promotion (Week 3-4)

Live promotion checklist:
- Daily retrain cron wired (scripts/retrain_*.sh + launchd plist)
- Smoke test --once paper, picks look reasonable
- Promote strategy_config.json + golden in same commit, `.previous.json` backups
- Watch 7d paper, then 14d paper, then live equity if metrics hold

---

## 6. Reference reading (for next session start)

| Topic | File |
|---|---|
| End-of-session this doc | `doc/archives/sessions/2026-05-07-session-handoff.md` |
| Engineering principles | `CLAUDE.md` §5 (especially §5.5 rollback, §5.10 saturate, §5.11 time-to-answer) |
| Failed experiments log | `doc/research/failed-experiments-log.md` (E27, E29, E30) |
| Roadmap | `doc/roadmap.md` (P0 = Plan B Phase 1) |
| Status | `doc/STATUS.md` |
| QP architecture | `doc/components/portfolio-qp.md` §0 |

### Key papers / repos for Plan B Phase 3

- iTransformer: thuml/iTransformer (Liu et al. 2024 ICLR)
- AlphaPortfolio: Cong, Tang, Wang, Zhang (2021/2024)
- MASTER: SJTU-Quant/MASTER (AAAI 2024)
- Qlib reference: microsoft/qlib `pytorch_transformer_ts.py`
- PatchTST: yuqinie98/PatchTST (Nie et al. ICLR 2023)
- Empirical Asset Pricing via ML: Gu, Kelly, Xiu RFS 2020 (the foundational ML-on-cross-section paper)

---

## 7. First actions next session

If you want to continue Plan B from here:

```bash
# Activate env (the only one that works — .venv is Python 3.9 incompatible)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate renquant

# Verify production untouched
git log --oneline -10
launchctl list | grep renquant.104   # should be 5 plists
diff <(jq -S . backtesting/renquant_104/strategy_config.json) \
     <(jq -S . backtesting/renquant_104/strategy_config.golden.json)  # should be empty

# Phase 1 first step: expand sector map (blocker)
python scripts/build_sector_map.py --help
# Then re-run Stage 1, expect ~400 admitted
```

Or pick a different starting point — see "Open bugs" §4 for things that could be fixed first (sim dispatch bug, etc.).
