# RenQuant — Roadmap

**Single source of truth for what's next.** All items ranked by **expected ROI (Sharpe-lift ÷ effort)** with status flag. Past dated plans archived to [`doc/archives/plans/`](archives/plans/) for provenance.

**Last consolidated**: 2026-05-18 NIGHT (consolidated 7 plan/backlog files → 1)
**Last incremental sync**: 2026-05-18 18:35 PT (anti-churn 4-layer, Platt switch, PatchTST scorer + DOE in flight, sentiment fully wired ON)

---

## 🔴 PRIME DIRECTIVE (locked 2026-05-14)

**RenQuant is a regime-conditional strategy.** Every knob lives at `regime_params.<REGIME>.<knob>`, every experiment asks "which regime?", every evaluation reports per-regime BEFORE pooled. Pooled-mean buries the actionable signal — 2026-05-14 shorts NEITHER pooled but regime-stratified gave clear deploy/skip per regime.

Full rules: [`CLAUDE.md`](../CLAUDE.md) PRIME DIRECTIVE section.

---

## 📍 Current state (2026-05-18)

| | |
|---|---|
| Production model | `panel-ltr.alpha158_fund.json` — XGBoost rank:pairwise, **169 features** (alpha158 + 5 fund + 3 PEAD + 3 SUE) |
| NGBoost head | Trained + promoted, val_IC +0.0352, σ-calib +0.274 (commit `e267101`). σ-wire stays OFF in golden per 3-condition A/B |
| Calibrator | pool_IC +0.094, er.y ∈ [-0.105, +0.200] (post 5/15 P0 refit, train-site clip) |
| Regime detector | 5-day BEAR + vol-cluster CHOPPY (commit `0a192c4`); HMM hysteresis sticky N=10 bars |
| Walk-forward gate | ENFORCED: daily train STAGES only; weekly Saturday 04:00 PT promote with full WF + sanity battery |
| Tax / lot accounting | HIFO default (commit `bc18795`, Berkin-Jeffrey 1990); IRC §1091 wash-sale + §1233 short-sale rules wired |
| Live broker | Alpaca LIVE (real money). 11 launchd plists active. PAPER mandate overridden ONLY for explicit `--broker alpaca` e2e invocations |
| **C5 news sentiment** | **2026-05-18 LIVE**: 14 regime entries `enabled`; panel-LTR retrained 169→172 feats; daily refresh cron staged |
| **C1 options-IV** | **2026-05-18 ACCUMULATION**: fetcher + parser shipped, daily snapshot cron staged. Wait n_daily_rows ≥ 120 (~6mo) |
| **PatchTST sequence model** | **2026-05-18 P0 ACTIVE**: scorer + registry + shadow shipped; DOE sweep in flight (PID 7998, ETA 20:15 PT) |

---

## ✅ DONE — recent ships (last 2 weeks, for context)

| Item | Date | Commit | Impact |
|---|---|---|---|
| **Detector**: 5-day BEAR + vol-cluster CHOPPY | 2026-05-17 | `0a192c4` | Catches SVB / DeepSeek / Aug-2024 |
| Per-regime σ-wire kernel + hysteresis (DORMANT) | 2026-05-17 | `0a192c4`, `e267101` | Infra ready; flag OFF per A/B |
| HIFO default lot selection | 2026-05-17 | `bc18795` | Tax-optimal; expected +0.5pp APY |
| Walk-forward gate enforcement | 2026-05-17 | `96af42b` | Daily train STAGES only; weekly does promote |
| Sunday-sweep / monthly-cal acceptance gates | 2026-05-17 | `477b94c`, `637594e` | Pre-refit backup + IC-non-collapse + rollback |
| DDV disabled globally (HXZ 2020) | 2026-05-17 | `d318060` | Distress anomaly fails to replicate |
| min_share_floor for high-price stocks | 2026-05-17 | `d318060` | Unblocks $700+ share-price names |
| STATE-EXT-SELL pending-order false-positive fix | 2026-05-17 | `e267101` | HON / META no longer wash-sale-blocked |
| P-FEATURE-COVER per-regime preflight gap | 2026-05-17 | `e267101` | NGB feat coverage now per-regime |
| Calibrator P0 (er.y clip + saturation guard) | 2026-05-15 | `b16e2a1` | er.y was up to +1.0 corrupting Kelly μ |
| Phase 3 μ/σ wiring + regime_momentum + deep_drawdown_veto | 2026-05-15 | activated in golden | Kelly no longer mu_none=N |
| NGBoost SUSPECT→CONFIRMED via 5-seed proper Duan 2020 | 2026-05-15 | retrain run | val_IC +0.0352 ± 0.0036 (significant) |
| Bug C: T+2 pending NAV omission | 2026-05-11 | `29e34b0` | Phantom ±sale_amount returns; corrupted all prior sim metrics |
| 6 silent-feature bugs fixed | 2026-05-09 | various | 70 regression tests + 4 universal model contracts |
| QP refactor to cvxpy + CLARABEL | 2026-05-06 | — | Boyd cvxportfolio reference |
| Sentiment data fetcher (Alpaca News, 14-day chunked) + FinBERT scorer | 2026-05-18 | `a6b8080` | 57827 rows backfilled, 0 sanity fail |
| C5 sentiment WIRED ON: 14 regimes enabled + panel-LTR retrained 169→172 + daily cron | 2026-05-18 | `e539a17`, `e8168f7`, `aad7b97`, `b3809b0`, `82c3be5` | Regime-conditional gate; HIGH_SPIKED expected primary lift |
| wl200 watchlist promotion (replace wl103, 142-ticker overlap, quality-first) | 2026-05-18 | `a21904e` | Breadth √(142/103)=1.17× IR per Grinold-Kahn |
| MCD anti-churn 4-layer: min_reentry_days=5 + calibrator-saturation NEW-BUY abstain | 2026-05-18 | `7e17e65`, `91e188a` | Blocks same-day rebuy + abstains when prob flat |
| Calibrator isotonic→Platt switch + flat-region preflight (P-CALIBRATOR-FLAT-REGION) | 2026-05-18 | `1428c26`, `1d0317a`, `467e824` | Eliminates flat-region tie-breaks; n_unique_prob_y 100 |
| Live inference μ̂ 17× tighter than training — root cause documented | 2026-05-18 | `f71f8ae` | Foundation-level finding driving Platt switch |
| PatchTST infra: scorer + model registry + shadow scoring (MLflow) | 2026-05-18 | `c7f710e`, `9a6cca1`, `a80ba96`, `5bad1a8` | One-config swap xgb↔patchtst; zero-risk shadow rollout |
| Momentum features: J-T 12-1 + 52w-distance + sector_mom (pandas_ta_classic) | 2026-05-18 | `fc32385`, `a355a69`, `72` | Mature library, 11 tests |
| 3 LIVE Alpaca orders FILLED at Monday open (HON, EQIX, META) | 2026-05-18 | — | $2,099 / $10,578 equity (~20% rebalance) |
| CLAUDE.md §0: execute immediately, never wait for next session | 2026-05-18 | `a7bb9b7` | User mandate locked in |

## ❌ CLOSED / REJECTED — do NOT re-open without new evidence

| Item | Date | Verdict | Reason |
|---|---|---|---|
| Long-short (#0) | 2026-05-17 | **SKIP** | Bottom decile 60d-ann = +0.58% (positive!); Kelly-Gu-Xiu 2020 needs −10 to −15%. Saves 3-4 weeks engineering |
| Vol-adjusted label (#5) | 2026-05-17 | **NEGATIVE** | val_IC=−0.0189 vs raw +0.0352 (Δ=−0.054). Hypothesis rejected by quality gate |
| Insider trading (#6) retest | 2026-05-18 | **NEGATIVE** | Both insider_signal and insider_xs_rank lower val_IC + placebo persistence 100-142% |
| Multi-horizon ensemble (#11 / E42 / B4) | 2026-05-18 | **MARGINAL** | fwd_20d val_IC +0.0291 vs fwd_60d +0.0333 — saves 1 week |
| σ-wire A/B 3-condition (global / per-regime / +hysteresis) | 2026-05-17 | **ALL NULL/NEG** | global +3.01pp NULL; per-regime −4.70pp; +hyster −7.89pp |
| Per-sector pure-alpha label | 2026-05-09 | NO-GO | Persistence 89%; pure-alpha drops to +0.005 |
| NGBoost σ-aware sizing E55 | 2026-05-09 | NO-GO | 27-mo A/B −3.78 APY pp / −0.14 Sharpe — but root cause was σ wire bug; theory now confirmed |
| wl183 bottom-up | 2026-05-05 | NO-GO | Transfer-coef halving (E26); replaced by quality-first plan (#4 below) |
| wl1640 R2K full Russell | 2026-05-08 | NO-GO | XGB IC dropped 75% |
| Macro overlay v1-v4 | various | NO-GO | All variants net negative IC at 103 panel size |
| Asset embeddings T2-2 | 2026-04-29 | NO-GO | +0.0001 IC delta |
| Boyd rotation T2-4 | various | NO-GO | -2.5 APY pts |
| Track F regime alpha overlay | various | NO-GO | +98bp was regime-persistence fitting |
| meta-label classifier | 2026-05-11 | OFF (theory) | AUC 0.55 ≈ random; theory not measurement |
| Phase A-C QHead variants E51/E52 | 2026-05-09 | NO-GO | Architectural changes don't break panel ceiling |
| max_position_pct sweep 8% | 2026-05-11 | NO-GO | Bug-C-flipped: post-fix 8% LOSES by 8.2pt vs baseline 20% |
| Grinold-Kahn α→μ (HIGH_CALM/HIGH_SPIKED) | 2026-05-12 | DEFERRED | +18% in HIGH_CALM / −32% in HIGH_SPIKED; blocked on regime-aware conditional deploy (now possible post-detector-fix; revisit if needed) |

---

## 🎯 ACTIVE — next P0 items (ROI ranked)

### 1. 🔬 PatchTST sequence model — **DOE sweep in flight** (2026-05-18 PM)

**Status (2026-05-18 PM)**:
- Scorer shipped (`c7f710e`): `PatchTSTPanelScorer` with `score_with_history()` end-to-end working in prod inference (OMP fix at module top for xgboost↔PyTorch coexistence).
- Model registry shipped (`9a6cca1`): one-config-knob swap `kind: xgb | patchtst` via decorator pattern; extensible for LGBM/NGBoost/future.
- Shadow scoring shipped (`a80ba96`, `5bad1a8`): alt-model runs full pipeline + records via MLflow, **no order submission** — safe rollout pattern.
- 5-seed prototype verdict (`4ee09c3`): seed-42 val_IC +0.050 / seed-43 val_IC −0.143. High variance ≠ confirmed signal; **not a winner over XGB at current data scale**.
- Optuna sweep tried + DELETED today (PRIME DIRECTIVE violation: pooled-mean objective picks regime-fragile models).
- **DOE sweep IN FLIGHT** (PID 7998, started 18:27 PT): 2^(4-1) Resolution IV fractional factorial + 1 center = 9 points × 3 seeds = 27 runs, ETA ~20:15 PT.
  - Knobs: `lr ∈ {1e-4, 1e-3}`, `weight_decay ∈ {1e-4, 1e-1}`, `warmup_epochs ∈ {2, 6}`, `seq_len ∈ {16, 60}`
  - **Objective = min-across-regime val IC** (PRIME DIRECTIVE compliant, NOT pooled mean)
  - Output: `artifacts/patchtst_doe/{runs.csv, main_effects.csv, interactions.csv, summary.md}`

**Pass gate (per CLAUDE.md §5.14.4 + PRIME DIRECTIVE)**:
- DOE finds a point where **min-regime val_IC ≥ XGB-equivalent min-regime val_IC** (per-regime, not pooled)
- 3-seed confirmation at predicted optimum with std/mean ≤ 0.5 (low variance)
- DSR > 0.5 OR PBO < 0.5 at confirmation (Bailey-Lopez de Prado 2014)
- User criterion: "PatchTST 只要不是比 xgb 差很多就换" — even tie acceptable for architectural diversification

**Next steps after DOE finishes**:
- IF best point passes: BBD (Box-Behnken) zoom-in around best knob region → predicted optimum → confirmatory run → **shadow registration** (user 2026-05-18 decision: HF PatchTST = first real shadow model, 1-2 wk MLflow log of rank divergence vs XGB primary, THEN decide promote-to-primary)
- IF best point fails per-regime gate: close research thread, archive to `failed-experiments-log.md`

**Variance-reduction protocol for HF DOE (preprocessing + ensemble)**:

Current DOE shows pt_00 best point with **mean min_regime=+0.078 ± 0.058** (CoV 0.7) — high upside but high seed variance. Standard literature stack to reduce variance (CLAUDE §5.12 canonical refs):

Preprocessing (bake into `scripts/patchtst_hf.py` data loader):
1. **CSRankNorm** features per-day (Kelly-Gu-Xiu 2020 RFS standard): `df.groupby('date')[feats].rank(pct=True) - 0.5`
2. **Label Winsorize ±3σ** (`scipy.stats.mstats.winsorize`)
3. **HF default `scaling=std`** ≡ RevIN (Kim 2021 ICLR) — already free

Training-side (in DOE script):
4. **N-seed Ensemble** (predict-average across 5 seeds) — Lakshminarayanan 2017 NIPS; variance ↓ √N → pt_00 σ 0.058 → 0.026
5. **SWA** (`torch.optim.swa_utils.AveragedModel + SWALR`) for confirmatory final training only — Izmailov 2018 UAI; 5-15% variance reduction without ensemble cost

NOT doing (ROI low / off-mandate):
- T-Fixup init (§5.12 violation — custom code)
- Snapshot ensemble (SWA strictly better)
- Time-series augmentation (ranking task incompatible)
- Sector residualization (overlaps with sector_mom feature)

**Shadow model wiring (sequencing after HF DOE)**:
1. Train confirmatory HF PatchTST on full 5-cut walk-forward at predicted optimum
2. Add `hf_patchtst` kind to `kernel/panel_pipeline/model_registry.py`
3. Write `HFPatchTSTScorer` adapter (sequence input → score; mirror existing `PatchTSTPanelScorer` API)
4. Register in `golden.json::ranking.panel_scoring.shadow_models` as kind=hf_patchtst
5. e2e Alpaca live run → verify `mlruns/renquant_104_shadow/` populated
6. Daily: prod runs primary XGB; ApplyShadowScoringTask logs HF rank + mean_diff + corr + top5_overlap to MLflow
7. Weekly review: `optuna-dashboard` or `mlflow ui` → if HF shadow consistently leads XGB primary on per-regime metrics, propose promote-to-primary via config flip

References: Nie et al 2023 ICLR *PatchTST*, Lim et al 2021 ICLR *TFT*, Box-Hunter-Hunter 2005 (FrFact), Qlib `pytorch_*_ts.py`.

### 1c. ~~News-sentiment integration~~ — **SHIPPED 2026-05-18** (was P0 #1)

Moved to DONE: regime-conditional gate live across 14 regimes; panel-LTR retrained 169→172 features (`sentiment_pos_share`, `mean_sentiment`, `n_articles`); daily refresh cron staged. Per-regime A/B verdicts will accumulate via prod data; revisit pass-gate (HIGH_SPIKED ΔSharpe ≥ +0.10) once we have 8+ live weeks.

Original IC verdict (regime-stratified): [`doc/research/2026-05-18-news-sentiment-ic-verdict.md`](research/2026-05-18-news-sentiment-ic-verdict.md)

### 1b. Options-IV integration — accumulation phase

**Status**: fetcher shipped; today's EOD snapshot done; daily cron staged. Alpaca Free Options gives current-snapshot only (no historical chains), so panel integration requires ~6 months of daily accumulation. Resume when n_daily_rows ≥ 120.

Refs: Bali-Hovakimian 2009 *MSci*, Cremers-Weinbaum 2010 *JFQA*, Goyal-Saretto 2009 *JFE*.

### 2. ~~Quality-first watchlist expansion to wl200~~ — **SHIPPED 2026-05-18** (was P0 #2)

**Outcome**: wl200 (142 ticker, quality-first) promoted to LIVE; replaced wl103 (commit `a21904e`). Panel-LTR retrained on expanded universe. Breadth lift √(142/103) ≈ 1.17× per Grinold-Kahn `IR = IC × √Breadth`.

**Followup (still open)**: 16-window WF Sharpe vs old baseline — accumulating as prod runs. Revisit if Sharpe ≥ baseline floor not met after 8 weeks.

References: Grinold-Kahn 1999 §6, Hou-Xue-Zhang 2020 *RFS*, Cakici-Cooper-Schmidt 2023 *JFE*.

### 3. ~~Smart-orders integration (VWAP execution wiring)~~ — **DEFERRED**

**Status (2026-05-18 re-scope)**: `kernel/execution/smart_orders.py` module exists + 20 tests; **DEFERRED until account scale ≥ $50k**.

**Why deferred**: at current $10k account, typical trade = 1-3 shares × $300 notional on liquid names (AAPL, SPY, EQIX). Slippage at this scale is already ~1bp (1 cent / share on a $200 share). Smart-orders' 5-10bps savings emerges at $50k+ accounts with multi-thousand-share orders. ROI/effort below other P0 items.

**Resume trigger**: account NAV crosses $50k OR per-trade notional exceeds 1% of ticker ADV (would hit the Almgren-Chriss price-impact regime).

References: Almgren-Chriss 2000 *J. Risk* §4 — linear price-impact regime ≤ 1% ADV/slice.

### 4. LightGBM with GICS sector encoding (~+0.05-0.10 Sharpe, 3 days)

**Status**: BLOCKED on GICS sector mapping. E48 LGB retest was on alpha158+5fund+3PEAD; marginal NO-GO without categorical handling.

**Theory**: Ke et al 2017 NeurIPS — LightGBM native categorical encoding handles sector membership without one-hot blow-up. Cat embedding learns sector-level mean offsets / risk parameters automatically.

**Engineering** (3 days):
- D1: source GICS sector data (SimFin or yfinance.Ticker.info)
- D2: add `sector` column to panel; retrain LGB with `categorical_feature=['sector']`
- D3: WF + sanity battery vs current XGB baseline

**Pass gate**: ΔIC ≥ +0.01 (5-seed std) AND placebo persistence < 70%.

References: Ke et al 2017 NeurIPS *LightGBM*, Prokhorenkova et al 2018 *CatBoost*, Qlib `LGBModel`.

### 5. Student-t σ̂ calibration (~+0.10 Sharpe via Kelly fix, 1 week)

**Status**: prerequisite for any future NGB σ-wire re-activation. Today's audit showed ±2σ̂ coverage = 88% (Gaussian expects 95%) → fat tails.

**Theory**: Lopez de Prado 2018 AFML §15. Replace Gaussian residual likelihood with Student-t. Better tail calibration → less Kelly under-sizing in normal regime, less over-sizing in tail.

**Lower priority** because NGB σ-wire is currently DORMANT (A/B failed). Only useful when next σ-wire attempt happens; that needs separate motivation.

References: Lopez de Prado 2018 AFML §15, Romano-Patterson-Candès 2019 NeurIPS *Conformalized Quantile Regression*, `mapie` library, scipy.stats.t.

---

## ⏳ DEFERRED — P1 architecture (medium-cost, less-clear ROI)

> **Note**: PatchTST moved to **P0 #1** (2026-05-18) — DOE sweep in flight, scorer + registry + shadow infra shipped.

### 7. Microstructure features (paid data, 1-2 weeks, $10/mo)

Easley-O'Hara 1987 / Kyle 1985 / Hasbrouck 2007 / Cont et al 2014. Order-flow imbalance, depth-3 book features, Kyle lambda.

**Risk**: built for sub-daily horizon; may not help daily 5d/60d strategies. Low priority for current model class.

### 8. renquant_105 — 30-min model (future, separate strategy)

Pre-requisites:
- renquant_104 stable on live for ≥ 3 months with WF Sharpe ≥ 1.0
- Minute-bar panel > 1M rows (103 tickers × 2y × ~16 bars/day)

Independent strategy dir `backtesting/renquant_105/`. Easley-Lopez de Prado-O'Hara 2012 microstructure ML literature.

---

## 🔧 P2 operational hygiene (compound, no Sharpe lift)

| Item | Status | Effort |
|---|---|---|
| Side-config DB migration | Started; live + retrain cron still read JSON | 2-3 days |
| Test suite hygiene (trim slow, add sim-level integration) | ~14k tests; some slow; coverage gaps in sim path | ongoing |
| Daily sentiment + IV refresh crons | After integration | 1 day (`task #60`) |
| 12 pre-existing test failures | Kelly DD scale / vol-target / dashboard / calibrator clip | 1-2 days |
| **架构肃清: 删所有 self-written + unused 代码** | **NEW 2026-05-18**: user mandate "所有代码". See dedicated section below | 1-2 days |

### 🧹 Architectural cleanup — delete all self-written + unused code

**User mandate 2026-05-18**: "自己写的，且没有用的代码都尽量删掉" + "我是说所有代码！". Defer批量执行后再做; precondition is current DOE finish + HF pivot land. Until then, list candidates so nothing slips.

**Tier A — confirmed dead today (deferred until DOE finishes ~23:15 PT 2026-05-18)**:
- `scripts/transformer_v4.py` — 784 LOC custom transformer/iTransformer/PatchTST/LSTM/MLP. Replaced by `scripts/patchtst_hf.py` (HF transformers backbone).
- `scripts/qlib_transformer_v5.py` — another custom impl, untested in prod
- `scripts/patchtst_doe_sweep.py` + `tests/test_patchtst_doe_sweep.py` — custom DOE orchestrator; will be replaced by `scripts/patchtst_doe_hf.py` (5-cut walk-forward + HF backbone)
- `backtesting/renquant_104/kernel/panel_pipeline/patchtst_scorer.py` — custom scorer; replace with HF version
- `kernel/panel_pipeline/model_registry.py` PatchTST handler — point at HF script

**Tier B — broader audit pending (when bored)**:
Per §5.13.2 "any new module is dead until grep proves prod imports it" — sweep entire codebase, list every `kernel/` or `scripts/` file with 0 non-test imports. Likely candidates from memory:
- Old `archive/` / `legacy/` / `_old.py` files (if any)
- Stub scripts in `scripts/` that haven't been touched since pre-Bug-C era
- Test fixtures that test deleted code

**Tier C — custom impl candidates for lib replacement** (not necessarily dead, but violates §5.12 "use canonical refs"):
- Anywhere we hand-roll regression/classification/optimization when sklearn/cvxpy/qlib exists
- Hand-written feature engineering when `pandas_ta_classic` / `qlib.contrib.data.handler` covers it
- Custom NaN/inf guards when pandera/numpy native handles it

**Protocol** (per §5.6 "fixed = 24h audit clean"):
1. Sweep `grep -rln 'import X\|from X' --include='*.py' | grep -v __pycache__ | grep -v 'tests/'` for each candidate
2. If 0 non-test prod imports → mark for delete
3. Delete in commit of ≤10 files at a time (rollback-friendly)
4. Verify full test suite green after each batch
5. Update this section as items are cleared

---

## 📦 Bug-C reassessment backlog (added 2026-05-11; revisit when bored)

Bug C (commit `29e34b0` 2026-05-11) inflated Vol by ~75× and corrupted every pre-fix sim verdict. These items were rejected on contaminated numbers; not all need retest, only the ones where the rejection magnitude was within today's noise floor.

| Item | Pre-fix verdict | Why revisit | Cost |
|---|---|---|---|
| B1 Time-decayed stop tightening | −4.33 pp APY | Sharpe / MaxDD numbers inflated; theory (Schwager 1992) defensible | 20-line code + 9 sims |
| B2 SDL skip-if-winner | +0.05 pp APY "noise" | Real noise floor was Bug-C-inflated; +0.05 might be real | 6 sims |
| B5 Trend overlay | "fires but redundant" | Pre-fix DD halt was hyperactive; trend has more room post-fix | 6 sims |
| B6 DD-Kelly scaling | "didn't fire" | Bug-C MaxDD inflated; binds at smaller `dd_max` post-fix | 9 sims |
| B7 CVaR λ re-sweep | ±7.6 pp noise (n=3) | Wide σ; could resolve with more windows | 25 sims via Box-Behnken |
| B8-B10 stop-loss / CVaR / Phase4 ablations | various | Pre-Bug-C measurements | 18+ sims |
| B12 27-mo baseline rerun | Non-reproducible | Reliable post-fix | 1 sim, ~3h |

**Drain protocol** (per CLAUDE.md §5.13.4):
1. One at a time (no batch — Bug C taught us contamination compounds)
2. Mean ± std from ≥ 5 runs minimum
3. Box-Behnken only after single-knob shows ≥ +0.10 Sharpe
4. Document outcome in `failed-experiments-log.md`

---

## 📜 Methodology lock-ins

| Discipline | Reference | Tool |
|---|---|---|
| 3-tier promotion gating (Tier 1 REJECT / Tier 2 SCREEN / Tier 3 LIVE) | [`doc/research/promotion-methodology.md`](research/promotion-methodology.md), CLAUDE.md §5.13.4a | `scripts/analyze_experiments.py` |
| §5.2 sanity battery (A/A + shuffled-label + time-shift) | CLAUDE.md §5.2 | `scripts/ic_eval_*.py` |
| Walk-forward gate + DSR/PBO | Bailey-Lopez de Prado 2014, Bailey-Borwein-LdP-Zhu 2015 | `kernel/model_acceptance.py`, `weekly_wf_promote.sh` |
| Box-Behnken for multi-knob | Box-Behnken 1960 *Technometrics*, CLAUDE.md §5.14 | `pyDOE2.bbdesign` |
| Regime-stratified analysis (PRIMARY) | CLAUDE.md PRIME DIRECTIVE | `scripts/analyze_regime_stratified.py` |
| Rigorous batch analyzer (Newey-West HAC + block bootstrap) | Romano-Wolf 2005, López de Prado 2018 | `scripts/analyze_panels_rigorous.py` |
| Acceptance gates on every promote artifact | CLAUDE.md §5.13.15 | `_check_wf_gate`, `weekly_wf_promote.sh` |

---

## 🎯 End-goal target

**APY 20% / Sharpe 1.5** long-only US equity. Per Grinold-Kahn 1999 §5 `IR = IC × √breadth`:

- IC +0.15 @ 103 breadth (4× current — single-name alpha) — items #1, #2, #4 working together
- IC +0.07 @ 500 breadth (~2× current — moderate breadth + IC) — item #2 wl200 + maybe wl500
- IC +0.034 @ 2000 breadth (current IC, huge data) — would need data infrastructure overhaul

**Honest backstop** (Hou-Xue-Zhang 2020 RFS — 65% factor replication failure rate):

| Outcome | Probability | Final Sharpe | APY |
|---|---|---|---|
| Hit stretch | 15% | 1.5+ | 20%+ |
| Hit baseline | 35% | 1.0-1.4 | 13-18% |
| Partial | 35% | 0.7-1.0 | 10-13% |
| Adverse | 15% | <0.7 | <10% |

---

## How to use this doc

1. **Pick topmost active item by ROI rank** (currently #1 news+IV integration)
2. **Open small branch**, ship smallest reversible step, commit
3. **Walk-forward 3-cut + §5.2 sanity** — no single-cut promotes
4. If pass gate cleared → **promote** in the same commit, update status here
5. Otherwise → **document failure** in `failed-experiments-log.md`, mark closed in this doc

**Working rhythm**: ship don't ponder; one P0 item in flight; commit each meaningful chunk; walk-forward EVERY claim.

---

## 📁 Provenance

Old plan files archived to [`doc/archives/plans/`](archives/plans/):

| File | Era covered |
|---|---|
| `roadmap-2026-05-17-pre-consolidation.md` | The full 789-line predecessor of this file |
| `2026-05-13-master-plan.md` | Mid-May audit, walk-forward gate enforcement design |
| `2026-05-13-afternoon-plan.md` | Same-day refinement |
| `2026-05-14-shorts-master-plan.md` | Long-short Phase 1-3 design (closed via empirical SKIP) |
| `2026-05-15-regime-reeval-plan.md` | Regime overlay sequential queue (largely resolved) |
| `2026-05-16-experiment-master-plan.md` | A/B/C lanes — A done, B partial, C ongoing |
| `2026-05-18-tier-c-planning.md` | C1/C3/C5 scoping (C1+C5 fetchers shipped today) |
