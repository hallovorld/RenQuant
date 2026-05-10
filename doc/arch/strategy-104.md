# renquant_104 — Panel-LTR Cross-Sectional Ranking

**Status:** Active daily strategy.
**Last updated:** 2026-05-09 EOD (post BUG #6/#7/#5 + cost-aware wash-sale + NGB on/off A/B revert + WF 3-cut)
**Based on:** renquant_103 (adaptive regime multi-stock, kept for rollback)

---

## Production snapshot (2026-05-09)

| | |
|---|---|
| Active model | **XGB rank:pairwise on 169 features** (alpha158 + 5 fund + 3 PEAD + 3 SUE) |
| Artifact | `artifacts/panel-ltr.alpha158_fund.json` fingerprint `4f1e25989d475225` |
| Feature count | 169 (158 alpha158-faithful per Qlib + 5 SEC fund + 3 PEAD + 3 SUE) |
| 7-cut WF mean IC | **+0.039 ± 0.046** (par with Qlib alpha158 benchmarks) |
| Pure alpha (post-persistence) | **~+0.018** (E53/E55) |
| Watchlist | 103 live / 292 train panel / wl162 quality-first selected |
| Panel size | 715,629 rows × 292 tickers × ~2455 dates |
| Portfolio QP | cvxpy + CLARABEL (Boyd/Stanford cvxportfolio.SinglePeriodOpt idiom) |
| QP no-trade band | `max(min_dw=2%, min(0.05, 1.0σ × σ̂))` — capped at 5% per BUG #7 |
| Wash-sale | Cost-aware per IRC §1091 (gain → no cost; loss → NPV deferred-tax cost) |
| NGBoost head | DISABLED (27-mo A/B: -3.78 APY pp / -0.14 Sharpe; 63% persistence per E55) |
| Promote gate | `wf_gate_metadata.passed=True` required (commit 5b8c891) |

---

## Performance — TBD pending bug-fix baseline (audit 2026-05-09)

Prior "27-mo APY +6.77% / Sharpe +0.40" and "3-cut WF mean +5.26%/+0.32" claims were single-measurement, **not reproducible** when re-run on same config+artifact. See `doc/AUDIT_2026-05-09.md` for root-cause analysis.

**Until a multi-seed A/A baseline is established (CLAUDE.md §5.2 mandate), no APY/Sharpe number from this strategy can be cited as ground truth.** All historical numbers in commit messages, prior STATUS.md, and the failed-experiments-log are upper-bound exploratory measurements, not reproducible benchmarks.

Today's 5 fix commits closed: cost-aware wash-sale (sim+QP+selection), broker-tagged DB, stale panel-ltr.json. Remaining: BUG #5 parquet regen, WF gate cron schedule.

---

## Pipeline

Three pipelines own the decision logic (`kernel/pipeline/` and `kernel/panel_pipeline/`):

### InferencePipeline / SellOnlyPipeline (LEAN, live, sim)

```
Preflight (8 HARD checks)
↓
DataFreshnessGate → Regime detection (SPY-GMM) → Drawdown halt
↓
Buy gates (Sharpe floor, vol cap, wash-sale cost-aware, earnings blackout)
↓
Sell jobs (parallel) — model_sell + path rules + SellGateB
↓
Candidate jobs (parallel) — earnings + wash-sale + features + score + threshold + RS
↓
PanelScoringJob:
  AssembleInferenceMatrix → ApplyScores (XGB rank) → ApplyNGBoost (skipped — disabled)
  → ApplyGlobalCalibration → ApplyKellySizing → SortCandidates
  → JointPortfolioQPJob (cvxpy CLARABEL) → EmitOrders
↓
Universal model contracts (post-predict diversity, pre-predict input variance — guards BUG #1/#2/#6 class)
```

### FullTrainingPipeline

`BaselineTournamentJob → PanelTrainingJob → RecalibrationJob`

### PanelTrainingPipeline

`PanelDataJob → PanelFeatureJob → PanelAssemblyJob → PanelModelJob → RefreshPanelCalibratorJob`

---

## What's currently OFF (NGB-off baseline)

| Feature | Status | Reason |
|---|---|---|
| NGBoost head + σ-aware Kelly | DISABLED | E55: 27-mo A/B -3.78 APY / -0.14 Sharpe; 63% persistence; pure-alpha too weak vs friction |
| edge_sharpe_floor (Conformal Gate B) | DISABLED | Pure-alpha ceiling makes target FDR=0.30 unachievable |
| Macro factor frame v1-v4 | DISABLED | All variants net-negative IC at panel size 103 |
| Asset embeddings (T2-2) | DISABLED | +0.0001 IC delta = no lift |
| LightGBM panel (E48) | REJECTED | -60% IC vs XGBoost; with sector categorical still net-negative |
| Boyd rotation (T2-4) | DISABLED | -2.5 APY pts default OFF, infra retained |
| Triple-barrier label (E25) | REJECTED | val_ic negative + placebo matches real |
| Multi-horizon ensemble (E42) | REJECTED | Shorter horizons dilute H=60 (today retest reproduced) |
| Per-sector excess label | REJECTED | 89% persistence, pure-alpha drops to +0.005 |
| Vol-adj label (Lim 2021) | REJECTED | -5bp on raw_y eval |
| Insider features (E22) | REJECTED | 8% panel coverage; needs full EDGAR backfill + opportunistic split |

---

## Acceptance gates (`kernel/model_acceptance.py`)

11 gates run by daily retrain (`scripts/train_104.py`):

```
G1  schema compatibility
G2  calibrator non-collapse (n_unique_prob_y ≥ 10)
G3  pool IC > 0
G4  OOS IC ≥ prior × (1 - 5%)            HARD
G5  score range coverage
G6  inference smoke
G7  OOS IC absolute floor (≥ 0.02)        HARD
G8  per-ticker variance
G9  sim APY drop < 1.0 pp                 HARD
G10 sim Sharpe drop < 0.10                HARD
G11 turnover ratio < 1.5x prior
```

Plus walk-forward gate (post 2026-05-09): `wf_gate_metadata.passed=True` required for promote(). Daily cron uses `RQ_ALLOW_NO_WF=1` override (cheap gates only); manual / weekly promote runs `scripts/run_wf_gate.py` first.

---

## Bug fixes shipped 2026-05-09

| Bug | Symptom | Fix | Commit |
|---|---|---|---|
| #1 | Runtime fund features all-zero in production | x-sec median imputation matches training | 507cef6 |
| #2 | SEC date misalignment caused panel build leak | hard-fail when alpha panel max date > sec max | 507cef6 |
| #5 | asset_growth 93.9% zero (XGB gain 0%) | `pct_change(periods=4d)` → `periods=252d` per Cooper-Gulen-Schill 2008 | 42e3adb |
| #6 | μ̂ collapse — all candidates identical prediction | ApplyScoresTask stamps rebuilt panel matrix to ctx | ac468e7 |
| #7 | σ-derived no-trade band 24% locked-out high-σ holdings | cap σ-band at 5% (`qp_no_trade_band_cap`) | ebbc158 |

Universal model contracts (`training_panel/model_contract.py`): post-predict diversity guard + pre-predict input variance guard on `QuantileHead`, `NGBoostHead`, `PanelScorer`, `ApplyGlobalCalibrationTask`. 70 new regression tests.

---

## References

**Implementation:** `kernel/pipeline/`, `kernel/panel_pipeline/`, `kernel/portfolio_qp/`, `training_panel/`

**Docs:**
- Roadmap: `doc/roadmap.md` (P0 #1–#9 ROI-ranked with paper citations)
- Failed experiments: `doc/research/failed-experiments-log.md` (E1–E55)
- IC eval methodology: `doc/research/ic-evaluation-methodology.md`
- Live status: `doc/STATUS.md`
- Operations: `doc/ops/usage.md`, `doc/ops/golden-config.md`

**Literature anchors:**
- Microsoft Qlib (alpha158 features) · Chen-Guestrin 2016 (XGBoost rank:pairwise)
- Bernard-Thomas 1989 (PEAD) · Foster-Olsen-Shevlin 1984 (SUE)
- Boyd cvxportfolio 2017 · Markowitz 1952 + Almgren-Chriss 2000 (QP + execution)
- Brown-Smith 2011 + Berkin-Jeffrey 1990 (tax-aware) · IRC §1091 (wash-sale)
- Lopez de Prado AFML 2018 §7/§3.6/§15 · Bailey-Lopez de Prado 2014 (DSR)
- Hou-Xue-Zhang 2020 RFS · Grinold-Kahn 1999 §5 (Fundamental Law)
