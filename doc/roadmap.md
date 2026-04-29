# RenQuant 104 — Planned Work

**Single source of truth for what's next.** Ordered by priority tier (P0 → P3), then by expected value within tier.

## ✅ RESOLVED (2026-04-28)

- [x] BUG-CV-1/2/3: FIXED 2026-04-28 — integer div, best_iter guard, eval-set alignment

## 🎯 End goal (user spec 2026-04-24)

**Golden config target:** **APY 41% | Sharpe 2.0**  (user revised down from earlier 141% target on 2026-04-24 PT — current v4.1 sweep ≈ 39.82%, needs +1-2 APY pt)

Current state:
- Golden v4.1: **+39.82% sweep APY** (`allow_fetch=False` handicap), **~+65% expected live**
- Sharpe: not yet formally tracked → first 3-day priority is instrumenting it

**Gap:** live APY ~+65% → target +141%. ~2.16× lift. Not achievable via flag flips alone; requires cumulative gains from:
- Better decision logic (thesis-A if A/B wins, Kelly-full-sweep optimization)
- Feature enrichment (hourly pruning, future: options flow, analyst revisions)
- Risk-adjusted optimization (Sharpe 2 requires specifically managing variance, not just mean)
- Panel model improvements (transformer when data permits)

Every 3-day push item is framed below as "gap closure" toward this target.

Working rhythm: pick topmost unblocked → flip 🟡 → ship smallest reversible step + commit → test vs golden → if APY ≥ +2 pts (or satisfies CLAUDE.md §2a exception) promote golden in same commit → flip ✅ → drop result + sha → move to History.

---

## Current golden state — v4 (Kelly half + A-gate)  ⭐

**Commit:** `eb8fab5` (promote)
**Frozen snapshot:** `backtesting/renquant_104/strategy_config.golden.json`
**Sweep result (27-mo OOS, `allow_fetch=False` handicap):** **+37.82% APY** vs v3 +25.91% → **+11.91 APY pts**
**Expected live (fetch-enabled):** ~+65% APY (`+44.2% × (37.82 / 25.91)`) — confirm on next Tue/Thu/Sun retrain
**Panel:** 47k × 31 hourly-enhanced feature rows, OOS mean IC (CPCV 15) +0.0326, win 85%, max no-trade streak 43d
**Key config:** `tiered_thresholds = [0.27, 0.45, 0.60]` (A-gate), `kelly_sizing.fractional = 0.5` (half-Kelly), `max_concentration = 0.35`

Full details: `doc/ops/golden-config.md`. Training history: `doc/experiments/panel-training-runs.md`. Methodology: `doc/components/panel-ltr.md`. Environment: `doc/ops/environment.md`.

---

## 🔥 P0 — Honest backtest framework (2026-04-24 PT, blocking all OOS claims)

**Problem (audit 2026-04-24 PT):** every per-ticker model on disk has
`live_train_end = 2026-04-17`, but the sim runs `2024-01-01 → 2026-03-26`.
The entire 28-month "OOS" window is **inside** the training set. Every
APY/Sharpe number we've reported (62.3/2.13, the 39.82 sweep number, every
A/B comparison) is **pure in-sample**. We have no defensible OOS metric.

**User decision (2026-04-24):** keep the current single-train-many-eval
workflow as the dev-time sanity check, but build a parallel honest path.

### B1 — Walk-forward sim runner (production-mirroring)  🔴
Mirror production retrain cadence (Tue/Thu/Sun) inside the sim loop:

```
sim/walk_forward_runner.py:
  for today in bt_dates:
      if _is_retrain_day(today, config):
          with snapshot_artifacts_ctx(strategy_dir) as snap:
              cfg = dict(config); cfg["sample_end"] = today.isoformat()
              FullTrainingPipeline().run(FullTrainingContext(
                  config=cfg, strategy_dir=snap, force_retrain=True))
              adapter = SimAdapter(..., strategy_dir=snap)
      ctx = adapter.make_context(today)
      InferencePipeline().run(ctx)
      adapter.commit(ctx)
```

Required new code:
- `_is_retrain_day(date, config)` — read `training.cadence`, return True on cadence days
- `sample_end` plumbed through `FullTrainingContext` → `RunBaselineTask` → `tournament.run_tournament_all` so `train_df.index.max() < sample_end`
- artifact dir snapshotting per retrain (already exists via `snapshot_artifacts_ctx`)
- SimAdapter must reload models when artifact dir changes mid-loop (currently artifacts loaded once at init — need a reset)

Cost: ~140 retrain points × 20 min/retrain ≈ 47 hours per backtest. Mitigations:
- Optional `tournament_cadence` separate from `panel_cadence` — retrain per-ticker tournament monthly, panel-LTR weekly
- Parallel retraining (multiple retrain points concurrent)
- `--cache-from previous-walk-forward.pkl` to incremental-extend the curve

### B2 — Hold-out backtest (single-cut sanity check)  🔴
Cheap version of walk-forward: train once with `sample_end = backtest_start - 1day`,
then sim. Useful as "does the strategy framework even make sense?" gate
before investing in walk-forward. Cost: ~30 min per run.

```
scripts/holdout_backtest.py --train-end 2023-12-31 --sim-start 2024-01-01
```

Drives FullTrainingPipeline once with the cutoff, then `run_backtest`. Should
produce a strict lower bound for walk-forward (production retrains see more
data; hold-out doesn't).

### B3 — Reporting separation  🔴
Split metric provenance in every doc + every chart caption:
- `apy_in_sample` — all-data fit + same-data sim, **never quoted as expected return**
- `apy_holdout` — fixed-cut hold-out, **conservative lower bound**
- `apy_walk_forward` — production-mirroring, **the real number for committee**
- `apy_live` — actual deployed, when available

Update `golden_config_*.md`, `panel_training_runs.md`, `ab_experiments.md`
to use these labels. Drop bare `"APY"` mentions.

### B4 — Hold-out for Round-3 panel audit fixes  🔴
After the P0 panel-LTR audit fixes (separate section below) ship, re-run
hold-out as the first honest measurement of whether the fixes helped.
In-sample 62.3% is **noise** for evaluating strategy changes.

### Why this is P0
Until B1-B3 ship, we cannot:
- Tell if panel-LTR helps or hurts (in-sample says doesn't matter)
- Validate any of Round-3 audit's panel fixes
- Promote a "v5 golden" honestly
- Make a defensible claim about live performance

Every "+X% APY uplift" in the roadmap below is currently **in-sample noise** until B1-B3 land.

---

## 🧠 2026-04-26 — Trade-evaluation DB + RL off-policy evaluation

**User ask 2026-04-26:** *"我想要一个db，来存储我的trade，这样7天，14天，
28天后可以re evaluate我的trade的合理性，用这个数据来校验我的model，用点
强化学习的概念理解我的需求"*

**Status:** 🔴 Design only. Full spec: `doc/components/trade-evaluation.md`.
Treats trades as RL `(s, a, r)` tuples; uses off-policy evaluation
(Sutton-Barto, Jiang-Li 2016, Doroudi-Thomas-Brunskill 2017,
López de Prado 2018) for time-delayed validation.

**6 phases:**
1. Schema + write-path (P1, ~3h) — 3 new tables: `trade_outcomes`,
   `policy_versions`, `policy_evaluations`.
2. Nightly backfill at 1d/5d/7d/14d/28d horizons (P1, ~2h).
3. Weekly rollup + ntfy on >1σ degradation (P1, ~3h).
4. OPE estimators: importance sampling + doubly robust (P2, ~5h).
5. Dashboard (P2, ~2h).
6. **Closed-loop policy improvement** (P3, weeks) — auto-nominate
   config changes that beat golden via OPE. Deferred until 6+ months
   of trade data. Requires safe-RL constraints + confidence bounds
   (Bottou et al. 2013).

**Cross-refs:** roadmap §B1-B4 (honest backtest — same problem from
SIM side), §144 (streak → db — same "db is canonical" theme).

---

## 📦 2026-04-26 — Model metadata DB + artifact cloud backup (DEFERRED)

**User ask 2026-04-26 evening**: "每个模型的metadata应该进数据库，模型artifact应该有云备份".

**Status**: ⏸️ DEFERRED — full plan written (`doc/components/metadata-db-and-backup-plan.md`), implementation moved to next session per user direction "下次再处理".

**What's already designed** (ready to implement):
- 7 columns added to `training_runs`: sim_apy/sharpe/calmar/max_dd/turnover, promoted_at, demoted_at, replaced_run_id, sha256_hex, cloud_backup_url
- 2 new tables: `training_run_gates` (per-gate verdicts), `tournament_rankings`
- Cloud backup tiers: Hot (per-promote, immutable run_id), Warm (daily db backup), Cold (weekly tar)
- Recommended provider: Backblaze B2 (~$0.50/year for everything, S3-compatible API)

**Implementation phases** (~6 hours total):
1. Phase A (1.5h): schema migration + write-path in `record_training_run` and `model_acceptance.promote()`
2. Phase B (1h): backfill from existing `panel-ltr*.json` artifacts
3. Phase C (30min): `scripts/model_history.py` CLI
4. Phase D (1h): `kernel/cloud_backup.py` B2 client
5. Phase E (30min): `promote()` upload integration
6. Phase F (1h): daily/weekly cron + restore script

**Decisions still needed from operator** (block implementation):
- Provider confirmation (B2 default? alternative?)
- Bucket name (`renquant-models` default?)
- Retention agreement (keep tier 1 forever, ~$1/decade)
- Trigger: only-promoted vs all-trained
- Encryption: server-side only vs client-side keys

---

## 🛡 2026-04-26 — Cloud backup plan (operational hygiene)

**User ask 2026-04-26:**
1. "统计一下需要备份的文件和信息，制定一个云备份的计划" — inventory + cloud plan
2. Clarification: "live_state.json应该至少备份在db里，而且类似的关键文件和artifacts应该有云备份" — db mirror for state + cloud for artifacts

**Status:** 🔴 Not started — designed only. P1 because total data loss = months of state gone.

### Current state (2026-04-26)

- ✅ **DB write-path exists**: `live_state_snapshots` table in `runs.db` is populated by `record_live_state_snapshot` (kernel/persistence.py:732) on every bar via `adapters/runner.py:1041`. Per-bar full `state_json` blob + indexed columns (regime, equity, drawdown).
- ❌ **DB read-path missing**: if `live_state.json` is corrupt/missing on startup, runner has NO fallback to db. Today, missing JSON = reset to defaults (lose streak, HWM, regime).
- ❌ **No off-machine backup** for `runs.db` itself — db is the canonical store of state, but it lives only on this laptop.

### B-Tier 1 — DB-canonical state (P0, ~2 hours)

The db is already the WRITE-side mirror; close the loop with a READ-side restore.

1. **`scripts/restore_live_state_from_db.py`** — read latest `live_state_snapshots` row for renquant_104, write `live_state.json`.
2. **Runner startup hook** (`adapters/runner.py::__init__`): if `live_state.json` missing OR `last_bar_date < today - 7d` (stale), auto-restore from db before initializing.
3. **`tests/test_live_state_db_recovery.py`** — delete json, run runner once, assert state recovered correctly.
4. **Stop using JSON as source of truth — db wins on conflict.**

This makes `runs.db` the authoritative store. The JSON becomes a fast cache + human-readable view, but loss of JSON ≠ loss of state.

### Inventory (sizes as of 2026-04-26 17:45 PT)

| File / dir | Size | Category | Replaceable? |
|---|---|---|---|
| `data/runs.db` | 45 MB | Live trade DB (tds, score_dist, calibration, snapshots) | ❌ NO — irreplaceable historical |
| `backtesting/renquant_104/live_state.json` | 4 KB | Sell streaks, regime, HWM, entry_dates | ❌ NO — current portfolio state |
| `live/logs/renquant-104/{date}.json` | 492 KB cum | Per-trade log (fills, signals, prices) | ❌ NO — audit trail |
| `logs/live_104/audit.jsonl` | <1 KB | Sustainability stream → ntfy alerts | ❌ NO — drives weekly_apy_check |
| `.env` | 4 KB | Alpaca API keys | ❌ NO — but **must be encrypted vault, NOT cloud** |
| `data/sim_runs.db` | 3 MB | Sim A/B journal | 🟡 Partial — sims could re-run but $$$ |
| `backtesting/renquant_104/artifacts/*.json` | 22 MB | panel-ltr, ngboost-head, calibrator, .pt | 🟡 Reproducible — Sun retrain ~30 min |
| `backtesting/renquant_104/models/{T}/*.json` | 535 MB | Per-ticker tournament artifacts (in git) | 🟡 Reproducible — Tue/Thu/Sun retrain |
| `data/ohlcv/` | 12 MB | yfinance daily bars | ✅ Re-fetchable from yfinance |
| `data/intraday/` | 83 MB | yfinance hourly bars | ✅ Re-fetchable (slow) |
| `data/{fundamentals,earnings_surprise,insider_trades}/` | 2 MB | OpenBB / yfinance / SEC | ✅ Re-fetchable |
| `backtesting/data/equity/usa/` | 229 MB | LEAN data zips | ✅ Derivable from data/ohlcv |

**Tier classification:**

- **Tier 1 — daily backup, irreplaceable:** ~50 MB delta/day, 30-day retention → **~1.5 GB/month**
  - `data/runs.db`, `live_state.json`, `live/logs/`, `logs/live_104/audit.jsonl`
- **Tier 2 — weekly backup, reproducible-but-expensive:** ~600 MB/week, 5-week retention → **~3 GB/month**
  - `data/sim_runs.db`, `artifacts/`, `models/` (dedupe-friendly via restic)
- **Tier 3 — never back up, derivable from source:** 326 MB on disk
  - `data/ohlcv/`, `data/intraday/`, `data/fundamentals/`, `data/earnings_surprise/`, `data/insider_trades/`, `backtesting/data/equity/usa/`
- **Credentials separately:** `.env` → 1Password CLI vault (NOT in cloud bucket)

### Cloud provider comparison

| Provider | Storage | Egress | Total est | Verdict |
|---|---|---|---|---|
| AWS S3 Standard | $0.023/GB | $0.09/GB | ~$0.12/mo | OK; egress hostile if restoring |
| AWS S3 Glacier IR | $0.004/GB | $0.03/GB | ~$0.05/mo | Cheap but 90-day min charge |
| Backblaze B2 | $0.005/GB | $0.01/GB | **~$0.05/mo** | ✅ Recommended |
| Cloudflare R2 | $0.015/GB | $0 (free) | ~$0.07/mo | Free egress good for restore drills |
| GitHub LFS | $0.07/GB | $0.07/GB | ~$0.35/mo | Bundled but expensive |
| iCloud 50GB | $0.99/mo flat | included | $0.99/mo | Manual, no API; useful as 2nd target |

**Recommendation:** **Backblaze B2 + restic** (client-side AES-256 encryption + deduplication).
- ~$0.05/mo at projected 5 GB/month volumes
- restic dedupe means weekly model snapshots only store deltas (~10 MB after 1st week)
- 3-2-1 rule: B2 primary + iCloud secondary (manual quarterly) + local Time Machine

### Implementation phases

**Phase 1 — Tier 1 nightly (P1, ~2 hours)**
- `scripts/backup_tier1.sh` — restic to B2, runs nightly via launchd at 02:00 PT (after daily_104 + intraday wraps)
- Source list: `data/runs.db`, `backtesting/renquant_104/live_state.json`, `live/logs/`, `logs/live_104/audit.jsonl`
- Encryption key in 1Password (separate from B2 app key)
- `scripts/restore_tier1.sh` — interactive restore from latest snapshot
- ntfy alert on backup failure (silent on success per ops contract)

**Phase 2 — Tier 2 weekly (P1, ~1 hour)**
- `scripts/backup_tier2.sh` — same restic repo (deduplicates against Tier 1)
- Source list: `data/sim_runs.db`, `backtesting/renquant_104/artifacts/`, `backtesting/renquant_104/models/`
- Sunday 02:30 PT (after retrain_panel.sh completes at Sun 10:00 PT… actually run BEFORE so we capture pre-retrain state)
- Reframe: Sun 09:55 PT (catches the previous week's models before they're overwritten)

**Phase 3 — Credentials to 1Password (P0, ~30 min)**
- `op` CLI install + login
- Move `.env` contents to 1Password "RenQuant Alpaca Live" item
- `scripts/load_env_from_1password.sh` — sources at runtime
- `.env` stays as a fallback during transition; remove after 2 weeks confirmed working

**Phase 4 — Restore drill (P1, ~2 hours)**
- Quarterly: spin up empty dir, restore Tier 1 + Tier 2, verify daily_104 runs end-to-end against restored state
- Document drill output in `doc/backup_drill_{date}.md`

**Phase 5 — Monitoring (P2, ~1 hour)**
- `scripts/check_backup_freshness.py` — daily cron, fires ntfy WARN if latest Tier 1 snapshot >36h old or Tier 2 >9d old
- Append snapshot age to weekly sustainability ntfy

### Files to create

- `scripts/backup_tier1.sh`
- `scripts/backup_tier2.sh`
- `scripts/restore_tier1.sh`
- `scripts/load_env_from_1password.sh`
- `scripts/check_backup_freshness.py`
- `doc/backup_runbook.md` — operator restore procedures + key location

### Out of scope (explicitly not backed up)

- `data/ohlcv/`, `data/intraday/`, `data/{fundamentals,earnings_surprise,insider_trades}/` — re-fetchable (CLAUDE.md path)
- `backtesting/data/equity/usa/` — derivable via `export_lean_watchlist.py`
- `backtesting/renquant_104/img/` — chart PNGs, regenerate via analyze_backtest
- `data/intraday_wash/`, `data/intraday_wash_panel/` — derived caches
- Test artifacts under `/tmp/`

### Manual interim (until Phase 1 ships)

- Daily `cp data/runs.db /tmp/runs_$(date +%F).db` (manual; user runs)
- Weekly `cp -r backtesting/renquant_104/artifacts /tmp/artifacts_$(date +%F)` (manual)
- `.env` already secure on local laptop; copy to encrypted USB drive

---

## 🆕 2026-04-25 — Panel-LTR ceiling: 4 promising upgrades + 2 research items

**Context:** Today's deep audit of renquant_104 panel-LTR cross-sectional ranking found 12+ implementation bugs (committed separately, not in roadmap scope) and identified that **the panel-LTR XGBoost backend has reached a ceiling around OOS IC ~0.066**. Web research surfaced four evidence-backed upgrade paths and two longer-horizon research items. Tier 1 (config-only changes) executed today; Tier 2-4 (real engineering) tracked here.

**Theoretical references** (cite when justifying experiments):
- Bagnara 2024 (Journal of Economic Surveys, "ML in asset pricing — a critical review") — documents weaknesses: data quality, overfitting, regime instability, statistical-vs-economic gap.
- Gu-Kelly-Xiu 2020 (RFS, "Empirical Asset Pricing via Machine Learning") — NN > trees > linear; key drivers momentum/liquidity/volatility.
- Poh-Lim-Zohren-Roberts 2020 (arXiv 2012.07149) — cross-sectional learning-to-rank, listwise > pairwise.
- CIKM 2025 (arXiv 2510.14156, "On Evaluating Loss Functions for Stock Ranking") — confirms listwise > pairwise.

**A3 (XGBoost rank:ndcg listwise) — BLOCKED on implementation work (2026-04-27)**:
A3 retrain crashed with `XGBoostError: When using relevance degree as target,
label must be either 0 or positive integer.` XGBoost's `rank:ndcg` objective
requires integer non-negative relevance labels, but our pipeline passes
continuous forward returns. Same issue LightGBM already solves via
`lgbm_ltr.py::_bucketize_labels` (per-date rank bucketization 0-10). To
unblock A3: copy that helper into the XGBoost FinalFitTask path
(`pp_panel_training.py::FinalFitTask`) and apply when `objective` starts with
`rank:ndcg` or `rank:map`. ~30 LoC, low risk. Then re-run A3 with --force.
Artifact preserved at `strategy_config.xgb_listwise.json`; production
unaffected (XGBoost crashed before SaveArtifactTask).

### Tier 2 — 2-4 weeks, evidence-backed wins

| # | Item | Evidence | ETA | Status |
|---|------|---|---:|---|
| ~~**T2-1**~~ | ~~LightGBM LTR replacement~~ — **REJECTED 2026-04-27** based on S2 retrain. On the **current panel** (99-ticker, 28-feature, 753-date, no-macro), LightGBM v1 scores OOS IC **+0.0193** vs XGBoost prod **+0.0482** (-60%). The historical "+128% IC" claim from the 491-date hourly-era panel does NOT transfer. With macro added, LGBM gets even worse: 0.0224 (-53%). **Bonus signal**: S2's RefreshPanelCalibratorJob FAILED with calibrator collapsed to 4 unique probability values (round-7 G2 floor = 5 unique). pool_ic = +0.0017 (noise floor). LGBM v2 attempt (4-fix hyperparam audit per `lgbm-implementation-audit.md`: exp label_gain, truncation 50, num_leaves=8, min_data_in_leaf=100) gave **+0.0014** OOS IC — even WORSE than v1. Conclusion: LGBM lambdarank is fundamentally unsuited to this small (~75K row) panel regardless of tuning. Artifacts preserved at `panel-ltr.lgbm-no-macro.bak.json` and `panel-ltr.lgbm-v2.bak.json` for forensics. | rejected | n/a | ❌ |
| **T2-2** | **Contrastive asset embeddings as features** — Dolphin et al. 2024 KDD ("Contrastive Learning of Asset Embeddings from Financial Time Series", arXiv 2407.18645). Pairwise-correlation contrastive learning yields per-asset embeddings; +3 pts F1 on sector classification, 19% volatility reduction in hedging. Plan: train 16-dim embeddings on watchlist OHLCV history → use as additional input features to existing XGBoost panel-LTR (no architecture change). Lowest-risk win. **2026-04-27 update**: 16-D InfoNCE embedding trained; artifact at `artifacts/asset-embeddings.json`. Raw IC 16-dim avg = 0.0147 (9/16 dims > 0.01); Residual IC (after removing momentum/vol) = **0.0356** (16/16 dims > 0.01) → embedding carries **independent alpha signal**, orthogonal to existing features; OLS A/B: **+26% linear IC**. Next step: enable `panel_ltr.asset_embeddings.enabled: true` → re-run PanelModelJob. Expected OOS IC 0.043–0.047 (+5%~+14%). | +3 pts F1 sector / -19% vol hedging | READY TO EXECUTE | 🟡 |
| **T2-3** | **Regime-conditional ensemble (mixture of experts)** — Two Sigma 2024 (https://www.twosigma.com/articles/a-machine-learning-approach-to-regime-modeling/) documented 4-state t-distributed mixture model on macro+style factors. Plan: train separate panel-LTR per regime (BULL_CALM / BULL_VOLATILE / CHOPPY / BEAR), select at inference via existing `ctx.regime`. `regime_state` infra already in place. Note: prior Plan F (regime-conditional calibration) shelved at -3.78 APY pts due to in-sample overfit; this is a **separate model per regime**, not just per-regime calibration — different failure mode. | qualitative (Two Sigma case study) | 1 week | 🔴 |
| **T2-4** | **Rotation Phase 3 — Boyd-style convex optimization** — replace Phase 2's greedy joint-action sorter with a true convex program (cvxpy). Decision variable: trade vector Δw (per-ticker buy/sell qty, signed). Objective: maximize μᵀΔw - γ·Δwᵀ Σ Δw - cost(\|Δw\|). Constraints: weight bounds, turnover cap, leverage cap, sector cap, correlation cap. Reference: Boyd "Markowitz Portfolio Construction at Seventy" + Gârleanu-Pedersen 2013 "aim in front of the target + trade partially towards aim" — slow-decay signals get higher weight in target, fast-decay signals get lower weight. Practical: 20% Sharpe lift over static one-period optimization per G-P empirical. Risk: cvxpy solve time per bar must stay <1s for live use; integer share constraint (we trade whole shares not fractions) breaks pure-convex assumption — likely need MILP relaxation or rounding heuristic. Already implemented infrastructure: panel_score, μ, σ from current panel-LTR + NGBoost; correlation matrix from `watchlist-correlation.json`. Effort: 1 week implementation + 1 week sim validation. | qualitative (G-P claims +20% Sharpe vs static) | 2 weeks | 🔴 |

### Tier 3 — Research project, 2+ months, speculative

| # | Item | Evidence | Risk | Status |
|---|------|---|---|---|
| **T3-1** | **TGNS (Transformer + Graph Neural Network)** — claims +12-22% IC over SOTA on Chinese A-share data. Concept Graph Attention + Stock Graph Attention modules. Reference: https://www.sciencedirect.com/science/article/abs/pii/S0020025525006887. **Risk:** validated on Chinese A-share, not US equity; may not transfer. Also needs the data-volume gate that's blocked Plan H (transformer panel) — currently 47k rows, transformer needs >200k. | +12-22% IC (CN A-share only) | High (geo + data) | 🔴 |
| **T3-2** | **FASCL (Future-Aligned Soft Contrastive Learning)** — Feb 2026 paper (arXiv 2602.10711), 4229 US equities, "outperforms 13 baselines across all future-behavior metrics". Pairwise future return correlations as continuous supervision. **US equity validation makes this higher-priority than T3-1 once code releases** ("available soon" per paper). Action: monitor arxiv listing + GitHub releases; revisit when reference implementation appears. | qualitative US equity claims | Medium (waiting on code) | 🔴 (waiting) |

### Tier 4 — Speculative, infra-heavy

| # | Item | Evidence | Cost | Status |
|---|------|---|---|---|
| **T4-1** | **LLM-generated factor features** — ICLR/NeurIPS 2025 trend. Use LLMs to generate trading factors from news/earnings text. Requires NLP infra (text data ingestion, prompt engineering, factor backtesting). Mixed historical results in literature. Probably not worth it until at least one of T2/T3 lands and we've squeezed numerical-feature alpha. | Mixed (paper trend) | 2+ months infra | 🔴 |

### Sequencing note

~~T2-1 (LightGBM) was the highest-confidence win~~ — **REJECTED 2026-04-27** (see the strikethrough row above). With T2-1 dropped, the new sequence is: **T2-2 (asset embeddings) → watchlist 99→200 → T2-3 (regime ensemble, gated on panel > 150k rows) → T2-4 (Boyd rotation, only when IC is high enough that rotation doesn't cost −2.5 APY/event)**. Design docs for T2-2 and T2-4 already written: [`components/asset-embeddings-design.md`](components/asset-embeddings-design.md) and [`components/boyd-rotation-design.md`](components/boyd-rotation-design.md). T3-1/T3-2 are research bets — don't start until Tier 2 saturates or a clear data-volume unblock (panel > 200k rows) lands. T4-1 deferred indefinitely.

**2026-04-27 Macro closure**: all macro variants exhausted (v1 broadcast → zero gradient; v2 per-ticker β post-3-bug-fix → −23% IC; v3 30 ETF + FRED → −29% IC). Macro re-evaluation gated on watchlist ≥ 200 tickers. See `doc/archives/sessions/2026-04-27-decisions.md` for full record. **Do not re-open macro experiments without first expanding watchlist.**

---

## 🆕 2026-04-25 — long-term training-stack architectural items

User spec 2026-04-25 (post NGBoost+Transformer audit retrain crash):
"这个要改的！下次优先改！你甚至可以用别的语言写训练算法！"

The current Python+PyTorch training stack hits Apple Silicon (MPS)
gaps repeatedly + ThreadPool+GIL leaves us at 1-core utilisation
during per-ticker work. Long-term refactor candidates, in order of
expected ROI:

- 🔴 **ThreadPool → ProcessPool for `run_panel_ticker_parallel`**
  (~80 LoC, medium risk). 5-10× wallclock speedup on the 99-ticker
  per-ticker chain. Breaks the GIL bottleneck (P3-1 in panel-ml audit).
  Requires picklable TickerPanelContext + worker `sys.path` setup.
- 🔴 **Vectorize panel build (drop per-ticker loop entirely)**
  (~300 LoC, high risk). 10-20× speedup; rewrites Feature/Neutralize/Factor
  as cross-sectional ops on the full panel. Algorithmic change; needs
  re-validation of every existing factor.
- 🔴 **Native (Rust/C++/Julia) rewrite of training core**
  (multi-week, very high risk + reward). Addresses both ThreadPool/GIL
  AND PyTorch-MPS gaps. Best for the tight numeric loops:
  transformer attention, NGBoost gradient boosting, panel z-score.
  Long-term play — also gives clean MPS/Metal Compute Shader path.
- 🔴 **Replace PyTorch+MPS with mlx (Apple's native ML lib)** —
  alternative to full native rewrite. mlx has first-class Metal
  support, no MPS-fallback gaps, similar API. Smaller migration than
  Rust but still a multi-day effort.

Current shipped workaround: `enable_nested_tensor=False` on
`nn.TransformerEncoder` (T-MPS-1 fix). Sidesteps one specific MPS
gap; doesn't address the broader pattern.

## 🆕 2026-04-24 PT — late-session pending queue (post-compact)

Ordered roughly by my own recommended shipping sequence. Items marked 🟡 are in flight; ✅ done; ⏳ waiting on A/B or wall-clock; 🔴 not started. Updated at every ship.

### 🟡 In-flight / just shipped (post-compact session)
- ✅ **TTL gate** `training.model_ttl_days` (per-ticker skip when fresh) — default 1.
- ✅ **Drift-guard hook** `scripts/install_git_hooks.sh` + daily_104 integration.
- ✅ **run_backtest(snapshot=True)** default — notebook sims auto-isolate.
- ✅ **pytest-xdist + OMP=1** — full suite 25 min hang → 14 sec.
- ✅ **net_safety daemon threads** — no more teardown hangs.
- ✅ **RENQUANT_NO_NOTIFY** env — tests don't spam user's ntfy.
- ✅ **10-min bar infra** — MinuteBarStore + fetch_minute_bars.py + compute_minute_features (10 features w/ `m_` prefix). Default off, flag `panel_ltr.minute.enabled`.
- ✅ **sim/analysis.strip_top_n_trades** — notebook robustness check for lucky-winner alpha.
- ✅ **Kelly-pure A/B** — result: **ΔAPY 0** (flag is no-op under current golden; conv/σ_mult already ≈1.0). Verdict: shelve.
- ✅ **panel_conviction_exit A/B** — result: **ΔAPY 0, 0 panel_exits fired**. The gate (panel < 0.20 AND μ ≤ 0.0) is never both true simultaneously on current holdings. Verdict: shelve until gate thresholds tuned — worth revisiting if holdings ever show panel degradation without μ inversion.
- ✅ **Rotation V1 gates shipped** (`9eb188b`) — `rotation.min_raw_advantage_pct` + `rotation.persistence_bars`, both default off.
- ✅ **Rotation V1 A/B** — result: **ALL variants 0 rotations.** Current panel ER distribution too tight even at threshold=0.005 — can't evaluate V1 on unchanged panel. Diagnosis: previous roadmap data (3 rot @ threshold 0.005 → -4.93 APY) was collected on a pre-retrain panel; user retrained since. Need wider-spread driver.
- ✅ **Rotation V2 + V3 gates shipped** (`0000b91`, `f674b3f`) — μ−λσ scoring mode + regime gate + held-DD gate. All flag-gated, all compose.
- ✅ **Rotation force A/B v2** — result: **ALL 4 variants still 0 rotations.** Threshold=0.001 + μ−λσ + all V1/V3 gates flipped on or off → same outcome. **Diagnostic (key):** rotation.BuildPairsTask requires `ctx.ranked` non-empty, but candidate admission (`ScoreBuyTask` + tiered_thresholds A-gate at 0.27) rejects all candidates on most bars. Rotation is starved of rotate-TO candidates, not of threshold headroom. **Implication for user's "rotation is core APY lever" hypothesis:** under current panel + config, rotation cannot fire regardless of gate. Unlocking rotation requires EITHER (a) looser candidate admission (reverses past A-gate wins), OR (b) a panel model that produces more >0.27 rank candidates. Neither are pure rotation-algorithm changes — they're upstream model/gate changes that would themselves need independent A/B. V1/V2/V3 infra is shipped + tested + ready when panel evolution opens the path.
- ✅ **Panel exit V2** shipped (`b022ad6`) — `risk.panel_exit.trigger_mode` = "and" (default) | "or". V1 AND never fired; OR mode queued for A/B (`/tmp/panel_exit_v2_ab.py`).
- ✅ **Feature-parity invariant test** (`58abd05`) — pins notebook / LEAN / live all use `kernel.indicators.build_feature_frame`; 4 tests guard against accidental fork (user contract: 保证 notebook feature integrity with lean).
- ✅ **Rotation V4 thesis-symmetric** (`709032d`) — full 4-point (A_entry+A_today+B_entry+B_today) via DB lookup of candidate_scores on A's entry date. User's own design. Lit-grounded (Avellaneda-Lee pair-trading, Gu-Kelly-Xiu ML ranking). 6 tests; default off; wiring deferred until A/B can fire.
- ✅ **Rotation research doc** (`709032d`) — `doc/research/rotation-research.md` — literature review: Jegadeesh-Titman, Moskowitz TSMOM, Barroso-Santa-Clara, Daniel-Moskowitz crashes, Avellaneda-Lee pairs, Grinold-Kahn breadth, López de Prado ML. 6 ranked implementation proposals.
- ✅ **V4 own-momentum gate** (`9463e4c`) — Proposal 1 shipped: A's own 63d return must have broken AND B's must be intact before rotating. Jegadeesh-Moskowitz compliance. 4 new tests.
- ✅ **10-min bar fetch complete** — 744k rows × 50 symbols × 2yr cached. Prereq for transformer retry (>200k row gate) satisfied.
- 🟡 **10-min panel retrain A/B running** (`/tmp/minute_panel_retrain_and_ab.py`). **Preliminary**: CPCV OOS IC = **+0.0355 (+0.003 vs baseline)** — hourly+minute panel beats hourly-only on OOS IC. NGBoost + sim phases pending. Full verdict ~10 min.

### ✅ ALL ROTATION + PANEL + DATA WORK SHIPPED THIS SESSION

Final result: **Sim APY 28.82% → +30.90%** on 27-mo OOS.
Panel CPCV OOS IC: 0.0391 → **+0.0536** (+37%).
Watchlist 43 → 99. Win rate 80.5%. Rotations finally fire (2x).
Full session detail: `doc/archives/sessions/2026-04-24.md`.

### ✅ Status of items in the previous "not shipped" list

- ✅ **Rotation V1** depth + persistence — shipped + tested (default off).
- ✅ **Rotation V2** μ-λσ direct mode — shipped + tested (default off).
- ✅ **Rotation V3** regime + held-DD gates — shipped + tested (default off).
- ✅ **Rotation V4** thesis-symmetric — shipped + tested + DB lookup wired (default off).
- ✅ **V4 own-momentum gate** (Jegadeesh/Moskowitz) — shipped + tested.
- ✅ **Sharpe scoring mode** (Barroso) — shipped + tested.
- ✅ **10-min panel retrain end-to-end** — done. +9.57 APY on isolated A/B; +2.08 APY on clean main retrain.
- ✅ **Transformer retry** — done. 0.89× XGBoost on 43-ticker panel — shelved again (panel still under transformer's data threshold; revisit at watchlist 120+ or 10-min training window).
- ✅ **Rotation algorithm review** — done. `doc/research/rotation-research.md` with academic refs + 6 proposals.

### ⏭ Remaining for next session

| Task | Est | Notes |
|---|---|---|
| **Deep audit** rotation + panel pipelines | 1 day | task #61 — walk every Task/Job for silent failures. 17 documented bugs await. |
| **Bug 18 fix**: NGBoost dropna before fit | 30 min | silences `overflow encountered in square` warning |
| **Bug 22/24 cleanup**: dead code (build_spy_context scalar, log-price size) | 30 min | low risk |
| **A/B remaining**: Q rotation 2D sweep, J hourly pruning, AB-trim, Kelly-tier-tune, Multi-entry cap, Kelly-full-sweep | each 1-3 h sim | nothing critical now that panel is the dominant alpha source |
| **Watchlist Wave 2** (99 → ~120) | 1 day | only after 1-2 weeks of live data on 99-ticker config |
| **Feature registry refactor** (pattern A1) | 1 day | prevents future silent feature drops |

### 🟡 P2 — analysis / diagnostic (no sim, notebook-friendly)

- **L** per-ticker hourly effectiveness (leave-one-out OOS IC)
- **Panel-IC-drift** diagnosis (±0.03 day-over-day)
- **BULL_CALM streak watch** (≥20d audit)
- **sector sub-buckets** (semis / software / cloud)
- **K** CHOPPY regime re-diagnosis (in-sample -0.116 vs live +0.0354 — real or calibrator artifact?)

### 🟢 P3 — passive / wall-clock gated

- **I** accumulate 4 weeks of Sun live-sustainability data
- **Transformer gate re-open** when panel > 200k rows (unlocks via 10-min data)

---

## ✅ 2026-04-24 — shipped today (30+ commits)

**Day 1 (audit foundation):**
- ✅ **M⁺** (`3dd903a`) — schema migration for `training_runs` 9 missing columns. Live DB now has 21 cols.
- ✅ **P** (`11be4cd`) — per-ticker rejection reason persists in `candidate_scores.blocked_by`.
- ✅ **AA** (`429298d`) — decision-factor DB + `analyze_decision_factors.py`. Backfilled 15,976 forward returns on 27-mo window. **Tier 1 @ 0.27 empirically validated: 78.9% P(fwd>0) vs v3's 71.1%.**

**Kelly loop closure:**
- ✅ **Partial-sell infra** (`efcca83`) — `ExitSignal.quantity` + adapter hooks.
- ✅ **AB-trim** (`6d8a52c`) — `TrimHeldTask` emits partial sells when over-weight. Default hysteresis 0.10.
- ✅ **BC** (`09468e3`) — Kelly-delta rotation gate alongside panel-delta gate.
- ✅ **CUSUM-cooldown-v2** (`10f788a`) — Design C confidence-scaled sizing, flag-gated.
- ✅ **S** (`b07a81c`) — `live_state_snapshots` append-only audit table.

**Cleanup + architecture:**
- ✅ **N** (`34ccac2`) — golden doc consolidated (v4 top, v1-v3 history).
- ✅ Analyzer enhancement (`a42344f`) — tier-usage + selected-bucket diagnostics.
- ✅ **CUSUM-v2 PROMOTED v4.1** (`1bb5ae1`) — +1.97 APY pts. Live/sim parity fix.
- ✅ **Thesis-A infra** (`e519177`) — entry-baseline rotation. Flag-gated.
- ✅ **BC audit guards** (`6b06dcd`) — ported AB-trim lessons preventively.
- ✅ **DB split** (`56083da`) — runs.db (live permanent) + sim_runs.db (ephemeral).
- ✅ **doc/components/databases.md** — full schema reference with migration rules.
- ✅ Docs sweep (`c73a4b5`) — CLAUDE.md test count + models.md Kelly/partial/trim/thesis-A + architecture.md adapter table.

**Tests added:** +120 (including regression, infra, and audit guards). Full test count ~1307 collected.

**Golden verified intact** at the end: A_GOLDEN_v4 sim = +37.85% APY = pre-session 37.82% (0.03 pt noise). All new code is flag-gated default-off OR routed through the sim DB.

---

## 🎯 Macro plan — 3-day push (2026-04-24 to 2026-04-27)

**Scope clarification (user 2026-04-24):** focus on what's achievable in 3 days. Multi-strategy / covered-call / international / cross-strategy allocation are OUT of scope — "model 还有 100 个 bug 没找到" 先 solidify 现在.

### Day 1 (today — remaining)
- 🟡 **Thesis-A A/B** verdict (sim running, ~20 min more)
- Plan B AA-calibrated placeholder (roadmap entry only)

### Day 2 — Pure-code high-ROI wins

**Priority-reordered to serve APY=1.41 / Sharpe=2 target:**

- **Portfolio-level risk metrics in DB** ⭐ TARGET-CRITICAL — without Sharpe tracking, we can't measure the "Sharpe=2" goal. New `portfolio_daily_metrics` table: daily Sharpe, rolling 3-month Sharpe, max-DD, VaR95/99, realized_vol, beta_to_SPY. Computed from `pipeline_runs.portfolio_value` history. Backfill from existing data → first snapshot of where we stand vs Sharpe=2 target.
- **Feature cache optimization** — 5-8x sim speedup enables running many more A/B experiments per day → faster convergence to target.
- **Performance regression test suite** — `tests/test_golden_preservation.py` opt-in `RENQUANT_REGRESSION=1`. Asserts sim APY ≥ 37.0% (v4.1 − 1 pt tolerance). Guards the floor during target push.
- **J hourly-feature pruning** A/B — one concrete gap-closure experiment: +0.005 OOS IC target if it wins.

### Day 3 — Strategic automations
- **Ticker rotation automation** ⭐ user approved — `scripts/screen_watchlist.py` weekly cron: 6-month realized Sharpe per watchlist ticker, sector ETF comparison, flag underperformers for drop + high-Sharpe non-watchlist names for add. Output: markdown report + ntfy.
- **Real-time retrain triggers** ⭐ user approved — `scripts/check_retrain_triggers.py` already sketched in past sessions; implement + plist. `SPY.|daily Δ| > 2%` or `VIX.|daily Δ| > 5%` → `train_104.py --force --trigger=anomaly_{spy,vix}`. Fires at 13:10 PT to let training land before 13:55 PT daily run.
- **Q rotation 2D sweep** — `{14, 21, 30} × {0.0, 0.02, 0.05}` config matrix
- **Kelly-full-sweep** — 10-point notebook grid (fractional × max_concentration)
- **Flag-drift alert** — fail pre-commit hook if any `default-off` flag moved to default-on without explicit promotion commit

### Parked (explicitly out of 3-day scope, per user)
- ❌ Multi-strategy parallel live — not needed
- ❌ Cross-strategy capital allocation — not needed
- ❌ Covered-call overlay — too advanced, "100 bugs first"
- ❌ International equities — too advanced
- ❌ Sector-guard sub-buckets — my invention, not validated priority

### Parked (deferred, time-gated)
- **Plan B** (AA-calibrated rotation) — needs ~3 months live data for sample
- **Plan F retry** (regime-conditional calibrators) — same data constraint
- **Transformer panel backend** — until panel > 200k rows (47k now)
- **Live sustainability accumulation (Plan I)** — passive, Sun 12PT plist already wired; review after 4 Sundays

---

## Detailed spec — Feature cache optimization (Tier 2)

**Problem:** Every `InferencePipeline.run()` bar rebuilds per-ticker feature frames from scratch. For each of 570 bars × 42 tickers, `build_feature_frame()` recomputes the FULL indicator history on the OHLCV slice up to that bar. 24k redundant recomputes per sim → 9 min/variant.

**Solution:** Pre-compute per-ticker full-range feature frames ONCE at sim start; per-bar tasks just index by today.

**Design:**
1. New `SimAdapter.__init__` step: for each ticker in ohlcv, call `build_feature_frame(full_stock, full_spy, spec)` once. Store in `self._feature_cache[ticker] = full_df`.
2. `TickerInferenceContext` gets optional `feature_cache_frame: pd.DataFrame | None` field.
3. Adapter's `_make_ticker_ctx` passes the cached frame.
4. `BuildFeaturesTask.run` — if `tc.feature_cache_frame` is set, `tc.features = tc.feature_cache_frame.loc[:tc.today]`; else fall back to `build_feature_frame(...)` (live runner path, where feature cache isn't prebuilt).

**Correctness test (critical):** Run golden-config sim with cache vs without, assert identical equity curve + trade log. Needs a new `tests/test_feature_cache_equivalence.py`.

**Projected speedup:** Conservative 5x (9 min → 1.8 min/variant). 100 notebook runs/day = 15 hr → 3 hr.

**Risk:** If `build_feature_frame` is NOT fully deterministic from full history (e.g. has lookahead or depends on `today`-specific logic), cache can diverge. Must verify equivalence before shipping.

**Not in this session** — too risky to squeeze in at end of 30-commit sprint. Scheduled for next session with full correctness test + sim verification.

---

## 🟠 P1 — Remaining Kelly completion

| # | Item | Impact | Est. | Prereqs |
|---|------|---|---:|---|
| **AB-trim A/B** | Run sweep (GOLDEN vs hysteresis 0.10 vs tight 0.0) to pick default | parameter defense | 1h sim | ✅ infra shipped |
| **CUSUM-v2 A/B** | Run GOLDEN vs `cusum_cooldown_mode: wall_time` to measure ~2 APY pt expected lift | validate design C | 1h sim | ✅ infra shipped |
| **Kelly-tier-tune A/B** | AA shows [0.35, 0.45) is the 80.7% hit-rate sweet spot. Test raising tier 1 from 0.27 → 0.35. | empirical-driven | 1h sim | ✅ AA data in hand |
| **Kelly-full-sweep** | 10-point grid on fractional × max_concentration from notebook | parameter defense | 1h sim | — |
| **Kelly × conviction** | `max_pct = kelly × conviction × σ_mult` may double-count μ and σ. Audit + decide. | design cleanup | 1h code | — |
| **Multi-entry accumulation** | Let Kelly target build up over sessions; per-entry 35%, cumulative 65%. | concentration headroom | 2h code | — |

## 🔴 CRITICAL finding (2026-04-24 late) — rotation currently hurts APY

**A/B result (Route A, lowering rotation ER threshold):**

| Variant | APY | Rot | ΔAPY |
|---|---:|---:|---:|
| A_GOLDEN_v4.1 (threshold 0.03) | +39.82% | 0 | — |
| B_rot_loose_005 (threshold 0.005) | +34.88% | 3 | **−4.93** |
| D_loose + thesis_loose (0.15/0.05) | +34.80% | 2 | **−5.02** |

**Per-rotation impact: ≈ −2.5 APY pts.** Tax drag + missed continuation on held side outweighs realized advantage from swapping to candidate. The golden's **0 rotations is protective, not a bug** — the strict ER threshold (0.03) is blocking net-negative trades.

**Implication for APY=1.41 goal:** rotation is NOT an APY lever under current model. APY lift must come from:
1. Better panel/NGBoost predictions (higher-quality μ estimates)
2. Feature selection (J hourly pruning — pending)
3. Better exits (panel_conviction_exit — pending A/B)
4. Ticker rotation (watchlist adds/drops — screen_watchlist shipped)

**What to do with rotation infra:** keep it (it's dormant at threshold 0.03). BC + thesis-A + Kelly-rot-advantage + thesis-primary flags all stay as dormant options. If future panel improvements produce more accurate swap signals, these gates can be activated one at a time and A/B'd.

**Not a shelve — a re-prioritization.** Rotation research is valuable for its audit logs (every swap decision traces to a rationale) even if rotations don't fire. Focus shifts to model + feature + exit improvements.

## ✅ Resolved from AA data (2026-04-24)

| # | Item | Resolution |
|---|------|---|
| ~~**BULL_VOL-reversal**~~ | **Shelved.** A/B (commit `efcca83` infra + post-sim): 3 variants (GOLDEN vs defensives_only vs full_cash) on 27-mo OOS. defensives_only wins **+0.44 APY pts** only (below +2 pt promotion floor). full_cash = GOLDEN exactly — the upstream filters (tier + Kelly + universe floor) were already keeping BULL_VOL buys near zero. In-sample IC = -0.17 on 445 rows (0.8% of sample) — too thin to move portfolio APY. Infra kept behind `regime.bull_vol_block_offensive` flag (default off). |
| ~~**K** (CHOPPY IC)~~ | **Demoted.** ~~CHOPPY IC=−0.116~~ in F's in-sample calibrator fit was an artifact; live data shows CHOPPY IC = **+0.0354** (1366 rows). Pooled calibrator fine for CHOPPY. |

## 🟠 P1 — Short-term strategic wins (non-Kelly)

| # | Item | Impact | Est. |
|---|------|---|---:|
| **Q** | `min_rotation_hold_days × rotation_advantage` 2D sweep (3×3: `{14,21,30} × {0.0,0.02,0.05}`) | adaptability vs churn | 1 h sim |
| **J** | Hourly-feature pruning — drop 3 weakest (`morning_drift_z` / `overnight_gap_z` / `vol_ratio_z`, all `|IC|<0.016`) + A/B retrain | OOS IC maybe +0.005 | 4 h |
| ~~**S**~~ | ~~live_state DB mirror~~ | ✅ shipped `b07a81c` |
| ~~**CUSUM-cooldown-v2**~~ | ~~Design C confidence-scaled sizing~~ | ✅ infra shipped `10f788a` — needs A/B |

## 🟡 P2 — Analysis / diagnostic

| # | Item | Impact | Est. | Prereqs |
|---|------|---|---:|---|
| ~~**K**~~ | ~~CHOPPY regime diagnosis~~ | ✅ resolved 2026-04-24: AA analysis shows CHOPPY IC=+0.0354 (positive, 1366 rows). F's in-sample -0.116 was calibrator-fit artifact, not a real anti-prediction. Demoted. | — | — |
| **L** | Per-ticker hourly-feature effectiveness (leave-one-out OOS IC) — which tickers carry +4.18 APY from hourly? | watchlist review | 1 day | — |
| **Panel-IC-drift** | Day-over-day panel IC swings ±0.03 under identical hyperparameters; likely per-ticker tournament drift changing feature frame distribution | stability | 4 h | AA helpful |
| **BULL_CALM-streak-watch** | Currently alerts at 15d. F's run hit 52d BEAR streak (valid). BULL_CALM should rarely ≥20d — monitor + per-ticker ScoreBuyTask audit. | gate sensitivity | 4 h | — |

## 🟡 P2 — Housekeeping

| # | Item | Impact | Est. | Prereqs |
|---|------|---|---:|---|
| **N** | Golden config doc consolidation — v1/v2/v3 inline → `## History` section; top reads "current v4 only"; delete v1.md and inline its config block | cleanup | 30 min | — |
| **sector-guard-review** | `max_positions_per_sector=6` with 22 tech tickers = 27% admission rate (tight). Consider sub-sector buckets (semis / software / cloud). | sector design | 2 h | — |

## 🟢 P3 — Passive / blocked on wall-clock

| # | Item | Impact | Prereqs |
|---|------|---|---|
| **I** | Accumulate 4 weeks of live sustainability data (Sun 12 PT plist fires `weekly_apy_check.py`). Decision rule: 30-day live APY < 25% for 2 consecutive Sundays OR drawdown > 20% for 5 consecutive days → postmortem | sustainability trend vs sim | wall-clock (4 weeks) |
| **Transformer revisit gate** | Shelved until panel > 200k rows (currently 47k). Need 4× more data — hourly growth or watchlist expansion. | — | data growth |

---

## ✋ Open decisions — user answered 2026-04-24

1. ~~SKH alternative ticker~~ — **dropped** per user.
2. **Max concentration** — 65% OK, but built up via **multiple sessions** (one signal ≠ one 65% buy). Per-entry cap 35%, cumulative cap 65%. → drives new **Multi-entry accumulation** item.
3. **Kelly sweep comparison** — **run from notebook**; notebook output is the comparison surface.
4. **AB-trim aggressiveness** — **run both experiments** (tight vs hysteresis), pick empirical winner.
5. **CUSUM cooldown v2 design** — **C 锁定** (confidence-scaled sizing, no hard block). Full spec under *Detailed specs → CUSUM cooldown v2*.

---

## Recommended sequencing

**Day 1 — data foundation + quick wins**
1. M⁺ (1 h) — training_runs schema fix
2. P (2 h) — `blocked_by` DB field
3. **AA (1 day) — decision-factor DB** ← keystone; unblocks Kelly-tier-tune, K (CHOPPY), Panel-IC-drift

**Day 2 — Kelly loop closes**
4. Partial-sell infra (2 h)
5. AB-trim (4–6 h)
6. BC — Kelly rotation delta (2 h)
7. Kelly-tier-tune + Kelly-full-sweep (2–3 h) — consumes AA data

**Victory lap (≤1 h):** Q rotation 2D sweep.

**+1 day of buffer:** J (hourly pruning) + S (live_state DB mirror) + CUSUM-v2.

**+1 week:** K (CHOPPY) + L (per-ticker hourly effectiveness).

**Stop rule:** if item's 2× budget blows with no completion path, mark blocked + diagnosis, move to Deferred.

---

## Detailed specs (items needing more than a table row)

### AA. Decision-factor DB  🔥 P0

**Goal:** every candidate the selection loop saw gets its forward 10d return logged, so tiers / Kelly params / rotation thresholds can be tuned from **actual** distributions, not theory.

**Plan:**
1. New SQLite table `ticker_forward_returns` (columns: `run_id, date, ticker, rank_score, panel_score, mu, sigma, kelly_target_pct, rank, fwd_1d, fwd_5d, fwd_10d, fwd_20d, regime, admitted, blocked_by`).
2. Populated by a new `LogDecisionFactorsTask` at end of `RankingJob` (reads current bar candidates) + a backfill job that computes fwd returns once enough bars have accrued.
3. `scripts/analyze_decision_factors.py` — quantile-sliced IC, base-rate-by-tier, Kelly-edge realisation, regime-conditional IC breakouts.

**Acceptance:** one end-to-end run writes ≥ 20 rows, `analyze_decision_factors.py` renders a tier-calibration table showing empirical `P(fwd_10d > 0 | rank_score > tier)` by bucket.

### AB-trim. Partial Kelly rebalance  🟠 P1

**Goal:** when Kelly target for a held position drops below current weight (e.g. big rally pushed weight to 45% but new Kelly says 20%), trim.

**Design alternatives (user decision item #3):**
- **Tight:** trim to exact Kelly target every bar.
- **Hysteresis:** only trim when `current_pct > kelly_target + 10%`. Reduces churn, accepts mild overweight drift.

**Prereq:** partial-sell infra — `AlpacaBroker.place_order` and `PaperBroker.place_order` need a `quantity_override` (sell N shares instead of closing the position).

**Acceptance:** new `TrimHeldTask` in SelectionJob emits `ExitSignal(exit_type="kelly_trim", quantity=...)`; paired alignment tests in `test_policy_alignment.py::TestAbTrimAlignment` (≥6 each side).

### CUSUM cooldown v2  🟠 P1 — design **C** locked

Current: `RegimeState.countdown` is bar-based. 3 intraday bars = 1h, not 3 days. Result: live cooldown is far shorter than sim.

**Design C — confidence-scaled sizing (no hard block):**
- Store `cooldown_start: datetime` in `RegimeState` (persists across runs via `live_state.json`, like other regime fields from R fix `dc7be6f`).
- `cooldown_progress = min(1.0, (now − cooldown_start) / 3 days)`.
- In `SizeAndEmit` + `EmitRotationsTask`: multiply `max_position_pct` by `(1 − cooldown_progress)` when cooldown active. Just after switch → ×0 (effectively no buys). 3 days later → ×1 (full size).
- No hard block on `ScoreBuyTask` — Kelly sizing + confidence scaling does the job together.

**Rationale:**
- Aligns with Kelly philosophy: let size encode uncertainty, not binary gates.
- Fully reproducible (no probability sampling like B).
- Live and sim read the same `datetime` field → behaviour identical.

**Acceptance:** paired alignment tests in `test_policy_alignment.py::TestCusumCooldownAlignment` (≥6 each side); `tests/test_regime_state_persistence.py` extended for `cooldown_start`; expected ~2 APY pt live-sim drift closure.

### K. CHOPPY regime diagnosis  🟡 P2

F's per-regime calibrator fit showed CHOPPY `pool_IC = −0.116` on 646 rows — **direction-wrong**. The pooled calibrator handles it in practice but it's a standing anomaly.

**Hypotheses:**
1. Universal sign flip → add CHOPPY-specific `tiered_thresholds` offset (+0.05).
2. Ticker-dependent → per-ticker CHOPPY behavior table, possibly exclude flippers.
3. Feature-dependent → `vwap_premium` or `morning_drift` carries the inversion.

**Acceptance:** one hypothesis confirmed + 1–2 APY pt A/B win from derived fix. Otherwise, document as irreducible noise.

### J. Hourly-feature pruning  🟠 P1

4 of 6 hourly features have `|IC| < 0.016` (below typical noise). Drop weakest 3 (`morning_drift_z`, `overnight_gap_z`, `vol_ratio_z`); keep `intraday_realized_vol_z`, `vwap_premium_z`, `afternoon_drift_z`.

**Plan:** add to `panel_ltr.drop_cols` → `python scripts/train_104.py --skip-baseline --skip-recalibrate --force` → A/B sim via `/tmp/test_hourly_ab.py` pattern.

**Acceptance:** OOS IC lifts ≥ +0.005 (0.033 → ≥0.038) with live-sim APY ≥ v4 baseline. If yes, promote golden v5.

### I. Live sustainability accumulation  🟢 P3

Already shipped (`67e95af`): `scripts/weekly_apy_check.py` fires Sun 12 PT via `~/Library/LaunchAgents/com.renquant.weekly-apy104.plist`. Computes 30-day APY + drawdown streak, ntfy alert.

**Decision rule:** 30-day live APY < 25% for 2 consecutive Sundays OR drawdown > 20% for ≥5 days → open postmortem against v4 sim baseline (~+65% expected).

**Action:** none. Accumulate 4 weeks, review trend.

### N. Golden config doc consolidation

`doc/ops/golden-config.md` grew v1 → v2 → v3 → v4 inline. Refactor: top reads "current = v4 @ +37.82% sweep APY"; v1–v3 tables move to bottom `## History`. Also: delete `doc/golden_config_2026-04-23.v1.md` (redundant) and update rollback reference in main golden doc to inline v1 config block.

---

## Watch items (monitor, not planned work)

- **Training audit `elapsed_sec` column drift** — see M⁺. Currently log noise only; eliminated when M⁺ ships.
- **Panel IC variance across daily retrains** — 0.03–0.04 day-to-day under identical hyperparameters. Acceptable unless day-over-day Δ > ±0.01 — investigate feature drift.
- **`NoCandidateAlert` streaks ≥ 20 days in BULL_CALM** — F's run showed 52-day BEAR streak (valid); BULL_CALM should rarely hit 15d. Alerts at 15d. Real BULL_CALM 20d streaks → per-ticker `ScoreBuyTask` thresholds may be too tight.
- **Kelly max-streak 43d vs monitoring threshold 15d** — by design; Kelly is disciplined and skips low-μ/σ² bets. If streaks routinely hit 60d+, investigate `base_rate` or tier-1 threshold.
- **Transformer revisit gate — panel > 200k rows.** Currently 47k. Would need ~4× more dates or watchlist expansion to ~100 tickers.

---

## Completed (archive — condensed)

### Session of 2026-04-23 (17 commits)

| # | Item | Commit(s) | Result |
|---|------|---|---|
| **Golden v4** | **A-gate + half-Kelly** | `eb8fab5` | **+11.91 APY pts over v3 (37.82% sim, ~+65% expected live).** Kelly sizing promoted; tiered_thresholds re-anchored to base_rate 0.273. |
| Kelly stack | `kernel/kelly.py` + `ApplyKellySizingTask` + `TopUpHeldTask` + `SizeAndEmit` refactor | `4787825`, `7601b5c` | Continuous-Kelly `f* = μ/σ²`, half fractional, top-up when Δ(kelly_target, current_pct) > 5%. |
| **G** | **Hourly-bar panel features** | `8c65537`, `f03d1eb`, `0c80443`, `3b1d2e2`, `e65b081` | **+4.18 APY pts (40.02 → 44.20% pre-Kelly). Panel 25 → 31 features.** `intraday_realized_vol_z` top-5 by \|IC\|. |
| A | HWM guard | `ab1006d` | `resolve_hwm()` snaps stale HWM to equity. +10 tests. |
| B | LightGBM A/B | `8d6b08a`, `67e95af` | Shelved −12.7 APY pts. Per-row-weight bug fixed on way in. |
| C | σ-penalty sweep | `0c80443` | Shelved — λ=0.25 only +2 pts (below +3 promotion floor). |
| D | Sustainability watch | `67e95af` | JSONL audit + Sun-12-PT plist. Infra for Plan I. |
| F | Regime-conditional calibration | `26c40ae`, `7f68a40` | **Shelved −3.78 APY pts.** In-sample IC 3.5×–17× pooled but didn't survive OOS. Kept behind off-by-default flag. |
| H | Transformer on hourly panel | `c9ee50b` | Shelved again — 0.20× XGBoost (was 0.49× daily-only). Needs > 200k rows. |
| O | Defensive gate in non-BEAR | `52bf718` | XLU BUY in BULL_VOLATILE fixed. `blocks["defensive_non_bear"]` counter. +10 regression tests. |
| R | `regime_state` persisted across live runs | `dc7be6f` | CUSUM countdown wasn't persisted → re-tripped every run. `live_state.json["regime_state"]` now carries 6 fields. +6 tests. |
| T | `entry_dates` fallback persisted | `c5a2ff7` | Legacy positions had hold_days=0 forever — `entry_dates.get(ticker, today)` returned fresh today but never wrote back. Fixed + persisted. |
| V | Held tickers exempt from universe_floor | `369973b` | AMZN unblocked. |
| B² | CUSUM cooldown only on regime switch | `013200a` | Moved into pipeline `CUSUMTask` / `RegimeFinalizeTask`. |
| W / W+ | Network-safety layer (yfinance/OpenBB hangs) | `67b8d64`, `632f3cd` | Per-call + per-ticker + batch timeout. |
| ntfy | Trade-level + decision-level + truthful + retrain-only-Tue/Thu/Sun | `a07f76b`, `d79b6c2`, `3578908`, `d302e5a` | Every live order notified; no false "trained" ntfy on non-retrain days. |
| Env | `requirements.lock.txt` + `doc/ops/environment.md` | `c9ee50b` | Python 3.10.20, xgboost 3.2, ngboost 0.5.10, torch 2.11. 310 pinned packages. |
| — | live_state contract | (earlier) | 9 attributes reviewed end-to-end. `tests/test_live_state_contract.py` (21 tests). |
| — | Watchlist +5 semis | `73a9327` | INTC/MPWR/TXN/NVTS/WDC added. |

### Prior sessions (condensed)

| Item | Commit(s) | Result |
|------|---|---|
| Run 3 — lookahead=10d + regularization | `5fdba09` → T4 | OOS IC 0.025 → 0.040; all 5 folds positive. |
| Global calibrator on panel | pre-session baseline | Pool IC 0.071 on 89k rows. |
| SQLite decision-trace DB | pre-session | 5 tables + sim/live hooks at `data/runs.db`. |
| BaselineTournament winner by IC | pre-session | `oos_single_ticker_ic` metric; default still `sharpe`. |
| Alpaca intraday sell overlay | pre-session | IEX feed, 20 slots 07:00–12:30 PT Mon-Fri. |
| Cross-sectional transformer panel backend | `8f38f80` → `908019a` | Infra kept; shelved at 0.49× XGB daily-only (0.20× hourly — H). |
| Universe floor regression fix | `2df4e21` | `_eval_sharpe` prefers tournament sharpe over noisy live_holdout_sharpe. APY 2.4 → 10.1%. |
| ConfidenceVeto disabled | `33c0e9b` | GMM posterior capped ~0.25; threshold 0.30 → 0 unblocks offensive buys. |
| DrawdownCircuitTask resets on recovery | `e586018` | THE bug behind 153-day no-trade streak. APY 10.9 → 33.1%. |
| Golden v1 snapshot (33.1%) | `d3ef68f` | First post-fix golden. |
| T4 — xgb_params revert (40.1% v2) | `ee4faab` | Reverted to pre-regression panel `xgb_params`. |
| recalibrate_scores.py race shield | `bc81360` | Merge-on-write. |
| PanelScoringJob reorder | `339944b` | NGBoost before calibration. |

---

## Deleted / no-action

- **Revert `lookahead_days` 10 → 5** — prior evidence shows 5d regresses.
- **E — re-fit calibrator on μ−λσ** — conditional on C winning; C shelved.

---

## 🌙 2026-04-28 — M-Series structural fixes (response to 2026-04-27 ridiculous trades)

**Context:** 2026-04-27 daily run produced unreasonable trades (CAT closed mid-trend, NET bought with 20d -1.5% return, TSM trimmed at 52w high). Root cause: model has a single 5d horizon + uniform Gate B threshold + momentum-blind QP solver. User explicitly rejected hardcoded-threshold band-aids in favor of structural / data-driven solutions.

**Roadmap items: M1, M2, M3 in progress (2026-04-28). M4 deferred.**

### M1 — Multi-horizon panel-LTR ensemble (5d / 20d / 60d) — IN PROGRESS

Train three independent panel-LTR + NGBoost models at lookahead_days = {5, 20, 60} on the 227-ticker watchlist. Inference loads all three.

Why: a single 5d horizon over-weights short-term mean reversion, mis-rates trending stocks. 20d/60d horizons learn trend persistence directly from data — no hardcoded "20d ≥ 0" gate needed. NET (20d −1.5%) gets a NEGATIVE μ from the 20d/60d head, dragging blended μ below Gate B. CAT/TSM (positive 20d) get supported.

ETA: 1-2 days post-B1. Three side configs writing to `panel-ltr.{5d,20d,60d}.json` + `ngboost-head.{5d,20d,60d}.json`.

### M2 — Learned regime-conditional blender — IN PROGRESS (after M1)

Small MLP (3-layer, 32 hidden, dropout 0.3) takes [μ_5, σ_5, μ_20, σ_20, μ_60, σ_60, regime_one_hot, recent_realized_vol_z, hwm_drawdown] → final (μ, σ).

Trained on ~75k panel rows × 4 regimes against forward-20d realized return. Replaces hand-tuned blend weights — the data tells us which horizon matters most in each regime.

ETA: 0.5 day post-M1.

### M3 — Conformal-calibrated Gate B (per-regime dynamic τ) — IN PROGRESS

Replace fixed `gate_b_τ=0.10` with conformal prediction calibrated per regime. For each regime r:
- history = past 252 bars in r
- For each candidate threshold τ: count {candidates with edge_sharpe ≥ τ that produced negative 5d return}
- τ_r = smallest τ such that FDR(τ_r) ≤ target (e.g. 30%)

Result: BULL_CALM might land τ ≈ 0.08 (FDR forgiving), CHOPPY ≈ 0.15 (FDR strict), BEAR ≈ 0.25 (very strict). Threshold adapts to regime stress automatically — no hardcoded gate.

ETA: 0.3 day. Code path: `kernel/panel_pipeline/task_quality_floor.py::_gate_b_edge_sharpe` — read τ from a regime-keyed JSON, fitted nightly.

### M4 — DEFERRED — End-to-end RL portfolio policy

**Status: ROADMAP ONLY. Not implementing now.**

**Sketch:**
- Replace QP solver with a small policy network: `state = [panel features, current weights, regime, market state] → action = Δw vector`.
- Reward: forward 20d portfolio return − transaction cost − drawdown penalty.
- Train via PPO or off-policy with replay buffer of historical bars.
- Self-learns: "don't churn trending positions", "only enter on confirmed momentum", "size up in BULL_CALM, size down in CHOPPY".

**Why deferred:**
- 75k panel rows is dangerously small for stable RL convergence.
- M1+M2+M3 should already address most of the 2026-04-27 ridiculous-trade root causes structurally.
- If M1+M2+M3 don't fully fix the trend-blind QP behaviour, revisit M4 with synthetic data augmentation + transformer-based policy.

**Required infra (when we revive M4):**
- Off-policy evaluation harness (already partial via doc/components/trade-evaluation.md).
- Replay buffer in runs.db schema.
- Small GPU (M-series MPS or cloud A100 spot) for PPO update steps.
- Behavior cloning warmup against current QP outputs.

**Decision criterion to revive:** if post-M1+M2+M3 OOS Sharpe < 1.5 OR live trades still mismatch market intuition on >20% of bars, revisit M4.

---

## 🕐 盘中买入路线图 (Intraday Buy Execution) — 2026-04-29

**背景**：当前所有市场时段 cron（开盘/盘中/收盘前）均运行 `--sell-only`，系统从未执行买入。模型信号基于日线 OHLCV，"算法买入还是要看日线，所以不能盘中交易"——这是当前阶段的正确约束，但阻止了所有新仓建立。本路线图规划如何在日线信号框架内安全启用买入。

**设计原则**：
1. 买入信号仍基于 T-1 日线 OHLCV（前一交易日收盘后计算），不引入实时定价假设
2. 执行时点选择：最大化信号有效性 vs 执行质量的权衡
3. 每阶段独立可验证，后阶段不依赖前阶段的强假设

---

### Phase 1 — 最小改动：开盘执行 T-1 信号 (P0，~1 day)

**目标**：恢复买入，使用昨日收盘后计算的面板分数，在今日开盘执行。

**信号有效性**：
- 面板 LTR 分数在每日 retrain 后（当晚或隔日 6:00 PT 前）计算完毕
- 60d horizon 信号：60 天内的排名稳定性高，T+0 开盘执行 T-1 信号完全合理
- 10d horizon 信号：T-1 排名在 T+0 仍有效（1 天衰减可忽略）

**实现**：
1. 移除 `com.renquant.open104.plist` 中的 `--sell-only` 标志
2. 确保 retrain plist（`retrain-panel104.plist`）在 open plist 之前完成（当前已是当晚运行，无需改动）
3. pre-flight 检查模型 `trained_date` 是否为 T-1 或更新（防止 stale 信号进场）

**风险与缓解**：
- Gap risk（隔夜跳空）：面板分数不含隔夜信息。缓解：`max_position_size` 已设 35%，regime 门已启用
- 信号 stale：pre-flight `trained_date` 检查（已有 P-MODEL-ARTIFACT 检查，扩展日期验证即可）

**验收标准**：`--once` dry-run 产生 ≥1 个 candidate，无 pre-flight 报错。

---

### Phase 2 — 预市信号刷新 + 开盘执行 (P1，~3 days)

**目标**：在市场开盘前，用最新 OHLCV 数据（含 T 日盘前价）刷新面板分数，减少隔夜信息损失。

**背景**：yfinance / Alpaca 提供盘前数据（6:00-6:30 PT），可将面板分数更新到 T 日盘前。

**实现**：
1. 新 cron `retrain-intraday-panel104.plist`：每日 6:00 PT，仅跑 `PanelDataJob + PanelFeatureJob + PanelModelJob`（不跑 NGBoost/calibrator，约 7 min）
2. 使用 side config `strategy_config.intraday_signal.json`，artifact 路径隔离，不覆盖 production
3. 开盘 cron 优先读取盘前刷新的 artifact（如存在且日期为今日），fallback 到前晚 artifact

**关键判断**：盘前刷新 CPCV IC 提升幅度 vs 7min 延迟成本。需要 A/B sim 验证（盘前刷新 vs 前晚信号 APY 差异）。

---

### Phase 3 — 信号热度检测 + 动态执行时点 (P2，~1 week)

**目标**：根据信号强度和市场微结构，选择最优盘中执行时点（不只是固定 6:30 开盘）。

**设计**：
- `SignalFreshnessScore`：panel_score 在过去 3 日的稳定性（rank 变化幅度）
- 高稳定性信号：在 VWAP 附近分批成交（减少冲击）
- 低稳定性信号：等待开盘后 30min 价格稳定再执行
- 结合 `intraday_realized_vol_z`（已有特征）判断当日波动率状态

**注意**：这阶段开始引入盘中数据依赖，需要 Alpaca WebSocket 实时 feed 或高频 polling。

---

### Phase 4 — 真正盘中信号（长期，T2/T3 级别）

**目标**：基于分钟/小时级特征的盘中排名模型，支持真正的日内信号更新。

**前提条件**：
- 面板 > 150k 行（当前 ~75k，需 watchlist 扩到 200+ 或积累 2+ 年日内数据）
- Phase 1-3 稳定运行 ≥3 个月，建立基准
- 独立的分钟级 panel-LTR 训练 pipeline（与日线模型完全解耦）

**模型选型备选**：
- Transformer（hourly feature panel，T2 级别）
- TGNS（图注意力 + 分钟级，T3，参见路线图 T3-1）

**执行架构**：
- `InferencePipeline` 增加 `IntradaySignalJob`：每 30min 刷新分钟特征 → 更新 candidates
- 信号变化超过阈值才触发换仓（避免过度交易）

---

### 决策树（何时推进各阶段）

```
[当前: sell-only]
       ↓
Phase 1 ──→ 开盘买入恢复 ──→ 观察 5-10 交易日买入质量
                              ├── OK → 推进 Phase 2（盘前刷新）
                              └── 问题 → 诊断后再推进

Phase 2 ──→ 盘前刷新 A/B ──→ APY 提升 > +1 pt → 升级为默认
                              └── 无提升 → 保留 Phase 1，Phase 2 作 fallback

Phase 3 ──→ 需要 Alpaca WebSocket infra，评估建设成本 vs 收益

Phase 4 (renquant_105) ──→ 独立策略版本，见 renquant_105 设计文档
```

---

### 当前 Action Items（Phase 1，随时可做）

1. **[5 min]** 在 `com.renquant.open104.plist` 移除 `--sell-only`，改为完整推断模式
2. **[30 min]** pre-flight 加 `trained_date` 检查：`today - trained_date ≤ 2 trading days`（超过则告警但不阻止）
3. **[1h]** `--once` dry-run 验证：候选股生成 → Gate A/B 通过 → 订单正确路由 → 不实际下单
4. **[视情况]** 在 Z9 alpaca-paper 验证通过后，paper 账户先跑 5 天，再切 live


---

### 版本边界说明（2026-04-29）

**renquant_104**（当前）：日线 OHLCV 信号，Panel-LTR 排名。盘中执行仅限"在最优时点执行日线信号"，不引入分钟级特征训练。Phase 1-3 均在 104 框架内完成。

**renquant_105**（下一版本）：30min 级别模型。独立训练 pipeline，独立特征体系，独立 IC/APY 评估基准。前提条件：
- renquant_104 Phase 1-3 稳定运行 ≥ 3 个月（建立基准）
- 分钟级面板 > 1M 行（103 ticker × 2 年 × 16 bars/日）
- 独立 `backtesting/renquant_105/` 目录，不复用 104 的 strategy_config.json

**105 的设计工作从 104 Phase 3 稳定后启动。**

