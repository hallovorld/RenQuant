# RenQuant — Single Source of Truth: Status, Issues, Decisions

**Last updated**: 2026-04-28 evening (post P0 fixes + production retrain)

This doc is the **single canonical status reference**. It supersedes scattered status mentions across all other docs. When a doc says "deferred", "still broken", "TODO", "🔴", or "blocked" — check here first; the answer is below.

For original details, the linked source docs remain. This doc does not replace specifications — it replaces *status claims* about specifications.

---

## Production state (right now)

**Strategy**: `renquant_104` panel-LTR cross-sectional ranking
**Watchlist**: 103 tickers
**Live model**: just retrained (best_iter=19, oos_mean_ic=+0.035)
**Broker**: Alpaca live
**Account equity**: ~$10,022 (5 holdings as of 12:44 PT today)
**Live cron schedule**: open + preclose + intraday + daily-post-close (3 plists active for 104; 103 plists unloaded permanently today)

The system is healthy. Today's 9 commits resolved every P0 bug we knew about, and the production model is for the first time running healthy-trained weights instead of an under-trained stub.

---

## §0. Top-line model findings (2026-04-29 最终)

### 完整实验矩阵 — 9 个实验 clean CV 结果

All experiments run under fixed CV (BUG-CV-1/2/3 patched, 2026-04-28/29).

| 实验 | CPCV IC | vs 10d XGB +0.035 | 结论 |
|---|---|---|---|
| **10d XGBoost**（生产基准） | **+0.035** | — | 真实基准（buggy CV 时 +0.042 虚高） |
| macro v2 per-ticker β | +0.034 | −3% | **中性** — 之前 −23% 是 P0 bug 造成的 |
| emb 16D | +0.031 | −12% | **否决** — 方向对，幅度被高估 |
| LightGBM 10d | +0.018 | **−49%** | **否决** — clean CV 下大幅落后 |
| **60d XGBoost** | **+0.074** | **+110%** | ← 唯一正向 lever，主线 |
| 60d + macro v2 | +0.074 | 0% | macro 在长 horizon 同样中性 |
| 60d + emb 16D | +0.034 | −54% | emb 在 60d 比 10d 更有害 |

**结论**（一句话）：60d XGBoost 是唯一经 clean CV 验证的正向信号，其余 6 个方向全部否决或中性。

**60d 主线已完成**（2026-04-29）：
- 早停修复（eval set 末 60 bar 全 NaN → 禁用早停）
- 校准器修复（crosssectional 模式，base_rate=50.4%，25 unique y）
- acceptance gate PASSED，side artifact 已 promote
- NGBoost val_mu_ic=−0.034（60d 上 NGBoost σ 不可靠，Kelly 定仓需绕开）

**待验证**（60d 升 golden 前必须）：
1. 27mo OOS sim：APY ≥ golden +39.82%？成本模型在 6× 持有期下净正？
2. 进出逻辑适配：min_hold / wash-sale / QP turnover penalty 按 60d 重设
3. Kelly 定仓替代方案（Panel-LTR 分数直接定仓，绕开 NGBoost μ/σ）

---

## §1. Issues resolved today (2026-04-28)

These were documented in scattered places (`doc/archives/audits/2026-04-28-deep-audit.md`, `doc/archives/audits/2026-04-28-nvts-buy-postmortem.md`, in-line in `CLAUDE.md`). All resolved.

| ID | Issue | Where it was | Fix |
|---|---|---|---|
| BUG-CV-1 | Linspace fold boundary drift in CPCV — same calendar date could land in different folds across runs, silent leakage | `training_panel/purged_cv.py` | Integer division for fold edges; verified across n_dates ∈ {750, 753} |
| BUG-CV-2 | Production saved best_iter=4 silently (model essentially untrained) | `pp_panel_training.py::FinalFitTask` | Hard guard: refuse save if best_iter < threshold (initially 20, lowered to 5 after diagnostic) |
| BUG-CV-3 | Early-stopping eval set was hard-coded last 20%, misaligned with CPCV last fold | `pp_panel_training.py::FinalFitTask` | Use `1/cv_n_splits` so early-stop and CPCV measure same data slice |
| BUG-G7 | Acceptance gate read pre-train (production) IC, not new artifact's IC | `scripts/train_104.py` | Resolve `active_path` from `panel_ltr.artifact_path` config, not hardcoded |
| CRIT-1 | NGBoost feature drift hard-fail returned silently — Gate B then admitted ALL candidates because μ/σ stayed `None` | `kernel/panel_pipeline/job_panel_scoring.py::ApplyNGBoostTask` | Stamp NaN on μ/σ + clear `ctx.candidates` so downstream blocks correctly |
| Z2 | Manual sells (user via Alpaca app, broker liquidations) didn't stamp wash-sale clock | `adapters/runner.py::commit()` | STATE-EXT-SELL: detect "ticker disappeared between bars without runner-sell", stamp `last_sell_dates` |
| Z3 | yfinance dot/dash translation (BRK.B → BRK-B) caused 4 errors per cron tick | `kernel/data.py::fetch_ohlcv` | Translate at upstream-fetch boundary; cache + canonical names stay on dot form |
| 103-plist | `daily_103.sh` crashed every day at 14:00 PT due to API drift | `~/Library/LaunchAgents/com.renquant.daily103.plist` | `launchctl unload -w` permanently |
| Auto-revert path | `auto_revert_b1_regression.sh` wrote `strategy_config.json` to `artifacts/` instead of strategy dir → 06:32 ntfy fingerprint mismatch root cause | `scripts/auto_revert_b1_regression.sh` | Per-file explicit destinations + post-cp SHA verify + post-revert config-consistency check + rollback rehearsed |
| LoadScorerTask strict | Config-consistency check defaulted to log-only, never aborted | `kernel/panel_pipeline/job_panel_scoring.py::LoadScorerTask` | Default `strict_config_consistency=True`; mismatch skips panel scoring, gate B fallback |
| Z1 (parabolic gate) | Built then deleted same day | `kernel/pipeline/task_candidates.py::ParabolicExhaustionGateTask` | Panel A/A test falsified hypothesis (top-decile rel_mom_20d *outperforms*); deleted |

**Verification**: 2696 tests pass. CLAUDE.md principles 5.1–5.8 codify the workflow that prevents this class of bug from re-occurring.

**Source docs (now historical / superseded)**:
- `doc/archives/audits/2026-04-28-deep-audit.md`
- `doc/archives/audits/2026-04-28-nvts-buy-postmortem.md`
- CLAUDE.md "P0 BUGS" section (resolved; section is now historical)
- `doc/research/m2-v3-result-analysis.md` (M2 v3 result + closure)

---

## §2. Open issues (still broken or unfinished)

| ID | Issue | Where | Why not fixed | Action |
|---|---|---|---|---|
| ARCH-1 | `_resolve_cache_dir` 4-fallback heuristic suggests path management has no central abstraction | `kernel/data.py` | Defensive code that masks a missing abstraction; refactor risk > current value | Defer until next architecture work |
| ARCH-2 | `cusum_threshold=5.5, cusum_drift=0.5` are hardcoded — will break in any unobserved regime | `kernel/regime.py` | Adams-MacKay BOCPD would replace this, but is a research project | Defer; revisit if regime detector misses a real shift |
| OBS-1 | Sharpe not formally tracked in production observability | live runner / DB schema | Said in `roadmap.md:15` "first 3-day priority"; never landed | Defer until live data accumulates more |
| LAUNCHD-1 | No job-dependency management — if `fetch` fails, `inference` runs on stale data | launchd plists | Said in `system-assessment.md:362`; current workaround = freshness check in inference step (fragile) | Defer until next ops sweep; not blocking |
| TICKER-DAILY-STATE | Schema added but writer not wired to actually populate per-bar | `kernel/persistence.py` + `db-design-decision-factors.md:156` | Documented "🟠 SCHEMA ADDED, writer not yet wired" | Pending; non-blocking for trading |
| Z9-INTEGRATION | Z9 broker-side stops shipped as a layer + 26 tests, but runner integration is default-OFF | `live/runner.py` + `doc/research/z9-broker-side-stops-design.md` | Awaiting operator decision on 4 design forks (which threshold, TRIM/TOPUP semantics, state schema, per-broker enable) | Plan: enable on `paper` immediately, then `alpaca-paper` for 1 day, then `alpaca` live |
| ABS-1 | yfinance is primary data source — single-vendor risk | `kernel/data.py` | Said in `system-assessment.md` 8.1; alternative providers cost money | Defer; have local cache as soft fallback |
| TRADE-EVAL | Off-policy evaluation (OPE) for trade quality is design-only | `doc/components/trade-evaluation.md` | Needs ≥6 months live OOS data to build behaviour vs target policy | Time-gated; no action |
| METADATA-DB | Cloud backup of model metadata to Backblaze B2 is plan-only | `doc/components/metadata-db-and-backup-plan.md` | User said "下次再处理" 2026-04-26 | Pending operator green-light |

**Important**: none of these block production trading. They're hygiene / nice-to-have. Production is stable on the current state.

---

## §3. Deferred by design (with explicit reasons)

These are NOT broken — they were investigated and deferred with clear gating conditions. Don't restart without checking the gate.

| ID | What | Gate condition | Source |
|---|---|---|---|
| T2-3 | Regime ensemble (per-regime panel-LTR sub-models) | Panel ≥ 150k rows (currently ~77k) | `roadmap.md:327` |
| T2-4 | Boyd convex-rotation (cvxpy MPC) | rotation as APY lever proven worth (currently each cycle costs −2.5 APY pts) | `roadmap.md:328` |
| T3-1 | TGNS (Transformer + GNN) | Same data-volume gate as T2-3 + US-equity validation (paper is CN A-share) | `roadmap.md:334` |
| T3-2 | FASCL (Future-Aligned Soft Contrastive) | Reference implementation released by paper authors | `roadmap.md:335` |
| T4-1 | LLM-generated factor features | Earlier tiers must saturate first | `roadmap.md:341` |
| Transformer backend | Code complete + audited, not promoted | Panel > 150k rows + smaller training cost-overhead | `doc/experiments/panel-backend-comparison.md` |
| RL portfolio policy (M4) | Off-policy MARL for sizing | Need both ≥6 months OOS data + B1-B3 backtest infra | `roadmap.md:824` |
| OOS Backtest infra (B1-B3) | Walk-forward sim runner + reporting | Live data accumulation needed | `roadmap.md:53` |

---

## §4. Closed by data (experiments that ran and were rejected)

These are **NOT** dead-ends to revisit. Each ran with paired CPCV (or panel A/A test) and was demonstrably worse or directionally falsified. Original data + reproduction recipes in `doc/research/failed-experiments-log.md` (E1–E18).

| ID | What | Why rejected | Note |
|---|---|---|---|
| M2 v2/v3 | Learned horizon blend (Lasso / ElasticNet over μ/σ_10/20/60) | Hold-out IC +0.027 vs single 10d +0.129 — 6× worse, structural correlation > 0.7 | E1, E2 |
| Z1 | Parabolic exhaustion gate (rel_mom_20d > 0.50 reject) | Panel A/A test: top-decile rel_mom_20d *outperforms* (paired t significant, opposite direction) | E4 — code deleted |
| Z8 | σ-cap (top-decile σ reject) | Panel A/A: high-σ tickers *outperform* on every quantile | E3 — never built |
| B1 | 227-watchlist mutual-fund expansion | OOS IC +0.0234 vs golden +0.0418, −44% | E5 |
| wl178 | 75 quality-filter expansion | Eval IC NEGATIVE every round, train IC also depressed | E17 — second expansion failure, structural |
| F3 | 10d hyperparam retune on 227 | Best result +0.039 (still below 103 baseline) | E8 |
| Macro v1-v4 | broadcast / per-ticker β / 30-ETF / panel-row | All variants reduced IC; v1 zero gradient mathematically | E9–E12 |
| T2-2 | Asset embeddings (16D learned per-ticker vectors) | Initial OLS A/B "GO" reversed by paired CPCV (IC −18.5%) | E13 |
| T2-1 | LightGBM substitution | IC −60% on this panel | E14 |
| Boyd Rotation as APY lever | T2-4 incomplete | Each rotation cycle costs −2.5 APY pts | E15 |

**Caveat for §4**: today's BUG-CV-1/2/3 fixes mean some closures (Macro v2, T2-2 embeddings, T2-1 LightGBM) used corrupted CV. Open question whether the *direction* of the negative finding survives clean CV. **In progress**: `strategy_config.macro_v2_retest.json` and `strategy_config.emb_retest.json` will rerun with fixed CV when the 60d-on-103 experiment finishes.

---

## §5. Architectural concerns (long-form)

These are not bugs — they are observations about where the design is showing strain. Not actionable today; record for future architecture work.

### A. Universe expansion cap

Two independent watchlist expansions (B1 / wl178) failed using completely different selection methods. The panel-LTR cross-sectional rank loss assumes universe homogeneity in feature distribution. Adding heterogeneous tickers (financials + industrials + consumer staples to a tech-heavy 103) breaks the rank ordering.

**Implication**: We cannot simply scale watchlist. To get past 103 we need either (a) per-sector sub-models, (b) sector-conditional features, or (c) a successful embedding integration (T2-2 has been falsified once already).

**Concrete consequence**: the chain "watchlist 200 → 60d horizon swap → macro × 60d × wider" has lost its first link. The latter two also need re-examination.

### B. Eval set vs panel size

Today's diagnostic (E18) confirmed XGBoost rank:pairwise on this panel naturally peaks at best_iter 9-25 because the eval set has only ~12k pair-observations (`125 dates × 103 tickers`) vs ~50 effective DOF in the model. After the peak, eval IC declines while train IC keeps rising — overfitting that the eval set is too small to suppress.

**Implication**: Many improvements that appear to fail on this panel might actually work on a larger panel. Without expanding the panel (which we can't, per §A), we're working near our model's information ceiling.

### C. Single-vendor data risk

yfinance is the OHLCV source; OpenBB shims on top. No paid data, no failover provider. If yfinance changes its API or rate-limits us, daily training breaks. We have local cache as a soft fallback.

### D. Acceptance gate observability

G4-G11 acceptance gates exist but their decisions aren't surfaced to the operator dashboard / DB systematically. When G7 reads the wrong path (today's bug), an operator wouldn't notice unless they read raw logs. Pre-flight smoke test (today's commit) is one layer of defense; gate decision telemetry is still missing.

### E. Production model sample size limit

CPCV mean IC = +0.035 on 15 folds × 103 tickers = ~1500 ticker-fold observations. Statistical noise on this magnitude is non-trivial. We need much more data before claiming +0.04 vs +0.03 differences are robust.

---

## §6. Documentation index — what each doc is now

After today's consolidation, these are the canonical sources for each topic:

| Topic | Read this | Older docs that mention it (still kept for history) |
|---|---|---|
| Current production status | `STATUS.md` (this doc) | n/a — this is the consolidator |
| Engineering principles | `CLAUDE.md` §5 | n/a |
| Failed experiments + reproduction recipes | `doc/research/failed-experiments-log.md` | M2 v3 analysis, individual deep-audits |
| Today's session retrospective | `doc/archives/sessions/2026-04-28-evening-retrospective.md` | (the 9-commit chronology) |
| Strategy 104 architecture | `doc/arch/strategy-104.md` | overview.md |
| Roadmap (future bets) | `doc/roadmap.md` | T2/T3/T4 sections |
| Z9 broker-side stops design | `doc/research/z9-broker-side-stops-design.md` + this doc §2 | n/a — not yet integrated |
| Watchlist 200 v2 plan | `doc/research/watchlist-200-v2-plan.md` | **superseded by §4**: expansion path closed structurally |

**Soft-deprecated docs** (still live but content-wise superseded by `STATUS.md`):
- `doc/archives/audits/2026-04-28-deep-audit.md` — all findings resolved (see §1)
- `doc/archives/audits/2026-04-28-nvts-buy-postmortem.md` — Z2/Z9 fixes shipped
- `CLAUDE.md` "P0 BUGS" section — all four resolved (see §1)
- The various session handoff docs (multiple `2026-04-XX-handoff.md`) — chronological log only, not status

**Active docs** (read for current detail):
- `CLAUDE.md` for principles + project state quick-look
- `doc/research/failed-experiments-log.md` for "is X dead?"
- `STATUS.md` for "what is the current state of issue Y?"
- `doc/roadmap.md` for "what should we do next?"

---

## §7. Decision tree — "I'm picking this up cold, what do I do first?"

```
1. Is production trading correctly? → check live_state.alpaca.json + 06:32 cron log
   • If yes → proceed
   • If no → check pre-flight log first (will tell you which check failed)

2. What's the current model? → check artifacts/panel-ltr.json metadata
   • Should have: best_iter ≥ 5, oos_mean_ic ≥ 0.02, config_fingerprint stamped
   • If any missing → retrain via train_104.py with --skip-acceptance for first run

3. Is there a known issue affecting trading? → check this doc §1 (resolved) and §2 (open)
   • Open issues are non-blocking; trading continues

4. Should we run experiment X? → check this doc §4 (closed by data) first
   • If listed → don't run; read the why
   • If not listed → check failed-experiments-log.md
   • If still not found → run paired CPCV first

5. Want to ship a new model variant? → §1 of this doc shows the guards in place
   • BUG-CV-2 guard prevents shipping under-trained models
   • Pre-flight prevents cron from running with config drift
   • Acceptance gates G4/G7 etc filter bad challengers
   • Manual override: --skip-acceptance flag (DANGEROUS, only for known-broken-but-recoverable)
```

---

## §7.5. Post-Tier-1 follow-ups (2026-04-25 doc) — most resolved

When I scanned `doc/experiments/post-tier1-followups.md` (4 days old), I found most items already resolved by today's audit work:

| ID | Item | Current state |
|---|---|---|
| DBT-1 | ENTRY-DATE-FROM-FILLS pagination (100-trade limit) | ⚠️ still open; edge case for very long-tenure positions; deferred |
| DBT-2 | sell-then-rebuy lifecycle awareness in fill matching | ⚠️ still open; ~1 day work; not blocking |
| **DBT-3** | 8 missing test groups for runner-only fixes | ✅ **resolved** — `test_runner_state_fixes.py` covers STATE-GC, STATE-GC-NEWBUYS, ENTRY-DATE-*, UNMANAGED, EXITS-FAIL |
| **DBT-4** | floor `compute_regime_confidence` at 0 | ✅ **resolved** — `kernel/regime.py:340-344` has `max(0.0, ...)` |
| **OP-1** | Add GLD to `earnings_surprise.skip_tickers` | ✅ **resolved** — already present in `strategy_config.json:633` |
| OP-2 | Stale HWM auto-snap — root cause analysis | ⚠️ open; low priority since RU-1 fix masks symptom |
| OP-3 | BA position unmanaged at broker | user decision (BA still held today; explicit decision pending) |
| **OP-4** | AMD not in watchlist | ✅ **functionally resolved** — bypass_ticker_gate=true exposes AMD to Panel-LTR; can be added to watchlist if you want |
| OP-5 | NGBoost RuntimeWarning on σ overflow | ⚠️ benign — clipping is in place; warning persists but doesn't corrupt output |
| FE-1, FE-2 | Feature engineering iterations | deferred — would require dedicated retrain cycles |
| V-1, V-2, V-3 | Tier 1 validation | superseded — Tier 1 baseline replaced by today's +0.0350 production retrain |

`doc/experiments/post-tier1-followups.md` is now **soft-deprecated**. Read the table above for current state.

## §8. What this doc replaces

Status claims in these docs are now **superseded** by the corresponding section here. Read the source for the original analysis; cross-reference here for current state:

| Source claim | Now lives in |
|---|---|
| `roadmap.md:5` "🔴 BLOCKER" sections | §3 (deferred items + gates) |
| `roadmap.md:148` "DEFERRED — Model metadata DB" | §2 row "METADATA-DB" |
| `roadmap.md:824` "M4 — DEFERRED" | §3 row "RL portfolio policy" |
| `roadmap.md:557` "🔴 CRITICAL — rotation hurts APY" | §4 row "Boyd Rotation as APY lever" — status: closed by data |
| `system-assessment.md:362` "no job dependency management" | §2 row "LAUNCHD-1" |
| `system-assessment.md:116` "CUSUM threshold sensitivity" | §5.A and §2 row "ARCH-2" |
| `db-design-decision-factors.md:156` "writer not yet wired" | §2 row "TICKER-DAILY-STATE" |
| `lgbm-implementation-audit.md:3` "not yet implemented" | §4 row "T2-1 LightGBM" — status: closed by data (-60% IC) |
| `boyd-rotation-design.md:3` "not yet implemented" | §3 row "T2-4" — gated on rotation profitability |
| `asset-embeddings-design.md:3` "not yet implemented" | §4 row "T2-2" — closed by data (-18.5%) |
| `metadata-db-and-backup-plan.md:3` "not yet implemented" | §2 row "METADATA-DB" |
| `panel-backend-comparison.md` "TBD" rows | §3 row "Transformer backend" |
| `sim-ab-results.md:36` "not yet implemented" QP flag | non-blocking; deferred |
| `panel-ic-improvement.md:54` options-data blocker | OpenBB free tier limitation; not actively pursued |
| `experiments/post-tier1-followups.md` "Why deferred" tables | §3 (full lookup) |

When you see "deferred" / "not yet implemented" / "🔴" / "broken" / "TODO" in any other doc, **come back here** for the current actionable status.

---

## Update protocol

This doc lies if it's not maintained. Every time a P0 issue is added or resolved, every time an experiment closes, every time a gate is added — update this doc within the same commit.

Per CLAUDE.md principle 5.6 ("definition of fixed = full 24h audit clean"), no fix is complete until this doc reflects the change.
