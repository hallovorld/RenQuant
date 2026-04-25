# RenQuant 104 — Planned Work

**Single source of truth for what's next.** Ordered by priority tier (P0 → P3), then by expected value within tier.

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

Full details: `doc/golden_config_2026-04-23.md`. Training history: `doc/panel_training_runs.md`. Methodology: `doc/panel_ltr_primer.md`. Environment: `doc/environment.md`.

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
- ✅ **Rotation research doc** (`709032d`) — `doc/rotation_research_2026-04-24.md` — literature review: Jegadeesh-Titman, Moskowitz TSMOM, Barroso-Santa-Clara, Daniel-Moskowitz crashes, Avellaneda-Lee pairs, Grinold-Kahn breadth, López de Prado ML. 6 ranked implementation proposals.
- ✅ **V4 own-momentum gate** (`9463e4c`) — Proposal 1 shipped: A's own 63d return must have broken AND B's must be intact before rotating. Jegadeesh-Moskowitz compliance. 4 new tests.
- ✅ **10-min bar fetch complete** — 744k rows × 50 symbols × 2yr cached. Prereq for transformer retry (>200k row gate) satisfied.
- 🟡 **10-min panel retrain A/B running** (`/tmp/minute_panel_retrain_and_ab.py`). **Preliminary**: CPCV OOS IC = **+0.0355 (+0.003 vs baseline)** — hourly+minute panel beats hourly-only on OOS IC. NGBoost + sim phases pending. Full verdict ~10 min.

### ✅ ALL ROTATION + PANEL + DATA WORK SHIPPED THIS SESSION

Final result: **Sim APY 28.82% → +30.90%** on 27-mo OOS.
Panel CPCV OOS IC: 0.0391 → **+0.0536** (+37%).
Watchlist 43 → 99. Win rate 80.5%. Rotations finally fire (2x).
Full session detail: `doc/session_summary_2026-04-24.md`.

### ✅ Status of items in the previous "not shipped" list

- ✅ **Rotation V1** depth + persistence — shipped + tested (default off).
- ✅ **Rotation V2** μ-λσ direct mode — shipped + tested (default off).
- ✅ **Rotation V3** regime + held-DD gates — shipped + tested (default off).
- ✅ **Rotation V4** thesis-symmetric — shipped + tested + DB lookup wired (default off).
- ✅ **V4 own-momentum gate** (Jegadeesh/Moskowitz) — shipped + tested.
- ✅ **Sharpe scoring mode** (Barroso) — shipped + tested.
- ✅ **10-min panel retrain end-to-end** — done. +9.57 APY on isolated A/B; +2.08 APY on clean main retrain.
- ✅ **Transformer retry** — done. 0.89× XGBoost on 43-ticker panel — shelved again (panel still under transformer's data threshold; revisit at watchlist 120+ or 10-min training window).
- ✅ **Rotation algorithm review** — done. `doc/rotation_research_2026-04-24.md` with academic refs + 6 proposals.

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
- ✅ **doc/database.md** — full schema reference with migration rules.
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

`doc/golden_config_2026-04-23.md` grew v1 → v2 → v3 → v4 inline. Refactor: top reads "current = v4 @ +37.82% sweep APY"; v1–v3 tables move to bottom `## History`. Also: delete `doc/golden_config_2026-04-23.v1.md` (redundant) and update rollback reference in main golden doc to inline v1 config block.

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
| Env | `requirements.lock.txt` + `doc/environment.md` | `c9ee50b` | Python 3.10.20, xgboost 3.2, ngboost 0.5.10, torch 2.11. 310 pinned packages. |
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
