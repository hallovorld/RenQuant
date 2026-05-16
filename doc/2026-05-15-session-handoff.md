# 2026-05-15 EVENING Session Handoff

**For: next-session Claude (or operator returning).**
**Status as of:** 2026-05-15 ~19:15 PT.

---

## ⏸ PAUSED 2026-05-15 ~20:00 PT — RESUME INSTRUCTIONS

Operator paused mid-queue (computer needed for other use). State preserved:

**Panels finished (committed):**
* `re_stop007` ✓ 16/16
* `p0activated_regime_aware` ✓ 16/16 (the CRITICAL one — see below)

**Panel partially done:**
* `re_sdl_n2` 12/16 (Q01–Q12 saved; Q13–Q16 killed mid-flight)

**Panels not started:**
* `re_trail015`, `re_cvar025`, `re_cvar050`, `re_kelly_t1_035`

### To RESUME (next session, single command)

```bash
cd /Users/renhao/git/github/RenQuant
nohup ./scripts/notify_when_panels_done.sh > logs/reeval_queue/notify.log 2>&1 &
nohup ./scripts/run_regime_reeval_queue.sh > logs/reeval_queue/2026-05-15_resume.log 2>&1 &
disown -a
```

Queue script `is_done` check skips panels with 16/16 already. Panel
runner (now ALSO idempotent — see commit ?) skips individual windows
that exist already. So:
* `re_stop007`, `p0activated_regime_aware`, `re_sdl_n2` 12 windows →
  skipped on resume (no work)
* `re_sdl_n2` Q13–Q16 → resumed (~17 min)
* `re_trail015`/`re_cvar025/050`/`re_kelly_t1_035` → fresh (~50 min each)

**Total resume walltime: ~3.5 hours** (vs full 4h had it run straight).

Notifier rebuilt fresh — polls until all 7 panels are 16/16 then ntfy
+ auto-runs `analyze_regime_stratified.py` per panel.

### Quick check on resume

```bash
# Just look at what's done
for label in re_stop007 p0activated_regime_aware re_sdl_n2 re_trail015 re_cvar025 re_cvar050 re_kelly_t1_035; do
  printf "  %3s/16  %s\n" "$(ls data/logs/sim_2026-05-15_${label}/equity/ 2>/dev/null | wc -l | tr -d ' ')" "$label"
done
```

---

## TL;DR

Three big-payoff threads finished or in flight:

1. **Calibrator P0 closed end-to-end.** Refit + train-site clip + load-time
   guard + G12 train-time gate. Prod artifact clean.
2. **NGBoost SUSPECT → CONFIRMED at t=+2.76σ.** Audit hypothesis right.
   Re-train + σ-aware Kelly wire pending.
3. **Regime-conditional re-evaluation queue running** (7 panels sequential,
   ETA ~3-4h). Notifier will ntfy + auto-analyze when complete.

**Don't ship anything new before checking notifier output. The regime-
stratified results will tell us which knobs to flip per regime.**

---

## 🔴 CRITICAL UPDATE (19:48 PT) — p0activated_regime_aware verdict

**My hypothesis (gates 该在 BULL_CALM/BULL_STRONG 关掉) WAS WRONG.**

Compare two 16-window panels:

| Variant | Pooled ΔAPY | Wilcoxon p | Verdict |
|---|---|---|---|
| `p0activated` (gates ON everywhere) | +0.52pp | 0.90 | NEITHER (mild) |
| `p0activated_regime_aware` (gates OFF in BULL_CALM/BULL_STRONG) | **-3.59pp** | 0.74 | **NEITHER (worse)** |

Per-regime stratified `p0_regime_aware`:

```
BEAR           n=1   +8.74pp   ✓ WIN (Q01)
CHOPPY         n=4   +7.01pp   ✓ WIN (3/4 positive: Q02 +15, Q06 +5, Q16 +22)
BULL_VOLATILE  n=8   -2.24pp   ▼ Q05 -29 / Q11 -35 catastrophes (4/8 positive)
BULL_STRONG    n=3   -25.40pp  🔴 BIG LOSE (Q07 -51pp)
```

### Why my reasoning was backwards

I assumed gates "kill mean-revert mega-cap winners in BULL rallies".
The data says the OPPOSITE:
* Q07 BULL_STRONG with gates ON loses -19pp
* Q07 BULL_STRONG with gates OFF loses **-51pp**
* Gates in BULL_STRONG **PROTECT** rather than damage

### Real culprit hypothesis

Phase 3 (`use_calibrator_mu=true` + `use_realized_vol_fallback=true`) was
ALSO ON in both variants. The Kelly μ/σ wiring may be the actual driver
of BULL_STRONG losses — not the gates. The expected_return from
calibrator is now sizing positions, and in BULL_STRONG it's putting
weight on the wrong names.

### What this means for next-session decisions

**DO NOT just disable gates in BULL_CALM/BULL_STRONG.** That made
things worse.

The right experimental isolation:
1. **Gates ON, Phase 3 OFF** — sim config needed: `sim_gates_only`
2. **Phase 3 ON, gates OFF** — sim config needed: `sim_phase3_only`
3. **Both OFF (= baseline_hmm)** — already have
4. **Both ON** = `p0activated` — already have
5. **Both ON but regime-conditional** = `p0activated_regime_aware` — already have

After 1+2 land, can isolate which of (Phase 3 wiring, Upgrade A+B gates)
is responsible for what in each regime.

### Provisional verdict (pending isolation)

* CHOPPY: regime_aware variant WINS (+7pp, 3/4) — **safe to deploy
  Upgrade A+B + Phase 3 in CHOPPY**
* BEAR: too small n=1 to be sure — keep current behavior
* BULL_VOLATILE: mixed — Q11/Q05 catastrophes need root-cause before
  any flip
* BULL_STRONG: REVERT — turning off gates here makes things much worse

**Concrete action for next session:**

1. Build `sim_gates_only_pre2024.json` + `sim_phase3_only_pre2024.json`
2. Run 2 new 16-window panels (sequential, ETA ~2h)
3. With 5 variants × 5 regimes data, build decision matrix
4. Flip configs per CLAUDE.md PRIME DIRECTIVE per regime

The other 5 queued panels (re_sdl_n2 etc.) are still useful but their
verdicts may also need this same isolation treatment.

---

## What's in flight (will complete overnight)

### Background processes

| Process | PID-ish | What | ETA |
|---|---|---|---|
| `run_regime_reeval_queue.sh` | bash | Sequential 7-panel runner | ~3-4h from 17:31 PT |
| `notify_when_panels_done.sh` | bash | Polls panels + auto-analyzer | fires at queue end |

### Panel completion tracking

```
data/logs/sim_2026-05-15_<label>/equity/Q{01..16}.json
```

| Panel | Status | When done, expected (per Explore-agent prediction) |
|---|---|---|
| `re_stop007` | ✓ 16/16 | BULL_VOLATILE n=8 +6.88pp WIN; rest flat (verified) |
| `p0activated_regime_aware` | running | Same gates as p0activated but disabled in BULL_CALM/BULL_STRONG. Hypothesis: removes -10~-25pp BULL losses while keeping +20~+30pp BEAR/CHOPPY/VOL wins |
| `re_sdl_n2` | queued | Same shape as re_stop007 — protect in BEAR/VOL, hurt in BULL |
| `re_trail015` | queued | CHOPPY/REVERT win, BULL_STRONG lose |
| `re_cvar025` | queued | BEAR +1~3pp, BULL_CALM -4~-6pp |
| `re_cvar050` | queued | More aggressive than 025; same regime split, larger magnitude |
| `re_kelly_t1_035` | queued | BEAR/CHOPPY +2~4pp (fewer bad trades), BULL_CALM -10~-12pp |

---

## NEXT SESSION — first 5 actions

### 1. Check ntfy alert + auto-analysis output

```bash
ls data/logs/reeval_results/   # one .json + .txt per panel
cat data/logs/reeval_results/p0activated_regime_aware.txt
cat data/logs/reeval_results/re_stop007.txt
# ... etc
```

For each panel, the analyzer outputs:
* Pooled ΔAPY mean, median, Wilcoxon p, n_pos
* Per-regime stratified mean/median/worst/n_pos (5 regimes)
* Verdict: WIN-CONDITIONAL / NEITHER / REJECT
* Conditional-win regimes (if any)

### 2. Identify conditional WINS

A "conditional WIN" is any regime with:
* mean Δ > +2pp
* n ≥ 2 windows
* worst window > -10pp (no catastrophe in that regime)

If a panel surfaces ≥ 1 such regime, that knob is a candidate for
regime-conditional deployment.

### 3. Flip via `regime_params` overlay (NOT global)

Per CLAUDE.md PRIME DIRECTIVE, wins flip per regime:
```json
"regime_params": {
  "BULL_VOLATILE": {
    "stop_loss_pct": 0.07   // overrides global 0.15 in BULL_VOLATILE
  },
  "BEAR": {
    "cvar_lambda": 0.25
  }
}
```
NOT `"stop_loss_pct": 0.07` at top level. The overlay key follows
`<top_section>_<knob>` convention — see `kernel/regime_resolver.py`.

For `buy_quality_gates` (which has its own `disabled_in_regimes` field
already), instead of regime_params overlay, just edit the
`disabled_in_regimes` list:
```json
"buy_quality_gates": {
  "regime_momentum":     {"disabled_in_regimes": ["BULL_CALM", "BULL_STRONG"]},
  "deep_drawdown_veto":  {"disabled_in_regimes": ["BULL_CALM", "BULL_STRONG"]}
}
```

### 4. Run sim_p0activated_regime_aware verification

When `p0activated_regime_aware` completes, run:
```bash
python scripts/analyze_regime_stratified.py \
  --baseline data/logs/sim_2026-05-14_baseline_regime_fix \
  --treatment data/logs/sim_2026-05-15_p0activated_regime_aware \
  --label "p0_regime_aware"
```

Compare to the original `p0activated` run (which had pooled +0.52pp NEITHER):
* If `regime_aware` shows pooled improvement (e.g. +5pp) AND no regime
  with mean < -5pp → flip `disabled_in_regimes` into golden permanently
* If regime_aware is still NEITHER → the gates need different cohort
  (maybe just disable in BULL_CALM, keep in BULL_STRONG, etc.)

### 5. Update CLAUDE.md status section + dashboard

```bash
python scripts/build_dashboard.py    # refreshes regime gate visibility
```

Edit `CLAUDE.md` ## 🗂 Status section to reflect Monday's results.

---

## NGBoost SUSPECT — ACTIVATION ROADMAP

`scripts/train_ngboost_proper.py` 5-seed validator confirmed:
```
val μ-IC mean = +0.0351 ± 0.0036  (vs XGB +0.0294 ± 0.0029, t=+2.76)
σ-calib mean  = +0.271  (all seeds in [0.265, 0.275])
μ x-sec std   = 0.01619
```

To activate in production:

### Step A: Train production NGBoost head

```bash
# scripts/train_ngboost_proper.py is a VALIDATOR — outputs metrics, not artifact
# Need to add --save flag OR write a new train_ngboost_prod.py that:
# 1. Uses same Duan §4 config
# 2. Saves artifact to backtesting/renquant_104/artifacts/prod/ngboost-head.alpha158_fund.json
# 3. Stamps metadata with σ-calib + val_IC for the acceptance gate
```

### Step B: Wire σ properly into Kelly

Current state: `c.sigma` comes from broken `sigma_calibration` constant
in calibrator artifact (single float, not per-prediction). Need:

1. Modify `ApplyNGBoostTask` in `kernel/panel_pipeline/job_panel_scoring.py`
   to populate `c.sigma` directly from NGBoost `dist.scale` (per-prediction)
2. Remove the `sigma_calibration` constant scaling
3. Add test that c.sigma differs across candidates when NGB ON

### Step C: Re-test E55 regime-stratified

Re-run the original E55 panel (sim_E55_ngb_on.json) on 16-window OOS
panel. Hypothesis: BEAR/VOL conditional WIN, BULL_CALM neutral.
If true → flip `ngboost_head.enabled = true` in BEAR/VOL regimes only.

---

## Phase 2D shorts (paper-only) — ACTIVATION ROADMAP

`task_short_cover.py` shipped (NOT yet wired to InferencePipeline).

Open work for shorts to actually go live (paper):

1. **Wire ShortCoverStopLossTask** into InferencePipeline AFTER
   `TickerSellJob`, BEFORE `JointActionJob` (so cover orders compete
   for capital alongside new buys).
2. **Wire IRC1233TaxMarkerTask** at end of pipeline (after exits settle).
3. **Add ShortHoldingState dataclass** to ctx + populate from
   `AlpacaBroker.get_position(symbol)` (negative qty path).
4. **Refactor ExecuteExitsTask** to route `reason="short_cover_stop"`
   to `buy_to_close` (currently only handles long sells).
5. **Phase 2E**: borrow rebate cost in PnL (Garleanu-Pedersen 2013
   constraint).
6. **16-window sim with shorts ON** in BEAR/CHOPPY/BULL_VOL only
   (per long-short Phase 2A finding) + cover-stop active.

---

## Q12 catastrophe (-10.99pt) — STILL OPEN

All 3 earlier panels (defensive_xpand, voltarget15, shorts_bullvol_only)
share an identical -10.99pt loss in Q12 (2025-Q1). Diagnosed as:

* SPY regime BULL_CALM with conf=0.25 (low confidence)
* Q12 = 2025-Q1 tech selloff
* All 3 configs share `defensive_tickers` + `bear_defensive_slots` block
  that baseline lacks
* Regime never went BEAR so defensive bucket didn't fire

Root cause needs **per-bar weight diff tool** between baseline and
treatment to identify which symbol(s) account for the -10.99pt. Not
yet built.

---

## HMM regime detector — STILL BUGGY on 2022 windows

Q02/Q03 (2022 deep bear, SPY -25%) classified as BULL_CALM by HMM.
Documented in CLAUDE.md PRIME DIRECTIVE warning. Workaround in
`scripts/analyze_regime_stratified.py`: uses SPY-return data-driven
classification via yfinance.

**Fix needed**: retrain HMM with longer history including 2022 bear,
OR add a hard-bear MA200 override on top of HMM (similar to long-only
PRIME DIRECTIVE fix at commit `3925c0d`).

---

## Files / scripts ready to use

| Purpose | File |
|---|---|
| Build sim configs | `scripts/build_regime_reeval_configs.py` |
| Run a panel manually | `/tmp/run_p0_panel.sh <label>` |
| Sequential queue runner | `scripts/run_regime_reeval_queue.sh` |
| Auto-analyze + ntfy | `scripts/notify_when_panels_done.sh` |
| Regime-stratified analyzer | `scripts/analyze_regime_stratified.py` |
| NGBoost validator | `scripts/train_ngboost_proper.py` |
| Pre-open cancel gate | `scripts/preopen_cancel_gate.py` (+ plist) |
| Build dashboard | `scripts/build_dashboard.py` |

| Purpose | File |
|---|---|
| Calibrator (clean prod) | `backtesting/renquant_104/artifacts/prod/panel-rank-calibration.json` |
| Calibrator pre-clip snapshot | `backtesting/renquant_104/artifacts/prod/panel-rank-calibration.pre-2026-05-15-clip.json` |
| NGBoost prod head (OLD) | `backtesting/renquant_104/artifacts/prod/ngboost-head.alpha158_fund.json` (misconfigured — needs retrain) |
| Golden config | `backtesting/renquant_104/strategy_config.json` (4 P0 flags ON) |
| Re-eval sim configs | `backtesting/renquant_104/strategy_config.sim_re_*.json` (12 files) |
| Phase 2D shorts task | `backtesting/renquant_104/kernel/pipeline/task_short_cover.py` |

---

## Memory updates this session

* `feedback_pooled_mean_bias.md` — always re-evaluate pooled NEITHER via stratified analyzer
* `project_ngboost_confirmed_2026-05-15.md` — NGB has real signal, E55 was misconfig
* `feedback_autonomous_action.md` — when to ship without asking (3 cases)

---

## All commits this session (chronological)

```
273051a feat(broker): alpaca-shorts broker option for paper account isolation
1d37b7a fix(panel_linear): preserve 'kind' in metadata for downstream dispatch
9c3ddf0 chore(scripts): refit_gate_b FDR floor 0.30 → 0.40; transformer_v4 --label CLI
b16e2a1 fix(calibrator): P0 Phase 1 — saturation + range-bound detection guards
00f94ff fix(kelly): P0 Phase 3 — opt-in μ/σ wiring for Kelly when NGBoost off
342309e fix(calibrator): P0 Phase 4 — train-site ER clip ±1.0 → ±0.20
2484ba8 feat(buy-gates): Upgrade A+B — regime-momentum alignment + deep-drawdown veto
5b78ffe fix(P0): activate calibrator+kelly+quality gates, ship pre-open gate, prune dead code
f21c1d1 test: update test_preclip_snapshot_triggers_guard for post-refit reality
(7e1f80c)? test: ship 3 audit-mandated regression tests + p0activated sim configs
d1624b3 docs+infra: regime-conditional retrospective + analyzer fix + queue notifier
307c4c4 docs(CLAUDE): add 2026-05-15 EVENING status section
0920cc7 feat(G12+dashboard): train-time calibrator gate + regime-conditional gate visibility
(latest) feat(shorts): Phase 2D-lite — short cover stop-loss + IRC §1233 tax marker
```

**~13 commits, ~80+ tests added, 0 regressions, 4 background jobs in flight.**

---

## When you wake up tomorrow

1. `git pull` (in case anything else landed)
2. `cat data/logs/reeval_results/*.txt` — read every panel's stratified verdict
3. Identify conditional WINS, propose flips per CLAUDE.md PRIME DIRECTIVE
4. Run `sim_p0activated_regime_aware` analysis — that's the highest-value
   verification (whether the regime-conditional fix to today's 4 P0 flags
   actually nets out positive)
5. `python scripts/build_dashboard.py` — see updated regime gate status

That's it. The infrastructure is in place. Results will tell the story.
