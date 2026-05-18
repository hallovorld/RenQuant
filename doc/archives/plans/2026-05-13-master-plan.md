# 2026-05-13 Master plan — efficient + scientifically rigorous path forward

## Goal hierarchy

1. **Find a Tier-3 promotable change** to baseline (−7.5pt/yr alpha gap to SPY).
2. **Re-evaluate 7 prior theory candidates** on top of the fixed baseline.
3. **Unblock Long-Short Phase 2** once baseline stable.
4. **Land structural improvements** (regime detector, σ-aware dead code, universe) regardless.

## ASCII timeline (§5.9)

```
[t=0 now]            [+4h]              [+12h]             [+36h]            [+2w]              [+4-6w]
──────────────────────────────────────────────────────────────────────────────────────────────────────
P1 4-panel screen → ANALYZE 4 HAC
  (in flight)          │
                       ↓ Tier-3? ──YES──→ P3 re-run 7 theory on new baseline ──→ P4 combine ──→ Phase 5 shorts
                       │
                       │ NO (likely)
                       ↓
                       P1.5 4-corner factorial (κ × EMA50 × min_dw) ─→ analyze
                                                              │
                                                              ↓ Tier-3?
                                                              │ ──YES──→ P2 BB-DOE 25-run optimum hunt ──→ P3/P4
                                                              │ ──NO ──→ pivot to P2.5 STRUCTURAL
                                                              ↓
                                                              P2.5 = regime fix + universe expansion + signal upgrade

PARALLEL tracks (no CPU dependency — code work during sim queues)
[T1] Regime detector audit + fix (Hurst/GMM)         ████████░░░░░░░░░░░░░░░░░░░░  (now → +24h)
[T2] σ-aware dead code triage (delete vs reactivate)         ░░████░░░░░░░░░░░░  (+4h → +12h)
[T3] wl183 / wl292 universe expansion code unblock         ░░░░░░████████░░░░░░  (+12h → +30h)
[T4] SEC fundamentals freshness audit                          ░░██░░░░░░░░  (+8h → +16h)
```

## Phase 1 — In flight (current, +4h to complete)

4 single-knob 16-window panels running. Each evaluated with paired-daily
HAC + stationary bootstrap + DSR/PBO via `eval_paired_returns.py`.

| Panel | Mechanism | Hypothesis | κ-trials | Predict |
|---|---|---|---:|---|
| κ=0.003 | mild friction penalty | barely binds | 100 | mostly no-op vs baseline |
| κ=0.05 | medium friction | between κ=0.001 and κ=0.1 | 100 | possible Tier 2 |
| min_dw_pct=0.05 | size-based filter (alt mech) | only trades ≥5% Δw | 100 | cleaner than κ |
| EMA50-off | gate bypass | recover bull-market lag | 100 | +bull / −bear |

Each gets 3 verdicts: REJECT / NEITHER / Tier 2 SCREEN / Tier 3 PROMOTE.

**Decision gate at +4h:**
- ≥1 Tier-3 winner → auto-promote per `feedback_auto_promote_to_prod.md`, branch to Phase 3
- All NEITHER → pivot to Phase 1.5

## Phase 1.5 — Two-way factorial (only if Phase 1 = NEITHER)

2-level fractional factorial 2³ (Plackett-Burman Resolution IV per
§5.14.1): screen interactions between the three knobs that showed any
direction in Phase 1.

Design matrix (4 panels):
```
                 cell  κ      EMA50    min_dw_pct
                 ────  ────   ──────   ──────────
                 A     0.05   on       0.05
                 B     0.05   off      0.02
                 C     0.001  off      0.05
                 D     0.001  on       0.02   (= baseline ish)
```

4 panels × 16 windows × 25min ÷ 5 sim-slots = **5h wallclock**.

Sanity-test triad (§5.2) for any winner:
- A/A test (same config × 3 seeds → σ noise floor)
- Shuffled-label sanity (panel_ltr.label_shuffle_seed → IC ≈ 0)
- Time-shift placebo (panel_ltr.label_shift_days → IC ≈ 0)

**Decision gate at +9h:**
- Tier-3 winner: branch to Phase 2 to find optimum
- NEITHER: branch to Phase 2.5 structural

## Phase 2 — Box-Behnken optimum hunt (only if Phase 1.5 winner)

Tier 2 winner identified ⟹ map the response surface around it. 3-knob
Box-Behnken (Box & Behnken 1960) at 3 levels per knob.

`pyDOE2.bbdesign(3, center=3)` → 13 design points + 3 center replicates
= 15 points. Each = 16-window panel. Total 15 × 16 = 240 sims = 60
sim-hr ÷ 5 slots = **12h wallclock**.

Quadratic response surface fit: `y = β₀ + Σβᵢxᵢ + Σβᵢⱼxᵢxⱼ + Σβᵢᵢxᵢ²`.
Optimum via `scipy.optimize.minimize` on fitted surface, within DCP-legal
bounds. Confirmation runs: 3 sims at predicted optimum + DSR + PBO.

## Phase 2.5 — Structural pivot (if Phase 1.5 = NEITHER)

Single-knob optimization is exhausted. Pursue:

| Track | Code work | Sim work |
|---|---|---|
| Regime detector fix (BULL_CALM lock) | T1 (~16h) | After fix, re-test GK_conditional |
| Universe expansion (wl183 ≥ 162) | T3 (~18h) | After unblock, 1 panel |
| NGBoost reactivation experiment | T2 → reactivate path | 1 panel A/A |
| Signal upgrade (LightGBM, PatchTST) | New training pipeline | ~1 week |

## Phase 3 — Re-run 7 theory candidates on fixed baseline (~24h)

Trigger: any Tier-3 fix lands and stays for ≥4h post-promotion.

Re-run each of vt15, GK094, GK15, GK_conditional, GK_v2_LOO, wl174,
wl174_retrained on the new baseline. Same 16-window paired-daily HAC
framework. Each = 16 sims = 4 sim-hr.

7 candidates × 4 hr ÷ 5 slots × Amdahl ≈ **6h wallclock** if all queued.

**Trick**: re-train walkforward retains may be needed for the universe
variants (wl174). Schedule those first (walkforward manifests can run
in parallel since outputs go to different dirs).

## Phase 4 — Composite golden (~24h)

Top-K Phase 3 winners that are orthogonal (different mechanisms — e.g.
friction-fix + signal-fix + sector-fix). Combine into one golden config,
run 16-window confirmation + DSR/PBO + sanity triad. Pin in
`strategy_config.golden.json` + `strategy_config.json`.

## Phase 5 — Long-Short Phase 2 (4-6 weeks)

Trigger: stable Phase 4 baseline for ≥1 week.

| Sub-phase | Days | Description |
|---|---:|---|
| 2B | 1-2 | Short candidate task (full-universe panel scoring) |
| 2C | 0.5 | Smoke sim — 1 window short-only decile spread |
| 2D | 2 | Wash-sale §1233 + ST tax on shorts |
| 2E | 3 | Alpaca borrow/locate + Reg-T margin checks |
| 2F | 2 | Paper test full Phase 2 flow on Alpaca paper |
| 2G | 14 | Live paper test (2 weeks observed) |
| 2H | 0.5 | Live flip with explicit user OK |

## Parallel tracks (no sim dependency)

### T1 — Regime detector fix
**Why**: detector stuck at BULL_CALM since 2026-04-25 (3 weeks).
Disables regime-conditional configs entirely. Likely affects GK_conditional
re-evaluation in Phase 3.

**Approach**:
1. Reproduce stuck behavior on last 3 weeks of SPY data offline.
2. Hurst window/threshold audit — is it MOMENTUM-biased?
3. GMM regime probs — is BEAR weight collapsed?
4. Fix design: re-tune Hurst k-bands or refit GMM on longer history.
5. Pin invariant in test (e.g. 2022-Q1 should be BEAR, 2024-Q1 CHOPPY).

### T2 — σ-aware code dead-path triage
**Why**: NGBoost OFF in prod → state.sigma always None → σ-aware
stop-loss code never executes. 124 tests pass via fixture mocks but
prod path is dead (§5.13.1).

**Approach**:
1. Decision: reactivate NGBoost or delete σ-aware code?
2. If reactivate: needs an A/A test post-feature-drift-guard first.
3. If delete: remove σ-aware stop-loss + 124 tests + clean up exits.py.

### T3 — Universe expansion code
**Why**: wl183 prior failure user-flagged "可能是bug原因，开始无视".
Want to retest with current scoring.

**Approach**:
1. Audit wl183 ticker list for delisted / data gaps.
2. Walkforward retrain pipeline for wl183 (script invocation).
3. Side config sim_wl183.json with proper artifact_path isolation.

### T4 — SEC fundamentals freshness audit
**Why**: production reads fundamentals daily. Need to verify last-update
timestamp on `sec_fundamentals_daily.parquet` is recent.

**Approach**: 1-hour check. If stale, kick off refresh cron manually.

## Scientific rigor checklist (CLAUDE.md §5)

| Check | Applied to | When |
|---|---|---|
| §5.2 sanity triad (A/A, shuffled, time-shift) | every claimed winner | before promote |
| §5.13.4 mean±std OR HAC | every quoted APY/Sharpe/IC | in every report |
| §5.13.4a 3-tier promotion | every panel verdict | already wired in eval_paired_returns.py |
| §5.14 DOE Box-Behnken | multi-knob optimization | Phase 2 only |
| §5.11 range-find before optimize | single-knob sweeps | already done for κ |
| §5.13.3 regression test for every fix | bug bounty | each commit |
| §5.13.4 K_trials counter for DSR | multi-comparison | already wired (--k-trials) |
| §5.10 saturate hardware | every long run | OMP_NUM_THREADS=2 per sim, 5 slots |

## Hardware budget (M2 Pro, 10 cores, 32GB)

- 1 sim = 2 cores × 25 min = 50 core-min
- 5 sim-slots in parallel × 2 cores each = 10 cores saturated
- Floor: 1 16-window panel = 16 sims × 50 ÷ 10 = **80 min wallclock**
- Phase 1: 4 panels = 320 min = 5.3h (but staggered start → 4h)
- Phase 1.5: 4 panels = 5h
- Phase 2 (BB): 15 panels = 20h
- Phase 3: 7 panels = 9.3h
- Phase 4: 1 panel + 5 sanity = 1.5h

## Decision rules — explicit

1. **Tier 3 promote** (per `feedback_auto_promote_to_prod.md`): DSR > 0.5 OR PBO < 0.5 OR n_days ≥ 30 with t_pool > 3.0. Auto-promote in flight.
2. **Tier 2 screen**: mean ΔAPY > 0 + consistency ≥ 4/N. Send to Phase 2 (DOE).
3. **NEITHER / REJECT**: document in `failed-experiments-log.md`. Pivot.
4. **Exclude from auto-promote**: broker switch, risk-loosening (stop-loss wider, drawdown wider), architecture changes (Phase 2 shorts, ExecutionPipeline rewire).

## Failure modes + recovery

- If Phase 1 sim batch crashes mid-run → skip window, mark NaN, eval remaining
- If a panel produces NaN equity → log and re-run that one window
- If RAM bloats > 28GB → throttle to 3 sim-slots × 2 cores = 6 cores
- If sim wallclock > 60min per window (oversubscription severe) → reduce parallel slots
- If prod cron (14:06 PT daily) needs to fire → pause sims briefly, resume

## Status & next ping

- Phase 1 in flight; analyze at +60min on next wake.
- All side configs for Phase 1.5 / Phase 2 will be pre-generated before Phase 1 finishes (cheap).
- T1 / T2 / T3 / T4 code work to interleave between sim batches.
