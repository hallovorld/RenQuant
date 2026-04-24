# Next Session — Priority TODO (updated 2026-04-23 late PT)

Refreshed at end of second 2026-04-23 session. Kelly sizing + top-up
landed tonight; the list below excludes things we already did. Work
top-to-bottom.

**Current golden:** `c5a2ff7` → now Kelly-capable (`4787825`).
After-tax **+44.20% APY** (27-mo OOS) on daily-only; Kelly variant
still under validation (sweep running).

---

## 🔴 P0 — Blockers / trust

| # | Item | Impact | Est. |
|---|------|---|---|
| **AA** | **Decision-factor DB: forward-returns table + analyze script** | 🔥 HIGHEST — foundational. Without it, every threshold / size parameter is tuned blindly. Powers AA → AB-trim → Q. | **1 d** |
| **P** | Populate `candidate_scores.blocked_by` in DB | HIGH — right now audit can't answer "why XLU?". Part of AA's JOIN surface. | 2 h |
| **M⁺** | `training_runs.elapsed_sec` schema fix | MEDIUM — silent failure, 0 rows over a week of retrains. 1-line migration. | 1 h |

## 🟠 P1 — Strategic (Kelly completion + short-term wins)

| # | Item | Impact | Est. |
|---|------|---|---|
| **AB-trim** | Partial-sell held positions whose Kelly target < current weight | HIGH — completes the Kelly story. Without trim, over-concentrated positions stagnate. Requires partial-exit infrastructure (currently exits = full liquidation). | 4 h |
| **BC** | Rotation uses Kelly delta instead of raw `panel_score` delta | MEDIUM-HIGH — aligns 3 decision layers on same math. | 2 h |
| **Q** | `min_rotation_hold_days` × `rotation_advantage` 2D sweep | MEDIUM — user raised 2026-04-23, 30d may be too conservative. 3×3 grid. | 1 h |
| **Kelly-tier-tune** | Tune `tiered_thresholds` based on tonight's sweep result + AA data | MEDIUM — tier 1 = 0.10 < base_rate=0.273 is "admit below chance". Either drop tier logic and let Kelly min_edge be the gate, or re-anchor to base_rate. | 1-2 h |
| **J** | Hourly-feature pruning sweep (drop 3 weakest `|IC|<0.016` cols) | MEDIUM — OOS IC possibly lifts +0.005. Tight A/B. | 4 h |
| **S** | Mirror `live_state.json` → DB `live_state_snapshots` | MEDIUM — audit improvement; history is currently JSON-frozen in time. | 2 h |

## 🟡 P2 — Analysis / diagnostic

| # | Item | Impact | Est. |
|---|------|---|---|
| **K** | CHOPPY regime diagnosis (IC=−0.116 inversion) | MEDIUM — understand why panel score is directional-wrong in CHOPPY. Could unlock CHOPPY-specific edge. | 1 d |
| **L** | Per-ticker hourly-feature effectiveness (leave-one-out) | LOW-MED — which tickers carry the +4.18 APY? Candidates for watchlist review. | 1 d |
| **N** | Golden-config doc consolidation (v1/v2/v3 → History) | LOW — housekeeping. | 30 min |

## 🟢 P3 — Passive

| # | Item | Impact | Est. |
|---|------|---|---|
| **I** | Accumulate 4 weeks of live sustainability data | LOW (passive) | 0 h (wait) |

---

## Total recommended pickup order

Sequenced so each item unlocks the next:

1. **M⁺ (1 h)** — fastest win, unblocks training audit.
2. **P (2 h)** — unblocks every "why" question downstream.
3. **AA (1 d)** — produces evidence for Kelly-tier-tune, AB-trim, BC.
4. **Kelly-tier-tune (1-2 h)** — once AA data is in, tune tier thresholds empirically.
5. **AB-trim (4 h)** — completes Kelly story.
6. **BC (2 h)** — rotation uses Kelly delta.
7. **Q (1 h)** — quick rotation-hold-days sweep.
8. **J (4 h)** — hourly feature pruning.
9. **S (2 h)** — live_state DB mirror.
10. **K (1 d)** — CHOPPY diagnosis.
11. **L (1 d)** — per-ticker hourly effectiveness.
12. **N (30 min)** — doc cleanup.
13. **I** — passive.

**Total next-session budget estimate: ~5-6 full days of work** if every item shipped end-to-end. Realistic one-session goal: items **1-5** (P0 + Kelly completion) = **~2 days effective**.

---

## ✅ Shipped this session (2026-04-23 — 15 commits)

| Item | Result | Commit |
|---|---|---|
| **O** | Defensive-ticker gate in non-BEAR (XLU bug) | `52bf718` |
| **R** | `regime_state.countdown` persisted across live runs | `dc7be6f` |
| **T** | `entry_dates` fallback persisted (legacy positions unlocked) | `c5a2ff7` |
| **V** | Held tickers exempt from universe_floor (AMZN unblocked) | `369973b` |
| **B²** | CUSUM cooldown only on regime SWITCH (not every CUSUM fire) | `013200a` |
| **W+** | Network-safety layer: per-call + per-ticker + batch timeout | `67b8d64`, `632f3cd` |
| **Watchlist** | +5 semis (INTC, MPWR, TXN, NVTS, WDC) — SKH delisted, skipped | `73a9327` |
| **HWM resolver** | Stale-HWM auto-snap on live start (from earlier session) | `ab1006d` |
| **live_state contract** | 9-attribute audit + contract tests | (earlier session) |
| **Retrain ntfy** | Fires only Tue/Thu/Sun (true retrain days) | `d302e5a` |
| **Trade ntfy** | `live/runner.py::_notify_decision` fires every cycle (trade/decision/skip) | `a07f76b`, `d79b6c2` |
| **ntfy truthfulness** | Reads `orders_placed` (broker-confirmed), not `ctx.orders` (intent) | `3578908` |
| **Kelly sizing** | `kernel/kelly.py` (f*=μ/σ²) + `ApplyKellySizingTask` + `SizeAndEmitTask` wiring | `4787825` |
| **Kelly top-up** | `TopUpHeldTask` — add-to-existing when `kelly_target > current_pct` | `4787825` |
| **Kelly validation** | `scripts/kelly_param_validation.py` with `--quick` / `--full` modes | `4787825` |

**13 real bugs fixed + 2 major features (Kelly + tiered ntfy) + watchlist expansion.**

**Test count tonight:** +60 new (kelly_sizing 23, runner_trade_ntfy 15, universe_held_exemption 9, regime_state_persistence 6, live_state_contract 21, cusum_regime_switch 7, defensive_gate 10, net_safety 13). Full suite estimated ~1150 tests.

## 🧪 Running now (will finish ~22:55)

`scripts/kelly_param_validation.py --quick` (4 configs × ~8 min):

```
[1/4] GOLDEN             apy=+25.91% win=81% buys=144 streak=25d  ← baseline
[2/4] A+Kelly(default)   in progress  (fractional=0.25, max_conc=0.35)
[3/4] A+Kelly(half)      queued       (fractional=0.50, max_conc=0.35)
[4/4] A+Kelly(tight)     queued       (fractional=0.25, max_conc=0.20)
```

Note: sweep APYs lower than documented golden (+44.20%) because
`allow_fetch=False` disables fundamentals/earnings/insider. Relative
deltas are still meaningful. If any Kelly variant beats GOLDEN by
> +3 pts → promote + run `--full` 9-grid tomorrow.
