# Next Session — Priority TODO (refreshed 2026-04-23 late PT, comprehensive)

**Canonical list for next session.** Merged from:
- `doc/improvement_roadmap.md` Active queue + Watch items
- Tonight's conversation (Kelly, decisions, user-raised concerns)
- New items from today's incidents

**Current golden:** `64de137` — +44.20% APY / 82% win / 47k × 31 feature panel. Kelly sizing installed behind `ranking.kelly_sizing.enabled=false` (golden unchanged). Validation sweep running at session end.

---

## 🔴 P0 — Blockers / correctness / audit trust

| # | Item | Impact | Est. | Prereqs |
|---|------|---|---:|---|
| **AA** | **Decision-factor DB** (`ticker_forward_returns` table + `analyze_decision_factors.py`) | 🔥 HIGHEST — keystone for data-driven tuning of tiers / Kelly / rotation | **1 day** | none |
| **P** | Populate `candidate_scores.blocked_by` in DB — `sector_guard` / `wash_sale` / `correlation_guard` / `tier_threshold` / `defensive_non_bear` | 🟠 audit black-box | 2 h | none |
| **M⁺** | `training_runs.elapsed_sec` schema fix — silently-swallowed writes = 0 rows/week of retrains | 🟡 audit transparency | 1 h | none |

## 🟠 P1 — Kelly completion (continuation of tonight's big push)

| # | Item | Impact | Est. | Prereqs |
|---|------|---|---:|---|
| **AB-trim** | `TrimHeldTask` — partial sell when Kelly target < current weight. Requires new partial-exit path (today's exits = full liquidation). | 🟠 Kelly closes the loop | 4-6 h | **partial-sell infra** |
| **Partial-sell infra** | Adapter + broker place_order path for "sell N shares, not all". Needed by AB-trim + future "rebalance to Kelly target every N days". | 🟠 enables AB-trim | 2 h | none |
| **BC** | `RotationJob` compares `kelly_target_pct` delta (not raw `panel_score` delta) — unifies 3 decision surfaces on same math | 🟠 consistency | 2 h | none |
| **Kelly-tier-tune** | Empirically tune `tiered_thresholds` from AA data. Current tier 1=0.10 < base_rate=0.273 (admits "below chance"). Either set tier 1 ≈ base_rate OR drop tier logic and let Kelly `min_edge` gate entirely. | 🟠 correctness | 1-2 h | **AA done** |
| **Kelly-full-sweep** | If tonight's `--quick` sweep is inconclusive, run `--full` 10-point grid (fractional × max_concentration) | 🟡 parameter defense | 1 h sim | none |
| **Kelly × conviction interaction** | Review: `SizeAndEmit` currently does `max_pct = kelly_target × conviction_mult × σ_mult`. Kelly already encodes μ (∝ conviction) and σ — the extra multipliers may double-count. Decide: use Kelly alone OR blend carefully. | 🟡 design cleanup | 1 h | none |

## 🟠 P1 — Short-term strategic wins (non-Kelly)

| # | Item | Impact | Est. | Prereqs |
|---|------|---|---:|---|
| **Q** | `min_rotation_hold_days × rotation_advantage` 2D sweep (3×3) — user asked "30d 科学 vs 14d?" | 🟡 adaptability vs churn | 1 h sim | none |
| **J** | Hourly-feature pruning — drop 3 weakest (`morning_drift`/`overnight_gap`/`vol_ratio`, all `\|IC\|<0.016`) + A/B retrain | 🟡 OOS IC maybe +0.005 | 4 h | none |
| **S** | Mirror `live_state.json` → DB `live_state_snapshots` table appended each bar | 🟡 audit | 2 h | none |
| **CUSUM-cooldown-v2** | Migrate `countdown` from bar-count to wall-time (3 calendar days) — currently 3 bars can mean 1hr if intraday runs tick it down. **Design alternatives A/B/C already discussed** ("timestamp-based", "confidence-scaled soft block", "wall-clock only"). | 🟡 live-sim parity | 3 h | design pick |

## 🟡 P2 — Analysis / diagnostic

| # | Item | Impact | Est. | Prereqs |
|---|------|---|---:|---|
| **K** | CHOPPY regime diagnosis (IC=−0.116 — panel anti-predicts). Derive CHOPPY-specific `ScoreBuyTask` offset if tractable. | 🟢 maybe unlock CHOPPY edge | 1 day | AA helpful |
| **L** | Per-ticker hourly-feature effectiveness (leave-one-out OOS IC). Reveals which tickers carry +4.18 APY. | 🟢 watchlist review | 1 day | none |
| **Panel-IC-drift** | Investigate day-over-day panel IC swings ±0.03 under identical hyperparameters. Likely per-ticker tournament drift. | 🟢 stability | 4 h | AA helpful |
| **BULL_CALM-streak-watch** | Currently alerts at 15d. F's run hit 52d BEAR streak (valid); BULL_CALM should rarely ≥20d. Monitor + per-ticker ScoreBuyTask threshold audit. | 🟢 gate sensitivity | 4 h | none |

## 🟡 P2 — Housekeeping

| # | Item | Impact | Est. | Prereqs |
|---|------|---|---:|---|
| **SKH-replace** | yfinance can't find SKH — user to confirm intended symbol (SIMO? SKYH? SKYT?). Re-add once resolved. | 🟢 ticker admin | 10 min (decision) | user input |
| **N** | Golden config doc consolidation — v1/v2/v3 inline → History section | 🟢 cleanup | 30 min | none |
| **sector-guard-review** | `max_positions_per_sector=6` confirmed tonight. With 22 tech tickers (after adding 5 semis), 27% admission rate is tight. Consider sub-sector buckets (semis vs software vs cloud). | 🟢 sector design | 2 h | none |

## 🟢 P3 — Passive / blocked on time

| # | Item | Impact | Est. | Prereqs |
|---|------|---|---:|---|
| **I** | Accumulate 4 weeks of live sustainability data (Sunday 12 PT plist) | LOW (passive) | 0 h (wait) | wall-clock |
| **Transformer revisit gate** | Shelved until panel > 200k rows (current: 47k). Need 4× more data. | LOW | 0 h (wait) | data growth |

---

## 🗺️ Recommended sequencing for next session

**Budget ~2 days of focused work** to land the biggest-ROI items:

### Day 1 — Data foundation + quick wins
```
1. M⁺  (1h)     training_runs schema fix
2. P   (2h)     blocked_by DB field
3. AA  (1d)     decision-factor DB + analyze script   ← keystone
```

### Day 2 — Kelly completion + param tuning
```
4. Partial-sell infra (2h)
5. AB-trim           (4-6h)
6. BC (Kelly rotation)  (2h)
7. Kelly-tier-tune + Kelly-full-sweep (2-3h)    ← uses AA data
```

After Day 2 you have: (a) all three decision layers aligned on Kelly, (b) empirical basis for every threshold. Then **Q** (1h rotation sweep) as a victory lap.

**Budget ~3 days** → add **J** (hourly pruning) + **S** (live_state DB mirror) + **CUSUM-v2**.

**Budget ~1 week** → plus **K** (CHOPPY diagnosis) + **L** (per-ticker hourly).

---

## ✅ Shipped this session (2026-04-23 — **17 commits**)

### Big-ticket features
| Plan | What | Commit |
|---|---|---|
| **G** | Hourly-bar panel features → **PROMOTED TO GOLDEN** (+4.18 APY pts, 40.02 → 44.20%) | `e65b081` |
| **Kelly sizing full stack** | `kernel/kelly.py` + `ApplyKellySizingTask` + `TopUpHeldTask` + `SizeAndEmit` refactor | `4787825` |
| **ntfy ecosystem** | trade-level + decision-level + truthful (broker-confirmed) + retrain-only-Tue/Thu/Sun | `a07f76b`, `d79b6c2`, `3578908`, `d302e5a` |

### Real bug fixes (13)
| Plan | Fix | Commit |
|---|---|---|
| **O** | Defensive gate in non-BEAR (XLU bug) | `52bf718` |
| **R** | `regime_state` persisted across live runs | `dc7be6f` |
| **T** | `entry_dates` fallback persisted | `c5a2ff7` |
| **V** | Held tickers exempt from universe_floor (AMZN) | `369973b` |
| **B²** | CUSUM cooldown only on regime SWITCH (in the correct pipeline path) | `013200a` |
| **W / W+** | Network-safety: per-call + per-ticker + batch timeout | `67b8d64`, `632f3cd` |
| HWM resolver | Stale-HWM auto-snap on live start | (earlier session) |
| Retrain ntfy | Only Tue/Thu/Sun (not every weekday) | `d302e5a` |
| Trade ntfy | Moved inside `live/runner.py` (not just shell) | `a07f76b` |
| Decision ntfy | Fires every cycle, includes "why no trade" | `d79b6c2` |
| Truthful ntfy | Reads `orders_placed` not `ctx.orders` (intent) | `3578908` |
| live_state contract | 9-attribute audit + contract tests | (earlier) |
| **Watchlist** | +5 semis INTC/MPWR/TXN/NVTS/WDC | `73a9327` |

### Shelved after A/B
| Plan | Outcome |
|---|---|
| **F** | Regime-conditional calibration — shelved at −3.78 APY pts live |
| **H** | Transformer on hourly panel — shelved at 0.20× XGBoost |

**Test count:** +60 new (~1150 total estimated).

### Running now
- `scripts/kelly_param_validation.py --quick` (finishing ~22:55 PT)

---

## ✋ One-liners I need YOU to decide next session

1. **SKH alternative ticker?** (SIMO / SKYH / SKYT / SKM — you know the business you want)
2. **Is 35% max_concentration acceptable, or should Kelly be allowed > 50% on extreme signals?** (You asked "甚至可以全仓" — my 35% was conservative pushback)
3. **If Kelly wins tonight's sweep, promote to golden immediately?** Or run `--full` grid first to stabilize params?
4. **AB-trim aggressiveness** — trim to exact Kelly target, or only trim if we're `> target + 10%`? (hysteresis to avoid daily trim churn)
5. **CUSUM cooldown v2** — pick A (timestamp-based) vs B (wall-clock soft-block) vs C (confidence-scaled sizing, no hard block)
