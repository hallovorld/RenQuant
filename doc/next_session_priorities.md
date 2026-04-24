# Next Session — Priority TODO (carry forward from 2026-04-23)

End of session 2026-04-23. Five bugs shipped today (O / R / T / HWM / stale test); three fresh items discovered. Work top-to-bottom.

## 🔴 P0 — correctness / trust

| # | Item | Why it's P0 | Est. |
|---|---|---|---|
| P | Populate `candidate_scores.blocked_by` in DB | Currently all rows have empty `blocked_by`. Impossible to answer "why was XLU not selected?" or "why was X bought?" from audit. Without this, the XLU bug would have been undetectable from DB alone. Write `sector_guard` / `wash_sale` / `correlation_guard` / `tier_threshold` / `defensive_non_bear` from the Selection loop into the column. | 2 h |
| M⁺ | Training-run audit schema drift fix | `record_training_run` silently swallows `table training_runs has no column named elapsed_sec` → **zero rows** in that table across a week of retrains. Entire training history is ungaudited. Add `elapsed_sec` column + `ALTER TABLE` migration. | 1 h |

## 🟠 P1 — performance / structural

| # | Item | Why it's P1 | Est. |
|---|---|---|---|
| Q | Rotation hold-days × rotation_advantage 2D sweep | User asked 2026-04-23: "30 天科学吗？14 更好？" Current golden `min_rotation_hold_days=30, rotation_advantage=0.0` validated at +44.20% APY but not sweep-optimised. 3×3 = 9 configs × ~7 min each = 1 h. Pareto-optimal finder for adaptability vs noise churn. | 1 h |
| J | Hourly-feature pruning sweep | 4/6 hourly features have `|IC| < 0.016` (morning_drift, overnight_gap, vol_ratio, afternoon_drift). Drop weakest 2-3 via `panel_ltr.drop_cols`. Target: OOS IC lifts ≥ +0.005 (0.033 → ≥ 0.038) without APY regression. | 1 day |
| S | live_state.json → DB `live_state_snapshots` mirror | User's 2026-04-23 Q: "should JSON be replaced by DB?" Keep JSON for bootstrap + human debug; add `live_state_snapshots (run_id, run_date, state_json, created_at)` table appended each bar. Then "what was position_hwm on 2026-04-20?" becomes queryable. | 2 h |

## 🟡 P2 — analysis / documentation

| # | Item | Why it's P2 | Est. |
|---|---|---|---|
| K | CHOPPY regime diagnosis | Plan F leftover: per-regime calibrator showed CHOPPY `pool_IC = -0.116` on 646 rows — panel score *anti-predicts* in CHOPPY. Bar-level diagnostic: universal inversion vs ticker-scoped vs feature-scoped. If tractable, derive a CHOPPY-specific ScoreBuyTask tier offset. | 1 day |
| L | Per-ticker hourly-feature effectiveness | Leave-one-ticker-out on the hourly-enhanced panel → which tickers carry the +4.18 APY gain? Candidates for watchlist re-evaluation. | 1 day |
| N | Golden config doc consolidation | `doc/golden_config_2026-04-23.md` has v1/v2/v3 inline. Fold v1/v2 into a `## History` section so top of doc reads as current-golden-only (+44.20%). | 30 min |

## 🟢 P3 — passive / hygiene

| # | Item | Why it's P3 | Est. |
|---|---|---|---|
| I | Accumulate 4 weeks of live sustainability data | `scripts/weekly_apy_check.py` fires Sun 12 PT. After 4 Sundays we have 4 weekly 30-day APY snapshots to trend. Auto-alert below 25% for 2 consecutive weeks. Just wait. | 0 h |

## ✅ Shipped this session (2026-04-23)

Condensed — full commit trail in `doc/improvement_roadmap.md` History.

| Plan | What | Commit |
|---|---|---|
| **G** | **Hourly-bar panel features → PROMOTED TO GOLDEN** (+4.18 APY pts, 40.02 → 44.20%) | `e65b081` |
| **O** | Defensive-ticker gate in non-BEAR — fixed XLU 2026-04-20 bug | `52bf718` |
| **R** | CUSUM countdown persistence across live runs — fixed 3-day zero-trade streak | `dc7be6f` |
| **T** | `entry_dates` fallback persistence — legacy positions no longer show hold_days=0 | `c5a2ff7` |
| — | `resolve_hwm()` + `live_state_contract` audit — 9-attribute end-to-end review | `ab1006d`, `c5a2ff7` |
| **F** | Regime-conditional calibration — shelved after live A/B lost −3.78 APY | `7f68a40` |
| **H** | Transformer on hourly panel — shelved at ratio 0.20× XGB | `c9ee50b` |
| — | Environment lockfile + versioning doc | `c9ee50b` |

## Current golden state

```
strategy_config.golden.json  @ commit c5a2ff7
+44.20% APY after-tax
82% win rate
26 d longest no-trade streak
47k × 31 feature panel (hourly-enhanced)
OOS mean IC (CPCV 15-split) = +0.0326
```

---

## Quick-start for next session

```bash
cd /Users/renhao/git/github/RenQuant
conda activate renquant
# Read this file first:
cat doc/next_session_priorities.md
# Run tests to confirm golden is green:
python -m pytest tests/ -q  # expect ~1100 tests green
# Pick a P0 item (P or M⁺) → ship it → commit → push
```

User's ongoing ask: when finishing a piece, update this file + `doc/improvement_roadmap.md` + commit same-day.
