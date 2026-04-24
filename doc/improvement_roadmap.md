# RenQuant 104 — Planned Work

Single source of truth for what's next. Ordered by expected value.
When starting an item, flip status to **🟡**. When done, flip to **✅** and drop a
1-paragraph result + commit sha. Archive anything closed to the History
section at the bottom.

**Current golden state** — `strategy_config.golden.json` @ commit `c9ee50b`:
after-tax **+44.20% APY** / +125.57% total / **82% win** / 26d longest-no-trade-
streak on the 27-month 2024-01-01 → 2026-03-26 OOS sim. Panel-LTR is the
hourly-enhanced 47k × **31** feature panel; OOS mean IC (CPCV 15-split) = +0.0326.

Full golden-config details: `doc/golden_config_2026-04-23.md`.
Per-run training history: `doc/panel_training_runs.md`.
Panel-LTR methodology primer: `doc/panel_ltr_primer.md`.
Environment + lib versions: `doc/environment.md`.

---

## Active queue (next pickup → drop to the bottom)

Handle letter is arbitrary — pick the topmost unblocked item when starting the next session.

| # | Item | Priority | Est. | Expected value |
|---|---|---|---|---|
| ~~O~~ | ~~Defensive-ticker gate in non-BEAR regimes~~ | ✅ shipped (same session) | 1 h | Cause of 2026-04-20 XLU BUY in BULL_VOLATILE. `run_selection_loop` now rejects defensive candidates unless `bear_only=True`; `blocks["defensive_non_bear"]` counter shipped. 10 regression tests (`tests/test_defensive_gate.py`) including a replay of the 2026-04-20 incident. |
| Q | Rotation hold-days + rotation_advantage 2D sweep | 🟡 MED | 1 h sim | Current golden: `min_rotation_hold_days=30`, `rotation_advantage=0.0`. User raised 2026-04-23: 30 d is conservative; panel IC=0.033 means <0.01 score differences are noise. Sweep 3×3: {14, 21, 30} × {0.0, 0.02, 0.05}. Goal: find Pareto-optimal adaptability vs noise-churn balance without dropping +44.20% APY. |
| ~~R~~ | ~~Diagnose stuck `TransitionWindowTask` / CUSUM buy-block~~ | ✅ shipped — `dc7be6f` | 2 h | Root cause: CUSUM fires legitimately every bar (SPY 20-day window shifted -5.3% → +7.7%), and `RegimeState.countdown` wasn't persisted across live invocations → always re-tripped. Fix: `live_state.json["regime_state"]` now carries 6 fields. 6 regression tests in `test_regime_state_persistence.py`. |
| S | Mirror live_state.json → `data/runs.db::live_state_snapshots` | 🟡 MED | 2 h | User's 2026-04-23 question: "shouldn't this file be replaced by the DB?" Answer: keep JSON for fast bootstrap + human debug editability, but append every bar's snapshot to a new `live_state_snapshots` table so historical "what was position_hwm on 2026-04-20?" is queryable. No schema-shared gain but big audit gain. |
| T | entry_dates persist fallback for legacy positions | ✅ shipped — (this session) | 30 min | Pre-fix: `entry_dates.get(ticker, today)` returned a fresh today every run but never wrote back. Legacy positions (inherited from renquant_103) had hold_days=0 perpetually → min_hold_days / min_rotation_hold gates locked. Fix: on missing key, stamp today ONCE and persist. |
| — | Full live_state.json attribute audit | ✅ shipped — (this session) | 1 h | 9 attributes reviewed end-to-end. 3 real bugs + 1 structural issue (JSON/DB schema split → Plan S). New `tests/test_live_state_contract.py` (21 tests) codifies read-site + write-site + per-attribute lifecycle invariants. |
| P | Populate `candidate_scores.blocked_by` in DB | 🟠 HIGH | 2 h | `blocked_by` column exists but stays empty — impossible to answer "why was X not selected" or "why was Y bought" from audit table. Write `sector_guard` / `wash_sale` / `correlation_guard` / `tier_threshold` / `defensive_non_bear` into the column from the selection loop. |
| M⁺ | Training-run audit schema fix — `elapsed_sec` column | 🟡 MEDIUM | 1 h | `SaveArtifactTask` + `NGBoostSaveTask` silently swallow write failures: `table training_runs has no column named elapsed_sec`. Add column + `ALTER TABLE` migration in `kernel/persistence.py`. Currently 0 rows in the `training_runs` table → a week of retrain history is missing. |
| I | Accumulate 4 weeks of live sustainability data | 🟢 passive | 0h (wait) | `scripts/weekly_apy_check.py` fires Sun 12 PT. After 4 runs, we have 4 weekly 30-day APY snapshots to trend against the +44.20% sim baseline. First trigger will be any single weekly APY < 25% for 2 consecutive weeks OR drawdown > 20% for ≥5 days. |
| J | Hourly-feature pruning sweep | 🟢 MED | 1 day | 4 of 6 hourly features have `|IC| < 0.016` (morning_drift, overnight_gap, vol_ratio, afternoon_drift). Drop the weakest 2–3 via `panel_ltr.drop_cols`, retrain, A/B vs current hourly golden. Accept if OOS IC lifts ≥ +0.005 with no APY drop. |
| K | CHOPPY regime diagnosis | 🟢 MED | 1 day | F's regime-conditional fit showed CHOPPY IC = −0.116 (direction-wrong). That signal is itself interesting: in CHOPPY the panel score anti-predicts forward excess return on 646 rows. Investigate bar-level which tickers flip sign and whether a CHOPPY-specific buy-filter (e.g. `ScoreBuyTask` threshold+= 0.05 in CHOPPY) could convert the inversion into an edge. |
| L | Per-ticker hourly effectiveness breakdown | 🟡 LOW | 1 day | Compute per-ticker OOS IC lift from hourly features (on vs off) to see whether some tickers carry all the gain. If yes, consider ticker-scoped hourly enablement or concentrating the Alpaca intraday fetch on the top-N beneficiaries. |
| M | Training-run audit schema fix | 🟡 LOW | 1 h | `SaveArtifactTask` logs `table training_runs has no column named elapsed_sec` on every panel retrain. Add column + `ALTER TABLE` migration in `kernel/persistence.py`. Eliminates noise from scheduled runs. |
| N | Golden config doc consolidation | 🟡 LOW | 30 min | v1/v2/v3 are now inline in `doc/golden_config_2026-04-23.md`; fold older tables into a `## History` section so the top reads as "current golden only". |

All previous A–H items are archived below.

---

## Detailed specs

### I. Live sustainability accumulation  🟢 passive

**Mechanism already shipped** (commit `67e95af`): `scripts/daily_104.sh` appends audit JSONL at `logs/live_104/audit.jsonl`; `scripts/weekly_apy_check.py` computes rolling 30-day APY + drawdown streak and fires ntfy alerts.

**Schedule:** `~/Library/LaunchAgents/com.renquant.weekly-apy104.plist` runs Sun 12 PT.

**Decision rule:** if 30-day live APY drops below 25% for 2 consecutive Sundays **or** drawdown exceeds 20% for 5 consecutive days, open a postmortem — golden expected **+44.20% APY** after-tax, so sustained sub-25% is a structural regression worth diagnosing against the +44.20% sim.

**Action:** no code change. Each Sunday's alert (or clean run) surfaces via ntfy. Accumulate 4 weeks of data, then review trend.

### J. Hourly-feature pruning sweep  🟢 MED

**Motivation:** 4 of 6 hourly feature columns carry |IC| < 0.016 — below the typical noise threshold. Including them adds XGBoost feature interactions that could be overfit-prone. Dropping them may lift OOS IC without hurting APY.

**Plan:**
1. Add `panel_ltr.drop_cols` entries for the weakest 2–3: `morning_drift_z`, `overnight_gap_z`, `vol_ratio_z` (keep `intraday_realized_vol_z`, `vwap_premium_z`, `afternoon_drift_z` — the three with |IC| ≥ 0.015).
2. `python scripts/train_104.py --skip-baseline --skip-recalibrate --force` to retrain.
3. Run the A/B sim via `/tmp/test_hourly_ab.py` variant.

**Acceptance:** OOS IC lifts ≥ +0.005 (0.033 → ≥ 0.038) with live-sim APY ≥ 44.20% (no worse than current golden). If yes, promote as golden v4.

### K. CHOPPY regime diagnosis  🟢 MED

**Motivation:** Plan F's per-regime calibrator fit showed CHOPPY `pool_IC = −0.116` on 646 rows — direction-wrong. The pooled calibrator still works there (per live A/B), but that `−0.116` is a standing anomaly: our panel score's sign is *flipped* in CHOPPY.

**Hypotheses to test:**
1. Score inversion is universal in CHOPPY → add a CHOPPY-specific `ScoreBuyTask` tier offset (e.g., `tiered_thresholds` gets +0.05 in CHOPPY so only the strongest panel-score candidates buy).
2. Score inversion is ticker-dependent (e.g., `SPOT`, `COIN` flip; defensives don't) → per-ticker CHOPPY behavior table.
3. Score inversion is feature-dependent — `vwap_premium` or `morning_drift` carry the sign flip.

**Acceptance:** one of the three hypotheses confirmed + a 1–2 APY pt A/B win from the derived fix. If none is confirmed, document `−0.116` as irreducible noise and move on.

### L. Per-ticker hourly effectiveness breakdown  🟡 LOW

Run CPCV on the hourly-enhanced panel with each ticker masked out (leave-one-ticker-out). Compare each resulting IC to the full-panel IC. Tickers where removing them raises IC are actively hurting the panel → candidates for ticker-scoped hourly disable or watchlist review. Tickers where removing them drops IC are load-bearing → guard against dropping from watchlist.

### M. Training-run audit schema fix  🟡 LOW

`kernel/persistence.py::record_training_run` tries to write `elapsed_sec` but the SQLite schema lacks that column. Log noise only — no data loss. Add the column + `ALTER TABLE` guarded migration. Affects `scripts/daily_104.sh` retrain output.

### N. Golden config doc consolidation  🟡 LOW

`doc/golden_config_2026-04-23.md` grew through v1 → v2 → v3 inline. Refactor so the top reads "current golden = v3 at +44.20%" and older baselines move to a bottom `## History` section. Keep the v1/v2 detail retrievable for audit, but frontload the latest numbers.

---

## Watch items (not planned work, but monitor)

- **Training audit `elapsed_sec` column drift.** See Item M. Low-impact noise.
- **Panel IC variance across daily retrains.** IC swings 0.03–0.04 day-to-day under identical hyperparameters. Largely from per-ticker tournament model drift that changes the feature frame distribution. Acceptable but worth tracking if day-over-day Δ exceeds ±0.01 — investigate feature drift.
- **`NoCandidateAlert` streaks ≥ 20 days in BULL_CALM.** F's run B showed a 52-day BEAR streak, but BULL_CALM should rarely hit 15d. Current monitoring alerts at 15d. If real BULL_CALM 20d streaks start appearing, ScoreBuyTask per-ticker thresholds may be too conservative.
- **Transformer revisit gate — panel > 200k rows.** Currently 47k. Would need ~4× more dates (10 yr hourly) OR watchlist expansion to ~100 tickers. Neither is near-term.

---

## Working rhythm

1. Pick topmost Active item.
2. Flip to 🟡, announce in chat.
3. Ship smallest reversible step + commit before next step.
4. Test vs golden (sim + sustainability check for any APY-affecting change).
5. If item lifts APY ≥ 2 pts: promote golden (`strategy_config.golden.json` + `doc/golden_config_*.md`) in same commit.
6. Flip to ✅ with 1-paragraph result + commit sha.
7. Move item to History.

**Stop rule:** if an item's 2× cost budget blows and no path to completion, mark as blocked with diagnosis, move to Deferred.

---

## Completed (archive)

Condensed summaries. Full spec for each is in the commit body + per-doc trails.

### Session of 2026-04-23 — G promoted, F+H shelved

| # | Item | Commit(s) | Result |
|---|---|---|---|
| A | Clear stale live_state `high_water_mark` | `ab1006d` | `resolve_hwm()` snaps HWM to equity when stored > 1.5× equity. +10 tests. |
| B | LightGBM backend A/B | `8d6b08a`, `67e95af` | Shelved: LGBM −12.7 APY pts. Per-row-weight bug fixed on the way in. Infra retained. |
| C | σ-penalty λ sweep | `0c80443` | Shelved: λ=0.25 +2 APY pts only; λ≥0.5 starves the ranker. Keep `score_mode=additive`. |
| D | Sustainability watch (30-day live APY tracker) | `67e95af` | JSONL audit + Sun-12-PT plist + 4 smoke tests. Infrastructure for Item I. |
| E | Re-fit calibrator on μ−λσ distribution | — | Cancelled: conditional on C winning. |
| **F** | **Regime-conditional calibration** | `26c40ae`, `7f68a40` | **Shelved: −3.78 APY pts live.** In-sample per-regime IC looked great (BULL_CALM 3.5×, BEAR 17× pooled) but didn't survive OOS. Infra kept behind off-by-default flag (10 regression tests). Takeaway: in-sample IC is necessary-not-sufficient — always A/B live. |
| **G** | **Hourly-bar panel features** | `8c65537`, `f03d1eb`, `0c80443`, `3b1d2e2`, `e65b081` | **✅ PROMOTED. +4.18 APY pts (40.02 → 44.20%), win 79 → 82%.** Panel 25 → 31 features. 153k hourly rows cached × 44 syms × 2 yr. `intraday_realized_vol_z` top-5 by \|IC\|. New golden. |
| H | Transformer rerun on hourly panel | `c9ee50b` | Shelved again: ratio **0.20×** (was 0.49× on daily-only). More features widened the gap. Panel needs > 200k rows for revisit. |
| — | Environment lockfile + versioning doc | `c9ee50b` | `requirements.lock.txt` (310 packages) + `doc/environment.md`. Reproducible env: Python 3.10.20, xgboost 3.2, ngboost 0.5.10, torch 2.11, Docker 29.3, LEAN 1.0.225. |
| — | Stale test reconciliation | `d857195` | `test_mu_minus_lambda_sigma_defers_to_ngboost` rewritten to match post-reorder semantics (calibration always runs). |

### Prior sessions — condensed

| # | Item | Commit(s) | Result |
|---|---|---|---|
| — | Run 3 — lookahead=10d + regularization | `5fdba09` prep → T4 | OOS IC 0.025 → 0.040; all 5 folds positive. |
| — | Global calibrator on panel | (pre-session baseline) | Pool IC 0.071 on 89k rows. |
| — | SQLite decision-trace database | (pre-session) | 5 tables + sim/live hooks at `data/runs.db`. |
| — | BaselineTournament winner by IC | (pre-session) | `oos_single_ticker_ic` metric added; default still `sharpe`. |
| — | Alpaca intraday sell overlay | (pre-session) | IEX feed, 20 slots 07:00-12:30 PT Mon-Fri. |
| — | Cross-sectional transformer panel backend | `8f38f80` → `908019a` | Infra kept; shelved at 0.49× XGB on daily-only panel (now 0.20× on hourly — Item H). |
| — | Universe floor regression fix | `2df4e21` | `_eval_sharpe` prefers tournament `sharpe` over noisy `live_holdout_sharpe`. APY 2.4 → 10.1%. |
| — | ConfidenceVeto disabled | `33c0e9b` | GMM posterior capped at ~0.25; threshold 0.30 → 0 unblocks all offensive buys. |
| — | DrawdownCircuitTask resets on recovery | `e586018` | THE bug behind the 153-day no-trade streak. APY 10.9 → 33.1%. |
| — | Golden v1 snapshot (33.1%) | `d3ef68f` | First post-fix golden. |
| — | T4 — xgb_params revert (40.1%, v2 golden) | `ee4faab` | Reverted to pre-regression panel `xgb_params`; APY 33.1 → 40.1%. |
| — | recalibrate_scores.py race-condition shield | `bc81360` | Merge-on-write. |
| — | daily_104.sh panel info in ntfy | `d71ee5d` | `panel@DATE IC` added to retrain notification. |
| — | PanelScoringJob reorder | `339944b` | NGBoost runs before calibration. |

---

## Deleted / no-action

- **Revert `lookahead_days` 10 → 5** — prior evidence shows 5d regresses.
- **E — re-fit calibrator on μ−λσ** — conditional on C winning; C did not.
