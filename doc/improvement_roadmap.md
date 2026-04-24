# RenQuant 104 — Planned Work

**Single source of truth for what's next.** Ordered by priority tier (P0 → P3), then by expected value within tier.

Working rhythm: pick topmost unblocked → flip 🟡 → ship smallest reversible step + commit → test vs golden → if APY ≥ +2 pts promote golden in same commit → flip ✅ → drop result + sha → move to History.

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

## 🔴 P0 — Blockers / correctness / audit trust

| # | Item | Impact | Est. |
|---|------|---|---:|
| **AA** | **Decision-factor DB** (`ticker_forward_returns` table + `analyze_decision_factors.py`) | 🔥 keystone — data-driven tuning of tiers / Kelly / rotation | 1 day |
| **P** | Populate `candidate_scores.blocked_by` (`sector_guard` / `wash_sale` / `correlation_guard` / `tier_threshold` / `defensive_non_bear`) | 🟠 audit black-box without it | 2 h |
| **M⁺** | `training_runs.elapsed_sec` schema fix — `SaveArtifactTask` + `NGBoostSaveTask` silently swallow writes → 0 rows/week of retrain history | 🟡 audit transparency | 1 h |

## 🟠 P1 — Kelly completion (close the loop from session 2026-04-23)

Kelly sizing is **LIVE in golden v4**. Remaining work to fully close the loop:

| # | Item | Impact | Est. | Prereqs |
|---|------|---|---:|---|
| **Partial-sell infra** | Adapter + broker `place_order` path for "sell N shares, not all" (today's exits = full liquidation) | enables AB-trim | 2 h | — |
| **AB-trim** | `TrimHeldTask` — partial sell when Kelly target < current weight | closes Kelly loop | 4–6 h | partial-sell infra |
| **BC** | `RotationJob` compares `kelly_target_pct` delta (not raw `panel_score` delta) — unifies 3 decision surfaces on same math | consistency | 2 h | — |
| **Kelly-tier-tune** | Re-examine tiers from AA data. Current 0.27/0.45/0.60 anchored to `base_rate=0.273` (theory). AA gives empirical validation. | correctness | 1–2 h | **AA done** |
| **Kelly-full-sweep** | `--full` 10-point grid (`fractional × max_concentration`), driven from notebook so output is comparable side-by-side | parameter defense | 1 h sim | — |
| **Kelly × conviction** | `SizeAndEmit` currently does `max_pct = kelly_target × conviction_mult × σ_mult`. Kelly already encodes μ (∝ conviction) and σ — multipliers may double-count. Decide: Kelly alone OR careful blend. | design cleanup | 1 h | — |
| **Multi-entry accumulation** | Allow Kelly target to be approached over **multiple sessions**, not one big buy. Per-entry cap stays 35%; cumulative cap 65%. Needs `SizeAndEmit` to size the *delta* (kelly_target − current_pct) bounded by per-entry cap, repeated over days. | concentration headroom | 2 h | — |

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
| **S** | Mirror `live_state.json` → DB `live_state_snapshots` (append each bar) | audit history queryable | 2 h |
| **CUSUM-cooldown-v2** | Design **C** (confidence-scaled sizing, no hard block): `max_position_pct × (1 − cooldown_progress)`, 0→1 over 3 calendar days | live-sim parity, ~2 APY pt | 3 h |

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
