# Self-audit — bugs in this session's shipped work (2026-04-24 PT)

User prompted an audit. Below are bugs I found in my own shipped code
today, classified by severity, with fixes. Running this openly so the
second AI's audit doesn't have to re-discover them.

## HIGH-severity — silent no-ops

### Bug 1: `find_thesis_symmetric_pairs` shipped but NOT wired

**Commit introducing:** `709032d`

**Symptom:** user sets `rotation.scoring_mode="thesis_symmetric"` expecting V4 4-point logic. Actual behavior: the string "thesis_symmetric" didn't match any branch in `BuildPairsTask`, so execution silently fell through to ER mode. The V4 kernel function + 10 tests existed but were reachable only from unit tests.

**Fix:** `f3a0000` (this session, post-audit) — added `if rotation_mode == "thesis_symmetric":` branch that calls `find_thesis_symmetric_pairs` with DB lookup + own-momentum.

### Bug 2: `own_momentum` param wired into kernel but no caller populates it

**Commit introducing:** `9463e4c` (own-momentum gate)

**Symptom:** even if someone set `rotation.thesis_symmetric.own_momentum_enabled=true`, the `own_momentum` dict passed to the kernel was always None/empty, so the Moskowitz gate was a no-op regardless of flag state.

**Fix:** same commit as Bug 1 — the new `if rotation_mode == "thesis_symmetric":` branch builds `own_mom` dict from OHLCV close series (63d return) and passes to the kernel.

## HIGH-severity — false claim in roadmap

### Bug 3: baseline IC reported 0.0326 was wrong; actual is 0.0391

**Commit introducing:** `322f5aa` ("minute panel CPCV IC +0.0355 prelim")

**Symptom:** I wrote that minute-enabled panel IC of 0.0355 was "+0.003 lift over baseline 0.0326". Actual current golden `panel-ltr.json` OOS IC is **0.0391**. So minute-enabled panel is **0.0355 = -0.0036 WORSE**, not better.

**Root cause:** I pulled the 0.0326 number from an old doc snippet (`renquant_104_design.md` §5d Round 1-5 log) which was a prior panel's IC. The actual current golden is trained_date 2026-04-23 and has IC 0.0391.

**Correction:** added to roadmap; the 10-min A/B conclusion is **minute features HURT IC**, pending NGBoost + sim phases which might recover some. The "transformer retry unlock" argument still holds on row-count grounds (744k > 200k gate), but it's not backed by an IC lift on the current run.

## MEDIUM-severity — test fixture brittleness

### Bug 4: V2 test used held with entry_price != current_price

**Commit introducing:** `0000b91` (V2 scoring mode)

**Symptom:** `test_rotation_v2_scoring.py::test_default_er_mode_unchanged` initially failed because my `_held()` helper defaulted `current_price=110` (10% unrealized), causing tax_drag to eat the edge. Caught in pre-ship test run.

**Fix:** changed default `current_price=entry_price` in the test helper so tax_drag is zero unless tests explicitly set it.

## MEDIUM-severity — operational

### Bug 5: panel_exit V2 "OR mode" still fired 0 exits on current panel

**Commit introducing:** `b022ad6`

**Symptom:** even with `trigger_mode="or"` (fires when panel<0.20 OR μ≤0.0), the 27-mo OOS A/B showed 0 panel_exits. The thresholds are too tight for our holdings' score distribution. Gate shipped correctly but is practically non-firing under current panel.

**Not a code bug** — parameter-tuning issue. The AND→OR change is sound; thresholds need raising (suggest panel_sell_floor=0.30, mu_sell_ceiling=-0.01) to see any fires.

### Bug 6: rotation V1/V2/V3 also 0 rotations in A/B

**Commit introducing:** `9eb188b`, `0000b91`, `f674b3f` (all three rotation versions)

**Symptom:** all rotation variants produce 0 rotations in A/B. Earlier A/B (Route A in roadmap) had produced 3 rotations at threshold=0.005. Between sessions the panel was retrained; current panel's ER distribution is too tight for the existing threshold to fire.

**Not a code bug** — documented in roadmap as "candidate-supply bottleneck". The rotation machinery works; it's starved of candidates to rotate TO.

## LOW-severity — minor

### Bug 7: `rotation.bear_only` check order vs V3 regime gate order

**Status:** ctx.bear_only check runs BEFORE V3 enabled_regimes check. If `bear_only=True` AND `enabled_regimes=["BEAR"]`, V3 never reaches. Arguably correct behavior (bear_only is an explicit suppression) but worth documenting. Not fixing — bear_only takes precedence by design.

## Bug 8: minute features silently dropped from FactorZScoreTask

**Commits introducing:** `9fba853` (10-min infra)

**Symptom:** `TickerPanelFactorJob` correctly wrote `m_*` columns into
`raw_factor_frame`, but `FactorZScoreTask.raw_cols` (a hardcoded list
filtering which raw cols to z-score) DID NOT include `m_*`. Plus the
output assembly loop also has a hardcoded list — both needed updating.

**Impact:** the entire 10-minute panel infra (fetch + features + cache
+ context wiring) was a no-op. Panel feature_cols stayed 31 (no m_*).
The "minute panel" we trained was identical to hourly-only with same
random seeds. My initial -2.50 APY result was variance noise.

**Fix:** `a55df54` — added `m_*` to BOTH lists (raw_cols + output loop)
+ `tests/test_session_silent_bugs.py::TestMinuteFeaturesReachPanel`
guards against re-introduction.

**Real result post-fix:** minute panel = **+9.57 APY pts vs golden**
(38.39% vs 28.82% on 27-mo OOS) with minute_cols=10 in feature_cols.

## Bug 9: fetch_minute_bars skip-cached only checked end-of-cache

**Commit introducing:** `9fba853`

**Symptom:** "skip fully-cached symbols" check looked at
`end_target − cache.last_date ≤ 2 days`. Symbols seeded by an earlier
90-day smoke-test had recent data and got SKIPPED on the subsequent
730-day full fetch — leaving NVDA/AAPL/SPY + 10 sector ETFs with
3 months of 10-min data instead of 2 years.

**Fix:** `49d351f` — require BOTH cache.first ≤ start_target+2d AND
cache.last ≥ end_target-2d. Re-fetched the 13 affected symbols.

**Lesson:** "fully cached" is a 2-sided check. Don't trust your own
"last seen" heuristic.

## Bug 10 (deep-audit): non-atomic rotation pair commit

**Discovered:** 2026-04-24 deep audit of rotation pipeline (user prompt).

**Symptom:** `EmitRotationsTask` appended the SELL exit FIRST, then
constructed the BUY. If the buy failed (`price <= 0` or `shares < 1`),
the loop hit `continue` — but the sell exit was already on
`ctx.exits` and would execute downstream. Outcome: held position
closed for cash, no replacement bought.

**Severity HIGH:** silent capital loss in production. Hard to detect
because logs read "rotation" exit + sell, but the buy line is
missing — easy to miss in audit logs.

**Realistic trigger paths:**
- `price <= 0`: a delisted symbol crossed bars
- `shares < 1`: rotation triggered with insufficient cash to fund
  a 1-share replacement (high-priced ticker, low cash)

**Fix:** rearranged loop body — compute price, conviction, sigma_mult,
shares FIRST. If any check fails, `continue` BEFORE touching
`ctx.exits`. Sell exit only commits after buy is confirmed.

**Test:** `tests/test_rotation_atomic.py::TestAtomicRotation` (3 tests):
buy succeeds → exit commits, no price → entire pair skipped, low cash
→ entire pair skipped. The "no price" and "low cash" cases would
have failed pre-fix (showing exit committed without order).

## Bug 13: earnings-calendar.json not auto-refreshed in training pipeline

**Discovered:** 2026-04-24 audit while 99-ticker retrain ran.

**Symptom:** `earnings-calendar.json` artifact was last written 2026-04-22 (before
watchlist expansion). After we expanded watchlist 43 → 99 today,
new tickers had **0 earnings dates** in the artifact. EarningsFilterTask
gates buys ±3 days around earnings; with no dates, all new tickers
were trading completely unfiltered through earnings periods.

**Severity HIGH:** in production this would cause NVDA/META-class
positions to be opened the day before an earnings announcement and
get blasted by the post-earnings move.

**Fix:** ran `scripts/fetch_earnings_calendar.py` manually for 99
tickers. `earnings-calendar.json` now has dates for all of them.

**Bug 14 (related):** the fetch script is NOT inside FullTrainingPipeline.
Other artifact-producing tasks (CorrelationJob, fundamentals, hourly,
minute) ARE in the pipeline; earnings calendar is asymmetric — must
remember to run the script after every watchlist change. Recommended
fix: add `RefreshEarningsCalendarTask` inside `FetchPanelDataTask`
chain, or reject training start if artifact is older than the
watchlist's mtime.

## Bug 15: SimAdapter top-up averages entry_price (not FIFO lots)

**Severity LOW (known simplification):** when topping up an existing
position, `entry_price = (old_shares × old_entry + new_shares × price) /
total_shares`. IRS lot accounting is FIFO — selling X shares of NVDA
sells from the OLDEST lot first at OLDEST price. Our average-cost
approach gives a different tax_drag estimate than reality.

**Not a bug per se** — documented elsewhere. Cost: tax_drag in the
rotation primitive could be off by 1-3% in cases of multi-lot
positions held across the LT/ST boundary.

## Bug 16: LoadInsiderTradesTask auto-fetches in training (45-min delay)

**Symptom:** `panel_ltr.insider_trades.allow_fetch: true` (default)
makes `LoadInsiderTradesTask` fetch missing tickers from SEC during
training. The 99-ticker retrain blocked for ~45 minutes inside this
task because each missing ticker timed out at 45s × 59 tickers.

**Severity MEDIUM:** training time bloat, but correctness is fine
(fail-soft into NaN insider feature, sector-median fill).

**Fix candidate:** flip default to `allow_fetch: false` so training
ALWAYS reads cache only; require explicit `scripts/fetch_insider_trades.py`
run before retrain. Same pattern as hourly/minute fetches. Apply for
LoadEarningsSurpriseTask and LoadFundamentalsTask similarly.

## Bug 17: NOT a bug — Rotation prunes ctx.ranked

Audit confirmed `EmitRotationsTask` correctly prunes `ctx.ranked` to
exclude rotation buys (line 737 of task_rotation.py), preventing
SelectionJob from double-buying a candidate that was already
rotated-into.

## Bug 18: NGBoost train doesn't drop NaN rows

**Discovered:** while watching `RuntimeWarning: overflow encountered
in square` fire repeatedly during NGBoost training.

**Symptom:** `NGBoostHead.train()` (line 69-70) does
`X = panel[feature_cols].values.astype(float)` without `.dropna()`.
If panel has NaN feature values (e.g. a ticker with missing minute
data on a particular date), NaN propagates into NGBRegressor.fit()
and σ²calculation, causing the overflow warning + potentially
degrading the saved σ estimates.

**Severity MEDIUM:** σ used for sigma_multiplier sizing; if some σ's
are bogus, position sizing for those tickers misallocates.

**Fix candidate:** add `panel.dropna(subset=feature_cols + [label_col])`
before fit. Or `np.nan_to_num` like transformer_model does.

## Bug 24: `size` factor is log(price) not log(market_cap)

**Discovered:** 2026-04-24 audit while NGBoost trained.

**Symptom:** `compute_size_feature(ohlcv, shares_outstanding)` falls
back to `log(close_price)` when `shares_outstanding=None`. Our
TickerPanelFactorJob calls it with None — so the `size` factor is
actually log(price), not log(market_cap).

**Severity LOW — feature has near-zero IC anyway** (-0.0075 in latest
99-ticker panel). L1/L2 regularization wipes it out.

**Fix candidate:** either drop `size` from raw_cols, or wire shares_outstanding
from yfinance .info to compute real market cap.

## Bug 22 (followup): `build_spy_context` is dead code

`kernel/indicators.py:128` defines the SCALAR-broadcast version that
once introduced lookahead. Now unused — only `build_spy_context_series`
is called. Should be removed in next-session cleanup to prevent
accidental future use.

## What I'm NOT claiming

- No guarantee I caught every bug this session. Welcome the second AI's audit.
- No guarantee the fixes are themselves bug-free. Tests passing is necessary but not sufficient.
- No guarantee the rotation-V4 wiring is complete — DB lookup depends on `ctx._db` being set on the adapter (done for SimAdapter in this commit; NOT done for LeanAdapter or RunnerAdapter).

## Recommendations going forward

1. Every new Task/Job/kernel fn needs a **wire-up test** that proves the code path from config flag → kernel fn → output. Unit tests alone don't catch dead branches.
2. Every numeric claim in docs/commit messages should be traceable to a specific file + line + timestamp. The IC misquote came from referencing a stale summary doc.
3. A/B results should be reported with clean variance attribution (same panel, same cache, same sim length). "Baseline drift" between A/Bs is a warning that the comparison isn't apples-to-apples.
