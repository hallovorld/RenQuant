# STATE-EXT-SELL must stamp the ACTUAL broker fill date, not today

2026-07-01. Severity: P0 data-integrity (live trading, wash-sale compliance
clock). Customer impact: META's wash-sale re-entry block was silently
extended by 24 days (would have cleared 2026-07-02, was heading toward
~2026-07-26). No bad orders placed — the effect is purely a false BLOCK
(over-conservative), not a wash-sale violation. Caught by manual review of
`live_state.alpaca.json` vs Alpaca's authoritative order history; an
emergency operational correction was applied directly to the live state
file today, out of band from this PR (not redone here — this PR is the
permanent code fix so it cannot recur, for META or any other ticker).

## Symptom

`live_state.alpaca.json`'s `last_sell_dates["META"]` was `2026-06-26`.
Alpaca's order history shows META's real last SELL fill was `2026-06-02`.
The 30-day wash-sale window computed from the wrong date would have kept
META blocked from re-entry until ~2026-07-26 instead of the correct
2026-07-02.

## Root cause

`RunnerAdapter.commit()`'s `STATE-EXT-SELL` reconciliation block
(`backtesting/renquant_104/adapters/runner.py`) detects tickers that
disappeared from the broker's book without a runner-driven sell (manual
close, broker-side stop, corporate action, etc. — Z2, 2026-04-28 NVTS
post-mortem) and stamps `last_sell_dates` so the wash-sale guard blocks
re-entry for 30 days. Pre-fix, for every ticker in `disappeared` it:

```python
ext_sell_fills = self._lookup_ext_sell_fills(ctx, disappeared)
for t in disappeared:
    self._last_sell_dates_str[t] = today_str      # always "today"
    attribution = self._attribute_ext_sell(t, ext_sell_fills)
    log.warning("STATE-EXT-SELL: ... stamping wash-sale clock today (%s) ...")
```

`ext_sell_fills` (via `_lookup_ext_sell_fills`) already queries the broker's
recent SELL fill history for every disappeared ticker — but pre-fix that
result was used ONLY to build the log-line's `attribution` string; the
actual `filled_at` date it carries was discarded, and `today_str` (the date
*this reconciliation code happens to run*) was stamped instead.

This is silently wrong whenever `disappeared` first fires on a LATER bar
than the one the ticker actually left the book — the `entry_dates` state
was still stale from a previous bar. This session directly confirmed one
real, occurring cause of that gap: a full daily-run mid-pipeline failure
that fell back to a sell-only run, which can leave reconciliation stale for
one or more bars until a later run's `disappeared` check catches it. Each
day of delay before the catch directly translates into that many extra
days tacked onto the wash-sale block, with no cap — worst case, weeks.

`entry_dates` already had the correct fix for this exact failure mode
(`ENTRY-DATE-FROM-FILLS` / `ENTRY-DATE-BACKFILL`, Round 4 audit,
2026-04-25): the broker's fill timestamp is treated as AUTHORITATIVE over
"today". `last_sell_dates` never got the equivalent treatment — this PR
brings it in line with that established principle.

## Fix

- `backtesting/renquant_104/adapters/runner_ext_sell.py` — new
  `ext_sell_fill_date(fill)` helper: extracts the fill DATE (first 10 chars
  of `filled_at`, `YYYY-MM-DD`) from a single normalized fill record (the
  same per-ticker value `lookup_ext_sell_fills` already returns). Returns
  `None` when there's no fill or `filled_at` is missing/unparseable.
- `backtesting/renquant_104/adapters/runner.py`:
  - `RunnerAdapter._ext_sell_fill_date` — thin delegate to the above,
    matching the file's existing delegate pattern for
    `adapters.runner_ext_sell`.
  - STATE-EXT-SELL loop: reuses the ALREADY-fetched `ext_sell_fills` (no
    re-fetch) to look up each disappeared ticker's real fill date. If
    found, stamps `last_sell_dates[t]` with that date and logs "ACTUAL
    broker fill date" explicitly. If NOT found (genuine unknown-cause
    disappearance — corporate action, account transfer, or a disposition
    the broker API can't attribute to a dated fill), falls back to
    `today_str` exactly as before, but now logs it as a distinct
    "NO-FILL-FOUND FALLBACK" so an operator reading logs later can tell
    which of the two happened.
- Left the EARLIER, separate `last_sell_dates` stamp in `commit()` (the
  primary full-liquidation path, `if not is_partial: ... =
  self._last_sell_dates_str[ticker] = today_str`) untouched: that path
  stamps at the moment the runner's OWN sell order fills within the SAME
  `commit()` call, so `today_str` there is the actual fill date by
  construction — it is not vulnerable to the same bug. Verified this with
  a source-scan regression test asserting no fill-date lookup is inserted
  into that path.

### Beneficial side effect

If the real fill turns out to be older than the 30-day wash-sale window
(e.g. reconciliation was delayed by 35+ days), the existing wash-sale GC
sweep later in the same `commit()` call (`cutoff = today - 30d`) now
correctly drops that entry immediately, since it's genuinely stale — rather
than the pre-fix behavior of extending a 30-day block starting from
"today" regardless of how old the real sale was.

### Known residual limitation (not fixed here, flagged for follow-up) — SUPERSEDED

**Update 2026-07-01 (codex #428 review round, see section below):** this
turned out to be load-bearing, not a safe deferral — the 5-day window
could never have found the real META fill (24 days back), so the
original version of this PR did NOT actually fix the incident it cited.
Fixed in the review round below (`EXT_SELL_LOOKBACK_DAYS = 45`). Original
text preserved for the record:

> `_lookup_ext_sell_fills` bounds its broker query to a 5-day lookback
> from `ctx.today` (the reconciliation run date). If reconciliation is
> delayed by more than 5 days, the lookup itself may not surface the real
> fill at all, and this fix's NO-FILL-FOUND fallback (stamp `today_str`)
> applies — safer than the pre-fix behavior (which always did this) but
> not a full fix for multi-week gaps. Widening that lookback window is a
> separate, narrowly scoped follow-up if multi-week reconciliation gaps
> turn out to be common; out of scope here since the bug being fixed is
> "the date is discarded when available", not "the lookback window is too
> narrow."

## Tests

- `tests/test_runner_ext_sell.py::TestExtSellFillDate` — behavioral
  (non-source-string) coverage of `ext_sell_fill_date` and its composition
  with `lookup_ext_sell_fills`, including:
  - the exact D1-vs-D2 reconciliation-delay scenario (broker SELL fill on
    D1, reconciliation actually runs on D2 > D1) — asserts the extracted
    date is D1, never D2 (the META-incident regression case);
  - the genuine no-fill-found case (broker has no matching SELL fill at
    all) — asserts the extraction yields `None` so the caller takes the
    fallback path;
  - edge cases: `None` fill, empty dict, missing `filled_at`, unparseable
    `filled_at`.
- `tests/test_runner_state_fixes.py::TestExternalSellUsesActualFillDate` —
  source-level regression guards (matching this file's established
  string-contract style for `RunnerAdapter.commit()`, which is otherwise
  too heavy to mock end-to-end): pins the fill-date-preferred stamp, the
  distinct ACTUAL-fill-date and NO-FILL-FOUND-FALLBACK log markers, the
  single (non-refetching) `_lookup_ext_sell_fills` call, and — the
  same-day-no-regression check — that the primary full-liquidation stamp
  path is untouched (no fill-date lookup inserted there).
- Existing `TestExternalSellWashSaleClock` / `TestStateExtSellPendingOrderFix`
  / `TestPreopenCancelDoesNotStampWashSale` string-contract tests in the
  same file continue to pass unmodified (the NO-FILL-FOUND fallback still
  contains the literal `self._last_sell_dates_str[t] = today_str` they pin
  on).

Full run: `tests/test_runner_ext_sell.py` + `tests/test_runner_state_fixes.py`
(81 tests) green, plus the broader runner-adjacent suite (`test_runner_sell_attribution.py`,
`test_runner_commit_save_state_config_arg.py`, `test_partial_sell.py`,
`test_runner_meta_label_wiring.py`, `test_runner_z9_integration.py`,
`test_production_runner_guard.py`; 57 tests) green. A full-repo `pytest`
run in this fresh bare clone shows pre-existing, unrelated failures
(`ModuleNotFoundError: No module named 'renquant_pipeline'` and similar) —
confirmed via `git stash` that these fail identically without this PR's
changes; they come from pinned-subrepo modules (e.g. `renquant_pipeline`)
that aren't wired into a bare umbrella-only clone, not from this change.

## 2026-07-01 review round: codex #428 CHANGES_REQUESTED — the fix above did NOT actually cover the cited incident

Codex's review of the first version of this PR found the "known residual
limitation" flagged above was actually load-bearing, not a safe deferral,
plus two more real gaps. All three are blocking; fixed in this round.

**1. Lookback window too short for the exact incident it cites.**
`lookup_ext_sell_fills` queried only `today - 5 days`. The real META
incident (fill 2026-06-02, discovered by reconciliation 2026-06-26) is a
24-day gap — the 5-day window can never find that fill, so the code still
fell back to `today_str` in the EXACT scenario it claimed to fix. The
prior regression test only exercised a 3-day delay with a mock broker
that ignored the `after=` parameter entirely, so it passed regardless of
window size — it proved nothing about the lookback boundary.

Fix: `EXT_SELL_LOOKBACK_DAYS = 45` (30d wash-sale window + 15d
operational buffer) in `adapters/runner_ext_sell.py`, replacing the
hardcoded `timedelta(days=5)`. New regression test
(`tests/test_runner_ext_sell.py::TestLookupExtSellFills::test_real_meta_incident_24_day_gap_through_real_lookback_boundary`)
reproduces the exact 2026-06-02 → 2026-06-26 gap through a broker mock
(`_AfterAwareBroker`) that actually filters on `after=`, so the test only
passes if the window genuinely covers the gap.

**2. The lookup accepted ambiguous fills without confirming SELL side.**
`lookup_ext_sell_fills` deliberately keeps rows with no `side`/`action`
field (the execution-subrepo broker schema) so `attribute_ext_sell`'s
log-only string can still name a candidate fill. That tolerance leaked
into the wash-sale-authoritative path too: an ambiguous fill (or a
BUY fill misread as the "most recent" fill) could set `last_sell_dates`.

Fix: `ext_sell_fill_date` now requires `fill.get("side") == "sell"` —
a CONFIRMED sell — before trusting the date. Ambiguous/BUY fills return
`None`, same as no fill at all; the lookup dict itself is unchanged (log
attribution still gets the ambiguous fill). No broker-enrichment lookup
was added (out of scope for this PR — flagged in code as the path if a
schema without `side` ever needs to be authoritative).

**3. Naive first-10-chars date slicing — off-by-one trading-date risk.**
`ext_sell_fill_date` did `_dt.date.fromisoformat(str(fa)[:10])` — a fill
at e.g. `00:30 UTC` was read as if that UTC calendar date were the NY
trade date, when it actually belongs to the PRIOR NY trading date.

Fix: new `_ny_trade_date_from_aware_timestamp()` helper — parses an AWARE
ISO-8601 timestamp (normalizing a `Z`/`z` suffix to `+00:00` for
pre-3.11 `datetime.fromisoformat`), converts via
`zoneinfo.ZoneInfo("America/New_York")`, takes `.date()`. Fails CLOSED
(`None`) on naive timestamps or unparseable strings — never guesses a
timezone. Same zone convention as `live/clock.py`'s `NY`/`trading_date()`
and `kernel/data.py`'s NYSE freshness checks (not imported directly —
`backtesting/renquant_104` doesn't depend on `live/`; independently
applied with the same zone name). Covered by
`tests/test_runner_ext_sell.py::TestExtSellFillDateTimezoneAware`: UTC
`Z`-suffix near-midnight (the review's exact 00:30 UTC example, asserts
the PRIOR NY date), explicit offset, lowercase `z`, DST spring-forward
boundary (2026-03-08 07:00 UTC transition, both sides), partial-fill,
naive-timestamp-rejected, and garbage-timestamp-rejected.

**Also addressed ("ALSO reconsider", non-blocking but straightforward):**
the no-fill fallback used to unconditionally overwrite an existing OLDER
`last_sell_dates` value with today's reconciliation date — "conservative"
in intent, but it destroys known evidence and recreates the
over-extension bug in a different form if the real fill is older than
even the widened lookback window. New `ext_sell_stamp_decision(fill_date,
prior_stamp, today_str)` pure function in `runner_ext_sell.py` returns a
3-way decision: `actual_fill` (confirmed fill wins), `unresolved_preserve`
(no confirmed fill but an older stamp already on file — keep it, log
UNRESOLVED, don't fabricate today), or `no_fill_fallback` (truly no
information at all — today is the conservative choice). `runner.py`'s
STATE-EXT-SELL loop now calls this via a new `_ext_sell_stamp_decision`
delegate instead of branching on `fill_date is not None` directly.

### Tests (this round)

- `tests/test_runner_ext_sell.py` — new/updated: `TestLookupExtSellFills`
  (lookback boundary + the exact 24-day META regression through a
  realistic `after`-respecting broker mock), `TestExtSellFillDate`
  (updated D1/D2 gap to the real 24 days), new
  `TestExtSellFillDateConfirmedSideRequired`, new
  `TestExtSellFillDateTimezoneAware`, new `TestExtSellStampDecision`.
- `tests/test_runner_state_fixes.py` — updated the two source-string
  assertions that changed shape (`test_stamps_today_str`,
  `test_prefers_actual_fill_date_over_today`) and added
  `TestCodex428ReviewFixes` (source-level regression guards for the
  widened lookback, confirmed-side gate, TZ-aware extraction, and
  preserve-on-unresolved).
- Full targeted run: `tests/test_runner_ext_sell.py` +
  `tests/test_runner_state_fixes.py` + `tests/test_state_ext_sell_fill_attribution.py`
  + `tests/test_broker_sync.py` (139 tests) green.
- Full-repo `pytest` in this fresh bare clone: compared FAILED-test-ID
  sets between this branch and the base commit (`fd99c8e8`) across two
  independent base runs — found ~10-15 tests flip pass/fail between
  *two runs of the identical base commit* (order/xdist-worker
  flakiness unrelated to any code change, mostly `test_sim_walkforward.py`
  / `test_walkforward_loader.py` / `test_training_modules.py`). None of
  the flaky or newly-failing tests touch `runner`, `ext_sell`, or `wash`;
  the 3 runner/wash-sale-adjacent failures present (`test_sim_partial_skips_last_sell_date`,
  a `runner_trade_ntfy` source check, a `state_store` delegate check) are
  IDENTICAL across the base commit (both runs) and this branch — confirmed
  pre-existing, not caused by this change.
- `git diff --check`: clean (no whitespace errors).
