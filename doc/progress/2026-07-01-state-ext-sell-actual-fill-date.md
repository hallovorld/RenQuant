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

### Known residual limitation (not fixed here, flagged for follow-up)

`_lookup_ext_sell_fills` bounds its broker query to a 5-day lookback from
`ctx.today` (the reconciliation run date). If reconciliation is delayed by
more than 5 days, the lookup itself may not surface the real fill at all,
and this fix's NO-FILL-FOUND fallback (stamp `today_str`) applies — safer
than the pre-fix behavior (which always did this) but not a full fix for
multi-week gaps. Widening that lookback window is a separate, narrowly
scoped follow-up if multi-week reconciliation gaps turn out to be common;
out of scope here since the bug being fixed is "the date is discarded when
available", not "the lookback window is too narrow."

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
