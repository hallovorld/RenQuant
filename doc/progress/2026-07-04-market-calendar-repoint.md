# Re-point umbrella scripts to the canonical NYSE market calendar (campaign B5)

Date: 2026-07-04
PR: fix(calendar): re-point to the common canonical

## What

Umbrella SCRIPTS ONLY (the kernel mirror is pipeline-owned and untouched):

- `scripts/preopen_cancel_gate.py` — `_previous_nyse_close` (strictly-before
  cash close, half-day aware) and `_is_nyse_session_date` are now composed
  over `renquant_common.market_calendar` (`sessions_between` /
  `session_bounds` / `is_session`) instead of hand-rolled
  `pandas_market_calendars` schedules. Fail-closed ValueError contract
  unchanged (same messages).
- `scripts/check_software_stops_liveness.py` — `market_session_open` reads
  the canonical `session_bounds` (close-inclusive check preserved); the
  lenient weekday 09:30-16:00 fallback is unchanged.

Zero `import pandas_market_calendars` remain under `scripts/` (ratchet test
included; the two orchestrator XNYS research scripts are out of scope per
audit #296 §4.1 rows 8-9).

## Evidence

- 10-year equivalence fixture (2016-01-01..2026-12-31): `_previous_nyse_close`
  and `market_session_open` at 3 intraday probes per date, plus
  `_is_nyse_session_date` on every date — identical to the pre-B5 hand
  copies, including half-day early closes and the exact-close boundary
  (strictly-before for the gate; close-inclusive for the liveness check).
- New lockstep tests in `tests/test_market_calendar_repoint.py`.

## Deploy notes

- Merge order: renquant-common `feat(calendar)` first, then the three
  subrepo re-points, this umbrella PR LAST.
- The scripts resolve `renquant_common` via the pinned subrepo runtime
  PYTHONPATH (`renquant_subrepo_pythonpath`); the renquant-common pin must
  be advanced to the market_calendar release (0.10.0) in
  `subrepos.lock.json` before or with this PR's deploy. The umbrella
  `.venv`'s installed renquant-common (0.8.1) predates market_calendar —
  runs that resolve common from the venv instead of the pinned checkout
  would fail loudly at import (fail-closed, never a wrong calendar answer).
