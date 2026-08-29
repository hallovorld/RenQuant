# 2026-08-29 — Entry date = CURRENT trip start; re-entry cooldown on every SELECT path (RenQuant#618 class B)

**Bottom line:** the live runner seeded `entry_dates` from the OLDEST BUY
fill ever per symbol, so a name that fully exited and re-entered inherited
its PREVIOUS trip's date (`ENTRY-DATE-SEED NVDA ← 2026-04-17` on 2026-08-25,
hold=130d one session after the buy) and `min_hold_days=5` never protected
it; `min_reentry_days=5` was enforced only by the QP wash mask, so the non-QP
`SELECT [slot N]` path re-bought VLO 7 h after its exit filled for a gain.
Five sessions (08-24..08-28) of VLO <-> NVDA ping-pong followed
(`[VERIFIED]` in RenQuant#618). This PR makes the entry date the first BUY
fill AFTER the last time the running position quantity reached zero (a
qty-only, order_id-deduplicated replay of the broker fill history,
broker-anchored when that history is inconsistent), clears entry state on
the REALIZED broker quantity rather than on intent, and applies the same
`min_reentry_days` rule the QP path uses to EVERY BUY that reaches the
runner (SELECT, rotation buy leg, QP). Order placement is otherwise
untouched: exits are never blocked, and the only new BUY-side block is the
cooldown the strategy config already declares. Nothing deployed by this
branch; the live tree and the `-run` checkout are untouched.

## 1. Incident (from RenQuant#618, all `[VERIFIED]` there)

| Defect | Where (pre-fix) | Effect seen live |
|---|---|---|
| `first_fill_map` = oldest BUY ever per symbol ("we don't currently track the trip-lifecycle") | `backtesting/renquant_104/adapters/runner.py:449-471` | `ENTRY-DATE-SEED NVDA ← 2026-04-17` (intraday 08-25, 08-27), `VLO ← 2026-08-05` (08-26, 08-28); `hold=130d` / `21d` one session after entry |
| seed/backfill override always took the older broker date | `runner.py:542-565` | `entry_dates.VLO = 2026-08-05` persisted through two re-entries |
| entry state cleared only on `not is_partial` (intent) | `runner.py:1351-1354` | a sell that leaves the broker flat but was sized as a trim kept the stale entry |
| `is_topup = ticker in self._entry_dates` | `runner.py:1610` | a fresh entry on a flat name with a stale entry read as TOPUP → previous trip's date preserved |
| `min_reentry_days` only in the QP wash mask | pinned `kernel/portfolio_qp/tasks.py:665`; SELECT path `kernel/selection.py::run_selection_loop` and rotation `task_rotation.py::admissible` apply only §1091 (gains skip) | VLO `SELECT [slot 1]` 2026-08-25 13:57, 7 h after its SELL filled at 13:30Z |

## 2. Change

### 2.1 `backtesting/renquant_104/adapters/runner_trip_lifecycle.py` (new, dependency-free)

* `replay_trip_lifecycle(fills, current_qty=…) -> (states, dropped)`:
  normalize (symbol / side / qty / NY trade date / order_id; both the
  umbrella `action`+`avg_price` and the execution-subrepo `side`+`filled_qty`
  schemas), de-duplicate on `(symbol, order_id)` (the umbrella page walk
  re-fetches the boundary order — class C), replay chronologically: BUY adds
  and opens a trip when flat; SELL subtracts and CLOSES the trip when the
  running qty reaches `<= 0` (date → `last_exit`). Price is never consulted,
  so a price-less SELL is NOT dropped (contrast `runner_tax_lots`, which
  needs a basis). Rows that cannot be placed on the timeline are counted in
  `dropped`, never silently lost.
* Broker-anchored correction: when the caller passes the broker's current
  quantities and the forward replay does not land on them (the standing
  `LIVE-TAX-LOTS: NVDA reconstructed lot qty 14 != broker 7` condition), the
  trip start is recovered by walking BACKWARD from the broker quantity to
  the BUY at which it reaches zero. The current trip's fills are the most
  recent ones, so this is robust to corruption in older history. If the
  walk never reaches zero the trip start is `None` (unknown) — the state
  date is then kept, never moved LATER by a guess.
* `resolve_entry_date(state, trip_start, today)` — the decision table:
  `seed` (no state) / `sentinel` (no state, no history: today−31d, as
  before) / `keep` (equal, or trip unknown) / `backfill` (state INSIDE the
  trip but later than its first fill → trip start; the pre-existing
  ENTRY-DATE-BACKFILL, now bounded by the trip) / `reseed` (state OLDER than
  the trip start = a previous trip → trip start; logs
  `ENTRY-DATE-RESEED <t> <old> → <new> (trip start)`). The old "state
  predates broker → preserve" rule survives ONLY inside the current trip.
* `reentry_blocked(...)` — the QP churn leg's rule verbatim
  (`0 <= days_since < min_reentry_days`), where the last full exit is the
  LATER of the persisted `last_sell_dates` entry and the replay's
  `last_exit`.

### 2.2 `backtesting/renquant_104/adapters/runner.py`

* `make_context` (ENTRY-DATE-FROM-FILLS block): `first_fill_map =
  trip_start_map(trip_states)`; per-symbol `ENTRY-DATE-TRIP` warning when the
  replay is inconsistent with the broker; the seed/backfill/reseed decision
  goes through `resolve_entry_date`; `self._trip_states` retained for
  `commit()`.
* `commit()` sell loop: after the fill, if `broker.get_position(ticker)`
  is 0 the sell is a FULL exit regardless of intent — `ENTRY-DATE-CLEAR`
  logged, `is_partial = False`, so the existing `if not is_partial:` block
  (unchanged; still the byte-window other tests pin) pops
  `entry_dates` / `entry_signals` / `position_hwm` / streaks and stamps the
  wash-sale clock. A `get_position` failure keeps the intent-based result
  and logs.
* `commit()` BUY loop: `is_topup` requires the name to be HELD at bar
  start (`ctx.holdings` or a positive positions-cache qty); a stale entry
  for a flat name is cleared (`ENTRY-DATE-CLEAR … stale entry state`) and
  the BUY is stamped as a fresh entry.
* `commit()` BUY loop, before any broker interaction:
  `_reentry_cooldown()` → `ANTI-CHURN <t>: BUY skipped — last full exit
  <d> is <n>d ago < min_reentry_days=<N>`; the order goes to
  `ctx.orders_skipped` with `skip_reason="min_reentry_days"`. Applies only
  to names not held at bar start (a top-up is not a re-entry); `min_reentry_days
  <= 0` disables it. The ledger read is `last_sell_dates` — a key the
  pipeline's `renquant_pipeline.kernel.live_state_v2` schema defines
  (`[VERIFIED]` `live_state_v2.py:78,122`) — plus the fill replay. No
  invented key.

### 2.3 Tests (`tests/test_entry_date_trip_lifecycle.py`, 50 tests)

* replay: single trip; exit + re-entry gives the re-entry date; partial
  sell keeps the trip; multiple round trips (the VLO ledger); price-less
  SELL not dropped; SELL-before-BUY closes; order_id de-dup; unusable rows
  counted; anchored correction (duplicate BUY → 14 vs 7 → latest BUY wins);
  anchored UNKNOWN when history does not reach flat; flat-at-broker with
  inconsistent history → latest SELL; execution-subrepo schema.
* `fill_trade_date`: aware UTC → NY trade date (00:30Z = previous NY day);
  naive / date-only fall back; garbage → None.
* `resolve_entry_date`: seed / sentinel / keep / backfill / reseed
  (the VLO 08-05 vs 08-26 and NVDA 04-17 vs 08-25 cases) / unparseable.
* `reentry_blocked`: days 0-4 blocked, 5+ allowed; later ledger wins;
  replay-only ledger; disabled / no-exit never blocks.
* REAL `RunnerAdapter.commit` path (fake broker): partial-intent sell with
  broker flat after → entry state cleared, wash-sale stamped, `SELL` not
  `TRIM`; partial with shares still held → unchanged; BUY blocked days 0-4
  (`skip_reason=min_reentry_days`, no `place_order`), allowed at 5/6/12;
  blocked from the replay ledger alone; top-up of a held name never
  blocked and keeps the trip start; cooldown disabled; stale entry on a
  flat name → fresh `BUY` stamped today.
* Source tripwires updated to the new anchors:
  `tests/test_runner_state_fixes.py` (the `f.get("action") != "BUY"`
  oldest-buy pin is now asserted ABSENT), `tests/test_live_state_contract.py`.

Run with `/Users/renhao/git/github/RenQuant/.venv/bin/python -m pytest -q
-o addopts=''` the way the CI workflows invoke pytest:

* focused (14 runner-touching files incl. the new one): **536 passed**.
* full `tests/` + `backtesting/renquant_104/tests` (`-m "not slow"`, xdist
  `-n 4`): fix worktree **15,384 passed / 159 failed / 73 errors**; the
  unmodified `origin/main` worktree run with the same flags in the same venv
  **15,452 passed / 111 failed / 2 errors**. The two xdist runs disagree on
  unrelated areas (shadow_scoring, walkforward_loader, short_cover …), so
  the 14 files whose failures differed were re-run SERIALLY in both
  worktrees, one after the other: **identical result — 62 failed / 171
  passed / 1 collection error (`tests/test_software_stops.py`:
  `renquant_pipeline` not importable outside xdist) in both**. Zero
  failures are introduced by this change; the residual set is pre-existing
  (walkforward loader/manifest/isolation, sim_walkforward, sim_wf_provenance,
  the `SimAdapter._provenance_sink` harness gap in
  `test_adapter_context_contract.py`) and none touches `adapters/runner*`.

## 3. Deploy step (NOT done here)

Merge → the operator/agent landing fast-forwards the live tree
`/Users/renhao/git/github/RenQuant` to `origin/main` (read-only checks
first, per the L6 live-deploy authorization) and, if the `-run` checkout
is the executing tree for the umbrella runner, the same ff there. Revert =
`git checkout <pre-ff sha> -- backtesting/renquant_104/adapters/runner.py`
plus removing `runner_trip_lifecycle.py`, or ff back to the previous main
sha. Expected first-run log lines on the current book:
`ENTRY-DATE-RESEED VLO 2026-08-05 → 2026-08-28 (trip start)` (if VLO is
still held), `ENTRY-DATE-TRIP …` warnings for the names whose lots already
mismatch (NVDA/VLO/PANW/APH), and `ANTI-CHURN` on any fresh BUY of a name
sold within 5 days.

## 4. Non-goals (separate lines)

* **Class C** — fill-replay drops / duplicates in `runner_tax_lots.py:124,140`
  and the `until_cursor` boundary re-fetch in `live/alpaca_broker.py:755-765`.
  This PR de-duplicates on `order_id` and anchors on the broker qty for the
  LIFECYCLE only; the tax-lot reconstruction still sees the raw list.
* **Class A** — the rotation buy leg is sized on the intended full sell and
  is not re-sized/cancelled when the sell leg is clamped by
  `qty_available` (`runner.py:1176-1209`); not touched.
* Pipeline-side `min_reentry_days` in `run_selection_loop` / rotation
  `admissible` (so the candidate never reaches sizing) — belongs to
  renquant-pipeline; the runner-side gate here is the umbrella's guard
  regardless of which pin runs.
