# 2026-08-28 — Live broker never silently truncates a quantity (fractional chain step 2)

**Bottom line:** the two order paths the live runner trades through
(`live/alpaca_broker.py::place_order` and `::place_stop_order`) no longer
build requests with `qty=int(quantity)`. A whole-share quantity produces a
payload, result dict, log line and client I/O that are byte-identical to
before (pinned against a verbatim legacy oracle); a fractional quantity is
NEVER truncated — it is either submitted exactly (confirmed-fractionable
asset, MARKET + DAY, 9dp grid rounded down) or refused with
`FractionalOrderRefused` before the account read, the breaker admit and the
submission. Every quantity is first validated finite and strictly positive
(`InvalidOrderQuantity`, before any I/O — review round 1). No flag is flipped; every
config in the repo still has `execution.fractional_shares.enabled` absent or
false, and on that path today's sizing emits exact integers, so production
behaviour is unchanged. This is **step 2 of 8** of the fractional dependency
chain (step 1 = RenQuant#610).

## Why this step exists

Step 1 gave the live broker `is_fractionable` + `is_no_submit_status` so the
capability gate can clear. Had the gate been flipped on top of the old order
path, a fractional BUY of 0.435578 shares would have been submitted as
`qty=0` and a 7.5-share SELL as `qty=7`, silently — exactly the failure mode
the gate fail-closes against today. The truncation had to go before any later
step could be allowed to produce a fractional intent.

## The contract implemented

| Rule | Where |
|---|---|
| **Basic validation first, before ANY I/O** (review round 1): every `quantity` must be finite and strictly positive — and not eps-integral to zero (`5e-10` would snap to `qty=0`) — else `InvalidOrderQuantity` (one WARNING). `place_order` runs it ahead of the G2 price read; `_resolve_submit_qty` runs it again (pure, idempotent) so the stop path is covered too. | `live/broker.py` `validate_order_quantity` / `InvalidOrderQuantity`; `live/alpaca_broker.py` `_require_valid_quantity`, `place_order` (call precedes `get_last_price`) |
| Eps-integral `quantity` (within `QTY_INTEGRAL_EPS = 1e-9` of an integer) → whole-share branch: submit `int(round(q))`; the request kwargs, the returned dict (`"quantity": int`), the INFO line and the I/O (one G2 price read, one pre-trade account read, **no** asset lookup, one G2 slot) are unchanged. A snap to **zero** (`5e-10` is eps-integral to 0) is refused with `InvalidOrderQuantity` by the validation above, before the price read (the branch keeps a defensive guard) — `qty=0` is never submitted. `is_whole_share` itself stays the owner's pure eps-integral predicate (0 is integral); validity is the caller's job. | `live/alpaca_broker.py` `_resolve_submit_qty` whole-share branch, `place_order` request/log/result, `place_stop_order` |
| Non-integral `quantity` → fractional preflight, in order: `order_type == "market"` and `time_in_force == "day"`; snap DOWN to the 9dp grid (never past the intent); known notional ≥ $1; asset CONFIRMED fractionable via `_lookup_fractionable` → `get_asset` (a lookup failure is its own refusal, never cached as a verdict). Any violation → exactly one WARNING + `FractionalOrderRefused(symbol, quantity, reason, status=<no-submit status>)`. | `live/alpaca_broker.py` `_resolve_submit_qty` |
| **What a refusal does and does not touch (precise):** `InvalidOrderQuantity` fires before ANY I/O — no price read, no asset lookup, no account read, no breaker slot, no submission. `FractionalOrderRefused` fires before the account read, the breaker admit and the submission — but the G2 price read (`get_last_price`) has already happened in `place_order`, and past the type/TIF rule the `get_asset` lookup happens; both are metadata I/O. On the stop path (no price read; type/TIF fires before the lookup) a fractional refusal is before any I/O. | `live/alpaca_broker.py` `place_order` preflight comment, `_resolve_submit_qty` docstring |
| `place_stop_order` routes through the same helper with `order_type="stop", time_in_force="gtc"` → every fractional qty is refused there (Alpaca fractional orders are DAY-only; broker-side GTC stops are whole-share only; the software-stop layer is stage 3). Whole-share stops unchanged. | `live/alpaca_broker.py:522-533` |
| Fractional submission result carries `"quantity": <float, 9dp>` plus `"requested_quantity"` (fractional path only — the whole-share dict stays key-for-key identical). | `live/alpaca_broker.py:378-381` |
| Vocabulary + helpers: `QTY_INTEGRAL_EPS`, `MAX_ORDER_DECIMAL_PLACES = 9`, `MIN_FRACTIONAL_NOTIONAL_USD = 1.0`, `FRACTIONAL_ORDER_TYPE/TIME_IN_FORCE`, the four `rejected_*` statuses (all members of `NO_SUBMIT_STATUSES`), `is_whole_share`, `snap_qty_to_broker_grid` (Decimal `ROUND_DOWN` on `repr(q)`), `FractionalOrderRefused(ValueError)`. | `live/broker.py:88-160` |

Semantics are a port of renquant-execution pin `91c7bf88`
(`src/renquant_execution/broker.py:62-75` Alpaca facts, `:81-88`
`is_whole_share`, `:122-171` `validate_fractional_order`;
`src/renquant_execution/alpaca_broker.py:397-480` `place_order`'s
whole-share snap / fail-closed branch and `:797-828` the whole-share-only
stop). Nothing is imported from the execution repo's order-submission code:
`live/runner.py` imports `live.alpaca_broker` directly (a deliberate
diverged pin, `live/alpaca_broker.py:31-33`), so the discipline has to live
on the class that actually trades. The only cross-repo import remains the
no-submit vocabulary established in step 1 (`live/broker.py:42-49`).

## Why the whole-share path is byte-identical

* Today's flag-off sizing emits exact integers
  (`adapters/runner_execmath.py` cash cap, `cap_affordable_qty`), and for an
  exact integer `int(round(q)) == int(q)`. The request kwargs are compared
  to a verbatim copy of the pre-change construction in
  `tests/test_live_broker_no_silent_truncation.py:176-235`
  (`_legacy_market_payload` / `_legacy_market_result` / `_legacy_market_log`
  and the stop equivalents) for BUY and SELL over `1, 5, 100, 2500, 5.0,
  12.0, 3.0000000004` — including `type(qty) is int`, the exact result dict
  (no extra key), the exact INFO string, zero WARNINGs, one account read, no
  asset lookup, one G2 slot.
* The INFO format changed from `%d` to `%s`; for an `int` argument the
  rendered string is identical (pinned by the same tests). The stop INFO
  keeps `%d` (its argument is always an `int` — a fractional stop never
  reaches it).
* **One deliberate deviation, made explicit** (`test_eps_noise_below_
  integer_snaps_to_nearest`): eps-noise *below* an integer, e.g.
  `2.9999999996`, is eps-integral (whole-share 3) and now submits 3; legacy
  `int()` floored it to 2 — a silent one-share truncation of float noise.
  That is the exec repo's `float(round(requested_qty))` snap and the same
  rule `supports_broker_side_stops` and `normalize_fill_qty` already use.
  Such inputs cannot arise on today's path.

## Runner mapping: a refusal is a no-submit outcome, never a crash

`FractionalOrderRefused` is raised, not returned as a `rejected_*` result
dict, on purpose: `adapters/runner_execmath.py:156-205`
(`broker_order_execution`) classifies any status outside its terminal
reject set as PENDING, so a returned no-submit dict would have been recorded
as an open order that never existed. The runner's existing handlers absorb
the exception on the existing surfaces:

* BUY — `backtesting/renquant_104/adapters/runner.py:1548-1555`:
  `except Exception` → `ctx.orders_skipped` with
  `skip_reason="broker_error:FractionalOrderRefused"`, `continue`.
* SELL — `runner.py:1216-1227`: `except Exception` → `ctx.exits_failed`
  with `error=str(exc)` (the message carries symbol, qty, reason and the
  no-submit status), `continue`; the position is not reaped, no partial
  truncated exit happens.
* Z9 — `adapters/z9_stops.py:169-173`: `except Exception` → WARNING, return;
  no stop is recorded. (The Z9 router already routes fractional qty away
  from the broker stop via `supports_broker_side_stops(symbol, qty)`; the
  refusal is defence in depth behind it.)

No runner change was made. The `skip_reason` therefore reads
`broker_error:FractionalOrderRefused` rather than a dedicated `no_submit:*`
label — the exception carries `.status` (a `NO_SUBMIT_STATUSES` member) so a
later step can classify it explicitly if the audit wants that distinction.

## Test evidence

`tests/test_live_broker_no_silent_truncation.py` (new, 89 tests, offline:
SDK request/enum modules stubbed via `sys.modules`, client stubbed on an
`AlpacaBroker.__new__` instance):

* `TestIntegralPathByteIdentical` — payload/result/log/I/O (one price read,
  one account read, no asset lookup, one G2 slot) vs the legacy oracle
  (market BUY/SELL × 7 inputs, stop × 4), the explicit eps-noise deviation,
  legacy stop guards untouched, and the source no longer contains the
  truncating casts.
* `TestInvalidQuantity` (review round 1) — `0, 0.0, 5e-10, -3, -0.5, NaN,
  inf, -inf` × BUY/SELL at `place_order` level: `InvalidOrderQuantity`, with
  **no price read, no asset lookup, no account read, no breaker slot, no
  submission**, exactly one WARNING; the `5e-10` snap-to-zero regression;
  negative whole numbers never become a negative int qty; the stop path
  refuses NaN and `5e-10` before the account read.
* `TestFractionalRefusals` — not fractionable / lookup failed (not cached) /
  not connected / known notional < $1 (refused before the lookup) /
  limit, GTC, IOC, stop, stop_limit: nothing submitted, no account read, no
  G2 slot, exactly one WARNING, `.status` in the no-submit vocabulary; the
  price read and (where reached) the asset lookup are asserted as having
  happened. Unknown notional (price feed down) does not block a
  fractionable order (mirrors the owner; the broker keeps that rejection).
* `TestFractionalSubmission` — exact float qty on MARKET + DAY,
  `requested_quantity`, cached verdict, 9dp floor never past the intent,
  and the real alpaca-py request model accepts a float qty (skips where the
  SDK is absent).
* `TestStopPath` — fractional stops refused before any I/O (no price read
  on that path; type/TIF fires before the lookup — price, asset, account and
  submit all asserted empty); the capability probe and the stop path agree
  on every quantity.
* `TestHelpers` — `is_whole_share` (as the pure predicate; zero-adjacent
  values deliberately NOT asserted as whole-share OK), `validate_order_
  quantity`, the snap, both exception shapes, and a drift tripwire equating
  every replicated constant / `is_whole_share` verdict with the pinned
  renquant-execution checkout.
* `TestRunnerMappingStatic` (runs everywhere) + `TestRunnerMapping`
  (end-to-end through the REAL `RunnerAdapter.commit` using the stage-0
  harness; needs the strategy deps): BUY refusal → `orders_skipped`, SELL
  refusal → `exits_failed` with the position kept, Z9 refusal → warning and
  no stop; the run completes and persists state.

Runs (all `[VERIFIED]` on this branch, 2026-08-28):

* Lean CI reproduction — throwaway venv with Python 3.10 + `pytest` only
  (what `.github/workflows/live-broker-fractional-contract.yml` installs):
  `pytest -o addopts='' tests/test_live_broker_fractional_contract.py
  tests/test_live_broker_no_silent_truncation.py` → **125 passed, 6 skipped**
  (owner tripwires without the sibling checkout, the real-SDK check, the 3
  runner e2e cases). The workflow now names the new file in both `paths`
  lists and runs it as a second step of the same job.
* Repo venv, `pytest.ini` (`-n auto`, `RENQUANT_NO_NOTIFY=1`): the new file +
  `test_live_broker_fractional_contract.py` +
  `test_s_frac_stage0_commit_contract.py` + every test importing
  `live/alpaca_broker` (`test_agent_breaker`, `test_broker_side_stops`,
  `test_live_alpaca_bounded_timeout`, `test_p0_fixes_regression_guards`,
  `test_round3_audit_fixes_2026_04_25`, `test_runner_state_fixes`,
  `test_state_ext_sell_fill_attribution`) → **473 passed, 1 failed**. The
  failure, `test_agent_breaker.py::TestAlpacaBrokerWiring::test_order_cap_
  blocks_before_api`, fails identically on a pristine `origin/main` worktree
  at the same time of day: the test pins `_day = dt.date.today()` (local,
  PDT) while `AgentBreaker.admit` rolls on `live.clock.trading_date()`
  (exchange date, `live/agent_breaker.py:60-64`), so between ~21:00 and
  24:00 PDT the counters reset and the breaker admits. Pre-existing, not
  touched here.
* Full umbrella suite, branch vs pristine `origin/main` worktree (same venv,
  same invocation): see the line below.

  Branch: **72 failed, 15383 passed, 7786 skipped, 2 errors**. Pristine
  `origin/main` (5ebe64d), run 1: **60 failed, 15308 passed, 2 errors**;
  run 2 (same code, minutes later): **66 failed, 15301 passed, 2 errors** —
  7 ids failed in run 2 that passed in run 1 (`test_training_modules`,
  `test_software_stops`), i.e. the suite is order-dependent under xdist on
  its own. Failing-id set-diff: branch minus (main run 1 ∪ run 2) = 18 ids
  in `test_score_audit`, `test_shadow_scoring`, `test_session_regressions`,
  `test_regime_ensemble_scorer`, `test_wf_loader_fingerprint_dispatch` —
  none touch `live/`. Their tracebacks are sibling-path pollution
  (`renquant_pipeline.kernel.panel_pipeline.shadow_scoring has no attribute
  _compute_shadow_summary`, `No module named renquant_pipeline.software_
  stops`): a worker's `sys.path` depends on which earlier test injected
  sibling checkouts. Evidence: the 13 collectable ids **pass serially on
  both worktrees** (`-o addopts=''`, 13 passed / 13 passed); the other 5 are
  in `test_wf_loader_fingerprint_dispatch.py`, which fails collection
  identically on main and branch (one of the 2 errors in every run). Seven
  `test_short_candidate_selection` ids failed on main and passed on the
  branch — the same class in the other direction. **Zero new failures
  attributable to this change.** The rewritten committed
  `backtesting/__pycache__/__init__.cpython-310.pyc` was restored before
  committing.

## Ambiguities resolved conservatively

1. "Byte-identical for integral within 1e-9" vs "the exec repo's
   whole-share snap": for exact integers both agree; for eps-noise below an
   integer the snap (round) wins over the legacy floor. Chosen: the snap,
   because keeping `int()` would keep a silent truncation in the code this
   change exists to remove and would disagree with `supports_broker_side_
   stops` (which already treats 2.9999999996 as 3). Pinned and documented.
2. `place_order` has no order-type / TIF parameters (it is always MARKET +
   DAY). The type/TIF rule is enforced in the shared `_resolve_submit_qty`
   and has a production caller through the stop path; the limit/GTC/IOC
   cases are tested against the helper directly.
3. Notional < $1 is refused only when the notional is KNOWN (the G2 price
   read succeeded). With the price unavailable the order proceeds (count-only
   G2 accounting, unchanged) and the broker remains the authority — same as
   the owner's qty-order preflight, which validates notional only when it is
   given.
4. Non-numeric / non-finite / zero / negative quantities surface as
   `InvalidOrderQuantity` (a `ValueError`) before any I/O, instead of the
   legacy `int()` `ValueError`/`OverflowError` after the account read (or,
   for 0 / negative integers, a submitted `qty=0` / negative qty the broker
   would bounce). Not valid integral quantities, so outside the
   byte-identical guarantee; the runner mapping is identical (same
   `except Exception` sites, `skip_reason="broker_error:InvalidOrderQuantity"`).

## Review round 1 (Codex, PR #612)

1. **Correctness hole fixed:** `is_whole_share` treats values within 1e-9
   of ZERO as integral, so `5e-10`, `0` and negative exact integers entered
   the whole-share branch and `_resolve_submit_qty` could return `0` or a
   negative `int` that `place_order` then submitted. Now: finite AND
   strictly positive is validated for every quantity ahead of the
   whole/fractional split and ahead of the price read
   (`live/broker.py::validate_order_quantity`,
   `live/alpaca_broker.py::_require_valid_quantity`), and a whole-share snap
   must be > 0. The test that asserted `is_whole_share(5e-10) is True` was
   removed; `TestInvalidQuantity` proves 0 / 5e-10 / −3 / −0.5 / NaN / inf
   are refused before breaker admission, before any price/account read and
   before submission. The byte-identical oracle class is unchanged and green.
2. **No-I/O claim corrected everywhere** (code comments, this doc, test
   names/docstrings): a fractional refusal is "no account read, no breaker
   slot, no submission" — the price read and the asset lookup may occur.
   Invalid-quantity refusals are the ones that happen before ANY I/O.

## Place in the fractional dependency chain

**Step 2 of 8** (list as in the step-1 doc,
`doc/progress/2026-08-28-live-broker-fractional-contract.md`). Remaining:
3 contract test (fractional BUY through the real commit path reaches the live
broker class un-truncated — the broker side of that is now
`TestRunnerMapping` + `TestFractionalSubmission`; the end-to-end wiring test
is still its own step), 4 `fractional_max_book_pct`, 5 pager install,
6 evidence packet, 7 `execution.software_stops` config row, 8
`execution.fractional_shares` flag row.

## Memory tier

No LONG/MID constraint changes. SHORT: the live order paths are
truncation-free as of this branch; a fractional intent is refused or
submitted exactly, never floored. Flag still OFF everywhere.
