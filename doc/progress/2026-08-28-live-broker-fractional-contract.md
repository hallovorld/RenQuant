# 2026-08-28 — Live broker exposes the fractional capability contract (gate leg (a))

**Bottom line:** the class the live runner actually trades through
(`live/alpaca_broker.py::AlpacaBroker`) now satisfies leg (a) of
`fractional_capability_gate` — callable `is_fractionable` + callable
`is_no_submit_status` — so the gate's `broker_fractional_contract` item can
clear. This is **step 1 of the fractional dependency chain**; nothing is
flipped, and the change is provably inert while
`execution.fractional_shares.enabled` and `execution.software_stops.enabled`
are both false (they are, everywhere, today).

## What changed

| File | Change |
|---|---|
| `live/broker.py:4-59` | No-submit status vocabulary. Imports `NO_SUBMIT_STATUSES` from the owner (`renquant_execution.broker`) when the pinned sibling checkout is on `PYTHONPATH`; otherwise falls back to `_FALLBACK_NO_SUBMIT_STATUSES` (`live/broker.py:28-40`), a verbatim copy of the owner vocabulary at renquant-execution `91c7bf88` (the `subrepos.lock.json` pin). `NO_SUBMIT_VOCABULARY_SOURCE` records which one was chosen. Module-level `is_no_submit_status` (`live/broker.py:52-59`) uses the owner's normalisation (`str(status or "").strip().lower() in NO_SUBMIT_STATUSES`). |
| `live/broker.py:184-205` | `BaseBroker.is_no_submit_status` staticmethod delegating to the module vocabulary (mirrors `renquant_execution.broker.BaseBroker.is_no_submit_status`, exec `broker.py:262-273`). **`is_fractionable` is deliberately NOT defaulted on the base**: the gate treats its presence as capability, so a base default would let every broker (paper, read-only wrapper, fakes) pass the structural probe and turn the gate fail-open. |
| `live/alpaca_broker.py:43-46` | `_FractionableLookupError` — raised, never cached, on a failed / not-connected asset lookup. |
| `live/alpaca_broker.py:81-83` | `self._fractionable_cache: dict[str, bool]` in `__init__` (also lazily created in `_lookup_fractionable` so `AlpacaBroker.__new__`-constructed instances — the pattern `tests/test_live_alpaca_bounded_timeout.py:43` uses — work). |
| `live/alpaca_broker.py:432-455` | `_lookup_fractionable(symbol)` — ported semantics of renquant-execution `src/renquant_execution/alpaca_broker.py:738-752` (pin `91c7bf88`): upper-cased cache key; only CONFIRMED verdicts are cached; not-connected or a raising `get_asset` → `_FractionableLookupError`. |
| `live/alpaca_broker.py:457-473` | `is_fractionable(symbol)` — ported from exec `alpaca_broker.py:754-766`: returns the cached/confirmed verdict; on lookup failure answers `False`, logs a WARNING, and does NOT cache so the next call retries. |
| `tests/test_live_broker_fractional_contract.py` (new, 42 tests) | Pins the contract on the REAL class (see evidence). |

Why a port and not an import: `live/runner.py` imports `live.alpaca_broker`
(`live/alpaca_broker.py:31-33` documents this — the pair is a deliberate
`diverged_pin`), so renquant-execution's implementation never reaches the
order path. The vocabulary, being pure data, IS imported from the owner when
available; the fallback exists only so the umbrella never crashes on import
without the sibling checkout (`adapters/runner_execmath.py:41-46` already
uses the same guarded pattern for `order_math`).

**Order submission is untouched.** The diff is insertions only (+144 / −0 in
the two `live/` modules). `place_order` (`live/alpaca_broker.py:270-343`) and
`place_stop_order` (`:371-417`) are byte-identical; the whole-share
`qty=int(quantity)` truncation at `:320/:330/:338` and `:401/:409/:415` is
**step 2** of the chain and must not ride along here.

## Why it is inert while both flags are false

`backtesting/renquant_104/adapters/commit_contract.py:208-212`:

```python
exec_cfg = (config or {}).get("execution") or {}
frac_cfg = exec_cfg.get("fractional_shares") or {}
enabled = bool(frac_cfg.get("enabled", False))
missing: list[str] = []
if enabled:
```

The broker probes (`callable(getattr(broker, "is_fractionable", None))`,
`:213`; the no-submit classifier, `:214-216`) and the software-stop probe
(`:219`) all sit under `if enabled:`. With the flag off the gate returns
`{"contract": "fractional-v2-stage0", "enabled": False, "ok": True,
"missing": []}` (`:221-226`) without reading a single broker attribute — so
adding attributes to the broker cannot change any flag-off decision. Even
with the flag ON the gate is a structural callability probe, not an asset
lookup (`TestFlagOffIsInert::test_gate_flag_on_is_a_structural_probe_not_a_lookup`
pins this with a client whose every attribute access raises). No code in the
umbrella order path calls `is_fractionable` / `is_no_submit_status`; they are
only read by the gate. `software_stops.enabled` is irrelevant to this change
(leg (b) is `software_stops_armed(...)`, `commit_contract.py:118-131`, which
this PR does not touch).

## Evidence

All runs with `/Users/renhao/git/github/RenQuant/.venv/bin/python -m pytest
-q` from the worktree (pytest.ini supplies `-n auto` and
`RENQUANT_NO_NOTIFY=1`). [VERIFIED]

* New file alone: `tests/test_live_broker_fractional_contract.py` — **42 passed**.
  Covers: the gate's literal probes on the class; the real
  `fractional_capability_gate` with the flag ON reporting `missing == []`
  against a real (offline) `AlpacaBroker` with armed stops, and
  `missing == ["software_stop_layer"]` with unarmed stops (leg (a) alone is
  satisfied); a bare `BaseBroker` subclass still failing leg (a); flag-off
  inertness against an exploding client for 5 config shapes; `is_fractionable`
  True/False cached, missing-field → confirmed False, lookup exception → False
  and re-queried on the next call (then cached once it succeeds), not-connected
  → False uncached, disconnect not poisoning the cache, case-insensitive key,
  typed error with `__cause__`; `is_no_submit_status` on 6 vocabulary members
  + 12 non-members, whitespace/case normalisation, every vocabulary entry
  round-trips; and the drift tripwire `_FALLBACK_NO_SUBMIT_STATUSES ==
  renquant_execution.broker.NO_SUBMIT_STATUSES` via the sibling checkout.
* Targeted set — the new file + `tests/test_s_frac_stage0_commit_contract.py`
  + every test file that imports `live/alpaca_broker` (`test_live_alpaca_bounded_timeout`,
  `test_broker_side_stops`, `test_agent_breaker`, `test_round3_audit_fixes_2026_04_25`,
  `test_p0_fixes_regression_guards`, `test_runner_state_fixes`,
  `test_analyze_decision_factors`, `test_state_ext_sell_fill_attribution`,
  `test_live_multirepo_entrypoints`): **421 passed, 4 failed**.
  The 4 failures are **pre-existing** — the identical 4 fail on a pristine
  detached `origin/main` (`31f2a4a0`) worktree with the same command:
  `test_agent_breaker.py::TestAlpacaBrokerWiring::test_order_cap_blocks_before_api`
  (breaker did not trip after the test pins `_day = date.today()`; wall-clock
  dependent — cause [GUESS]) and three
  `test_live_multirepo_entrypoints.py` source-text assertions against the
  PINNED orchestrator sibling checkout (`_locked_subrepo_source`), which do not
  read any file this PR touches.
* Full umbrella suite (`pytest -q tests`, xdist `-n auto`), branch worktree:
  **15308 passed, 60 failed, 7784 skipped, 3 xfailed, 2 collection errors in
  81s**. Same command on a pristine detached `origin/main` (`31f2a4a0`)
  worktree: 15236 passed, 81 failed, 7793 skipped, 2 collection errors in
  77s. Set difference of `FAILED`/`ERROR` ids: **zero tests fail on the
  branch that pass on main**; 21 tests failed on the main run and passed on
  the branch run (xdist-order / environment flakes in `test_sim_walkforward`,
  `test_preflight`, `test_umbrella_gates_ledger`, ... — cause [GUESS], not
  attributable to this change). The 60 branch failures are the same
  pre-existing set (data-coverage baselines, production-config assertions,
  sibling-checkout source-text pins, ...), none in a file this PR touches
  other than the 4 already characterised above.
* Vocabulary source, both ways [VERIFIED]: bare venv →
  `NO_SUBMIT_VOCABULARY_SOURCE == "local_fallback"`, 10 statuses; with
  `PYTHONPATH=<renquant-execution>/src:<renquant-common>/src` →
  `"renquant_execution"`, `NO_SUBMIT_STATUSES == owner` and
  `_FALLBACK_NO_SUBMIT_STATUSES == owner` both `True`.

CI note: the umbrella's workflows each name specific test files
(`.github/workflows/*.yml`); none runs the whole `tests/` directory, so the
new file runs in the operator's local suite but is not named by any workflow
yet. Naming it in a workflow is a separate, reviewable choice.

## Place in the fractional dependency chain

This is **step 1 of 8**. The remaining steps, each its own PR, in order:

2. **Truncation fix** — `live/alpaca_broker.py` `place_order` /
   `place_stop_order` stop casting `qty=int(quantity)`; fractional qty on
   TIF=DAY market orders, whole-share GTC stops unchanged.
3. **Contract test** — end-to-end test that a fractional BUY through the real
   commit path reaches the live broker class un-truncated.
4. **`fractional_max_book_pct`** — the sizing cap for fractional exposure.
5. **Pager install** — the operator alerting hook for the software-stop layer.
6. **Evidence packet** — the §9-style authorization package for the flag flip.
7. **`execution.software_stops` config row** (leg (b) of the gate).
8. **`execution.fractional_shares` flag row** — the last row; the gate then
   admits fractional BUYs only if steps 1-7 are all live.

## Memory tier

No LONG/MID constraint changes. SHORT: the live broker satisfies gate leg (a)
as of this branch; leg (b) (`software_stops_armed`) remains unmet by design.
