# Execmath fractional-aware cash cap — the last buy-path int truncation closed

STATUS: implemented, flag-conditional, default-inert. Closes **D7 gap
inventory item #1** (S-FRAC v2 audit; see renquant-orchestrator PR #444):
after stage 0 made the commit path float-preserving end-to-end
(`commit_contract.py`, `normalize_fill_qty`), ONE residual int truncation
remained on the buy path — `cap_buy_order_to_cash`
(`adapters/runner_execmath.py`): `affordable = int(cash // price)`. Under
fractional sizing (S-FRAC v2 stage 2, `execution.fractional_shares.enabled`)
a buy whose cost exceeds available cash was silently resized to WHOLE shares
— e.g. $50 cash against a $100 name truncated 0.5 affordable shares to 0 and
rejected the order outright.

## Ownership (reworked per this PR's review)

The sizing math is NOT owned here. Per the codex review of this PR (the
umbrella is being deprecated and must not gain new execution capability),
the implementation, its behavior pins, and the D7 #1 semantics moved to
**renquant-execution `order_math.cap_affordable_qty`
([execution#25](https://github.com/hallovorld/renquant-execution/pull/25))**.
The umbrella surface is a **time-bounded compatibility CALL-SITE only**:
`cap_buy_order_to_cash` keeps the runner's order-intent envelope (afford
epsilon, reason strings, resized-dict shape) and delegates BOTH modes —
whole-share and fractional — to the owner function.

* **Missing-owner fallback (fail-closed, tested)**: the umbrella imports
  renquant-execution via the pinned checkout on the live PYTHONPATH; a pin
  that predates `order_math` must not crash the commit path. On
  `ImportError` the call-site degrades: flag-off uses the inline legacy
  `int(cash // price)` (byte-identical — the ONE sanctioned truncation, on
  the auditor's flag-off `else` arm); `fractional=True` logs
  `EXECMATH-CASHCAP-FALLBACK` and degrades to the same conservative
  whole-share cap — never umbrella-local fractional math.
* **Sunset**: this call-site is deleted when RunnerAdapter order math moves
  into renquant-execution under the adapter-migration program;
  **renquant-execution owns the cutover** (the delegate function is already
  the single implementation, so deletion is call-site removal + pin bump,
  no math relocation).

## What changed

1. **`cap_buy_order_to_cash(order, remaining_cash, *, fractional=False)`**
   (`adapters/runner_execmath.py`) — explicit flag parameter; both modes
   delegate to renquant-execution `order_math.cap_affordable_qty`
   (execution#25) with the fail-closed fallback above.
   * `fractional=False` (the live default): byte-identical legacy behavior —
     `int(cash // price)`, reject below 1 whole share. Pinned by a frozen
     verbatim copy of the legacy implementation swept over the seeded input
     grid under BOTH wirings — owner delegate present AND missing-owner
     fallback (`tests/test_runner_execmath_invariants.py::TestCapBuyOrderToCashFractional::
     test_flag_off_is_byte_identical_to_frozen_legacy`), including the `int`
     type of resized shares.
   * `fractional=True`: affordable quantity floors to 6 decimal places —
     `floor(cash / price · 1e6) / 1e6` — the same quantization convention as
     renquant-pipeline `kernel/sizing.py::compute_position_size` (stage 2).
     Flooring (never round-to-nearest) keeps realized notional ≤ cash. A
     capped quantity whose notional lands below the ~$1 broker fractional
     minimum (`MIN_FRACTIONAL_NOTIONAL_USD` — now imported from the owner
     repo's `broker.py`, no longer a value-parity copy here) rejects as
     `cash_budget_exhausted` — the fractional analog of the whole-share
     `affordable < 1` reject. The $25 anti-churn dust floor is a sizing-time
     ENTRY convention (pipeline `fractional_dust_floor_usd`) and
     deliberately does not re-apply to a budget resize of an
     already-admitted intent.
2. **Caller wiring** (`adapters/runner.py`, buy loop in `commit`) — the flag
   comes from the SAME source of truth the commit path already uses:
   `frac_gate = fractional_capability_gate(...)` stamped at the top of
   `commit`. `fractional = frac_gate["enabled"] and frac_gate["ok"]` — flag
   off ⇒ legacy truncation; gate enabled-but-unsatisfied ⇒ outcome-neutral
   (every BUY fail-closes at `fractional_entry_fail_closed_reason` anyway).
   A fractionally-resized quantity then flows through the existing stage-0
   protections (fail-closed entry, stop routing, `normalize_fill_qty`).
3. **Truncation auditor extended**
   (`scripts/check_commit_path_no_int_truncation.py`) — beyond fill-quantity
   casts, it now flags `int()` casts on ORDER-SIZING quantities (`shares`,
   `affordable`, `cash`, `price`, ...) in `runner_execmath.py`. The ONE
   sanctioned appearance is recognized STRUCTURALLY: the `else` arm of an
   `if fractional:` split (the flag-off legacy branch) — not a function
   allowlist — so a planted `int()` on the flag-ON branch, or anywhere else
   in the module, fails the audit. Self-tested by re-deriving the current
   module with the flag-ON delegate call swapped back to `int(cash // price)`
   and asserting the audit fails. The inline fallback's legacy cast sits on
   the sanctioned flag-off `else` arm and stays audit-clean.

## Test evidence

Behavior pins for the math itself now live in the OWNER repo
(execution#25 `tests/test_order_math.py`, incl. the 4000-case byte-identity
grid). The umbrella keeps call-site pins:

* `tests/test_runner_execmath.py` — delegation examples (6dp floor value,
  sub-1-share resize admitted — the exact D7 #1 gap, sub-$1 reject,
  flag-off default = legacy int truncation) plus the missing-owner
  fallback pins: `fractional=True` without `order_math` → legacy
  whole-share result + `EXECMATH-CASHCAP-FALLBACK` warning, never a crash;
  flag-off silent and unchanged.
* `tests/test_runner_execmath_invariants.py` — seeded-grid sweeps (4000
  cases/property): flag-off byte-identity vs the frozen legacy copy under
  BOTH wirings (delegate present / fallback); fractional fallback
  never-crashes-and-stays-whole-share over the full grid; delegated
  fractional mode: never-overspend, resize lands on the 1e-6 grid, shrinks,
  clears the $1 minimum, reject ⇔ below-$1 notional, monotone
  non-decreasing in cash.
* `tests/test_s_frac_stage0_commit_contract.py` — commit-level wiring:
  flag-on E2E resize (1 share @ $100 with $50 cash → broker receives 0.5,
  via the owner delegate), the missing-owner E2E counterpart (same case
  fails closed to a whole-share reject + logged warning), flag-off E2E
  counterpart (whole-share rejected as legacy); auditor self-tests (planted
  regression on the flag-ON delegate fails, flag-off `else` arm is the only
  sanctioned sizing cast, scope limited to `runner_execmath.py`).
  One existing flag-ON fixture (`test_cash_accounting_is_float_exact`)
  adjusted: its $49 residual was asserted as a whole-share reject, which is
  the exact truncation this change removes — residual moved to $0.99 (below
  the $1 fractional minimum) preserving its discriminating power. All
  flag-OFF regression pins (`TestFlagOffWholeShareRegression`) unchanged.
* Delegate-dependent tests resolve the owner via the pinned
  `subrepos.lock.json` checkout (`tests/_order_math_owner.py`) and SKIP
  when the pin predates execution#25 — they go live with the pin bump.

Runs: touched-area subset (execmath + invariants + stage-0 contract):
89 passed with the owner module on PYTHONPATH (delegate + fallback wirings
both exercised); 79 passed / 10 skipped without it (fallback wirings only —
the pre-pin-bump CI state); adjacent runner subset (state-fixes + lean-order
+ z9) 110 passed; auditor exit 0.

## Repo-ownership note (time-bounded migration exception)

This fix lands in the umbrella tree because the bug it closes lives in
`adapters/runner_execmath.py`, and the entire `RunnerAdapter` order-math
layer is umbrella-resident legacy. Per Codex review on renquant-orchestrator
PR #444 (the D7 gap-inventory audit this fix closes item #1 of): **this is a
TIME-BOUNDED MIGRATION EXCEPTION, not a proposed architecture.** The target
owner for execution math is `renquant-execution`; the removal plan is the
adapter-migration program (moving `RunnerAdapter` order math, including this
module, into that repo). Until that migration lands, any further change to
umbrella-resident order math must carry this same exception label and must
not add new umbrella-owned capability beyond what's needed to close a
specific S-FRAC v2 contract gap — after the rework this PR adds NONE:
`cap_buy_order_to_cash` gains a flag parameter and delegates both modes to
the renquant-execution owner (execution#25); the umbrella holds zero
fractional sizing math (see the Ownership section above).

## Follow-ups

* Merge execution#25 first, then bump the renquant-execution pin so the
  live delegate resolves (until then the call-site runs its fail-closed
  legacy fallback — behaviorally identical to pre-D7 for everything the
  flag-off production path does).
* Delete this call-site when RunnerAdapter order math migrates to
  renquant-execution (adapter-migration program; renquant-execution owns
  the cutover).
* Stage-2 activation (turning the flag on in strategy-104) remains a
  separate operator decision gated on stage 3 (software stops) per the
  S-FRAC v2 design.
