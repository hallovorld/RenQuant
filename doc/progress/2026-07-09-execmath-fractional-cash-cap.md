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

## What changed

1. **`cap_buy_order_to_cash(order, remaining_cash, *, fractional=False)`**
   (`adapters/runner_execmath.py`) — explicit flag parameter.
   * `fractional=False` (the live default): byte-identical legacy behavior —
     `int(cash // price)`, reject below 1 whole share. Pinned by a frozen
     verbatim copy of the legacy implementation swept over the seeded input
     grid (`tests/test_runner_execmath_invariants.py::TestCapBuyOrderToCashFractional::
     test_flag_off_is_byte_identical_to_frozen_legacy`), including the `int`
     type of resized shares.
   * `fractional=True`: affordable quantity floors to 6 decimal places —
     `floor(cash / price · 1e6) / 1e6` — the same quantization convention as
     renquant-pipeline `kernel/sizing.py::compute_position_size` (stage 2).
     Flooring (never round-to-nearest) keeps realized notional ≤ cash. A
     capped quantity whose notional lands below the ~$1 broker fractional
     minimum (`MIN_FRACTIONAL_NOTIONAL_USD`, value-parity with the pipeline /
     execution#22) rejects as `cash_budget_exhausted` — the fractional analog
     of the whole-share `affordable < 1` reject. The $25 anti-churn dust
     floor is a sizing-time ENTRY convention (pipeline
     `fractional_dust_floor_usd`) and deliberately does not re-apply to a
     budget resize of an already-admitted intent.
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
   module with the 6dp floor swapped back to `int(cash // price)` and
   asserting the audit fails.

## Test evidence

* `tests/test_runner_execmath.py` — example pins: 6dp floor value, sub-1-share
  resize admitted (the exact D7 #1 gap), sub-$1 reject, flag-off default =
  legacy int truncation.
* `tests/test_runner_execmath_invariants.py` — seeded-grid sweeps (4000
  cases/property): flag-off byte-identity vs the frozen legacy copy;
  never-overspend under `fractional=True`; resize lands on the 1e-6 grid,
  shrinks, and clears the $1 minimum; reject ⇔ below-$1 notional; monotone
  non-decreasing in cash.
* `tests/test_s_frac_stage0_commit_contract.py` — commit-level wiring:
  flag-on E2E resize (1 share @ $100 with $50 cash → broker receives 0.5),
  flag-off E2E counterpart (same order whole-share rejected as legacy);
  auditor self-tests (planted regression fails, flag-off `else` arm is the
  only sanctioned sizing cast, scope limited to `runner_execmath.py`).
  One existing flag-ON fixture (`test_cash_accounting_is_float_exact`)
  adjusted: its $49 residual was asserted as a whole-share reject, which is
  the exact truncation this change removes — residual moved to $0.99 (below
  the $1 fractional minimum) preserving its discriminating power. All
  flag-OFF regression pins (`TestFlagOffWholeShareRegression`) unchanged.

Runs: touched-area subset (execmath + invariants + state-fixes + stage-0
contract) 151 passed; wide commit-path subset (runner*/z9*/broker*/
execution*/e2e) 497 passed, 5 failed — all 5 reproduce on pristine `main`
in this environment (runner_artifacts earnings/corr fixtures +
trade_ntfy source-level test; unrelated); auditor exit 0.

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
specific S-FRAC v2 contract gap — this PR adds none: `cap_buy_order_to_cash`
gains a flag parameter to close the one remaining truncation, nothing else.

## Follow-ups

None for this slice beyond the migration-exception note above. Stage-2
activation (turning the flag on in strategy-104) remains a separate operator
decision gated on stage 3 (software stops) per the S-FRAC v2 design.
