# Progress — disposed-lot tax netting fix (kernel/portfolio.py)

**Date:** 2026-07-27. **Type:** accounting bug fix in the shared sim/live
kernel. **Priority:** HIGH — latent LIVE fail-close risk.
**Pairs with:** the identical mirror fix in renquant-pipeline
(`src/renquant_pipeline/kernel/portfolio.py`, duplicated-kernel class).

## STATUS
delivered (PR open; merge ordering note below)

## BOTTOM LINE
`compute_disposed_lot_tax` taxed each positive-gain lot independently and
never netted losing lots within the same sell event. A mixed-sign multi-lot
disposal (top-up lot + original lot, full exit at a price between the two
bases) produced "net loss with positive tax" — an accounting impossibility
that trips the decision-trace integrity validator
`_sell_economics_are_valid` (fail-closed RuntimeError). Fix: net gains and
losses per rate bucket within the one sell event by reusing
`compute_netted_capital_gains_tax` (same file). The validator is CORRECT
and untouched.

## VERIFIED INSTANCE [VERIFIED]
MA 2025-06-24 sim sell: lot +126.9676 gain (taxed at 0.5 → 63.4838) + lot
−193.2083 loss (ignored) → gross −66.2407 with tax +63.4838. Found by the
G4 rerun batch — the first execution of the persistence-ON validation path
over a full window (the weekly gate's `--no-persist` never exercises it).
Deterministically reproduced; pinned as a regression test, including the
assertion that the PRE-fix triple fails `_sell_economics_are_valid`.

## LIVE EXPOSURE
This kernel copy is the RUNTIME for both sim and LIVE
(`adapters/runner_tax_lots.py:165,184`, `adapters/sim.py`,
`adapters/lean.py`). The live daily runner calls
`validate_decision_trace_integrity` (`adapters/runner.py:2351`), so the
latent bug is a LIVE fail-close risk for any full exit of a topped-up
position at a price between the lot bases.

## NEW TAX SEMANTICS
Exactly `compute_netted_capital_gains_tax`'s Schedule-D shape, applied
within the one sell event:
1. short-term lots net against short-term lots; long-term against
   long-term;
2. both buckets non-negative → `st_net·st_rate + lt_net·lt_rate` (each
   bucket taxed at its own rate — identical to pre-fix behavior for
   all-gain events);
3. both non-positive → 0 (identical to pre-fix for all-loss events);
4. opposite signs → `max(0, st_net + lt_net) ×` the gaining bucket's rate.
Cross-bucket offset (rule 4) is REQUIRED for validator safety: with
ST +100 / LT −150 the event gross is −50, so any positive tax would
recreate the "loss with positive tax" trip. Structurally guarantees both
validator invariants: loss ⇒ tax 0, and tax ≤ positive gross (rates ≤ 1).
The returned `short_term_gross_pnl` / `long_term_gross_pnl` splits stay
pure per-bucket sums (unchanged; annual-net reporting unaffected).
This also fixes the sibling failure: a net-GAIN mixed sell whose per-lot
tax exceeded net gross (validator invariant #4).

## TESTS [VERIFIED]
- NEW `tests/test_disposed_lot_tax_netting.py` (16 tests): the exact MA lot
  pair (tax MUST be 0.0, net = gross); pre-fix triple fails the validator;
  net-gain mixed case (tax = 0.5 × netted sum, ≤ invariant-#4 bound,
  including the deeper-loss case where pre-fix tax exceeded net gross);
  all-gain + all-loss regression (behavior identical to pre-fix); ST/LT
  bucket-rate separation; cross-bucket cases asserted EQUAL to
  `compute_netted_capital_gains_tax` outputs; validator run directly on
  fixed outputs.
- Scoped owning suite (netting + kernel_units + runner_tax_lots(+invariants)
  + tax_lots_g7 + short_cover_tax + persistence + partial_sell +
  kernel_parity vs the fixed pipeline copy): **272 passed, 0 failed**.
- Full suite in an isolated worktree: 14996 passed / 67 failed vs clean
  origin/main baseline 14949 passed / 99 failed in the same env — every
  fixed-branch failure either also fails on the clean baseline or passes
  when rerun serially (xdist/env flakiness; missing live artifacts). Zero
  regressions attributable to this change.

## PARITY / MERGE ORDERING
`portfolio.py` is NOT in the kernel-parity allowlist; the two copies are
byte-identical post-fix (`check_kernel_parity.py`: 78 identical, 91
allowed-drift, 0 NEW drift when pointed at the fixed pipeline branch).
kernel-parity-ci compares this PR against renquant-pipeline at the
`subrepos.lock.json` pin, so it stays RED until the mirror PR merges and
the pipeline pin advances past it — merge the pipeline mirror first, bump
the pin through the normal pin process, then re-run CI here. Do NOT
allowlist portfolio.py to silence it.
