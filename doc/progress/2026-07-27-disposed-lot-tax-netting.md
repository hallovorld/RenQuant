# Disposed-lot tax netting fix (kernel/portfolio.py)   (PR #532)

STATUS:    delivered
WHAT:      `compute_disposed_lot_tax` taxed each positive-gain lot
independently and never netted losing lots within the same sell event,
producing "net loss with positive tax" (an accounting impossibility) on
mixed-sign multi-lot disposals. Fixed by netting gains/losses per rate
bucket (short-term vs long-term) within the one sell event, reusing
`compute_netted_capital_gains_tax` (same file, single source of truth for
netting semantics, Schedule-D shape). The decision-trace integrity
validator `_sell_economics_are_valid` is correct and untouched.
WHY/DIR:   Latent LIVE fail-close risk. This kernel copy is the runtime for
both sim and live (`adapters/runner_tax_lots.py:165,184`, `adapters/sim.py`,
`adapters/lean.py`); the live daily runner calls
`validate_decision_trace_integrity` (`adapters/runner.py:2351`), so any full
exit of a topped-up live position at a price between the two lot bases
would fail-close the daily run. Found via the G4 rerun batch — the first
execution of the persistence-ON validation path over a full window (the
weekly gate's `--no-persist` never exercises it). Pairs with the identical
mirror fix in renquant-pipeline (`fix/disposed-lot-tax-netting-mirror`,
PR #217) — duplicated-kernel class per the triple-impl playbook.
EVIDENCE:  artifact: tests/test_disposed_lot_tax_netting.py (16 new tests,
deterministic MA 2025-06-24 reproduction: lot +126.9676 gain taxed at
0.5->63.4838 + lot -193.2083 loss ignored -> gross -66.2407 with tax
+63.4838 pre-fix; post-fix tax=0.0, net=gross). prod or exp: kernel
correctness fix, not a model/data performance claim — no IC/Sharpe number
involved. existing data: scoped owning suite (netting + kernel_units +
runner_tax_lots(+invariants) + tax_lots_g7 + short_cover_tax + persistence +
partial_sell + kernel_parity vs the fixed pipeline branch) = 272 passed, 0
failed. Full suite in an isolated worktree: 14996 passed / 67 failed vs
clean origin/main baseline 14949 passed / 99 failed in the same env — every
fixed-branch failure either also fails on the clean baseline or passes when
rerun serially (xdist/env flakiness, missing live artifacts); zero
regressions attributable to this change. best-known?: n/a — bug fix, no
variant comparison. scope: this is a correctness fix to the shared sim/live
kernel verified via the paired scoped + full-suite runs above, not a
performance/model claim.
NEXT:      Done — the renquant-pipeline mirror merged (PR #217, commit
`cb38a73`), and this PR now carries the pin bump via
`scripts/promote_pin.py bump --subrepo renquant-pipeline --commit
dbcab26556a0db474038ea8f9f2a76d85f944c12 --apply` (the merge-commit tip of
renquant-pipeline main, which includes the mirror). Regenerated
`doc/arch/strategy-104-snapshot.md` per the tool's staleness backstop (`make
snapshot`) — diff is exactly the pin-driven fingerprint change plus the
live prod calibrator's independently-drifting artifact fingerprint (both
expected, no other content changed). Re-ran `kernel-parity-ci`'s own test
locally against the freshly-synced pipeline checkout: `pytest
tests/test_kernel_parity.py -v` -> 1 passed (was previously failing with
"NEW drift detected: portfolio.py" because the branch's old pin, 9c5f48e,
predated the mirror fix). No further action needed; nothing left blocking
merge on this PR's side.

## Additional detail — new tax semantics

Exactly `compute_netted_capital_gains_tax`'s Schedule-D shape, applied
within the one sell event:
1. short-term lots net against short-term lots; long-term against
   long-term;
2. both buckets non-negative -> `st_net*st_rate + lt_net*lt_rate` (each
   bucket taxed at its own rate — identical to pre-fix behavior for
   all-gain events);
3. both non-positive -> 0 (identical to pre-fix for all-loss events);
4. opposite signs -> `max(0, st_net + lt_net) *` the gaining bucket's rate.

Cross-bucket offset (rule 4) is REQUIRED for validator safety: with
ST +100 / LT -150 the event gross is -50, so any positive tax would
recreate the "loss with positive tax" trip. Structurally guarantees both
validator invariants: loss => tax 0, and tax <= positive gross (rates <= 1).
The returned `short_term_gross_pnl` / `long_term_gross_pnl` splits stay
pure per-bucket sums (unchanged; annual-net reporting unaffected). This
also fixes the sibling failure: a net-GAIN mixed sell whose per-lot tax
exceeded net gross (validator invariant #4).

## Additional detail — parity / merge ordering

`portfolio.py` is NOT in the kernel-parity allowlist; the two copies are
byte-identical post-fix (`check_kernel_parity.py`: 78 identical, 91
allowed-drift, 0 NEW drift when pointed at the fixed pipeline branch).
kernel-parity-ci compares this PR against renquant-pipeline at the
`subrepos.lock.json` pin, so it stays RED until the mirror PR merges and
the pipeline pin advances past it.
