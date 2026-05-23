# 2026-05-23 QP Tax Soft-Sell Contract

## Finding

`rotation.joint_actions.qp_tax_aware` was set to `false`, but QP order
emission still called the shared LT/tax soft-exit guards. That meant the solver
used a zero tax-cost vector while the later sell-emission stage could still
block QP trims/closes for tax reasons.

This was a hidden decision-tree inconsistency:

- QP solve: tax-unaware.
- QP order emission: tax-aware through `risk.panel_exit.tax_adjusted_soft_exit`
  and root `lt_hold_gate_days`.

## Fix

QP soft-sell behavior is now explicit:

- QP still applies the thesis-age horizon guard. This is not tax logic.
- QP applies LT/tax suppression only when `qp_tax_aware = true`, or when an
  explicit `qp_soft_sell_guard.apply_tax_gates = true` override is set.
- Production config stamps `qp_soft_sell_guard.apply_tax_gates = false`.

This preserves the user's `tax = reporting only, no QP decision logic` mandate
while leaving an explicit research switch for a future tax-aware QP A/B.

## Literature Support

- Shefrin and Statman (1985), "The Disposition to Sell Winners Too Early and
  Ride Losers Too Long": tax-aware hold/sell rules can mechanically encode the
  disposition-effect failure mode if they are not explicitly justified.
  Reference: https://doi.org/10.1111/j.1540-6261.1985.tb05002.x
- Odean (1998), "Are Investors Reluctant to Realize Their Losses?": empirical
  evidence that investors realize gains more readily than losses; a QP gate
  that defers winner exits needs direct performance proof, not an implicit
  default.
  Reference: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=94142
- Magill and Constantinides (1976), "Portfolio selection with transaction
  costs": transaction costs justify no-trade regions, but they must be part of
  the stated optimizer objective/contract, not a hidden post-solve veto.
  Reference: https://doi.org/10.1016/0022-0531(76)90018-1

## Regression Tests

Updated `tests/test_joint_qp_task.py::TestQPSoftSellGuard`:

- `qp_tax_aware=false` ignores tax-drag suppression for QP trims.
- `qp_tax_aware=false` ignores LT tax gate suppression for QP trims.
- `qp_tax_aware=true` still blocks weak short-term winner trims.
- `qp_tax_aware=true` still respects the LT tax gate.
- The horizon gate still blocks young QP trims.

Focused verification:

- `pytest tests/test_joint_qp_task.py::TestQPSoftSellGuard -q`
  -> 6 passed
