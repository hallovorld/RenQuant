# 2026-05-23 QP Sector Metadata Audit

## Scope

Audited `renquant_104` portfolio QP decision layer:

- `kernel/portfolio_qp/job_qp.py`
- `kernel/portfolio_qp/tasks.py`
- `kernel/portfolio_qp/qp_solver.py`
- QP interaction with sector metadata, order attribution, and decision trace.

## Finding

The QP sector-cap matrix can only constrain tickers that have a sector row.
Before this fix, a ticker missing from `sector_map` was excluded from the
sector indicator matrix and could therefore receive new QP allocation if it
entered through a direct candidate path or existing broker holding state.

This is the same bug class as the BAC/WFC/D audit: missing metadata must not
become an implicit permission to add risk.

## Fix

Added `ApplySectorMetadataGuardTask` before sector/correlation constraints.
When `qp_sector_cap_enabled=true` and `sector_map` exists:

- unmapped new candidates get max QP weight `0`;
- unmapped existing holdings are capped at current weight;
- QP may hold or reduce an unmapped holding, but cannot increase it;
- candidate decision trace gets `blocked_by=missing_sector_map`.

The guard intentionally no-ops when no `sector_map` exists, preserving legacy
unit tests and explicit no-sector research paths. Production preflight remains
responsible for rejecting missing production metadata.

## Additional Fix

QP buy attribution now stamps the real emitting task:

`JointPortfolioQPJob.EmitOrdersFromQPSolutionTask`

Previously buys reported the compatibility shim `JointPortfolioQPTask`, which
made decision-tree tracing less precise.

## Second-Pass Finding

The first guard closed the obvious missing-sector allocation bug, but a deeper
review found another constraint-contamination path:

- `BuildSectorConstraintMatrixTask` anchored sector caps on
  `max(_qp_w_upper)` across all QP tickers;
- `BuildCorrelationGroupConstraintTask` used the same global max to set every
  high-correlation pair cap;
- a large unmapped broker holding could therefore avoid the sector row while
  still inflating mapped sector and correlation caps.

This is not a cvxpy solver issue. It is bad metadata leaking into constraint
construction.

Fix:

- sector caps now anchor only on tickers that actually belong to mapped sector
  rows;
- correlation pair caps now use the two assets' own upper bounds
  (`w_upper[i] + w_upper[j]`) instead of a global outlier max.

This matches the mature optimizer pattern: group constraints are functions of
group membership, not of unrelated assets outside the group.

## Production Data Repair

Active/golden/shadow configs now include sector metadata for:

- `SPY -> benchmark`
- `TLT -> defensive_bonds`
- `XLV -> healthcare`

And sector ETF metadata for:

- `benchmark -> SPY`
- `defensive_bonds -> TLT`

Coverage check after repair:

- missing `sector_map`: `0`
- missing `sector_etf_map`: `0`

## Regression Tests

Added complex QP regression coverage in `tests/test_qp_sector_constraint.py`:

- direct Task test caps missing-sector weights at current weight;
- full QP blocks a missing-sector new candidate even when it has the highest
  `mu`;
- full QP blocks top-up of a missing-sector existing holding;
- mixed book test: missing-sector names and non-BEAR defensive buys are
  suppressed, mapped positive-alpha candidates can buy, and mapped negative
  holdings can sell.
- sector caps are not inflated by an unmapped high-cap holding.

Added correlation-cap regression coverage in
`tests/test_qp_correlation_constraint.py`:

- high-correlation pair caps use pair-local upper bounds, not a global outlier.

Validation:

- QP sector/correlation suite: `29 passed`
- QP suite: `260 passed, 4 skipped`
- Config/QP focused suite: `116 passed`
- Full suite: `12646 passed, 8791 skipped, 1 xfailed`

Preflight after repair:

- sell-only: `P-SECTOR-MAP hard True`
- full: `P-SECTOR-MAP hard True`
- full still correctly hard-fails on `P-WF-GATE` because the active model
  artifact carries failed WF evidence. That is a model-promotion blocker, not
  a remaining QP sector-metadata bug.
