# Pin bump: governed diagnostic-only buy admission (pipeline #203 + strategy-104 #59)

Date: 2026-07-16

## What

- `renquant-pipeline`: `28be76c6` → `7108f514` (exactly 1 commit: #203, the
  governed operator-override mechanism for diagnostic-only buy admission —
  fail-closed validator, both enforcement points, expiry, scorer binding)
- `renquant-strategy-104`: `0e5d9891` → `0d45d960` (6 commits: #59 the
  operator authorization record [expires 2026-08-15, bound to
  sha256:656b70be…], plus three previously merged default-OFF/shadow-only
  config contracts: fractional contract redo #54, sleeve shadow logging
  #57, one-share floor staging #55 — all flags OFF, behavior-neutral)

## Why

Resolves the 07-16 incident's final layer: the 07-15 admission gate blocks
all buys for diagnostic-only scorers with no override path; the book
drained to 94% cash. This deploys the governed alternative — explicit
operator identity, hard expiry, single-scorer content binding, full audit
provenance — per the operator's decision (option b, task #62).

## Verification plan

Post-sync full daily rerun must show: P-WF-GATE PASS with override
provenance, buy funnel open, VetoWeakBuys and all economic gates unchanged.
