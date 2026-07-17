# Pin bump: renquant-orchestrator → Alpaca retry + control-plane fixes

Date: 2026-07-16

## What

`renquant-orchestrator` `bfb935e4` → `511edfa9` (5 commits): #522 (rq105
session-scheduler transient Alpaca API retry — RFC-9110 Retry-After both
forms, ContextVar per-invocation timeout, fail-loud unsupported callables;
closes the 07-13/14 `halted_tick_error` mid-session crash class), #519
(review-queue predicate fixes), #521 (F-8 docs), #517 (single-identity PR
guard).

## Why

The rq105 session scheduler crashed mid-session on 07-13/14 from transient
Alpaca timeouts — exactly the class #522 handles. The 105 diagnosis
(2026-07-16) confirmed no independent 105 breakage; deploying this closes
the remaining crash class before tomorrow's 06:25 PT session.

Strategy-104 snapshot is unaffected (orchestrator pin is not a snapshot
input beyond the lock table row, regenerated here).
