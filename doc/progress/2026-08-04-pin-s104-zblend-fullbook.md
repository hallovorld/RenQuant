# 2026-08-04 — pin advance 4: strategy-104 → 2358b56b (OPERATOR OVERRIDE: z-blend full book)

Carries s104#88: prod primary scorer → zblend(reversal + slow momentum), full
book, under the operator's explicit repeated directive and blast-radius choice
(整本切换). Authority, zero-OOS disclosure, rollback, and review condition
live in `configs/zblend_prod_artifact_manifest.json` (s104) and the s104
progress doc; GOAL-9 registration orch#794.

- `subrepos.lock.json`: renquant-strategy-104 `547fc49b` → `2358b56b` (sha
  read back from merge output)
- snapshot re-rendered at the new pin (blend primary now visible in the
  production table)

Pre-merge verification already on record (s104#88): suite 97 passed + readonly
FULL-FUNNEL sim on the exact config (rc=0, zero hard fails, governance PASS,
GOOG+VLO decisions, pending orders excluded).

Deploy after merge = batch 5 (grants-logged): live pull + runtime s104 sync.
Tomorrow 13:55 PT = the first full-book z-blend run.
