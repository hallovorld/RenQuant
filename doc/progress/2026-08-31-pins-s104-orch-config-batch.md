# Pin advance: strategy-104 79982124 + orchestrator 34336b16

STATUS: pin advance (subrepos.lock.json), two repos in one batch.

## strategy-104: d3c8026a -> 79982124

Commits included (newest first):

- **s104#105** `79982124` — `rotation.enabled=false` in active + golden +
  6 prod-mirror lanes (orch LONG-ledger row 2e, AUTHORIZATION COMPLETE
  2026-08-30 12:09 PDT). New pin
  `test_rotation_engine_is_disabled_until_validated`. Frozen arms untouched.
- **s104#106** `df74602` — `execution.buying_power_mode=settled_cash` in
  active + golden + 6 prod-mirror lanes (orch LONG-ledger row 2f,
  AUTHORIZATION COMPLETE 2026-08-30 12:09 PDT). New pin
  `test_buys_are_sized_on_settled_cash_never_margin`. Frozen arms untouched.

Both config changes carry first-hand, change-specific operator confirmation
(agent prompt naming both keys explicitly, operator verbatim reply).

**Ordering note (row 2f):** RenQuant#624 (buying_power_mode adapter fix)
MERGED before this pin advance — the adapter now honours the config key,
so sim and live are in ONE mode after this pin lands.

## orchestrator: 64238032 -> 34336b16

Commits included (top of the range):

- **orch#1097** `34336b16` — LONG-ledger row 2f (settled_cash authority,
  AUTHORIZATION COMPLETE).
- **orch#1095** `b432c1f8` — LONG-ledger row 2e (rotation OFF authority,
  AUTHORIZATION COMPLETE).
- **orch#1106** `dc7846b5` — GOAL-1 AC2/AC4 closeout research.
- Intermediate: AC4 r2 staged plan.

## Deploy

Normal ordered path — this PR advances the pins in subrepos.lock.json.
The live tree fast-forward is a separate reviewed step after this merges.

## Supersedes

RQ#626 (orch pin at f89f1519 only, stale — did not include row 2e/2f
authority commits or s104 config changes).

§4b: pin advance only (subrepos.lock.json), no code change in this repo.

## backtesting: e5f9bae3 -> d845da9e

- **bt#115** — WF-gate freshness fallback requires genuine_ic >= 0.02
  quality floor (RFC#210 Amendment A4). Carried forward from #626.

## model: bd0fa488 -> 36085810

- **model#229** — LATEST_MODELS refresher never writes into a pinned
  runtime checkout. Carried forward from #626.
