# 2026-07-10 — GOAL-1 batch pin bump: pipeline / strategy-104 / execution / orchestrator

## Bottom line

Durable record of the operator-machine live deploy already executed today via
`scripts/promote_pin.py` (all four subrepo runtimes synced + e2e-verified
before this PR). This PR converges git with the live tree: the committed
`subrepos.lock.json` is **byte-identical** to the live tree's (sha256
`13932efb…c1c9a0`), so the live-tree `git pull` after merge is a content
no-op. Follows the merged #455 pattern (lock + regenerated snapshot +
progress doc).

## Pin delta (four repos, all `main`)

| Repo | Old pin | New pin |
|---|---|---|
| renquant-pipeline | `b6139e6a3ad7` | `2b0eb0257b88` |
| renquant-strategy-104 | `8b2a592e53e4` | `0e5d989137b6` |
| renquant-execution | `43a8bdd36539` | `c41639840b2c` |
| renquant-orchestrator | `6a6a1bd371f6` | `e8fe46206025` |

## What ships per repo

| Repo | PRs | Live effect |
|---|---|---|
| renquant-pipeline | #179 Governor D2–D4, #180 governor harness, #181 arm tags | **Inert** — governor stages ship flag-OFF; harness + arm tagging are observational plumbing |
| renquant-strategy-104 | #53 arm configs, #46 fractional, #51 software_stops | **All inert except shadow arm-A** — fractional + software_stops keys default-off in prod/golden; the only armed change is the `strategy_config.shadow.json` arm-A veto flip (shadow-only) |
| renquant-execution | #25 order_math, #26 readonly param | **Inert on prod path** — order_math is the extracted shared math module; readonly param hardens read-only invocations |
| renquant-orchestrator | #451 two-arm runner, #446 freeze tool, #452 backup compression, + 240-commit catch-up to 2026-07-10 main | **Inert for daily prod decisioning** — two-arm runner drives shadow arms only; freeze/backup are ops tooling; catch-up commits were already individually merged through the reviewed-PR gate |

## Production behavior equivalence

All newly shipped behavior flags are **OFF** in the prod and golden configs;
the single armed change is the arm-A veto flip in
`strategy_config.shadow.json`, which only affects the shadow decisioning arm
(observational; never places orders). Prod decisioning inputs are therefore
behavior-equivalent before/after the bump. Snapshot source fingerprints
confirm the prod config change is limited to added default-off keys
(`strategy_config.json` sha256 `587b0afb…` → `2752de27…`, shadow
`ea52e7f2…` → `b7a4332…`).

## Deploy evidence

- Live deploy: `promote_pin.py bump --apply` per repo on the operator
  machine, 2026-07-10; runtime sync + verify green for all four.
- Four e2e verification runs (one after each bump): **each produced a
  committed decision; production paths untouched.**
- Snapshot: `doc/arch/strategy-104-snapshot.md` regenerated from the live
  pinned sources; `--check` (byte-exact), `--verify-pinned-declaration`
  (CI semantic check against the pinned configs + this lock), and renderer
  `--selftest` all **PASS**.
- Pin-related tests in this clone with the live venv:
  `tests/test_promote_pin.py`, `tests/test_subrepo_pin_guard.py`,
  `tests/test_render_strategy_104_snapshot.py` — see PR body for counts.

## Rollback

`scripts/promote_pin.py revert --apply` using the runtime backups from this
deploy (`promote-bak.20260710T09315x` series, one per repo), then revert this
commit.

## References

- Umbrella precedent: #455 (strategy-104 pin bump, merged 2026-07-10)
- renquant-pipeline PRs #179, #180, #181
- renquant-strategy-104 PRs #46, #51, #53
- renquant-execution PRs #25, #26
- renquant-orchestrator PRs #446, #451, #452
