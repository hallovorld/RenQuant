# 2026-07-09 — strategy-104 pin bump 74a643e9 → 8b2a592: one-share floor shadow arming

## Bottom line

Durable record of the operator-authorized live deploy already executed on the
machine via `scripts/promote_pin.py` (subrepo runtime synced + verified before
this PR). This PR converges git with the live tree: the committed
`subrepos.lock.json` is **byte-identical** to the live tree's (sha256
`fcd01700…e85366`), so the live-tree `git pull` after merge is a content
no-op.

## What ships in the pin delta (74a643e9a449 → 8b2a592e53e4)

| strategy-104 PR | Change | Live effect |
|---|---|---|
| #44 | S7 parking-sleeve config + fraction-resolution / max_sleeve_pct fixes | **Inert** — `sleeve.enabled=false` in all configs |
| #49 | `sizing.one_share_floor_enabled` key | **Prod OFF** (`false` in prod + golden), **shadow ON** (`true` in shadow) — S1 one-share-floor shadow data collection starts with the next shadow e2e run |
| #50 | `deployment_governor` config block | **Inert** — `enabled=false` in ALL THREE configs (prod / shadow / golden) |

## Production behavior equivalence (semantic diff proof)

Semantic diff of `configs/strategy_config.json` @74a643e9 (GitHub) vs the
live pinned runtime checkout @8b2a592 shows **only two ADDED blocks, both
default-off, zero CHANGED/REMOVED keys**:

- `ADDED: deployment_governor` — `enabled: false`
- `ADDED: sizing` — `one_share_floor_enabled: false`

Production decisioning is byte-equivalent in effect; the only armed change is
the shadow-config floor flag (observational only).

## Deploy evidence

- Live deploy: `promote_pin.py bump --apply` on the operator machine,
  runtime sync + verify green (committed decision; prod paths untouched).
- Read-only e2e: `check_readonly_e2e.sh` **PASS**, run_id
  `2026-07-09-live-192fbca9`.
- Snapshot: `doc/arch/strategy-104-snapshot.md` regenerated from the live
  pinned sources; `--check` (byte-exact) and `--verify-pinned-declaration`
  (CI semantic check against the pinned configs + this lock) both pass.
  The snapshot's `UMBRELLA WORKING-COPY DRIFT: kind=hf_patchtst vs pinned
  xgb` source warning is true and expected (the umbrella working-copy config
  is stale; the pinned config is what the daily run consumes) — kept.
  The snapshot also newly resolves artifact/calibrator metadata (XGB
  trained 2026-07-06, WF gate passed) because this render ran where the live
  artifacts exist; the previously committed render lacked them.

## Rollback

`scripts/promote_pin.py revert --apply` (runtime backup `20260709T223749`),
then revert this commit.

## References

- strategy-104 PRs #44, #49, #50
- orchestrator PRs #443, #444 (one-share-floor shadow wiring)
