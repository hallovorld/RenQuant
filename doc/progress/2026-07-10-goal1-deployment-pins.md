# 2026-07-10 — GOAL-1 batch pin bump: pipeline / strategy-104 / execution / orchestrator

STATUS:    delivered (reconciliation only — see "This PR is reconciliation
           only" below)
WHAT:      Durable git record of an already-executed, already-verified
           operator-machine pin bump across four subrepos (pipeline,
           strategy-104, execution, orchestrator). Corrected per Codex
           review 2026-07-10T17:06:27Z: the pipeline pin explicitly
           excludes #182 (not a gap — a versioned rollout decision, see
           below); the strategy-104 delta no longer claims #46 (still
           open); the shadow-config narrative is corrected (no veto flip
           anywhere in this bump — see below); reconciliation evidence
           (exact pre/post hashes, backup timestamps) added.
WHY/DIR:   GOAL-1 (Deployment Governor + D6 shadow-AB track). Makes git
           match the live tree after an operator-executed, e2e-verified
           sync; per orchestrator#454 (architecture compliance audit,
           in review), the umbrella remains a temporary pinned-subrepo
           consumer only — this PR's role is exactly that: consume and
           record pins, own no new behavior.
EVIDENCE:  see "Reconciliation evidence" below for the full pre/post
           hash chain and backup timestamps (not a model/data claim)
NEXT:      A separate, future pin-bump PR is required to pick up
           renquant-pipeline#182 (full L3 fee-aware replay fix, merged
           after this bump was executed) — that PR is its own
           live-tree sync requiring its own review and operator
           authorization; not executed by this PR.

## Bottom line

Durable record of the operator-machine live deploy already executed today via
`scripts/promote_pin.py` (all four subrepo runtimes synced + e2e-verified
before this PR). This PR converges git with the live tree: the committed
`subrepos.lock.json` is **byte-identical** to the live tree's (sha256
`13932efbb90b83841778ebfee31655c0ec08cdaf23c056b8dde9490473c1c9a0`), so the
live-tree `git pull` after merge is a content no-op. Follows the merged #455
pattern (lock + regenerated snapshot + progress doc).

**This PR is reconciliation only** — it records a pin state that was already
applied, synced, and e2e-verified on the operator machine before this PR was
opened. It does not request or execute any new live-tree mutation. Per this
project's standing rule, future pin changes must be reviewed and merged
BEFORE live apply, except under an explicitly operator-authorized incident
procedure (this bump predates that discipline being written down in
orchestrator#454's remediation roadmap R1 shim-governance section, now in
review).

## Pin delta (four repos, all `main`)

| Repo | Old pin | New pin |
|---|---|---|
| renquant-pipeline | `b6139e6a3ad7` | `2b0eb0257b88` |
| renquant-strategy-104 | `8b2a592e53e4` | `0e5d989137b6` |
| renquant-execution | `43a8bdd36539` | `c41639840b2c` |
| renquant-orchestrator | `6a6a1bd371f6` | `e8fe46206025` |

## What ships per repo (CORRECTED — verified against actual merge state,
not asserted)

| Repo | PRs actually included | Live effect |
|---|---|---|
| renquant-pipeline | #179 Governor D2–D4, #180 governor harness, #181 arm tags (10 commits, `git log --oneline b6139e6a..2b0eb025`, verified) | **Inert** — governor stages ship flag-OFF; harness + arm tagging are observational plumbing. **Explicitly EXCLUDED: #182** (full L3 fee-aware replay fix) — merged at `8775fec50e8ffd8d7e6b72a09b8f7c4d8ce7aa24` AFTER this bump's pipeline pin was taken; verified NOT an ancestor of `2b0eb0257b88` (`git merge-base --is-ancestor 8775fec 2b0eb025` → exit 1). This is a versioned rollout decision, not an oversight: picking up #182 requires its own future pin-bump PR, its own review, and its own operator authorization for the live-tree sync — not executed here. |
| renquant-strategy-104 | #51 software_stops PREPARE (merged, commit `f9f0dae5`), #53 arm configs (merged, commit `0e5d9891`) | **All inert.** **CORRECTED: #46 (fractional) is NOT included** — verified via `gh pr view 46`: `state=OPEN`, `mergedAt=null`. The prior draft of this doc incorrectly claimed #46 shipped; it does not, and cannot, since it isn't merged. See "Shadow-config narrative" below for the corrected veto-flip claim. |
| renquant-execution | #25 order_math (merged `80763e18`), #26 readonly param (merged `c4163984`) | **Inert on prod path** — order_math is the extracted shared math module; readonly param hardens read-only invocations |
| renquant-orchestrator | #451 two-arm runner, #446 freeze tool, #452 backup compression, + 240-commit catch-up to 2026-07-10 main (`git log --oneline 6a6a1bd3..e8fe4620` → 240, verified) | **Inert for daily prod decisioning** — two-arm runner drives shadow arms only (and is UNINVOKED — no launchd entry ships anywhere in this bump); freeze/backup are ops tooling; catch-up commits were already individually merged through the reviewed-PR gate |

## Shadow-config narrative (CORRECTED — Codex review: the prior draft's
"arm-A veto flip" claim was false)

**There is no veto flip, arm-A or otherwise, anywhere in this pin bump.**
The LEGACY `configs/strategy_config.shadow.json` (Step-4 ops shadow, broker
tag `alpaca_shadow`, still invoked daily by `daily_104.sh` independent of
the D6-§2a experiment) changes hash across this bump —
`ea52e7f2873e87d59e648f812be6ce5bccfc2f8631c345a3c1b502259908902d` →
`b7a4332652184f58a41c2eab3903f501496fb790d8eeef62ceb59409ffbe06e0` — but
**the entire diff is one additive, default-`enabled:false` `software_stops`
block from #51** (verified via `git diff 8b2a592e 0e5d9891 --
configs/strategy_config.shadow.json`: the only change is a 6-line addition
under the existing `t2_settlement` section, identical in kind to the
already-documented prod-config `software_stops` addition). No `buy_floor`,
`buy_floor_std_mult`, or any D6-§2a key is touched in this file, in this
bump, at all.

The actual D6-§2a two-arm shadow A/B experiment (strategy-104#53) lives
ENTIRELY in two NEW files this bump adds: `configs/strategy_config.shadow_a.json`
(treatment, buy_floor_std_mult=0.5) and `configs/strategy_config.shadow_b.json`
(control, buy_floor_std_mult=1.0) — both **UNARMED**: no launchd entry, no
`daily_104.sh` change, no scheduled invocation ships anywhere in this bump.
Arming requires orchestrator#451's two-arm runner to actually be invoked in
production, which is a separately-gated future step (#451 itself remains
uninvoked code even after this pin bump lands it).

## Production behavior equivalence

All newly shipped behavior flags are **OFF** in the prod and golden configs.
The prod config hash change (`strategy_config.json` sha256 `587b0afb…` →
`2752de27…`) is limited to the same additive, default-off `software_stops`
keys described above — not a behavior change. Prod and legacy-shadow
decisioning inputs are therefore behavior-equivalent before/after the bump.
No arm is armed by this bump; the D6-§2a experiment configs exist on disk
but are not invoked by anything scheduled.

## Reconciliation evidence (ADDED — Codex review)

This bump was applied as four SEQUENTIAL per-repo `promote_pin.py bump
--apply` runs on the operator machine, each preceded by a pre-bump backup of
`subrepos.lock.json`. The backup files (found on the live tree,
`RenQuant/subrepos.lock.json.promote-bak.<timestamp>`) give an exact,
verifiable timestamp/order/content chain:

| Timestamp | Backup sha256 | Lock state captured (pins at that moment) |
|---|---|---|
| `20260710T093150` | `fcd0170046eaba9af9dba86403d37cb20ea5643490e114bc2ba3b92099e85366` | Pre-bump baseline: all four repos at their OLD pins (matches this PR's "Old pin" column exactly) |
| `20260710T093312` | `a69365519b7738da982beb92f0918cedd041d2a7a83f4cd06b552034d8aed163` | After pipeline bump: pipeline `2b0eb025`, others still old |
| `20260710T093436` | `587a1824857d00e192744a7e1773da11b7b5b37849ad0df2c7b3f4bd48fc0749` | After strategy-104 bump: strategy `0e5d9891`, pipeline `2b0eb025`, others still old |
| `20260710T093619` | `45e1aa22d950f5b5c6854fadfcab8174607b7ad14d69c9f6dce6f75abca00c06` | After execution bump: execution `c4163984`, prior two updated, orchestrator still old |
| (final, this PR) | `13932efbb90b83841778ebfee31655c0ec08cdaf23c056b8dde9490473c1c9a0` | After orchestrator bump: all four at their "New pin" values (this PR's committed state) |

Sequence: **pipeline → strategy-104 → execution → orchestrator**, four
bumps completed within a 4.5-minute window (09:31:50–09:36:19, plus the
final orchestrator bump shortly after). Each bump's own e2e verification run
(see "Deploy evidence" below) executed between its backup and the next.

## Deploy evidence

- Live deploy: `promote_pin.py bump --apply` per repo on the operator
  machine, 2026-07-10, in the sequence and at the timestamps documented
  above; runtime sync + verify green for all four.
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
deploy — the exact four files are
`subrepos.lock.json.promote-bak.20260710T093150`,
`.20260710T093312`, `.20260710T093436`, `.20260710T093619` (see
"Reconciliation evidence" above for their content/hashes) — then revert
this commit.

## References

- Umbrella precedent: #455 (strategy-104 pin bump, merged 2026-07-10)
- Architecture context: orchestrator#454 (compliance audit — umbrella is a
  temporary pinned-subrepo consumer only, in review)
- renquant-pipeline PRs #179, #180, #181 (merged, included); **#182 merged
  but explicitly EXCLUDED from this bump — see pipeline row above**
- renquant-strategy-104 PRs #51, #53 (merged, included); **#46 still OPEN,
  NOT included — corrected from the prior draft's false claim**
- renquant-execution PRs #25, #26
- renquant-orchestrator PRs #446, #451, #452
