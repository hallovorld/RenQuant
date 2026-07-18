# G1 burned-sessions manifest — companion note (v5 prereg §4.7 rule 1)

Status: **PREP** — pilot registration is NOT authorized (turnkey after the
epoch-5 refreeze). The auditable object is the committed JSON next to this
note, [`burned-sessions-manifest.json`](burned-sessions-manifest.json); this
note is the human-readable companion. At the pilot-registration commit the
JSON is frozen, `pilot_registration_commit` is stamped, and any session
recorded between this compilation (2026-07-18) and that commit is appended
first.

Prereg: [`doc/experiments/2026-07-17-equal-weight-deployment-prereg-v5.md`](../2026-07-17-equal-weight-deployment-prereg-v5.md)
(merged as RenQuant#494). Rule 1 burns EVERY paired session observed before
the registration commit — explicitly including the "~4 unblinded pairs"
behind the #485 provisional parameterization and anything analyzed in any
memo — and requires this enumeration ("the manifest, not prose, is the
auditable object").

## Bottom line

**14 session entries enumerated; ALL burned; zero pilot-eligible sessions
exist.** Only **3** were ever valid pairs (all 2026-07-11 — a Saturday,
re-observing Friday's close). Only **1** intact *discrete session record*
of a valid pair survives on disk (epoch-2); the epoch-3-minting 09:43 PT
pair's discrete record was overwritten, but its sealed inputs AND full
per-arm output trees survive unblinded in the live harness root. The pilot
starts at n = 0 and needs ≥ 40 fresh single-epoch paired sessions strictly
postdating the registration commit.

**Scope note (completeness surface):** this enumeration covers every
*observed* session — including sessions whose discrete records were
overwritten by later same-date attempts and sessions that exist only in
logs. The PART-B orchestrator telemetry counts RECORDED sessions only
(safe in direction: an unrecorded session can never enter the pilot), so
overwritten/log-only sessions are invisible to it — **this manifest is the
completeness surface** and its compilation must sweep the logs, not just
the record files. The 09:43 PT entry below was missed by the first
compilation for exactly this reason and recovered in Codex review.

## Session table

| # | Session date | Epoch | Pair status | What happened | Burned because (beyond §4.7 rule 1) |
|---|--------------|-------|-------------|---------------|--------------------------------------|
| 1 | 2026-07-10 | pre-arming | failed | arm_failure both arms (exit 4), ~19:13Z | failed pair; no epoch freeze |
| 2 | 2026-07-10 | pre-epoch-1 | failed | precheck: patchtst_shadow artifact unresolvable (22:17 PT) | no observation; diagnosed in the artifact-store memo |
| 3 | 2026-07-11 | epoch-1 | **valid** (record overwritten) | 01:58–02:00 PT preflight pair (start 08:58:16Z), exit 0 at session-log line 163 (the log's FIRST run — both `launchd.err.log` exit=0 lines belong to #6/#7, not this pair), minted epoch-1 freeze | unblinded + analyzed (write-containment forensics, weekday-gate memo); Saturday world; epoch retired with zero counted pairs |
| 4 | 2026-07-11 | epoch-1 | failed | ~04:5x PT kickstart: strategy-104 tree DIRTY (`?? logs/`) — poisoned by #3's own stray byproducts | no observation; incident analyzed in a committed memo |
| 5 | 2026-07-11 | epoch-1 | failed | frozen_fingerprint_mismatch after the #470 orchestrator pin bump (d4d27863 → fb3b69ff); counters 4/3 at retirement | no observation |
| 6 | 2026-07-11 | epoch-2 | **valid** (intact record) | 07:20–07:22 PT pair, exit 0, both arms full native chain, as-of 14:20:31Z, digest f215af9b, universe 145; minted epoch-2 freeze; the ONLY intact discrete valid-pair record | unblinded (#485 evidence count); Saturday world; epoch-2 retired ~2h20m later with zero counted pairs |
| 7 | 2026-07-11 | epoch-3 | **valid** (record overwritten; sealed + per-arm artifacts survive) | 09:43–09:45 PT pair (start 16:43:06Z), exit 0 at session-log line 819, **minted the epoch-3 freeze** (frozen_at 16:43:08Z = start+2s); sealed snapshot digest 36bffe24, universe 145; discrete record overwritten by #8, but sealed inputs + full `arm_alpaca_shadow_a/b` trees survive UNBLINDED in the live root; never counted by the counters. *Missed by the first compilation; recovered in Codex review.* | unblinded (per-arm outputs on disk); Saturday world; never counted; epoch-3 schedule mismatch (model#60) |
| 8 | 2026-07-11 | epoch-3 | failed | 14:35 PT scheduled run: same_world_violation vs **#7's** recorded bundle (36bffe24 ≠ bde4f1cd) — first compilation mis-attributed the collision to #6, whose digest is f215af9b and whose bundle was already archived at the 09:43 PT epoch-3 mint | no observation |
| 9 | 2026-07-12 | epoch-3 | failed | precheck: renquant-model tree DIRTY (`M README.md`); Sunday | no observation |
| 10 | 2026-07-13 | epoch-3 | failed | identical dirty-tree precheck abort; counters reach 4/3 (the "#485 ~4 pairs" state) | no observation |
| 11 | 2026-07-14 | epoch-3 | no_record | launchd: `shadow_ab_daily.sh: No such file or directory` | no observation exists (silent stall, epoch-4-refreeze memo) |
| 12 | 2026-07-15 | epoch-3 | no_record | same scheduler failure | no observation exists |
| 13 | 2026-07-16 | epoch-3 | no_record | same scheduler failure | no observation exists |
| 14 | 2026-07-17 | epoch-3 | failed | wrapper PRECHECK: all 8 repos off the frozen manifest (GOAL-5 pins) → triggered the epoch-4 refreeze | no observation |

Counts: by status — valid 3 / failed 8 / no_record 3; by epoch — pre-arming 1,
pre-epoch-1 1, epoch-1 3, epoch-2 1, epoch-3 8, **epoch-4 0 (zero sessions —
nothing to burn yet)**. (The first compilation's "epoch-3 6" was itself a
miscount of its own 7 enumerated epoch-3 entries; the corrected by-epoch
counts sum to the 14 entries.)

## Analytical exposure (why "burned" is not a formality)

- **RenQuant#485** (§4.6 power simulation, merged 35982da) + its committed
  artifact `doc/experiments/2026-07-16-ew-prereg-power-simulation-results.json`
  + `doc/progress/2026-07-16-g1-power-simulation.md` count "~4 paired
  sessions" from this harness as the evidence base for the PROVISIONAL
  sigma_d = 25 bps AR(1) DGP. Honesty note: sigma_d was NOT numerically
  fitted from these pairs (too few — that is #485's own caveat); the pairs
  are burned as referenced/analyzed evidence, which is exactly the §4.7
  rule 1 standard ("analyzed in any memo").
- **orchestrator `doc/progress/2026-07-11-shadow-ab-write-containment.md`**
  reads the 02:15 PT valid pair's on-disk byproducts (58k admission-shadow
  JSONL) — direct unblinded contact with arm-level output.
- **orchestrator `doc/progress/2026-07-11-shadow-ab-weekday-gate.md`**
  analyzes the 2026-07-11 Saturday paired world (duplicate of Friday's
  close, zero information) — a committed analysis of all three valid
  pairs' observation world.
- **The 09:43 PT pair's unblinded surface is the live root itself**: its
  sealed inputs and full per-arm output trees (native inference/execution,
  admission-shadow JSONL, live-state contracts) sit on disk readable by
  any operator or agent — burn-relevant exposure independent of any memo.
- **orchestrator `doc/progress/2026-07-17-shadow-ab-epoch4-refreeze.md`** +
  **renquant-model#60**: all epoch-3-era paired sessions are additionally
  schedule-mismatched under the close-anchored as-of contract (not poolable
  with epoch-4+ regardless of burning).

## Method / provenance

Compiled read-only from `/Users/renhao/renquant-shadow-ab` (session JSONs,
per-date bundles, the three epoch archives + EPOCH-NOTEs, pre-arming
archive, counters, freezes, `logs/*.log`, `run_manifest.json`). Attempt
numbering within 2026-07-11 follows the session log's six run blocks
(starts 08:58:16Z / 11:59:33Z / 13:17:09Z / 14:20:31Z / 16:43:06Z /
21:35:05Z, exits 0/3/3/0/0/3), cross-checked against the per-epoch counter
arithmetic (1,1 → 4,3 in the arming/epoch-1 series; 1,0 in epoch-2; 2,1 →
4,3 in epoch-3 — which never counted the freeze-minting pairs), freeze
`frozen_at` stamps, and `launchd.err.log` exit lines (2026-07-11 reads
3,0,0,3: both exit=0 lines belong to the 07:20/09:43 PT pairs); where a
discrete record was overwritten the entry says so and cites its evidence. **No DB-backed session records exist** — no `*.db` file
exists anywhere under the harness root (the per-arm `runs.<tag>.db` is only
created inside an arm's native-live-run step, which no surviving on-disk
session reached), so the JSON records above are the complete session set;
there is no separate orchestrator session DB for two-arm pairs.

Companion deliverable (PART B, orchestrator repo): per-epoch/per-role
`n_paired_sessions` telemetry (§4.7 rule 4) with the "no manifests →
everything burned" safe default, deriving counts mechanically **from
recorded manifests alone** — by design it cannot see overwritten/log-only
sessions (safe direction), which is why this manifest, not the telemetry,
is the completeness surface for the burn set.
