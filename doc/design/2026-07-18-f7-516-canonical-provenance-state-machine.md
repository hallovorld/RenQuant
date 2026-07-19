# Design (RenQuant #516): canonical-publication provenance — the pair state machine

Issue: RenQuant#516 "F-7 gate: integrate canonical publication snapshot
before artifacts pin advance". Supersedes the mis-homed orch#559 (closed).
Date: 2026-07-18. Status: RFC — design review required before implementation.
Owner: drafted personally per design-review policy. Authored under
hallovorld@users.noreply.github.com (the authorized identity).

SINGLE-SOURCE protocol: the transition table in §3 is authoritative; prose
elsewhere only explains it. (v2 rewrite — replaces the earlier layered draft
so no section contradicts another.)

## 1. Ground truth: the canonical pair is published by TWO independent, individually-atomic jobs — NOT jointly atomic

Verified on umbrella `main`:
- **Scorer** (`panel-ltr`/`panel-rank`): published by `weekly_wf_promote.sh` / `manual_promote.sh` from a `panel-ltr.staging.json` staging path.
- **Calibrator** (`panel-rank-calibration.json`): published by `monthly_calibrator_refresh.sh` via `monthly_calibrator_atomic_swap.py` (unique staging path -> verify-before-write -> `os.replace`), with a production-overwrite guard + rollback snapshot.

Each artifact publish is individually atomic; the **pair is not jointly atomic** (weekly vs monthly cadence). That non-joint-atomicity IS the calibrator<->scorer orphan class that drained the book on 2026-07-16 (RenQuant#505 fixed the `calibrator_sha256` binding; the durable fix is the pair-atomic seal). The F-7 run-intent therefore binds at the **pair-atomic publish point = AC4's bundle seal** (`renquant_orchestrator.bundle_seal`, **renquant-orchestrator#556** — the only point the pair is one unit), NOT retrofitted into the two legacy jobs. The producer hook is therefore an ORCHESTRATOR-owned addition to that repo's AC4 sealing workflow (§7), invoked under the designated producer CI principal — NOT umbrella code.

**Hard dependency (activation primitive):** orch#556 P1 seals only BYTE-IDENTICAL (genesis) content — its own review-fix deferred a truly pair-atomic CHANGED-content activation (per-generation directory / single-pointer flip) to a larger AC4 P2/P3 change (tracked as the orchestrator changed-content activation follow-up; codex refs it as orch#558 — to be filed/confirmed). F-7 MUST NOT claim any producer or activation path is READY until that changed-content activation lands and the exact orchestrator pin carrying it is part of DEPLOYMENT validation (§2). P1's byte-identical seal alone is an unsafe activation primitive for a changing pair.

## 2. Two distinct validations (the fix for the activation cycle)

Two validation operations, at different times, against different targets — conflating them created a circular dependency:
- **DEPLOYMENT validation** (pre-activation): run against the CANDIDATE pinned checkout BEFORE the pair is ACTIVE. Proves the candidate is registered, immutable, its snapshot clean, its pair valid (§4). Success is a PRECONDITION of activation.
- **ADMISSION validation** (runtime): the daily run validates ONLY the already-ACTIVE pinned snapshot each session; never a candidate. On failure it fails closed for that run and rolls back (§3).

`ACTIVE` = a single mutable pointer (in the pinned umbrella lock) naming the generation the daily run admits. Everything else (sealed bundles, registry entries) is immutable/append-only. ACTIVE advances only by the operator ACTIVATION action (§3.1), only after DEPLOYMENT validation passes.

## 3. The single authoritative state machine (per pair generation N)

| # | state | actor | transition guard | on failure |
|---|-------|-------|------------------|-----------|
| 1 | STAGED | scorer job + calibrator job | members produced to staging paths; prod untouched | job non-zero -> prod untouched; prior ACTIVE stays |
| 2 | PAIR-VALIDATED -> SEALED | orch AC4 seal hook (producer principal) | §4 pair checks ALL pass; compose bundle gen N (identity = bundle_id+manifest_digest+member digests). **SEALED is staging-only / NON-admissible.** | any pair check fails -> NO identity; not sealed; prior ACTIVE stays |
| 3 | RUN-INTENT PREPARED | producer principal | `write_canonical_run_intent` + `build_canonical_provenance_reference` (digest from bytes); write a CANDIDATE entry to the producer's staging ref ONLY (§5) — never the live INDEX | prepare fails -> discard staging ref; prior ACTIVE stays |
| 4 | REGISTERED (immutable) | **verified-publisher** (§5), NOT the producer | publisher proves the candidate came from the producer principal (§5), `verify_canonical_run_intent` + append-only + digest checks pass; APPENDS to protected-branch `INDEX.json` | publisher refuses -> no append; prior ACTIVE stays |
| 5 | COMMITTED + PINNED | operator | append-only registry commit pushed; artifacts subrepo pin advanced past it | push/pin fails -> prior pin ACTIVE; re-run |
| 6 | DEPLOYMENT-VALIDATED | operator (pre-activation) | validate the CANDIDATE pinned checkout: registered, clean snapshot, pair valid | fails -> candidate NOT activated; prior deployment ACTIVE (last-known-good) |
| 7 | ACTIVATED | operator (atomic, §3.1) | advance ACTIVE -> gen N | — |
| — | ADMISSION (runtime) | daily run | validate the ACTIVE pinned snapshot each session | post-activation runtime failure -> fail closed for this run + rollback to last-known-good (§3.1) |

### 3.1 Activation + rollback (last-known-good)
ACTIVE and the artifacts pin advance together. If a single umbrella commit can carry both the lock pin bump and the ACTIVE pointer, that commit IS the atomic activation; if they live in separate files that cannot be one commit, activation is the reviewed commit updating BOTH, gated on DEPLOYMENT-VALIDATED — never one without the other. The prior (gen N-1) pinned deployment is retained as **last-known-good**; runtime ADMISSION failure at gen N reverts ACTIVE+pin to gen N-1 (a reviewed revert commit), which is still registered+pinned+valid, so rollback can never orphan a binding.

## 4. Pair validation at SEAL (core 07-16 protection)
SEAL refuses a bundle identity unless ALL pass: **binding** (calibrator fit against THIS scorer — `calibrator_sha256`, RenQuant#505); **digests** (exact member content digests); **fingerprints** (code/config/data of both runs present + mutually consistent); **freshness/cadence** (reject a fresh weekly scorer beside a stale monthly calibrator beyond a declared bound). A valid pair is not "whatever two files are staged."

## 5. Control-plane trust boundary (producer + publisher + operator)
- **Producer principal**: the AC4 seal hook runs under a dedicated CI principal permitted to create ONLY a candidate staging ref, namespace `refs/candidates/canonical/<gen>-<run_intent_digest>`, retention = pruned after promote or a fixed TTL. NO write to the protected default branch, NO pin authority.
- **Verified-publisher**: a single code-reviewed CI job with a distinct least-privilege credential — NOT interactive agent tokens, NOT a shared branch. SOLE writer of the live `INDEX.json`. Proves candidate provenance by restricting candidate-ref creation to the producer principal and re-verifying `run_intent` against the recorded environment before it appends. Target = protected default branch; required checks = append-only invariant test + digest match + allow-list membership.
- **Operator**: the only actor advancing the artifacts pin + ACTIVE (§3.1).
Three-principal split (producer prepares · publisher appends · operator activates) eliminates the multi-account/shared-branch failure mode.

## 6. Policy: ONE source of truth (no second window)
Reuse artifacts' EXISTING `PROVENANCE_REQUIRED_AFTER` + `RQ_REQUIRE_PROVENANCE` (owned by `experiment_registry`) as the single enforcement contract; the canonical path opts IN through it. NO second `CANONICAL_PROVENANCE_REQUIRED_AFTER`/`RQ_REQUIRE_CANONICAL_PROVENANCE` window. Governed rollout: opt-in -> the existing dated window -> consumer suites (backtesting/model/orch) in the gate-introducing PR -> never a flag-day (the artifacts#24 lesson).

## 7. Ownership, artifacts preconditions, phasing
Repo ownership (corrected r4): the **umbrella** owns this cross-repo PROTOCOL + the lock/pin integration + integration acceptance; **renquant-orchestrator** owns the seal-hook producer AND the admission adapter; **renquant-artifacts** owns the registry invariant.
- **Producer hook** = **renquant-orchestrator** (an addition to that repo's AC4 sealing workflow around `renquant_orchestrator.bundle_seal`, orch#556, run under the producer CI principal); sequences AFTER the AC4 P1-seal live cutover (operator-gated) AND after the orch changed-content activation follow-up lands (§1 hard dependency).
- **Admission adapter** = renquant-orchestrator (narrow): supply the snapshot + call `validate_artifact_manifest`; proceeds independently of the producer.
- **Registry** = renquant-artifacts. **PRECONDITIONS (not open items):** the `INDEX.json` append-only + content-addressed invariant AND the producer allow-list must be specified, implemented, and independently tested in renquant-artifacts before the producer is "ready". artifacts#29 landed the record SHAPE but NOT the append-only INDEX invariant — so #29 is NOT "done" for this gate. A follow-up renquant-artifacts issue will track the invariant + allow-list + acceptance tests; the artifacts pin stays FROZEN past 0b67302f until they land.
- **Umbrella** = this protocol + `subrepos.lock.json` pin/ACTIVE integration + the cross-repo integration-acceptance test.
- **Pin gate** = operator; unchanged.

Phasing: (P0) artifacts append-only INDEX + allow-list + tests -> (P1) admission adapter in orchestrator (opt-in OFF) -> (P2) orch producer hook on AC4 seal (after BOTH the AC4 P1 cutover AND the changed-content activation follow-up) -> (P3) dated-window enforcement via the EXISTING policy -> (P4) operator pin advance + ACTIVE activation.

## 8. References
- RenQuant#516 (this issue); orch#559 (closed superseded RFC).
- renquant-artifacts#29 (record shape merged, commit 0b67302f) — append-only INDEX invariant still OWED (new artifacts issue to file).
- **renquant-orchestrator#556** (AC4 P1 bundle_seal — byte-identical genesis only); the orchestrator CHANGED-content activation follow-up (deferred by #556's review-fix; codex refs orch#558 — file/confirm) is a HARD dependency of the activation primitive (§1). RenQuant#505 (calibrator_sha256 binding).
- Incident 2026-07-16 (book drained to 94% cash — the orphan class this gate prevents); artifacts#24 (the flag-day lesson governing §6).
- artifacts pin-gate: do not advance the artifacts pin past 0b67302f until §7 preconditions + the integration land.
