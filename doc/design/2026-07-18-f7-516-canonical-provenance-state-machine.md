# Design (RenQuant #516): canonical-publication provenance — the pair state machine

Issue: RenQuant#516 "F-7 gate: integrate canonical publication snapshot
before artifacts pin advance". Supersedes the mis-homed orch#559 (closed;
codex correctly rejected: wrong repo, premature "buildable", duplicate policy).
Date: 2026-07-18. Status: RFC — design review required before implementation.
Owner: drafted personally per design-review policy. Authored under the
authorized identity (fixes the orch#559 commit-email defect).

## 1. Ground truth: the canonical pair is published by TWO independent,
individually-atomic jobs — NOT jointly atomic

Verified on umbrella `main`:
- **Scorer** (`panel-ltr` / `panel-rank`): published by `weekly_wf_promote.sh`
  / `manual_promote.sh` from a `panel-ltr.staging.json` staging path.
- **Calibrator** (`panel-rank-calibration.json`): published by
  `monthly_calibrator_refresh.sh` via `monthly_calibrator_atomic_swap.py`
  (unique staging path -> verify-before-write -> `os.replace`), with a
  production-overwrite guard + rollback snapshot.

Each artifact's publish is individually atomic, but the **pair is not jointly
atomic** — the two jobs run on different cadences (weekly vs monthly). That
non-joint-atomicity IS the calibrator<->scorer binding-orphan class that
drained the book on 2026-07-16 ([[calibrator-scorer-fingerprint-triple-impl-bug]]),
and it is exactly what AC4's pair-atomic bundle seal exists to eliminate.

**Design consequence (the key decision):** the F-7 run-intent must bind at the
**pair-atomic publish point**, i.e. **AC4's bundle seal** (`bundle_seal.py`,
the ONLY point where the scorer+calibrator are one atomic unit / one bundle
generation), NOT retrofitted into the two legacy jobs. This makes AC4's bundle
generation the pair identity the run-intent references, and confirms #55's
producer is an AC4-bundle-seal hook — not independent wiring.

## 2. The pair-publication state machine

States (per canonical pair generation N):

1. **STAGED** — both members produced to staging paths by their jobs
   (scorer staging, calibrator staging). No prod mutation. (existing behaviour)
2. **SEALED** — AC4 `bundle_seal` composes the members into bundle generation
   N (pair identity = `bundle_id` + `manifest_digest` + member digests). This
   is the pair-atomic unit. (AC4 P1/P2 — the coupling point.)
3. **RUN-INTENT WRITTEN** — at seal time, the reviewed producer entrypoint
   calls `write_canonical_run_intent(...)` recording THIS run's code pins +
   config/data fingerprints; `build_canonical_provenance_reference` computes
   `run_intent_digest` from the actual bytes.
4. **REGISTERED (immutable, append-only)** — `register_canonical_publication`
   writes an APPEND-ONLY, content-addressed entry binding
   `run_intent_digest <-> pair(bundle) digest` into the registry `INDEX.json`.
   Append-only + content-addressed is the publisher-authorization answer
   (§7.2-b of the v1 draft): no per-run human review; the run-intent's
   re-verifiable evidence (pins/fingerprints vs the actual environment) IS the
   review. A mutating (non-append) write is refused.
5. **COMMITTED + PINNED** — the append-only registry commit is pushed; the
   artifacts subrepo pin advances past it (**operator-gated deployment**,
   [[artifacts-pin-gate-f7-canonical-snapshot]]). This is the produce->clean-
   pinned-snapshot transition #29 requires.
6. **VALIDATED (admission)** — the daily-run hydration resolves
   `CanonicalPublicationSnapshot` from the pinned registry checkout and calls
   `validate_artifact_manifest(..., canonical_publication_snapshot=snapshot)`
   before admitting the pair. Currently ZERO such call site exists; it is
   introduced here.

Failure/rollback (append-only makes this clean):
- A failed member publish leaves prod untouched (existing atomic-swap guard).
- A failed seal/register leaves the prior generation ACTIVE (no partial state).
- Rollback = re-activate a PRIOR pinned generation entry; the registry is never
  mutated in place (append-only), so rollback can never orphan a binding.

## 3. Policy: ONE source of truth (no second window)

Reuse artifacts' EXISTING `PROVENANCE_REQUIRED_AFTER` + `RQ_REQUIRE_PROVENANCE`
(owned by `experiment_registry`) as the single enforcement contract. The
canonical path opts IN through that existing contract; NO second
`CANONICAL_PROVENANCE_REQUIRED_AFTER`/`RQ_REQUIRE_CANONICAL_PROVENANCE`
(the orch#559 defect codex flagged — it would create unsynchronized windows
and repeat the flag-day break). Governed rollout stays (opt-in -> the existing
dated window -> consumer suites in review, never a flag-day — the artifacts#24
lesson).

## 4. Ownership + phasing (right repos this time)
- **Producer hook** (umbrella): the AC4 bundle-seal entrypoint gains the
  run-intent write + append-only registration. Rides AC4 P1/P2.
- **Admission** (orchestrator, narrow consumer-adapter): the daily-run
  hydration supplies the snapshot + calls validate. This is the ONLY piece
  that returns to orchestrator (a small adapter design), per codex.
- **Registry** (artifacts): DONE (#29). Append-only INDEX semantics must be
  confirmed/added (open item §5).
- **Pin gate** (umbrella lock): operator-gated; unchanged from
  [[artifacts-pin-gate-f7-canonical-snapshot]].

## 5. Open items to close in implementation-design (honest)
- Confirm the registry `INDEX.json` is (or is made) APPEND-ONLY + content-
  addressed — the state machine's rollback-safety depends on it; #29 provides
  the record shape but the append-only INDEX invariant must be verified/added.
- The producer hook depends on AC4 bundle-seal being the pair-publish point,
  which is live only after AC4 P1 cutover (operator-gated). So #55's PRODUCER
  sequences after AC4 P1; the ADMISSION adapter + the registry append-only
  invariant can proceed independently now.
- Exact `write_canonical_run_intent` `producer` allow-list value for the
  seal entrypoint.

## 6. AC4 coordination (resolved, not just noted)
#55 is NOT parallel plumbing to AC4 — its producer IS an AC4 bundle-seal hook,
because the pair is only atomic there. This supersedes the v1 §5 "sequence
after AC4 or go independent" framing: the PRODUCER must ride AC4; only the
admission adapter + registry-invariant work is independent. The operator's AC4
P1-seal cutover timing therefore gates #55's producer (not the whole thing).

## 7. r2 — resolutions to codex round-2 review (making it an implementable gate)

### 7.1 SEALED is staging-only; only ACTIVE is live (point 1)
A SEALED bundle generation is **staging-only and NON-admissible**. It cannot
become live until ALL of REGISTERED (immutable) → COMMITTED+PINNED → VALIDATED
succeed AND the ACTIVE pointer (§7.3) is advanced to it. A crash at any point
before ACTIVE advances leaves the PRIOR generation live and admissible; a
half-sealed / registered-but-unpinned / pinned-but-unvalidated generation is
never admitted. Admission (state VALIDATED) is gated on `ACTIVE == this
generation AND pinned-snapshot validates`, so an unproven pair can never be
served.

### 7.2 Authorized writer + transport for the registry commit (point 2)
`re-verifiable evidence IS the review` was wrong as an AUTHORIZATION claim;
it only justifies not needing a per-run *human content review*, not who may
write. Concrete transport:
- The daily producer (AC4 seal hook) only PREPARES the run-intent + candidate
  append-only entry into a **staging ref** in the artifacts registry repo — it
  never writes the live `INDEX.json` on the default branch.
- A single, code-reviewed **verified-publisher** (a dedicated CI job with a
  dedicated least-privilege credential — NOT the interactive multi-account
  agent tokens, NOT a shared branch) promotes the staged entry by APPENDING to
  `INDEX.json` on the protected default branch, only if
  `verify_canonical_run_intent` + append-only + digest checks pass. This is the
  only writer of the live registry; it eliminates the multi-account/shared-
  branch failure mode ([[agent-pr-merge-control-plane]], [[dual-identity-commit-emails]]).
- The subrepo **pin advance** past the new registry commit is a SEPARATE
  operator action ([[artifacts-pin-gate-f7-canonical-snapshot]]) — the producer
  and verified-publisher never advance the umbrella pin.

### 7.3 ACTIVE pointer state + recovery table (point 3)
Add **ACTIVE** = the single mutable pointer naming the pinned generation the
daily run admits. It lives in the pinned config/lock (advanced only by the
operator pin action after VALIDATED). Mutable surface = ONLY the ACTIVE
pointer; everything else (sealed bundles, registry entries) is immutable/
append-only.

| failure at | detection | recovery (prior ACTIVE stays live) |
|---|---|---|
| seal | seal job non-zero; prod untouched (existing atomic-swap) | prior generation ACTIVE |
| verify/register | verified-publisher refuses; no INDEX append | prior generation ACTIVE; staged ref discarded |
| commit/push | protected-branch push fails | prior generation ACTIVE; re-run publisher |
| pin advance | operator pin step fails / not run | prior pin ACTIVE (daily run unaffected) |
| validation | admission `validate_artifact_manifest` raises | fail-closed: refuse admit; prior ACTIVE stays served |

No path advances ACTIVE without a validated, pinned, registered generation.

### 7.4 Pair VALIDATION at seal, before pair identity (point 4)
SEALED is not "compose two staged files"; the seal MUST first validate the pair
or refuse to assign an identity:
- **binding**: the calibrator was fit against THIS scorer (the
  `calibrator_sha256` scorer-binding, RenQuant#505 / [[calibrator-scorer-fingerprint-triple-impl-bug]]);
- **digests**: exact member content digests recorded;
- **fingerprints**: code/config/data fingerprints of both producing runs
  present + consistent;
- **freshness/cadence**: reject a fresh weekly scorer beside a stale monthly
  calibrator beyond a declared freshness bound (a valid pair is not "whatever
  two files are staged"). This is the CORE 2026-07-16 orphan-class protection.
A pair failing any check is refused a bundle identity (never SEALED).

### 7.5 append-only + allow-list are artifacts PRECONDITIONS, not open items (point 5)
Reclassifying §5: the `INDEX.json` **append-only + content-addressed**
invariant and the **producer allow-list** are hard PRECONDITIONS of the
REGISTERED state — not implementation details. They must be specified,
implemented, and INDEPENDENTLY TESTED in `renquant-artifacts` (its own issue +
acceptance tests) before the producer is called ready. artifacts#29 landed the
record SHAPE but NOT the append-only INDEX invariant — so #29 is NOT "done" for
this gate. The artifacts pin stays FROZEN past 0b67302f until these invariants
land ([[artifacts-pin-gate-f7-canonical-snapshot]]). Follow-up artifacts issue
to be filed for the append-only INDEX + allow-list + tests.
