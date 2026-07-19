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
