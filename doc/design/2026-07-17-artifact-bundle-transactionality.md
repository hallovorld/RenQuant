# Design: transactional artifact bundles for the 104 serving pair (GOAL-5 AC4)

Date: 2026-07-17
Status: RFC — design review required before any implementation
Owner: operator-delegated (GOAL-5 P0); drafted personally per design-review policy

## 1. Problem

The 104 serving state is a LOGICAL PAIR that must be internally consistent:
the panel scorer artifact (`panel-ltr.alpha158_fund.json`), its pooled
calibrator (`panel-rank-calibration.json`), and the WF-gate metadata
embedded in the panel. Four independent writers mutate these as SEPARATE
files with SEPARATE per-file rollbacks:

1. weekly WF promote (stage → verdict → per-file rollback on FAIL)
2. monthly calibrator refresh (atomic per-FILE swap, not per-pair)
3. restamp tooling (binding/fingerprint edits, per file)
4. manual ops / incident response

Consequence, observed four times (2026-05-27, 06-22, 07-01, 07-14→16):
a writer or its rollback leaves a calibrator whose scorer binding no longer
matches the live panel → `LoadGlobalCalibrationTask` fail-closes → zero buy
candidates → the book bleeds to cash through exits (94% cash on 07-16). The
07-13→16 churn is the canonical trace: promote-REJECT rollback restored the
panel, the calibrator went through refresh + restamp + rollback separately,
and the pair ended orphaned. Per-file atomicity cannot fix a PAIR-level
invariant.

## 2. Design (r2 — normative protocol; review round 1 findings adopted)

### 2.1 Store layout and host model

```
backtesting/renquant_104/artifacts/prod/bundles/<bundle_id>/
    manifest.json
    <members...>
backtesting/renquant_104/artifacts/prod/ACTIVE      # pointer file
backtesting/renquant_104/artifacts/prod/bundles/.lock
```

**Host model (normative):** single-host POSIX filesystem (APFS on the
live machine). `flock` is the store lock under this model ONLY; the store
is NOT supported on network filesystems, and the writer refuses to start
if the store path is not a local mount (checked at open).

### 2.2 Bundle identity (normative)

`manifest.json` fields (all REQUIRED; unknown fields REJECTED):

- `schema_version: 1`
- `members`: the EXACT allowed set for schema v1 —
  `{panel-ltr.alpha158_fund.json, panel-rank-calibration.json}`; any
  missing or extra file in the bundle directory ⇒ invalid bundle.
- per-member `{sha256, bytes}` (algorithm pinned per schema version)
- `bindings` (as v1 draft, digest basis = renquant-common v1 impl)
- `authorization` (§2.4)
- `parent_bundle`, `created_at`
- `manifest_digest`: sha256 over the CANONICAL serialization (UTF-8,
  sorted keys, no insignificant whitespace, LF) of the manifest WITHOUT
  this field.
- `bundle_id = <utc-ts>Z-<first 16 hex of manifest_digest>` — timestamp
  for human ordering, digest for content identity; collision ⇒ abort.

**Run-bundle binding:** every daily run's persisted run bundle records
`{bundle_id, manifest_digest, member digests, pointer_generation}` so any
historical run can be replayed against the exact archived bundle.

### 2.3 Writer protocol (normative, with durability order)

```
1  flock(bundles/.lock, EX)                # writers AND GC serialize here
2  write members into bundles/<id>.tmp/    # fsync each file
3  write manifest.json                     # fsync file
4  fsync(bundles/<id>.tmp dirfd)
5  rename bundles/<id>.tmp → bundles/<id>; fsync(bundles/ dirfd)
6  validate: re-read manifest, verify member digests, run the PUBLIC
   pair-validation API (§2.5); failure ⇒ delete dir, abort (still locked)
7  write ACTIVE.tmp = "<generation+1> <bundle_id>"; fsync(ACTIVE.tmp)
8  rename ACTIVE.tmp → ACTIVE; fsync(prod/ dirfd)
9  append an immutable operation record (§2.4); fsync; unlock
```

Crash at/before step 8 ⇒ previous ACTIVE intact (worst residue: orphan
dir, GC'd later). Crash after 8 before 9 ⇒ new state serves; the missing
operation record is detected by the auditor (record generation gap) and
alarmed — state is consistent, provenance is loudly incomplete, never
silently wrong. The kill-injection suite (§4) crashes at EVERY numbered
step and after each individual fsync.

**Pointer format:** `<generation> <bundle_id>` — generation is a
monotonically increasing integer; readers and run bundles record it,
making pointer flips totally ordered and stale-pointer rollback
detectable.

### 2.4 Writer authorization and audit (normative)

`authorization` in the manifest and the append-only operation log
(`bundles/OPERATIONS.jsonl`, one fsync'd line per commit/rollback):

- `tool` + `tool_version` (the committing script; "hand-edit" is not a
  value — manual response uses the break-glass tool below)
- `actor` (OS user + configured operator identity)
- `source`: for wf_promote — the WF run/verdict IDs; for
  monthly_refresh — the fit-input fingerprints; for restamp — the
  incident/task reference (REQUIRED field, per the containment protocol)
- `inputs`: content digests of everything the writer consumed.

**Break-glass path:** `bundle_breakglass` is the ONLY sanctioned manual
mutation tool: it takes an incident/task reference (mandatory), performs
the same protocol (§2.3), marks `authorization.tool=bundle_breakglass`,
and its use ALWAYS alarms via the drift sentinel (a break-glass commit is
by definition a containment event under the AC3 protocol). Rollback is
the same tool with `--rollback-to <bundle_id>`, restricted to ancestors
reachable via parent_bundle.

### 2.5 Validation API (boundary-correct)

renquant-pipeline exposes a VERSIONED PUBLIC API (new module
`renquant_pipeline.bundle_contract`): `validate_pair(manifest, member_paths)
-> Verdict` — internally the same logic as the runtime loader's matcher,
exported deliberately (the private `_assert_calibrator_matches_scorer` is
NOT imported by umbrella code; finding 5). The pipeline reader and the
umbrella writer call the SAME public function, and a contract fixture in
renquant-common pins the verdict semantics both sides test against.

### 2.6 Reader protocol and GC (race-defined)

Reader: read ACTIVE (generation + id) → open bundles/<id>/ by dirfd →
read manifest via that dirfd → verify member digests → serve. Because GC
serializes on the store flock and NEVER deletes (a) the current ACTIVE
target, (b) any ancestor within the retention window, (c) any bundle
referenced by a run bundle in the last 90 days (queried before delete),
a reader that resolved ACTIVE holds a directory that cannot disappear
mid-read on the single-host model (unlink-after-open keeps the dirfd
valid on POSIX regardless). GC deletions are operation-log records too.

### 2.7 WF status semantics (finding 6)

The manifest's `bindings.wf_gate_verdict` is a VERBATIM COPY of the
panel's stamped metadata — bundle validity asserts PAIR CONSISTENCY ONLY
and is EXPLICITLY NOT a buy-admissibility statement. Admission remains
the preflight P-WF-GATE's job (incl. the governed diagnostic-only
override, pipeline#203); nothing in this design feeds it. P1's seal of
the current pair asserts "this is the pair the operator is knowingly
serving today", not "this pair passed the WF gate".

## 3. Migration (census-first; unchanged phases P0-P3 otherwise)

**P2 entry criterion (finding 7):** a committed census document listing
EVERY reader/writer of the flat paths — daily runner, sim, shadow arm,
preflight, calibrator refresh, wf_promote, restamp tools, per-ticker
tournament, manual-recovery scripts — each classified {migrates to
bundle API | keeps flat VIEW}. Views are read-only (mode 0444 files
regenerated on pointer flip); a legacy WRITER discovered post-census is
a migration blocker, not a workaround. Rollback invariant for every
phase: reverting the phase's commit restores the previous serving
behavior without artifact surgery (P1's flip is reverted by pointing
ACTIVE back; flat files remain until P3).

## 4. Verification (AC4 acceptance; expanded per review)

Kill-injection at every §2.3 step and after each fsync; reader/GC race
tests (reader holding dirfd across a GC pass; pointer flip during read);
invalid-schema/extra-member/missing-member injection; stale-pointer
rollback (generation regression must be detected and refused without
--rollback-to); break-glass path leaves record + triggers sentinel;
run-bundle replay: resolve a 30-day-old run's recorded
{bundle_id, digests} against the archive and re-verify. Plus the
incident-replay and live-drill items from r1.

## 5. Ownership boundaries (amended)

- renquant-pipeline: `bundle_contract` public API + reader resolution.
- renquant-common: digest impl (M6) + the validation contract fixture.
- umbrella: store, writer tools, break-glass, GC, census, views.
- Shadow arm: out of scope for v1 (unchanged).

## 6. Open questions (narrowed)

1. Retention: last N=8 bundles + 90-day run-bundle references — confirm.
2. Per-ticker tournament store: explicitly out of scope for schema v1;
   revisit only if it ever feeds the serving pair.
