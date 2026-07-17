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

(r3 note: the authoritative store lives in the renquant-artifacts registry per §5; the layout below describes the store SCHEMA and the umbrella's read-only materialization uses the same shape.)

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
7  append PREPARE record {generation+1, bundle_id, authorization}
   to OPERATIONS.jsonl; fsync                    # BEFORE the flip
8  write ACTIVE.tmp = "<generation+1> <bundle_id>"; fsync(ACTIVE.tmp)
9  rename ACTIVE.tmp → ACTIVE; fsync(prod/ dirfd)
10 append ACTIVATE record {generation+1, bound to the PREPARE}; fsync;
   unlock
```

**Activation-audit invariant (r3, review finding 2):** a generation may
serve ONLY if its PREPARE record exists. The reader resolves ACTIVE and
verifies the generation has a PREPARE record in OPERATIONS.jsonl; a
generation with PREPARE but no ACTIVATE is a detected crash interval —
the reader serves it (state is valid and was fully prepared+validated)
and the next writer/auditor MUST commit a RECOVERY record naming the
interval before any further mutation, with an alarm. A generation with NO
PREPARE record is REFUSED (fail-closed, never serve unaudited state).
Crash at/before step 9 ⇒ previous ACTIVE intact. The kill-injection
suite (§4) crashes at EVERY numbered step, after each fsync, and
SPECIFICALLY in the rename→ACTIVATE interval, proving the next reader
never serves a generation without its PREPARE record.

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

Reader: read ACTIVE (generation + id) → verify the generation's PREPARE
record (§2.3) → open bundles/<id>/ by dirfd → read manifest via that
dirfd → verify member digests → serve.

**Retention guarantee (r3, review finding 3 — reference-rooted GC):** the
replay guarantee is scoped precisely: ANY run whose persisted run bundle
is retained can be replayed against its exact archived bundle. GC
therefore deletes a bundle ONLY if (a) it is not the ACTIVE target, (b)
not an ancestor within the rollback window, and (c) NO retained run
bundle references it — determined by querying the run-bundle store for
the bundle_id before delete, with no time cutoff. Run-bundle retention
itself is the single knob: bundles live exactly as long as any run that
used them remains auditable. The GC acceptance test proves this by
constructing an old-run reference and demonstrating the bundle survives
GC while an unreferenced sibling is collected. GC serializes on the store
flock; unlink-after-open keeps a mid-read dirfd valid on POSIX. GC
deletions are operation-log records.

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

## 5. Ownership boundaries (r3 — registry-correct; review finding 1)

Per RENQUANT_REPOS.md, renquant-artifacts is the artifact registry and
promotion-status owner; the prior draft wrongly created a second
registry inside the umbrella working tree.

- **renquant-artifacts**: AUTHORITATIVE bundle publication/registration —
  immutable bundle identity, the manifest schema, the authorization/
  operation log contract, retention policy and GC. Bundles are published
  into the artifacts store and resolved by immutable reference.
- **renquant-pipeline**: the `bundle_contract` public pair-validation API
  and the runtime reader (resolve → verify → serve).
- **renquant-orchestrator**: resolves the registered bundle at run time
  and records `{bundle_id, manifest_digest, member digests, generation}`
  in each run bundle (the existing provenance surface).
- **umbrella**: this RFC, the pins, and an explicitly READ-ONLY
  deployment materialization of the registered active bundle for the
  local runtime (a cache, never a publication authority); the census and
  compatibility views live here because the legacy flat paths do.
- Writer tools (promote/refresh/break-glass) INVOKE the artifacts-owned
  publication API; they do not own the store.
- Shadow arm: out of scope for v1 (unchanged).

## 6. Open questions (narrowed)

1. Retention: last N=8 bundles + 90-day run-bundle references — confirm.
2. Per-ticker tournament store: explicitly out of scope for schema v1;
   revisit only if it ever feeds the serving pair.
