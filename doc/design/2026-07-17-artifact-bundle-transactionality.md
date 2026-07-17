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

## 2. Design

### 2.1 Bundle store

```
backtesting/renquant_104/artifacts/prod/bundles/<bundle_id>/
    manifest.json
    panel-ltr.alpha158_fund.json
    panel-rank-calibration.json
    [panel-rank-calibration.<regime>.json ...]   # future
backtesting/renquant_104/artifacts/prod/ACTIVE   # pointer file, one line: <bundle_id>
```

`bundle_id` = `<utc-ts>-<short-digest>` (content-derived, collision-free).
POINTER FILE, not a symlink: atomic via write-tmp + `rename(2)`, identical
semantics on every filesystem, and trivially auditable in git history if
tracked.

### 2.2 Manifest (the pair-level contract)

```json
{
  "bundle_id": "...",
  "created_at": "...", "created_by": "wf_promote|monthly_refresh|restamp|manual",
  "parent_bundle": "<bundle_id|null>",
  "members": {"panel-ltr.alpha158_fund.json": "sha256:<file>", ...},
  "bindings": {
    "panel_model_content_v1": "sha256:...",       // renquant-common v1 impl (M6)
    "calibrator_scorer_binding_v1": "sha256:...", // MUST equal the line above
    "config_fingerprint": "sha256:...",
    "wf_gate_verdict": {"passed": true, "diagnostic_only": true, "...": "..."}
  }
}
```

Manifest creation VALIDATES the bindings (runs the real pipeline matcher
in-process — the same `_assert_calibrator_matches_scorer` that guards the
daily); a manifest whose bindings do not verify cannot be written. The v1
digest basis is the single renquant-common implementation (M6 unification
is a dependency of phase 3, not of phases 0-2).

### 2.3 Writer protocol (all four writers)

```
with store_lock():                      # flock on bundles/.lock
    build bundles/<new_id>/ fully       # members + manifest, fsync
    validate manifest bindings          # fail → delete dir, abort
    write ACTIVE.tmp; rename → ACTIVE   # the ONLY mutation of serving state
```

Rollback = pointer swap to `parent_bundle`. A crash at ANY step leaves the
previous bundle fully active; the worst residue is an orphan directory
(GC'd by a janitor, never load-bearing). Concurrent writers serialize on
the lock — today's weekly/monthly/restamp interleavings become impossible.

### 2.4 Reader contract

The runtime resolves the pair ONLY via `ACTIVE` → manifest → members, and
verifies member file digests against the manifest at load (fail-closed with
a named remedy). The existing calibrator/scorer matcher stays as
defense-in-depth; after this design it should never fire, and its firing
becomes a page-worthy anomaly (sentinel hook).

## 3. Migration (default-ON at the end, no dark shipping)

- **P0 (reader fallback)**: pipeline loader learns the pointer path; if
  `ACTIVE` is absent → legacy flat paths + WARN. Ships inert-safe.
- **P1 (seal + flip)**: seal the current verified-good pair as the first
  bundle (offline validation = the 07-16 sandbox procedure, now code);
  flip `ACTIVE`. Legacy flat paths remain as COPIES temporarily.
- **P2 (writers)**: wf_promote, monthly refresh, restamp tooling emit
  bundles + pointer swaps; their private backup/rollback files retire in
  favor of `parent_bundle`. Flat paths become generated views for any
  unmigrated reader (sim/scripts inventory required — census step).
- **P3 (fail closed)**: remove the fallback; missing pointer/digest
  mismatch aborts the run. Requires the M6 v1-only flip
  (`accept_legacy_stamps=false`) to land with it.

## 4. Verification (AC4 acceptance)

1. **Kill-injection**: harness crashes the writer at every step boundary
   (after member write, after manifest, mid-rename); after each crash the
   loader must resolve a fully consistent pair. Automated in CI.
2. **Concurrency**: two writers under contention → serialized, both
   outcomes consistent.
3. **Incident replay**: scripted re-enactment of the 07-13→16 churn
   (promote-reject rollback + refresh + restamp interleaved) against the
   bundle store → the serving pair never desynchronizes.
4. **Live drill**: one monthly-refresh cycle through the new protocol with
   the sentinel watching; the calibrator-mismatch class alert count stays 0.

## 5. Ownership boundaries

- umbrella: bundle store layout, writer tooling, seal/flip scripts, GC.
- renquant-pipeline: reader resolution + manifest verification in the
  loader path (kernel-owned; no umbrella-local loader logic).
- renquant-common: the single v1 digest implementation (M6).
- Shadow arm: OUT of scope for v1 (its own pair gets a bundle in a
  follow-up; shadow breakage never blocks prod).

## 6. Open questions for review

1. Track `bundles/` + `ACTIVE` in git, or filesystem-only with the drift
   sentinel watching? (Draft position: filesystem-only + sentinel;
   git-tracking model weights re-arms the checkout-clobber trap.)
2. GC policy: keep last N bundles + all bundles referenced by run bundles
   in the last M days?
3. Does the two-arm freeze (artifact_store contract) reference flat paths
   that P2 must preserve as views?
4. Where does the per-ticker tournament model store fit — same mechanism
   later, or explicitly never?
