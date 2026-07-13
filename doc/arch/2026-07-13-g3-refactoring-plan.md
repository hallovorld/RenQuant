# G3 REDEFINED: Incremental Architecture Refactoring

Moved here from `renquant-orchestrator` (orchestrator PR #503/#507 review):
this is cross-repo operating-model material, not an orchestrator-local
source of truth.

Status: DRAFT — requires operator review.
Date: 2026-07-13.
Prior scope: one-shot audit (DONE — orch #492 merged 2026-07-12).
New scope: incremental refactoring, simplest-first, zero regressions.
Source: violation registry `doc/design/2026-07-10-architecture-compliance-registry.md`.

---

## 1. Goal statement

Incrementally refactor code across all RenQuant repos to align with the
multi-repo + pipeline architecture, using the violation registry as the roadmap.
Start from the simplest, lowest-risk fixes. Every change must:

- **Break zero functionality** — no behavior change unless explicitly pre-registered
- **Introduce zero regressions** — every fix carries a test proving equivalence
- **Be independently mergeable** — each PR is self-contained and useful alone

---

## 2. Acceptance criteria (goal-level)

| AC | Criterion | Metric | Source | Threshold |
|----|-----------|--------|--------|-----------|
| AC-1 | Zero regressions from any G3 PR | `make test` pass count ≥ pre-PR count in EVERY touched repo | CI + local | 0 new failures |
| AC-2 | Violation count decreases monotonically | Count of open violations in the registry | registry doc updates | Net decrease per phase, never increase |
| AC-3 | No behavior change without pre-registration | Every behavioral PR has a shadow comparison or golden-vector test | PR evidence block | 100% coverage |
| AC-4 | Each phase exit reviewed by operator | Progress doc with evidence block | doc/progress/ | Operator ACK before next phase |

---

## 3. Phases — simplest to hardest

### Phase A — Tripwires + parity tests (S, ~1 week)

**What:** Add mechanical drift detection BEFORE moving any code. Makes every
subsequent migration alarm-protected instead of review-protected.

**Registry items:** R0 (tripwires first).

| Task | Registry | Repo(s) | Risk | Test |
|------|----------|---------|------|------|
| A1: Parity hash tests for live/ broker twins | C3 | execution + CI | ZERO — read-only tests | Assert file hash equality; fail if drift detected |
| A2: Parity tests for duplicated constants ($1 floor, thresholds, tax rates) | C4, C5, B4 | common + pipeline + execution | ZERO — read-only tests | Import both sides, assert == |
| A3: Calendar implementation inventory test | B7 | common + pipeline | ZERO — read-only tests | Count calendar callsites; assert all use canonical |
| A4: Unknown-config-key warning counter | T7 | pipeline | ZERO — warning only, no reject | Log count of unknown keys per run |

**Exit criterion:** All parity tests committed and passing in CI across repos.
Failing tests document the CURRENT drift — they are evidence, not blockers.

### Phase B — Single-source contract consolidation (S-M, ~2 weeks)

**What:** Consolidate duplicated constants and contracts to single sources in
`renquant-common`, with importing sites replacing local copies. Start with
items that have NO behavioral impact (same value, just one source of truth).

**Registry items:** R6 (single-source contracts), B4, B7, C4, C5.

| Task | Registry | Repo(s) | Risk | Test |
|------|----------|---------|------|------|
| B1: Tax rates → single source in common | B4 | common + pipeline + execution | LOW — same values, single import | Before/after: identical tax calculation on 10 historical trades |
| B2: Calendar → adopt `renquant_common.market_calendar` at all 7 sites | B7 | pipeline + execution | LOW — fixes settlement holiday bug as side-effect | Before/after: identical session dates for 2024-2026; ADD holiday regression test FIRST |
| B3: `MIN_FRACTIONAL_NOTIONAL_USD` → single source | C4 | common + execution | ZERO — same value | Assert import == hardcoded value, then replace |
| B4: `compute_parent_intent_id` → single source | B3 | common + pipeline + execution | LOW — verified identical already | Golden-vector: 100 historical intent IDs match |

**Exit criterion:** All targeted constants/contracts have ONE source; parity
tests from Phase A now pass by construction (import, not duplicate).

### Phase C — Dead code and misplaced code cleanup (S, ~1 week)

**What:** Remove orphaned code and move misplaced code to correct repos. ZERO
behavioral impact — the code is either unused or behind permanent gates.

| Task | Registry | Repo(s) | Risk | Test |
|------|----------|---------|------|------|
| C1: Delete orphaned `parking_sleeve.py` in strategy-104 | D5 | strategy-104 | ZERO — 0 importers verified | grep confirms no imports |
| C2: Clean 5 dead top-level config keys | T7 | pipeline | ZERO — already ignored | Config schema update; silent-default count −5 |
| C3: Classify orchestrator scripts/ (62 files, 9 fitting models locally) | D8-9 | orchestrator | ZERO — inventory only | Manifest file with owner/keep/delete/migrate per script |

**Exit criterion:** Dead code removed; script inventory complete; all repos
`make test` green with same or higher pass count.

### Phase D+ — Larger migrations (M-L, future phases, requires operator approval)

These are NOT in scope for the initial G3 push. They are listed for sequencing
only; each requires its own design doc and acceptance criteria:

- **R1: launchd cutover** — scheduler only, kernel-identity preserved
- **R2: Kernel cutover** — the P0, staged per-module (exits → sizing → preflight → runner)
- **R3: Broker stack unification** — port protective features to execution
- **R4: Training migration** — Sunday tournament → model repo
- **R5: Fingerprint fail-closed** — behavior change, needs shadow period

---

## 4. Safety rails

1. **Golden-vector testing**: for any contract consolidation, run the
   canonical test vectors through BOTH old and new code paths; assert
   identical output. Commit the vectors as regression fixtures.
2. **One-repo-at-a-time PRs**: never bundle cross-repo changes in one PR.
   If B2 (calendar) touches pipeline AND execution, that's 2 PRs with the
   common change merged first.
3. **Test count monotonicity**: `make test` pass count must be ≥ pre-PR in
   every repo touched. A net test-count decrease blocks merge.
4. **No behavior change without shadow**: if a change COULD alter behavior
   (even if analysis says it won't), shadow-compare for ≥1 daily run
   before merging.

---

## 5. Progress reporting format (per governance process)

```
PHASE: A (Tripwires) — IN_PROGRESS
BLOCKER: none
METRICS:
  - Violations addressed this phase: 4/4 targeted
  - Regressions introduced: 0
  - Parity tests added: X
  - Test count delta: +Y across repos
DELIVERABLES: A1 merged, A2 in review, A3/A4 pending
NEXT: A2 codex review, then A3
RISKS: none identified — Phase A is read-only tests
```
