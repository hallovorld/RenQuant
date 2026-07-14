# Goal Governance Process — Binding Operating Rules

Moved here from `renquant-orchestrator` (orchestrator PR #503/#507 review):
this applies to all goals and all agents across the RenQuant system, so it
belongs as umbrella-repo operating-model material, not an orchestrator-local
source of truth. The orchestrator scheduler incident that triggered it is
recorded separately at
`renquant-orchestrator/doc/progress/2026-07-13-g2-premature-deployment-retro.md`.

Status: DRAFT — requires operator review before becoming binding.
Date: 2026-07-13
Trigger: G2 premature-deployment incident + operator directive on process gaps.

**Current program proposal:**
[`2026-07-13-four-goal-program-reset.md`](2026-07-13-four-goal-program-reset.md)
defines the active G1-G4 dependency order, evidence contract, and stop rules.
It supersedes this document's incomplete G1/G2/G4 execution assumptions while
this governance process remains the lifecycle and reporting baseline.

---

## 1. Purpose

This document defines the MANDATORY process for defining, tracking, delivering,
and verifying goals in the RenQuant system. It exists because:

1. Goals were reported as "done" or "near-ready" without measurable criteria.
2. Infrastructure work was conflated with capability delivery.
3. No formal gate existed between "code merged" and "goal delivered."

These rules are NON-NEGOTIABLE once approved. They apply to all goals (G1-G4 and
future) and all agents operating in the RenQuant system.

---

## 2. Goal lifecycle

Every goal passes through exactly these stages, in order:

```
DEFINED → IN_PROGRESS → PHASE_COMPLETE(n) → DELIVERED → VERIFIED → CLOSED
```

No stage may be skipped. Each transition requires specific evidence.

---

## 3. Goal definition (DEFINED stage)

A goal is NOT defined until it has ALL of the following, documented in a design
doc or RFC:

### 3.1 Acceptance criteria

A numbered list of conditions that are ALL true when the goal is delivered.
Each criterion must be:

- **Measurable**: a number, a test result, a binary state — not "works well" or
  "is ready"
- **Verifiable**: an operator or automated check can confirm it independently
- **Unambiguous**: two reasonable people reading it reach the same verdict

Example (G4):
```
AC-1: Phase A runner produces a verdict on ≥200 common dates
AC-2: If L1_BEATS_CHAMPION at p<0.05, L2 design doc filed within 1 week
AC-3: If CHAMPION_RETAINED, experiment closes with no further work
```

Anti-example:
```
❌ "Crypto trading works"
❌ "Model is good enough"
❌ "Pipeline supports crypto"
```

### 3.2 Measurable metrics

Each goal specifies the metrics it will be judged by, with:

- **Metric name**: what is measured
- **Source of truth**: where the number comes from (which DB, which script, which
  log)
- **Threshold**: the value that separates pass from fail
- **Measurement method**: how to reproduce the number

Example (G2):
```
M-1: OOS IC (rank, cross-sectional) > shuffled-label placebo + 0.005
     Source: WF gate output artifact
     Method: run_wf_gate.py --scorer crypto_xgb --mode oos_ic
M-2: Net-of-cost Sharpe > 0 over ≥60 calendar-day shadow period
     Source: shadow_replay_audit.jsonl
     Method: scripts/crypto_shadow_audit.py --min-days 60
M-3: Stage-0 battery: all step checks PASS including signal quality
     Source: battery CLI output
     Method: python scripts/crypto_stage0_battery.py --full
```

### 3.3 Deliverables list

Concrete artifacts (PRs, files, configs, services) that must exist. Each
deliverable has:

- **ID** (e.g., D-C3)
- **Repo**
- **Description**
- **Acceptance test**: how to verify THIS deliverable is done (not the goal)
- **Dependency**: which other deliverables must be done first

### 3.4 Phase gates

Goals with >4 deliverables MUST be broken into phases. Each phase has:

- A subset of deliverables
- Its own acceptance criteria (a strict subset of the goal's AC)
- An explicit exit criterion
- A dependency on prior phases

---

## 4. Progress reporting (IN_PROGRESS stage)

### 4.1 Report structure (mandatory)

Every progress report follows this format, in this order:

```
1. PHASE STATUS: Phase N of M — [NOT_STARTED | IN_PROGRESS | BLOCKED | COMPLETE]
2. BLOCKER CHAIN: what is preventing the next deliverable (if any)
3. METRICS (current vs target): measured values against acceptance criteria
4. DELIVERABLES: done / in-progress / not-started counts
5. NEXT ACTION: the single most important thing to do next
6. RISKS: what could prevent delivery
```

### 4.2 Prohibited reporting patterns

The following are BANNED:

- **Done-list leading**: "We merged 7 PRs!" without stating what's MISSING
- **Infrastructure-as-progress**: reporting plumbing (scheduler, battery, CI) as
  goal progress when the decision-making chain doesn't exist
- **Implicit readiness**: any language suggesting operational readiness ("deployed",
  "live", "running") when acceptance criteria are not met
- **Vague status**: "making progress", "almost ready", "close to done"
- **Percentage claims**: "70% done" without a denominator definition

### 4.3 Honest vocabulary

| Say | When |
|-----|------|
| "Phase 0 BLOCKED on D-C3" | A specific deliverable prevents progress |
| "7/13 deliverables merged, 0/4 acceptance criteria met" | Infrastructure done but no capability |
| "NOT STARTED — no work has begun on this phase" | Truth |
| "Phase A COMPLETE — verdict: L1_BEATS_CHAMPION (p=0.0295)" | Measurable result achieved |

| Don't say | Why |
|-----------|-----|
| "G2 is making strong progress" | No metric attached |
| "Scheduler deployed and running" | Implies capability that doesn't exist |
| "Model is nearly ready" | No measurement of "nearly" |
| "Should be done soon" | No timeline, no evidence |

---

## 5. Delivery claim (DELIVERED stage)

### 5.1 Delivery gate

A goal CANNOT be claimed as delivered until:

1. **ALL acceptance criteria are met** — with [VERIFIED] evidence for each
2. **ALL deliverables in the final phase are merged** to their respective mains
3. **ALL metrics meet their thresholds** — with source data cited
4. **A delivery report exists** at `doc/progress/<date>-<goal>-delivery.md`
   containing the evidence block
5. **Operator has reviewed** the delivery report (for goals requiring sign-off)

### 5.2 Evidence block format

Every delivery claim includes an evidence block:

```markdown
## Delivery evidence

| AC | Criterion | Result | Source | Verified |
|----|-----------|--------|--------|----------|
| AC-1 | OOS IC > placebo + 0.005 | 0.042 > 0.009 [PASS] | wf_gate_output_20260801.json | [VERIFIED] |
| AC-2 | Net-of-cost Sharpe > 0 | 0.31 [PASS] | shadow_audit_20260815.json | [VERIFIED] |
| AC-3 | Battery all PASS | 11/11 [PASS] | battery_20260815.log | [VERIFIED] |

Deliverables: 13/13 merged (list with PR numbers)
Metrics: 3/3 passing (table above)
Operator sign-off: [pending/approved] — date, method
```

### 5.3 What "delivered" does NOT mean

- ❌ "Code is merged" — merged ≠ delivered; it must also be INTEGRATED and VERIFIED
- ❌ "Tests pass" — tests verify code correctness, not goal completion
- ❌ "It runs without errors" — running ≠ producing value
- ❌ "Infrastructure is in place" — infrastructure without decision capability = 0 value

---

## 6. Verification (VERIFIED stage)

After delivery is claimed:

1. **Independent verification**: the operator (or Codex adversarial) re-checks
   the evidence block — can the numbers be reproduced?
2. **Metric recomputation**: metrics are re-run from scratch, not taken from the
   delivery report at face value
3. **Integration test**: the delivered capability is exercised end-to-end in the
   actual production environment (paper or shadow mode)

Only after verification passes does the goal move to CLOSED.

---

## 7. Retrospective trigger

A goal governance retrospective is MANDATORY when:

1. A goal is reported as delivered but later found to not meet acceptance criteria
2. A phase is skipped or its exit criterion is not checked
3. Metrics are cited without source data
4. Infrastructure is deployed before the decision chain is complete
5. The operator identifies a reporting failure

The retrospective must identify:
- Which section of this process was violated
- Why the violation was not caught
- What mechanical check would prevent recurrence

---

## 8. Application to current goals

### G1 (Pipeline/S-FRAC)
- Acceptance criteria: UNDEFINED — needs to be written
- Current status: cannot be accurately reported without AC

### G2 (Crypto trading)
- Acceptance criteria: defined in `doc/arch/2026-07-13-g2-phased-plan.md`
  (phase-level exit criteria) -- see that document's status note: it predates
  the SMA50 pivot and orchestrator#505 is still under architectural review
- Goal-level AC needed: what does "crypto trading is live" actually mean?

### G3 (Architecture refactoring) — REDEFINED 2026-07-13
- Prior scope: one-shot audit → DONE (orch #492 merged)
- New scope: incremental safe refactoring per violation registry
- Acceptance criteria: the active-runtime gate and its evidence contract are
  defined in `doc/arch/2026-07-13-four-goal-program-reset.md` §4; the
  incremental migration sequence remains in
  `doc/arch/2026-07-13-g3-refactoring-plan.md`.
- Phased: A (tripwires) → B (single-source consolidation) → C (dead code) → D+ (migrations)

### G4 (Ensemble)
- Phase A AC partially defined in design doc §4.5
- Goal-level AC: UNDEFINED — when does the ensemble experiment "conclude"?

**Action required**: G1, G2 (goal-level), and G4 need acceptance criteria
written and reviewed before any further delivery claims.
