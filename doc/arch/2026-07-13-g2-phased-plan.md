# G2 Crypto Trading — Phased Execution Plan

Moved here from `renquant-orchestrator` (orchestrator PR #503/#507 review):
this is cross-repo operating-model material, not an orchestrator-local
source of truth.

**Status note (added on move, 2026-07-13):** this plan predates the SMA50
trend-following pivot (base-data#45, orchestrator#505) and describes an
earlier XGB-model-based Phase 1. orchestrator#505's data/strategy/pipeline
ownership and design are still under active architectural review as of
this date (Codex CHANGES_REQUESTED on #505/#194/#506) — nothing below
should be read as an approved, settled design; treat the phase/deliverable
structure as historical context. The replacement program is
[`2026-07-13-four-goal-program-reset.md`](2026-07-13-four-goal-program-reset.md).

Post-incident re-plan. Replaces the implicit "build whatever is tractable" approach
with a dependency-driven, phase-gated plan. Each phase has a VERIFIABLE exit
criterion; the scheduler (D-C11) does NOT re-deploy until Phase 2 clears.

Reference: `doc/design/2026-07-10-crypto-trading-rfc.md` (the RFC is sound; the
execution order was not).

---

## Current state (2026-07-13)

**Done (infrastructure):** D-C1, D-C2, D-C4, D-C5, D-C6/C7 partial, D-C11,
D-C12 — 7 deliverables merged. All are plumbing/data/infrastructure.

**Not done (signal chain):** D-C3, D-C6/C7 remaining, D-C8a/b, D-C9, D-C10,
D-C13 — 6 deliverables. These form a serial chain: features → model → strategy.

**Scheduler:** UNLOADED. Will not re-deploy until Phase 2 exit.

**Open PRs:** exec#38 (enum fix), orch#502 (env rename) — non-blocking cleanup.

---

## Phase 0 — SHORT-TERM: data foundation (target: 1-2 weeks)

**Goal:** Produce a crypto alpha158 panel dataset with features and labels that a
model can train on.

| Task | Deliverable | Repo | Dependency | Exit criterion |
|---|---|---|---|---|
| Build crypto panel | D-C3 | base-data | D-C2 (done) | `crypto_panel.parquet` exists with ≥90 calendar days, ≥15 pairs, 158 features + h=20 calendar-day label |
| Pipeline calendar threading | D-C6/C7 rest | pipeline | D-C1 (done) | `make test` passes with `asset_class=crypto` config; freshness/hold-clock/annualization use 365d; wash-sale bypassed for crypto |

**Exit criterion for Phase 0:** A reproducible pipeline command produces a dated,
fingerprinted crypto panel parquet that passes schema + completeness checks.

**What Phase 0 does NOT include:** model training, strategy config, cost model,
deployment. Resist the temptation to build ahead.

---

## Phase 1 — MEDIUM-TERM: model + strategy scaffold (target: 2-4 weeks after Phase 0)

**Goal:** A WF-gated crypto XGB panel model trained on Phase 0 data, with a
minimal strategy repo that can consume it.

| Task | Deliverable | Repo | Dependency | Exit criterion |
|---|---|---|---|---|
| Cost model primitive | D-C8a | model-common | none | Unit-tested fee/spread/increment accounting; consumed by both replay and runtime |
| Crypto model training | D-C9 | model | D-C3 panel, D-C8a | Trained model passes WF gate (OOS IC > placebo, net-of-cost); model card published |
| Crypto WF evaluation | D-C8b | model | D-C8a, D-C9 | BTC-baseline comparison documented; promotion decision in gate harness |
| Strategy repo scaffold | D-C10 | NEW repo | D-C9 model | Configs (active/golden/shadow), validator, universe snapshot, sleeve budget cap, risk rails |
| Artifact registry entry | D-C13 | artifacts | D-C9 model | Crypto model registered with promotion contract |

**Exit criterion for Phase 1:** `make test` in the new strategy repo passes;
the crypto model artifact is registered and loadable; a DRY RUN (no orders)
through the full pipeline produces a non-trivial signal for ≥10 pairs.

---

## Phase 2 — LONG-TERM: integration + staged rollout (target: 4-8 weeks after Phase 1)

**Goal:** Paper-verified crypto trading with staged operator sign-offs.

| Stage | Description | Gate | Operator sign-off |
|---|---|---|---|
| 2a: Re-arm scheduler | Re-deploy D-C11 with real model + strategy config | D-C9/D-C10 loadable, dry run passes | YES — explicit re-authorization |
| 2b: Stage 0 battery | Full paper battery with real model (not just connectivity) | All step checks PASS including signal quality | NO (mechanical) |
| 2c: Stage 1 shadow | ≥5 trading days of shadow replay; no anomalies | Shadow log audit clean | YES |
| 2d: Stage 2 paper canary | Paper account live trades, $0 real risk | ≥10 trading days, no operational failures | YES |
| 2e: Stage 2.5 evaluation | Prospective economic evaluation per RFC §6.1 | Positive net-of-cost expectancy [VERIFIED] | YES — capital decision |
| 2f: Stage 3 live canary | $200 real capital, ≤3 pairs | ≥10 trading days, P&L within 2σ of paper | YES — capital at risk |

**EXIT CRITERION FOR PHASE 2:** Operator has signed off on Stage 3; system runs
autonomously 24/7 with monitoring and alerting.

---

## Anti-patterns to avoid (from the 2026-07-13 incident)

1. **Don't deploy inert scaffolding.** If a component can't produce decisions, it
   stays as code in a PR, not a running service.
2. **Lead with the blocker chain.** Progress report = "Phase 0 blocked on D-C3
   panel builder (ETA: X)", not "7 deliverables merged!"
3. **Don't build ahead.** Phase 1 work doesn't start until Phase 0 exits. Resist
   the urge to scaffold the strategy repo while waiting for the panel builder.
4. **Each phase exit is operator-reviewed.** No implicit "Phase 0 is done, moving
   to Phase 1." File a progress doc and get acknowledgment.

---

## Honest current ETA

- Phase 0 (data): **1-2 weeks** — D-C3 panel builder is the critical path.
  D-C2 bars ingestion is merged, so features can be computed from existing bars.
  Pipeline crypto threading (D-C6/C7) is parallelizable.
- Phase 1 (model): **2-4 weeks** after Phase 0 — depends on data quality. If the
  first training run shows signal (IC > placebo), fast. If not, research cycle.
- Phase 2 (live): **4-8 weeks** after Phase 1 — staged rollout with operator gates.

**Total realistic timeline: 7-14 weeks to Stage 3 live canary.** This is an
honest estimate, not an optimistic one. The largest risk is Phase 1: crypto may
not have enough cross-sectional signal for the XGB panel approach, and the
research cycle to find out is uncertain.
