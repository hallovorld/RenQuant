# Four-Goal Program Reset: Evidence Before Deployment

Status: **PROPOSED — replaces the current operating assumptions for G1-G4.**
Date: 2026-07-13
Owners: RenQuant umbrella for program governance; named subrepos for delivery.

This document is a corrective plan, not a progress report. A merged PR, a
passing unit test, an installed scheduler, or a completed batch is not evidence
that a goal has created economic value or is operationally safe.

It supersedes the execution assumptions in `2026-07-13-g2-phased-plan.md` and
the incomplete goal definitions noted in `goal-governance-process.md`. Those
documents remain useful historical records. Where they conflict with this
document, this document governs the four active goals.

## 1. Executive decision

1. **G3 is gate zero.** No model or strategy can be promoted, and no scheduler
   can be armed, until the authoritative runtime and promotion paths are
   demonstrated to be single-source, pinned, and fail-closed.
2. **G1 is two experiments, not one deployment claim.** Rounding/sizing
   fidelity and equal-weight policy are separately identified treatments. A
   result from one is not evidence for the other.
3. **G2 is a falsification program, not a serial implementation plan.** It may
   produce data and research evidence, but may not create a trading sleeve
   until an independently reproducible, net-of-cost, prospective decision
   chain exists.
4. **G4 is evidence-graph first.** Ensemble layers are not added because they
   are available. They are considered only after an immutable, leakage-safe
   comparison proves incremental value over the incumbent on untouched time.

All four goals are **research-only / no promotion** until their stated gates
are satisfied. This is a deliberate stop condition, not a lack of progress.

## 2. Common non-negotiable contract

Every accepted observation, comparison, and promotion decision must carry a
reproducible evidence tuple:

```
{code_commit_set, lockfile_digest, strategy_config_digest,
 data_manifest_digest, data_asof/watermarks, calendar_digest,
 feature_schema_digest, label_definition_digest, model_artifact_digest,
 calibration_digest, decision_trace_digest, cost_model_digest,
 experiment_plan_digest, run_id}
```

The producer and consumer must resolve the same immutable identity. A relative
path, suffix match, mutable default, environment fallback, latest alias, or
working-tree import is not an identity. Missing fields fail closed; a human
note cannot repair an incomplete run bundle after the fact.

For every experiment, freeze before execution:

- the hypothesis, regime scope, treatment and comparator;
- the primary metric, economic metric, downside metric, and minimum effect
  worth acting on;
- all candidate variants and the model/feature selection rule;
- split boundaries, embargo/purge rule, and the untouched confirmation window;
- transaction-cost, financing, tax, borrow, fill, and missing-data assumptions;
- a stopping rule and the exact decision each result may support.

Reports show regime-stratified results before pooled results. They include the
number of independent time blocks, not merely the number of rows or daily
observations. A confidence interval or p-value does not establish tradability;
the economic criterion must also pass after the predeclared costs.

## 3. Dependency order and permissions

| Gate | Required evidence | Enables | Explicitly does not enable |
|---|---|---|---|
| G3-0, canonical path | one deterministic producer-to-consumer run bundle and deliberate missing/ambiguous-input failures | offline research output registration | scheduler, paper or live order submission |
| G1 design gate | matched-state paired protocol and calendar/regime block plan | prospective shadow collection | allocation-policy deployment |
| G2 data gate | frozen UTC data/label manifest, coverage and lineage checks | offline model falsification | strategy/pipeline scheduling |
| G4 evidence gate | immutable score ledger for incumbent and candidates on common dates | exploratory comparison only | ensemble promotion or stacking |
| confirmatory gate | untouched, preregistered, net-of-cost external evaluation | an operator review | automatic promotion |

`renquant-common` owns reusable contracts and evaluators; `renquant-base-data`
owns source data, frozen calendars, manifests, and feature/label lineage;
`renquant-model` owns training and score artifacts; `renquant-artifacts` owns
artifact validation and promotion records; `renquant-strategy-104` owns policy
and sleeve constraints; `renquant-pipeline` owns runtime decisions and intents;
`renquant-execution` owns broker semantics and audit; `renquant-backtesting`
owns simulation and forensics; `renquant-orchestrator` only assembles pinned
components and persists the run bundle. The umbrella contains this program and
the integration/rollback lock, not new domain implementations.

## 4. G3: authoritative runtime and promotion path

### Problem to solve

The critical issue is not the number of audit violations closed. It is whether
the code that generated an accepted score, the artifact that was promoted, and
the code that would create a live intent are provably the same declared system.
The current dual-home kernel/live twins, training-from-umbrella path, and
runtime/promotion mismatch make that claim unsafe.

### P0 sequence

1. **Trace, do not move first.** Produce a machine-readable route inventory
   for training, walk-forward evaluation, promotion, simulation, and inference:
   executable entry point, imported package origin, subrepo SHA, artifact/data
   inputs, output, and any working-tree or fallback path. V-001, V-002 and
   V-014 stay open until the inventory is complete.
2. **Choose one authority per capability.** The model factory trains; artifacts
   publish; strategy supplies policy; pipeline makes decisions; execution mutates
   the broker; orchestrator wires pins. No consumer imports a model factory and
   no umbrella kernel is an alternate production implementation.
3. **Cut over one path at a time with a read-only parity harness.** Given an
   immutable fixture bundle, the canonical path must reproduce the approved
   decision trace. The old path is then disabled before its code is deleted.
   No dual-write or dual-authority interval is allowed.
4. **Enforce at boundaries.** Import rules, no-working-tree execution guards,
   artifact resolver checks, and run-bundle completeness checks are release
   gates. They are not substitutes for step 3.

### G3 acceptance

- Every full run resolves only lock-pinned subrepos and immutable artifact/data
  identities; an absent or conflicting identity aborts before an intent exists.
- A fixture proves training/evaluation provenance, artifact publication, and
  runtime consumption end-to-end; the output trace and all identity digests are
  reproducible.
- Fault injection for stale pin, missing manifest, duplicate authority, and
  fallback import fails closed with an auditable reason.
- No active entry point imports or executes the retired umbrella twins or a
  training working directory. V-001/V-002/V-014 are closed by route inventory
  plus parity and fault-injection evidence, not a file-count reduction.

## 5. G1: cash drag and equal weight

Cash drag is an outcome, not a diagnosis. It can arise from alpha selectivity,
risk limits, unavailable cash, rounding, price/lot constraints, stale data,
pending settlement, taxes, or intentional reserve policy. Equal weight does
not mechanically imply a positive return, and a better deployment ratio alone
does not identify an allocation benefit.

### G1-A: sizing-fidelity experiment

Question: conditional on identical eligible orders and target exposures, does
the current quantization/one-share rule suppress economically valid orders?

- Construct paired shadow arms from the exact same frozen account snapshot:
  cash, positions, lots, pending settlement, reservations, prices, order
  eligibility, costs, and model/strategy decision trace.
- Hold selection, gross/net targets, risk caps, and sell logic fixed. The sole
  treatment is the rounding/quantization implementation.
- Primary metric: difference in rejected or clipped eligible notional and its
  reasons. Safety metrics: cash debit reconciliation, no overspend, no
  displaced higher-priority order, and post-trade constraint satisfaction.
- This experiment can justify a sizing-correctness change only. It cannot
  justify equal-weight allocation or forecast an excess return.

### G1-B: equal-weight policy experiment

Question: at the same investable capital, risk constraints, and turnover/cost
model, does a simple equal-weight allocator improve a predeclared utility over
the incumbent policy within a specified regime?

- Start each arm from the identical state; do not compare one arm's initial
  cash with the other's live positions. Equalize target exposure or estimate
  the policy effect conditional on exposure; otherwise report the result as an
  exposure intervention, not allocator superiority.
- Predeclare the universe, selection set, rebalance cadence, maximum names,
  fractional versus integer share policy, reserves, taxes, slippage, and the
  no-trade treatment. Keep the score/selection model constant.
- Use non-overlapping calendar/regime blocks or a dependence-aware block
  bootstrap whose block length is fixed before the result. Five daily points or
  a Newey-West statistic on a few daily returns is insufficient evidence.
- Primary decision metric: net-of-all-costs regime-stratified utility versus the
  incumbent, with a minimum economic effect. Secondary: drawdown, turnover,
  exposure, concentration, tax, and capacity. Require a separate untouched
  confirmation period before any policy promotion.

G1 stops with `NO_ACTION` if either treatment lacks sufficient independent
blocks or fails the predeclared economic and safety criteria. It does not
continue by changing the allocator after inspecting the same holdout.

## 6. G2: crypto sleeve as a falsifiable capability

The prior plan must not treat a 90-calendar-day, 15-pair, 158-feature panel as
adequate evidence for a flexible panel model. It is a feasibility dataset, not
a basis for model selection, stacking, or a trading claim. The crypto calendar
is 24/7 UTC; labels described as "20 calendar days" cannot be implemented as
20 observed rows when gaps and pair availability differ.

### Ordered gates

1. **Data integrity:** base-data defines a frozen UTC calendar, pair inclusion
   policy, watermarks, observation masks, availability time, feature schema,
   and label horizon. Labels use the declared calendar or are explicitly named
   row horizons. Missing observations are visible and handled by a predeclared
   rule. The manifest digests the actual artifact and all inputs, not a short
   CSV-derived proxy.
2. **Generic evaluation:** common/backtesting provide one reusable net-cost
   evaluator. Fees, spread/slippage, funding/financing, turnover, unavailable
   bars, and delist/venue survivorship assumptions are logged per decision. A
   crypto-specific implementation must not silently create a second evaluator.
3. **Offline falsification:** model runs a preregistered walk-forward study with
   purging/embargo appropriate to the forward label, contemporaneous UTC
   coverage, time-shift and shuffled-label placebos, and regime-stratified
   metrics. The result must beat a simple frozen BTC and naive portfolio
   comparator after costs; otherwise the result is `REJECTED` or `DIAGNOSTIC`,
   not a request for runtime work.
4. **Policy and artifact:** only a surviving artifact is published. A dedicated
   strategy policy owns crypto universe, sleeve budget, risk and no-trade
   behavior. Pipeline consumes published policy/artifacts and emits facts and
   intents; execution owns venue/broker details.
5. **Prospective observation:** after G3, a dry-run may verify plumbing but does
   not count as economics. Shadow and paper stages use a predeclared duration
   and stopped-on-anomaly rules. Any capital decision needs explicit operator
   authorization after an independently reproduced net-of-cost report.

No scheduler, strategy scaffold, or broker route is a G2 capability milestone
until gate 3 is passed. A model that fails the falsification gate closes the
goal honestly; it does not trigger a search across more models on the same
small sample.

## 7. G4: multi-model ensemble

An ensemble can reduce variance only when member errors are usefully diverse
and the combination weights are estimated without leaking the evaluation
period. A generic L1 runner cannot establish this. Inverse-variance weights,
stacking, and regime routing add estimation degrees of freedom and therefore
increase selection risk; they are not a default improvement.

### Design

**Phase E0 — evidence substrate.** Base-data emits immutable input manifests;
model emits dated, as-of constrained scores for every candidate and the
incumbent; artifacts records model/config/calibrator identities; a ledger
records score rows, availability timestamp, common-date coverage, and all
digests. A path suffix match is forbidden. A canonical immutable URI resolved
exactly or a content digest is required.

**Phase E1 — exploratory comparison.** On a frozen set of common dates, compare
each predeclared base model to the incumbent using the same score-to-portfolio
mapping and cost model. Report coverage, score correlation/error correlation,
regime IC, turnover, and net portfolio metrics. E1 can select a limited,
predeclared candidate family for confirmation; it cannot promote an ensemble.

**Phase E2 — nested combination selection.** Within each training fold only,
fit weights and any meta-model, including regime router thresholds. The outer
fold is never used to choose members, normalization, weights, or hyperparameters.
Compare in this order: incumbent, equal-weight ensemble, a predeclared
shrinkage/regularized blend, then a stacked/regime-routed model. Later layers
run only if the simpler layer clears the same economic and statistical hurdle.

**Phase E3 — confirmation.** Freeze the winner, all code/data/model identities,
and the number of candidates tried. Evaluate once on a chronologically later,
untouched external window. Require the predeclared net-of-cost effect, downside
constraint, regime outcome, and selection-bias adjustment. Failure means
`CHAMPION_RETAINED`; no retuning on that window.

**Phase E4 — shadow only.** After G3 and E3, publish a shadow artifact and
observe it prospectively. Promotion is a new operator decision, not an
automatic consequence of an E3 result.

### G4 acceptance

- All compared scores are attributable to immutable, exact identities and share
  a verified as-of date/calendar; missing coverage fails closed rather than
  changing the sample.
- Combination weights are fit strictly inside the outer evaluation window.
- The external confirmation reports all candidates/trials and a multiple-testing
  or selection-bias adjustment, alongside economic net-of-cost results.
- The simplest surviving model wins by a predeclared material margin; otherwise
  the incumbent remains champion.

## 8. Why the bar is this high

This program does not assume that a positive backtest predicts a positive
future return. Bailey et al. show why ordinary holdouts are unreliable after
researcher selection and propose a direct probability-of-backtest-overfitting
framework. Harvey, Liu, and Zhu show that data-mined return signals require a
higher statistical hurdle than a conventional t-statistic. DeMiguel, Garlappi,
and Uppal document that sophisticated allocation methods do not consistently
beat naive diversification out of sample. These results do not prove that G1,
G2, or G4 will fail. They rule out treating an in-sample or loosely specified
win as sufficient proof that it will help.

References:

- Bailey, Borwein, Lopez de Prado, and Zhu (2016), *The Probability of
  Backtest Overfitting*, Journal of Computational Finance, DOI
  `10.21314/JCF.2016.322`.
- Harvey, Liu, and Zhu (2016), *... and the Cross-Section of Expected
  Returns*, Review of Financial Studies, DOI `10.1093/rfs/hhv059`.
- DeMiguel, Garlappi, and Uppal (2009), *Optimal Versus Naive
  Diversification*, Review of Financial Studies, DOI `10.1093/rfs/hhm075`.

## 9. Immediate work queue and stop rules

| Priority | Work | Owning repo(s) | Done only when |
|---|---|---|---|
| P0 | G3 route inventory and pinned-bundle parity/fault harness | orchestrator, pipeline, model, artifacts, umbrella lock | canonical execution is demonstrated and ambiguity fails closed |
| P1 | G1 matched-state protocol and block-level power calculation | backtesting, strategy, artifacts | protocol freezes treatment, exposure, state and independent-block requirement |
| P1 | G2 frozen UTC manifest and generic cost evaluation | base-data, common, backtesting | labels/coverage/lineage and costs are reproducible |
| P1 | G4 exact-identity score/evidence ledger | model, artifacts, base-data | candidate and incumbent scores are comparable on common dates without fallback identity |
| P2 | G2/G4 offline falsification studies | model, backtesting | an accepted or rejected evidence report exists |

Stop immediately and record `INVALID_EXPERIMENT` when a run has an ambiguous
code/data/model identity, an unavailable timestamp, a changed treatment after
inspection, a contaminated holdout, a missing cost assumption, or a runtime
fallback. Do not repair the result by editing the report or rerunning with more
variants; repair the contract, create a new preregistered run ID, and retain
the invalid run in the ledger.

## 10. Reporting template

Every goal update must start with the following, not a PR count:

```
STATE: [RESEARCH_ONLY | BLOCKED | INVALID_EXPERIMENT | CONFIRMATION | CLOSED]
NEXT GATE: <one named gate>
EVIDENCE: <run IDs / immutable digests / missing prerequisite>
DECISION: <continue, reject, or operator decision needed>
```

There is no percentage-complete metric. The only valid status transition is
from evidence satisfying the next named gate.
