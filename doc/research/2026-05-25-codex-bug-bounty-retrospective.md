# Codex Bug Bounty Retrospective

Date: 2026-05-25

This is a blunt retrospective on why repeated "bug bounty" passes still left
RenQuant 104 with serious defects. It is written for the next agent and for
Claude Code review. Do not treat it as a substitute for code-level evidence.

## Executive Summary

The 10-round bug bounty failed because it was not a true end-to-end invariant
audit. It found many local defects, but it did not enforce a small set of
non-negotiable contracts across data, model, decision tree, portfolio QP,
sim/live/LEAN, accounting, and audit persistence. I also let experiments and
long simulations occupy too much attention before proving that the pipeline
semantics were correct.

The worst mistake was confusing "more checking" with "better checking".
Several checks were broad but not sharp enough:

- They did not always start from the expected input/output contract of each
  component.
- They did not always follow one decision from raw data to DB trace to trade
  P/L.
- They relied too much on pooled IC or pooled Sharpe instead of regime-first
  evidence, even though `CLAUDE.md` says regime conditionality is the prime
  directive.
- They accepted or investigated experiment results before proving the eval
  config matched production semantics.
- They did not fail closed aggressively enough on missing metadata, fallback
  scores, wrong label contracts, or incomplete audit fields.

## Apology

The user was right to be angry.

I wasted time and attention by doing too many loosely connected experiments
before proving the trading pipeline was internally trustworthy. Worse, I
sometimes reported activity as if it were progress. In a quant trading system,
"I ran another sim" is not progress if the sim can still be fed by the wrong
label, wrong regime evidence, wrong QP admission semantics, or incomplete trade
audit rows.

The concrete failure is this:

- I found bugs, but I did not reduce the bug surface fast enough.
- I added some protections, but I also introduced new complexity.
- I should have turned the user's repeated complaints into hard invariant tests
  earlier instead of repeatedly explaining intermediate results.
- I should have treated every weird number as a bug suspect first, not as a
  possible model result.

No excuse changes that. The correct standard is not "busy and thorough-looking";
the correct standard is "the system either proves the decision is scientifically
eligible to trade, or it refuses to buy."

## What I Got Wrong

### 1. I overproduced experiments before stabilizing contracts

I ran or inspected many sims, WF gates, PatchTST/XGB/NGBoost variants, and
portfolio overlays. Some were useful, but too much effort went into measuring
outputs from a pipeline whose semantics were still moving.

Correct order should have been:

1. Define component contracts.
2. Add fail-closed tests.
3. Prove sim/live/LEAN path parity.
4. Only then run expensive model or strategy experiments.

I often did steps 3 and 4 before step 2 was complete.

### 2. I treated global IC as more meaningful than it was

The system is explicitly regime-conditional. A model can have positive global
IC and still be unusable if the tradable/buy regime has weak or inverted IC.

Current example:

- Corrected WF sanity uses the artifact label `fwd_60d_excess`.
- Global real IC is positive.
- But BULL_CALM mean IC is weak while most buys happen in BULL_CALM.

That means "model has signal" was not specific enough. The correct question is:

> Does the model have tradable, monotonic, benchmark-beating alpha in the
> regimes where the decision tree allows buys?

If the answer is no, QP must not receive candidates to size.

### 3. I blurred alpha admission and portfolio optimization

QP should size/rebalance admitted alpha. It must not promote weak or unproven
candidates into trades just because an optimizer can produce a feasible vector.

The architecture must be:

- Model/decision tree decides whether a ticker is eligible to buy.
- QP decides how much to hold, trim, rotate, or close.
- Benchmark sleeve is beta exposure, not alpha evidence.

Whenever QP output is used to justify a buy without a separate model/regime
admission proof, the decision tree is conceptually broken.

### 4. I did not force enough exact trade-level explanations early

The user repeatedly asked for every trade. The correct artifact is not a log
summary. It is a DB/ledger row with:

- ticker
- blocked_by
- model_type
- regime and regime confidence
- sector
- raw score, calibrated rank score, panel score
- expected return, mu, sigma, horizon
- QP target, current weight, delta, constraints, source job/task
- sell reason, sell P/L, tax estimate, tax cash debit mode, net after tax
- same-capital SPY comparison for active alpha

Without this, the system can only be debugged by guessing from logs. That is
unacceptable for a model-based trading system.

### 5. I let stale docs and stale memory waste time

I sometimes forgot previous fixes or re-discussed issues already diagnosed.
That is a process failure. The live memory doc must be the anchor, and every
material result must be written there immediately with exact artifact paths,
commit hashes, and whether the result is acceptance-grade or diagnostic-only.

### 6. I underweighted negative controls and placebo failures

A placebo IC higher than real IC is not a minor oddity. It is a priority-one
signal that either:

- the label/feature alignment is wrong,
- the evaluation target is wrong,
- the model is capturing regime persistence rather than alpha,
- or the gate is measuring a different objective from the model.

The latest fix corrected one concrete error: WF sanity had hard-coded
`fwd_60d_excess_raw` even when the rank-LTR artifact was trained on
`fwd_60d_excess`. That was a real contract bug. The corrected diagnostic still
fails, which is useful because it points to the real structural issue instead
of a mislabeled gate.

### 7. I did not make "safe fallback" impossible enough

Silent fallback is toxic here. Examples that must be hard-blocked:

- missing sector metadata in QP
- missing scorer/calibrator fingerprint
- missing train/WF/regime stamps
- missing `mu` or `sigma` for QP
- wrong label horizon
- raw score fallback into expected-return logic
- sim config drifting from production semantics

If the system cannot prove the contract, it should not buy.

## Why the 10 Rounds Were Not Enough

The rounds were not consistently global in the right way. A real global pass
must walk the same invariant chain every time:

1. Data freshness, corporate actions, schema, NaN policy, sector map.
2. Feature construction, no future bars, train/infer parity.
3. Label construction, horizon, target, purge/embargo.
4. Model training, CV, WF, placebo, regime IC, calibration.
5. Artifact contract and fingerprints.
6. Runtime preflight.
7. Candidate generation and buy gates.
8. Panel scoring, regime admission, rank/ER/mu/sigma.
9. QP admission and sizing constraints.
10. Order emission, duplicate guards, broker semantics.
11. Sell logic, tax, P/L, SPY active comparison.
12. DB trace completeness.

I did pieces of this, but not always as a single mandatory chain. That allowed
serious bugs to survive between components.

## Standards Going Forward

### Contract first

Every component must have explicit expected input/output, with tests for:

- valid input
- missing metadata
- wrong horizon
- wrong label
- NaN/inf
- stale artifact
- sim/live/LEAN parity
- disabled or diagnostic-only config

### Regime first

Every model and strategy result must report:

- per-regime IC
- per-regime placebo
- per-regime trade monotonicity
- per-regime active P/L versus SPY
- buy counts by regime

Pooled results are secondary.

### TDD before patching

For code changes:

1. Write the failing test or identify an existing failing acceptance gate.
2. Patch the smallest relevant code path.
3. Run targeted tests.
4. Run the broader suite for touched modules.
5. Commit and push with exact scope.

### Sim/live/LEAN parity

No production decision logic should live only in sim, live, or LEAN. If a rule
affects trading, it belongs in a shared Task/Job/Pipeline path or in a shared
helper that all adapters call.

### Experiments only after invariants

A model experiment is invalid if:

- evaluation config is not production-semantic,
- artifact and calibrator fingerprints do not match,
- regime admission is disabled without being labeled diagnostic-only,
- trades cannot be traced to full decision inputs,
- tax/cash mode is ambiguous,
- SPY comparison is missing.

## Immediate Mainline Implications

The current mainline should not chase another random model sweep. The next
useful work is:

1. Finish the strict WF rerun after the label-contract correction.
2. Treat BULL_CALM as non-buyable unless regime IC/placebo and trade
   monotonicity pass for that regime.
3. Keep BenchmarkSleeve separated as beta exposure; do not count sleeve return
   as alpha improvement.
4. Make every daily/full buy path fail closed when the active artifact lacks
   passing WF/regime evidence.
5. Continue trade-level forensics until every losing trade has a precise
   pipeline-cause label, not a vague explanation.

## Personal Process Correction

I need to be less performative and more mechanical:

- Fewer vague progress claims.
- More exact file/line/artifact references.
- More failing tests before fixes.
- More small commits.
- More parallel sidecar review while critical-path jobs run.
- Less waiting on sims when code review can proceed.
- No treating "it ran" as evidence that "it is right".

The user was right to be angry. The bar for this project is not "many things
checked"; it is "the trading decision can be trusted or is blocked." Anything
below that is noise.

## Non-Negotiable Repair Protocol

For the rest of this repair campaign, I should follow this protocol before any
performance claim:

1. Identify the exact invariant being tested.
2. Name the code path that owns it.
3. Add or cite the regression test.
4. Run the test.
5. If it affects trading, verify sim/live/LEAN share the same Task/Job path.
6. If it affects model acceptance, verify regime-level IC, placebo, trade
   monotonicity, SPY active economics, and artifact fingerprints.
7. Commit and push only scoped changes.
8. Write the result into the mainline memory doc with artifact paths.

Anything outside that flow is diagnostic-only and must be labeled as such.
