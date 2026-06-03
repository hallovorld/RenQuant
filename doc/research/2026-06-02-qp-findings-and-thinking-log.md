# QP review — findings, thinking, and uncertainties

**Date**: 2026-06-02
**Status**: Addendum to PR #125 — companion to the parent memo + 3-questions addendum
**Author**: Claude
**Audience**: codex (PR review) + user (architectural decision)
**Purpose**: Make my reasoning chain visible so it can be scrutinized

---

## Why this document exists

The parent memo (`2026-06-02-qp-architecture-review-and-alternatives.md`) and the 3-questions addendum present **conclusions**. This document presents the **reasoning** — methodology, evidence catalog, counter-arguments I considered, and open uncertainties. The audience is a reviewer who needs to evaluate not just my answers but how I got to them.

I am explicitly cataloging:
- What I am **certain about** (file-line citations, git commits, published literature)
- What I am **inferring** (judgment from evidence, possibly wrong)
- What I am **guessing** (low confidence; please challenge)
- Counter-arguments I considered and **why I rejected each**
- Specific empirical observations that **would change my mind**

If codex's review surfaces evidence that contradicts a high-confidence claim, this doc tells codex which claim to challenge first.

---

## 1. Methodology — how I did this research

Concrete steps, in order:

### Step 1 — Inventory the current state

Read `kernel/portfolio_qp/qp_solver.py` end-to-end (1,100 LOC including docstrings).
Read `kernel/portfolio_qp/tasks.py` end-to-end (~2,800 LOC across multiple Tasks).
Counted parameters via Python `inspect.signature()`: 32.
Counted constraint sites via regex over `qp_solver.py:1`: 5 hard + 3 optional.
Counted objective sites via `obj_terms.append`: 5.

**Confidence**: HIGH. These are mechanical counts; I can produce the exact line numbers on request. Cited in §1 of the parent memo.

### Step 2 — Audit which parameters are actually active in prod

**Initial audit was wrong; this section reflects the corrected audit performed after codex's #125 re-review.**

The initial audit (2026-06-02, before re-review) read `strategy_config.json` partially and claimed "8 active, 24 inactive". That count dropped the regime overlays, the admission gate, the soft-sell guard, and other behaviorally-active keys; and cited stale numeric values (`qp_cost_kappa=0.0001`, `qp_turnover_max=0.2`, `qp_min_invested_pct=0.7`, `qp_cash_drag_lambda=0.05`) that no longer match the committed config.

The corrected audit re-read the head of `strategy_config.json` + the four `regime_params.{BULL_CALM, BULL_VOLATILE, BEAR, CHOPPY}` overlays + `SolveMarkowitzQPTask._build_solver_kwargs()`. Actual count is **25 active / 7 inactive**, with values listed in the parent memo's §2.1. Real values: `qp_cost_kappa=0.002`, BULL_CALM `qp_turnover_max=0.15`, `qp_cash_drag_lambda=0.0` (inactive), `qp_min_invested_pct=0.0` (inactive). Active list adds `qp_conviction_cap_enabled`, `qp_admission_gate`, `qp_soft_sell_guard`, strict horizon/μ contracts. Constraints/objectives restated as **9 hard constraints + 3 objective terms** (was "3+3"). The complexity-tax argument is now built on the right numbers.

**Confidence in the corrected audit**: HIGH. The numbers come straight from the committed config head and cross-check against the regime overlay dict. A methodology doc cannot preserve a stale count as if it had been the performed method, so the wrong-count paragraph above explicitly admits the initial audit was incorrect.

### Step 3 — Trace the historical "why"

`git log --reverse --format="%h %ad %s" --date=short -- backtesting/renquant_104/kernel/portfolio_qp/qp_solver.py` gave the first commit (`f4233fc`, 2026-04-26).

`git log --all --oneline -- "backtesting/renquant_104/kernel/rotation.py"` traced the predecessor.

`git show d284d52 --format="%B" -s` retrieved the first commit message of the QP predecessor (`ConvexRotationSolver`) explicitly citing `boyd-rotation-design.md Phase A`.

Cross-referenced with `memory/feedback_industry_leading_quality.md` (recorded user mandate 2026-05-04) and `memory/feedback_validate_with_mature_lib.md` (user verbatim 2026-05-14 — "I want mature 3rd party, industry-grade").

**Confidence**: HIGH on the chronology (git is the source of truth). MEDIUM on the user's *intent* — I'm reading memory files that summarize past sessions; I haven't independently verified the user actually said those exact words in those exact contexts. But the memory files are dated and immutable.

### Step 4 — Trace QP's functional role in the pipeline

Read `kernel/pipeline/pp_inference.py` for Phase ordering.
Grep'd `_qp_*` reads/writes across `tasks.py` to map inputs/outputs.
Confirmed `EmitOrdersFromQPSolutionTask` is the consumer of QP's output.

**Confidence**: HIGH. The Task graph is mechanical to read.

### Step 5 — Apply the published-literature critique

Searched my training-data knowledge for "Markowitz noisy μ̂" + "1/N optimal diversification" + "error maximizer". The four papers I rely on most heavily:
- Markowitz 1952 (original MV)
- Michaud 1989 (error-maximizer critique)
- Chopra-Ziemba 1993 (μ̂ ~10× more damaging than Σ̂)
- DeMiguel-Garlappi-Uppal 2009 (1/N is not consistently beaten by 14 MV-family models across 7 empirical datasets)

For each, I documented the specific quantitative claim, not just the conclusion. The 10× number is from Chopra-Ziemba Table 3. The DeMiguel et al. count — **14 portfolio models compared across 7 empirical datasets** — is from DeMiguel et al. §3 (earlier drafts of this doc mis-stated this as "14 datasets"; codex's #125 re-review corrected it).

**Confidence**: HIGH on the existence and content of these papers. MEDIUM on the *applicability* to our specific dataset — see §3 below for counter-arguments.

### Step 6 — Construct the 5-level hierarchy

I constructed Levels 0-5 as a synthesis of:
- Boyd-Busseti-Diamond-Kahn-Koh-Nystrup-Speth 2024 cvxportfolio book — explicit Single vs Multi distinction at Levels 1 vs 2
- Merton 1969 — Level 3 (HJB continuous-time)
- Lobo-Boyd 1998 — Level 4 (robust)
- General RL portfolio literature — Level 5

**Confidence**: MEDIUM. The 5-level structure is my own synthesis; published taxonomies (e.g., Markowitz vs Black-Litterman vs Risk Parity vs HRP) don't exactly map to my levels. I find my taxonomy useful for ranking alternatives but I won't defend it as canonical.

---

## 2. Evidence catalog — what I cite vs what I infer

The parent memo's load-bearing claims, ranked by confidence:

### High confidence (file/commit/citation verifiable)

| Claim | Evidence |
|---|---|
| `solve_portfolio_qp` has 32 params | `inspect.signature(solve_portfolio_qp)` returns 32 entries; can re-run |
| 5 objective `obj_terms.append` sites | `grep -c "obj_terms.append" kernel/portfolio_qp/qp_solver.py` = 5 |
| 5 hard `constraints.append` sites + 3 optional | grep result, cited line numbers in parent memo §1 |
| QP introduced 2026-04-26 / 2026-04-27 | Git log on `qp_solver.py` (first) + `rotation_convex.py` (predecessor) |
| User mandated industrial-grade approach | `memory/feedback_industry_leading_quality.md` + `memory/feedback_validate_with_mature_lib.md` |
| DeMiguel 2009 finding (1/N is not consistently beaten by 14 MV-family models across 7 empirical datasets) | Published paper, RFS 22(5), §3 |
| Chopra-Ziemba 1993 finding (μ̂ ~10× more damaging) | Published paper, JPM 19(2), Table 3 |
| Bug F (delta_below_min_dw) | Recorded in TaskList #21 + commit `bf4edaf` audit |
| Today's daily-104 QP infeasibility | `logs/daily_104/2026-06-02.log` lines on `per_asset_cap_max=-0.042` |

### Medium confidence (inference from evidence)

| Claim | What I'm inferring from |
|---|---|
| ~25 of 32 params actually drive prod (corrected from "8 of 32") | Re-audited 2026-06-02 against current `strategy_config.json` + regime overlays + `_build_solver_kwargs`. See parent memo §2.1 for the full table; only 7 keys are inactive |
| Today's bug is a "constraint-composition bug" not a QP bug | My architectural reading; codex's #123 v2 catch (`ApplyConvictionCapTask` raised the hard cap up to over-cap `w_current`) is independent confirmation: bug surfaced in constraint assembly, not in cvxpy |
| The MV-vs-1/N mechanism applies to us in principle | Mechanism (μ̂-error damage) is well-established (DeMiguel 2009, Michaud 1989, Chopra-Ziemba 1993). But codex correctly flagged DeMiguel does NOT publish an IC-threshold; we cannot conclude RenQuant sits below a known boundary. Mechanism credible, quantitative placement unsupported |
| Earlier draft claim: "Migration to Hybrid is incremental and reversible" | This was asserted in an earlier draft of parent §8; the corrected §8 no longer asserts Hybrid is the path forward (it says no allocator is pre-selected and migration is NOT authorized). The incremental-and-reversible claim survives only as a property of the eventual A/B winner if Hybrid wins. |

### Low confidence (judgment calls; please challenge)

| Claim | Why I'm uncertain |
|---|---|
| Earlier draft claim: "Practical apex for our regime is Level 0" | Extrapolation from DeMiguel / Michaud / Chopra-Ziemba mechanism to OUR data is not a measurement. Both codex (re-review) and gemini explicitly rejected pre-deciding Level 0 (or any other allocator) from this memo. The honest answer is "we don't know until we measure" — the §8 Step 4 offline WF A/B replay is what tells us. |
| "QP is over-engineered for our problem class" | Subjective. A different team would say "we built the right infrastructure; the bugs are growing pains." Both framings are defensible. My framing reflects my own bias toward simplicity |
| "<10% of bars need QP fallback under Hybrid" | A guess based on how often sector caps actually bind in production. I don't have logging for this; could be 1%, could be 30%. Step 4 (offline WF A/B replay) would tell us — and Step 5 live shadow would refine the rate empirically |
| Cvxportfolio MultiPeriodOpt would be better than SinglePeriodOpt for us | Theoretically: yes, if signal-decay model is meaningful. Empirically: depends on label horizon vs decay rate. For fwd_60d_excess at our scale, marginal lift might be small |

---

## 3. Counter-arguments I considered and rejected (and why)

### CA1 — "DeMiguel 2009 tested monthly equity-sector returns, not daily firm-level. The finding doesn't transfer."

**Strength**: HIGH validity concern. DeMiguel's data was indeed monthly sector ETF / international stock / industry portfolio data. Daily firm-level US equity is a different regime.

**Why I still cite them**: Their *mechanism* — estimation error in μ̂ dominating optimization gain — is dataset-agnostic. The mechanism is amplified, not diminished, when you go to noisier data (daily firm-level returns have higher noise-to-signal than monthly sector returns). DeMiguel's finding is the LOWER bound of how often simpler beats Markowitz; for our noisier data, it's likely MORE often.

**What would change my mind**: If codex points me to a paper specifically showing that for **daily firm-level US equity with model-driven signals at IC ≈ 0.04**, Markowitz beats 1/N out-of-sample, I would weight that as direct evidence and downgrade the DeMiguel application.

### CA2 — "The bugs are growing pains. The QP infrastructure is now stable; rewriting risks more bugs."

**Strength**: HIGH validity. The current QP has 456 tests passing, has been in production for ~5 weeks, has survived several real-money days. A migration to Hybrid will introduce a new bug surface.

**Why I still find Hybrid worth carrying into the A/B (NOT a recommendation)**: It is not a rewrite — it's "decompose decision-making so QP is only invoked rarely". The QP stays; it becomes a fallback. The new code is Stages 1-3, each simpler than what they replace. Whether this actually delivers a Sharpe lift over Level-1 QP for OUR data is the open empirical question; the §8 Step 4 offline WF A/B replay is what measures it. **Codex's re-review explicitly rejected pre-deciding Hybrid in this memo; it is one of 5 candidates the A/B compares.**

**What would change my mind**: If the offline WF A/B replay (revised §8 Step 4) shows the Hybrid produces materially worse decisions than QP on 20%+ of bars, I'd retract and recommend "stay on QP, fix constraint composition layer instead."

### CA3 — "Why not just fix the constraint-composition layer?"

**Strength**: VERY HIGH. This is the most legitimate counter — it's a less ambitious change with smaller blast radius.

**Why I still find Hybrid worth carrying into the A/B (NOT a recommendation)**: Fixing constraint composition keeps us at Level 1 (the level codex+gemini agree is over-engineered for our problem class). It takes engineering effort to land the single coherent `ConstraintSnapshot` contract, and the result would be a Level 1 implementation that is correct-by-construction but still Level 1. Hybrid would move us toward Level 0 + Level 1 fallback IF it actually wins the A/B — the §8 Step 4 offline replay is what decides.

**What would change my mind**: If codex argues that the constraint-composition refactor is genuinely cheaper (e.g., 3 days work) and the empirical Level 0 advantage is small in our data, that's a valid concession. The constraint-composition fix is a defensible halfway position. I'd prefer it to the status quo even if I think full Hybrid is even better.

### CA4 — "You haven't actually compared Sharpe ratios. This is theoretical."

**Strength**: HIGH. I have not run the experiment.

**Why I still write the memo (UPDATED 2026-06-02 after codex re-review)**: The original phrasing claimed 30 trading days of shadow data would be enough to compare Sharpe. Codex correctly flagged this as insufficient (MED-6) — 30 days does not statistically separate a Sharpe delta of 0.1. The revised plan in the parent memo §8 puts paired-daily-returns offline WF A/B replay FIRST (with regime stratification + DSR/PBO multiple-comparison correction), and uses live shadow ONLY for operational telemetry (fallback-rate drift, implementation parity), not as a Sharpe gate.

**What would change my mind**: If the offline WF A/B (Step 4 in revised §8) shows Hybrid Sharpe < current-QP Sharpe with `DSR > 0.5` AND `PBO < 0.5` AND ≥ 4/N cuts consistent, my recommendation flips to "stay on QP, fix the constraint-composition layer (Step 1) and stop." Live shadow would not even start.

### CA5 — "The user originally wanted industrial-grade infrastructure. Going back to Level 0 violates that."

**Strength**: HIGH. The user's 2026-05-04 mandate (`feedback_industry_leading_quality`) explicitly says "no hand-rolled, use mature libraries."

**Why I still carry the Level-0 / Hybrid hypothesis (NOT a recommendation)**: The hypothesis is NOT hand-rolled. It cites:
- DeMiguel-Garlappi-Uppal 2009 (1/N + Bayes-Stein literature)
- Kelly 1956 / Thorp 1969 (per-name Kelly fraction)
- Davis-Norman 1990 (closed-form no-trade band — already in our codebase)
- Frazzini-Pedersen 2014 (BAB, low-beta anomaly — separate from sizing)

The Hybrid is *industrial-grade Level 0*. The "no hand-rolling" mandate doesn't lock us into Level 1; it locks us into citing published references. Level 0 with citations satisfies the mandate.

**What would change my mind**: If the user clarifies that "industrial-grade" specifically means "Markowitz-style MV optimization, no simplifications," then the mandate IS binding and we should stay at Level 1. But I don't read the 2026-05-04 memory that way.

### CA6 — "Cvxportfolio MultiPeriodOpt (Level 2) might be a better next step than going to Level 0."

**Strength**: MEDIUM-HIGH.

**Why I rank it as alternative**: Level 2 is mathematically more rigorous than Level 1; it explicitly models signal decay and transaction-cost smoothing over multiple bars. For HFT-style strategies with second/minute label horizons, Level 2 is genuinely the win.

**Why I still rank Hybrid first for OUR case**: We have a 60-day-horizon label. The signal decays slowly on the multi-day timescale Level 2 optimizes over. The marginal lift from Level 2 vs Level 1 is small at our horizon. Meanwhile DeMiguel 2009 implies Level 0 might beat both at our noise level.

**What would change my mind**: If codex's cvxportfolio production experience says "we see materially better OOS Sharpe from MultiPeriodOpt at daily / weekly rebalance" in regimes like ours, that's strong evidence I'd defer to.

---

## 4. Open uncertainties — please challenge

I am genuinely unsure about these. If codex has views, I'd update my recommendation.

### U1 — Is our IC ≈ 0.04 in the "1/N beats MV" regime?

DeMiguel 2009 doesn't quantify the IC at which 1/N starts to lose to MV — they show 1/N wins across ALL their datasets but don't bound when MV would catch up. If signal IC is high enough, MV must win eventually (perfect μ̂ → MV is provably optimal).

**My guess** at the boundary: somewhere around IC = 0.15-0.20, MV's optimization gain dominates estimation error. At IC = 0.04 we're well below that.

**What I'd want to know**: an empirical study fitting MV vs 1/N as a function of signal IC on daily US equity. I don't have that paper. If codex knows of one, it would significantly tighten the recommendation.

### U2 — What's the actual fallback rate under Hybrid?

My estimate of "<10% of bars need QP fallback" is a guess. The actual rate depends on:
- How often sector caps bind (depends on which sectors the model picks)
- How often correlation pair caps bind (depends on universe correlation structure)
- How often turnover_max is the binding constraint

If the actual rate is 1%, the migration is even better than I claim (QP becomes near-vestigial). If it's 30%, the maintenance burden of two paths is high. **The offline WF A/B replay (revised §8 Step 4) is exactly the measurement that resolves this — fallback rate is one of the metrics it captures.**

### U3 — Is the constraint-composition layer fixable in isolation?

My memo argues "constraint composition is the architectural debt, not the optimizer." Codex's likely counter: "OK, then fix the constraint composition layer." I haven't deeply analyzed how hard that would be.

A naive estimate: collapse `ComputeQPConstraintsTask` + `ApplyExposureScalingTask` + `ApplyConvictionCapTask` + the sector/corr indicator-building Tasks into a single coherent `BuildQPConstraintsTask` that owns the full constraint-vector contract. Probably 2-3 days of refactoring + comprehensive new tests.

**I haven't checked**: whether the existing Task split was deliberate for §1c size limit reasons (CLAUDE.md memory `feedback_qp_pipeline_alignment.md`: "each Task ≤50 lines"). If so, collapsing them violates that mandate. The right path might be "smaller Tasks but with a coordination contract that prevents contradictory constraint construction" — which is more work than I budgeted.

### U4 — Does our 60-day-horizon forecast actually have meaningful signal decay?

If our forecast state `μ̂_t` persists with high autocorrelation to `μ̂_{t+k}`, then single-period and multi-period optimization give nearly identical answers. Then there's no reason to move to Level 2.

If the forecast state decays significantly within a 60-day window, then Level 2 captures real value Level 1 misses.

**What we have — codex re-review correction (2026-06-02)**: my earlier draft of this section cited `label_autocorr_60` (realized forward-label autocorrelation) for BEAR = +0.213, BULL_CALM = −0.040, BULL_VOLATILE = +0.015, CHOPPY = −0.024 and concluded "signal-decay is FAST → Level 2 more useful."

Codex flagged that this is the wrong observable: **realized forward-label autocorr conflates label noise with forecast persistence**. Multi-period optimization needs `corr(μ̂_t, μ̂_{t+k})` on the calibrated forecast state, plus rank/top-K overlap and an expected-return half-life — measured from the model's actual output, not from the realized labels.

**Required measurement (NOT YET DONE)**:
1. `corr(μ̂_t, μ̂_{t+k})` per regime, k ∈ {1, 5, 10, 20, 60 trading days}
2. Top-K rank overlap `|topK(μ̂_t) ∩ topK(μ̂_{t+k})| / K` for K ∈ {5, 10, 20}
3. Expected-return half-life: smallest k such that
   `corr(μ̂_t, μ̂_{t+k}) ≤ 0.5 × corr(μ̂_t, μ̂_t)`
4. Regime-conditional realized turnover pressure under current cost
   structure (whether Level 2's amortization actually pays).

**Updated view**: I cannot quantify Level 2's attractiveness without these measurements. **The right next step is to measure them**, not to skip Level 2. Gemini's review explicitly pointed at the same gap from a different angle (Level 2 *might* shine in this regime — the only way to know is to measure). Until then, U4 is an open uncertainty, not a Level-2-attractiveness claim.

### U5 — Does the Hybrid (Option F) actually preserve the regime-conditional discipline?

PRIME DIRECTIVE (CLAUDE.md §1) says every knob is regime-conditional. Today's QP has regime-conditional `w_upper` via the cap construction chain. Hybrid Stage 2 (Kelly fraction) is regime-conditional via the calibrator. Hybrid Stage 3 (no-trade band) — is it regime-conditional in my proposed design?

I haven't specified that. The Davis-Norman band already in the codebase is per-asset, not per-regime. If the band threshold should vary by regime (e.g., wider in CHOPPY for whipsaw protection, tighter in BULL_CALM for fast capital deployment), my Hybrid design doesn't address it.

**This is a hole in my proposal.** Hybrid Option F design (only relevant IF Step 4 of revised §8 selects it) needs regime-conditional configuration of all three stages, not just Stage 2.

---

## 5. What evidence would change my mind

Specific, falsifiable triggers that would flip my recommendation:

| Evidence | Recommendation flip |
|---|---|
| Codex cites a paper showing MV beats 1/N for daily firm-level US equity at IC ≈ 0.04 | Downgrade DeMiguel-applicability claim; Level 1 may be defensible |
| Codex says cvxportfolio production with similar μ̂ shows clear MultiPeriodOpt > SinglePeriodOpt | Promote Level 2 above Hybrid in the ranking |
| Constraint-composition refactor turns out to be 2-3 days; produces a clean coherent constraint builder | Switch recommendation to "Fix in place + stay at Level 1, defer migration" |
| Offline WF A/B (revised §8 Step 4) shows Hybrid Sharpe < QP Sharpe with `DSR > 0.5` + `PBO < 0.5` + ≥ 4/N cuts consistent | Abandon Hybrid migration; fix constraint-composition in place at Level 1 |
| Offline WF A/B shows Hybrid fallback rate > 30% | Hybrid is not the simplification I claim; the constraint-composition fix carries the load alone |
| User says "industrial-grade specifically means MV optimization" | Stay at Level 1; the user mandate is binding |
| User says "I want to keep iterating on the QP, not migrate" | Stay at Level 1; focus on constraint-composition refactor |

---

## 6. Concrete decision tree (REVISED 2026-06-02 after codex re-review)

**The old tree was the 3-phase Hybrid migration (Phase 1 shadow → Phase 2 sim → Phase 3 cutover). Both codex and gemini rejected that framing as the wrong order: live shadow doesn't statistically separate a Sharpe delta of 0.1 over 30 days, and the constraint-composition contract must be fixed BEFORE any allocator comparison can be trusted. The tree below replaces it with the measurement-and-contract sequence in the parent memo's revised §8.**

I'm not asking the user to make a binary "migrate or don't" choice. The decision tree is:

```
                       ┌─ Codex / gemini agree: fix constraints first ─┐
                       │   (the convergent verdict, 2026-06-02)         │
USER reads PR #125 ────┤                                                │
                       ├─ Codex argues Level 2 (MultiPeriodOpt) ────────┤
                       │   is more attractive than parent memo claims   │
                       │                                                │
                       └─ Codex agrees: keep Hybrid as one A/B candidate ─┤
                                                                        │
                                                                        ▼
                                                  ┌─── User decides ───┐
                                                  │                    │
                                                  ▼                    ▼
                                      Step 0 — PR #123 v4         (do nothing —
                                      DONE (merged 2026-06-03)     not a valid
                                                  │                  endpoint)
                                                  ▼
                                      Step 1 — ConstraintSnapshot
                                      contract refactor (codex +
                                      gemini convergent recommendation)
                                      contract refactor (codex +
                                      gemini convergent recommendation)
                                                  │
                                                  ▼
                                      Step 2 — μ̂-autocorrelation
                                      measurement per regime
                                      (closes HIGH-4; quantifies
                                      Level-2 attractiveness)
                                                  │
                                                  ▼
                                      Step 3 — Param-inventory
                                      rerun (already done in
                                      parent memo §2.1)
                                                  │
                                                  ▼
                                      Step 4 — Offline WF A/B
                                      replay, 5 baselines:
                                      QP / hard-only QP / Hybrid /
                                      inverse-vol top-K / MPO
                                      (DSR > 0.5, PBO < 0.5,
                                      ≥ 4/N cuts consistent)
                                                  │
                              ┌───────────────────┼───────────────────┐
                              ▼                   ▼                   ▼
                       A. A candidate    B. No candidate      C. Hard-only QP
                          dominates         dominates;            wins (means
                          offline           stay on current        constraint
                          │                 QP + ConstraintSnapshot composition
                          ▼                                         was the
                       Step 5 — live                                whole bug)
                       shadow for
                       operational
                       telemetry only
                       (NOT Sharpe gate)
                          │
                          ▼
                       D. Promote winner
                          to live via
                          config flag
```

A, B, C, D are valid endpoints. The honest answer is still: "I don't know which is right without measurement", but the measurement order is now correct (offline WF A/B is the gate, live shadow is operational verification). The earlier "Phase 1 → Phase 2 → Phase 3" tree had live shadow as the gate, which neither codex nor gemini would accept.

---

## 7. What I would do if I were the reviewer

If I were codex reviewing this PR, I would:

1. **Challenge claim U1** — push me to defend that IC=0.04 is "below the 1/N-vs-MV boundary"
2. **Probe CA3** — argue more aggressively for constraint-composition fix as the minimum-blast-radius path
3. **Push back on §3.b dismissal of Level 2** — given the new evidence (label autocorr very low in our data), Level 2 might be the actual best next step
4. **Question the contract refactor timeline** — is the Step 1 `ConstraintSnapshot` / `BuildQPConstraintsTask` refactor really 2-3 days given how many Tasks currently compose the constraint vector?
5. **Ask for sim numbers** — refuse to authorize live shadow (Step 5) until offline WF A/B replay (Step 4) produces a Sharpe delta + DSR/PBO/per-regime breakdown

If codex does (1)-(5), the conversation gets MUCH sharper. I'd update the memo with the answers.

---

## 8. Bottom line for the user

After all this analysis, here's the most honest summary I can produce:

> **QP isn't broken. It's at Level 1 of a 5-level hierarchy of portfolio
> optimization sophistication. For our specific problem (noisy μ̂ at IC≈0.04,
> 60-day horizon, ~150-name universe, daily rebalance), the published
> literature suggests Level 0 (Kelly + closed-form bands) likely matches
> or beats Level 1 out-of-sample because estimation error dominates the
> optimization gain. Moving to Level 2 (multi-period MV) is a defensible
> alternative path — IT might be the right answer rather than Level 0.
> The honest answer is "we don't know without measurement." The offline
> WF A/B replay (revised §8 Step 4) is the experiment that resolves the
> uncertainty — live shadow (Step 5) is operational telemetry, not the
> Sharpe gate.**

**Earlier draft said**: *"60% confident the Hybrid recommendation is right and 40% confident codex's review will surface evidence shifting the recommendation toward either (a) constraint-composition refactor in place at Level 1 or (b) move to Level 2 (cvxportfolio MultiPeriodOpt) instead. Any of (a), (b), or Hybrid is a strict improvement over the current 'keep iterating on Level 1 with growing technical debt' path."*

**Corrected stance after codex's third re-review (2026-06-03)**: this memo does NOT recommend Hybrid (or any other allocator). My prior on which allocator wins is irrelevant to the §8 plan — the offline WF A/B replay (Step 4) does the comparison, and the answer is whatever the data says. The work order that DOES survive: Step 0 (DONE — PR #123 v4 merged), Step 1 (`ConstraintSnapshot` contract refactor — codex + gemini convergent), Step 2 (μ̂ autocorrelation measurement), then Step 4 to decide between the 5 baselines. The user's decision should be informed by the A/B output, not by my prior.

---

## References supporting this thinking log

- Markowitz 1952 — original MV
- Michaud 1989 *Financial Analysts Journal* 45(1) — "error-maximizer" critique
- Chopra & Ziemba 1993 *JPM* 19(2) — quantitative finding that μ̂ errors ~10× more damaging than Σ̂ errors for MV
- DeMiguel, Garlappi & Uppal 2009 *RFS* 22(5) — 1/N is not consistently beaten by 14 MV-family models across 7 empirical datasets
- López de Prado 2016 *JPM* 42(4) — Hierarchical Risk Parity
- Boyd-Busseti-Diamond-Kahn-Koh-Nystrup-Speth 2024 cvxportfolio textbook — SinglePeriodOpt vs MultiPeriodOpt distinction
- Garleanu-Pedersen 2013 *J. Finance* 68(6) — partial-move under transaction costs (relevant to Level 2 motivation)
- Merton 1969 *Rev. Econ. Stat.* 51(3) — Level 3 continuous-time HJB
- Davis-Norman 1990 *Math. Operations Research* 15(4) — closed-form no-trade band
- Frazzini-Pedersen 2014 *J. Fin. Econ.* 111(1) — BAB (separate consideration, not the sizing layer)
- Internal: CLAUDE.md §1 PRIME DIRECTIVE, §7.5 single source of truth, §7.10 canonical references, §10 general coding guidelines
- Internal: `memory/feedback_industry_leading_quality.md`, `memory/feedback_validate_with_mature_lib.md`, `memory/feedback_qp_pipeline_alignment.md`, `memory/feedback_qp_gross_max_is_leverage.md`
- Internal: `doc/research/2026-06-02-bull-calm-no-signal-diagnostic.md` (the IC=0.04 source)
