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

Read `backtesting/renquant_104/strategy_config.json` end-to-end and matched the 32 QP parameters against the config's `rotation.joint_actions.*` values. Counted: **8 active, 24 inactive** (default-zero, INACTIVE-by-policy like tax/leverage, or legacy kwargs).

**Confidence**: MEDIUM-HIGH. The audit relies on my reading of one config file. The config has a `regime_params` override layer I sampled but didn't exhaustively cross-check. If a per-regime override activates a parameter I marked "INACTIVE", that's a hole in my audit. Counter: I checked the obviously regime-conditional knobs (`drawdown_*`, `kelly_*`, `qp_dw_max`); the truly latent ones (CVaR, robust-μ, sqrt-impact) are not in any regime override I saw.

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
- DeMiguel-Garlappi-Uppal 2009 (1/N beats MV across 14 datasets)

For each, I documented the specific quantitative claim, not just the conclusion. The 10× number is from Chopra-Ziemba Table 3; the 14-dataset count is from DeMiguel et al. §3.

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
| DeMiguel 2009 finding (1/N beats MV in 14 datasets) | Published paper, RFS 22(5), §3 |
| Chopra-Ziemba 1993 finding (μ̂ ~10× more damaging) | Published paper, JPM 19(2), Table 3 |
| Bug F (delta_below_min_dw) | Recorded in TaskList #21 + commit `bf4edaf` audit |
| Today's daily-104 QP infeasibility | `logs/daily_104/2026-06-02.log` lines on `per_asset_cap_max=-0.042` |

### Medium confidence (inference from evidence)

| Claim | What I'm inferring from |
|---|---|
| 8 of 32 params actually drive prod | Reading `strategy_config.json` + cross-checking regime overrides; missed regime overlays may exist |
| Today's bug is a "constraint-composition bug" not a QP bug | My architectural reading; codex may argue the QP solver's contract should not allow `w_upper < w_current` to be acceptable at the boundary |
| Our IC=0.039 puts us "in the regime where DeMiguel wins" | DeMiguel's datasets were monthly equity sector / country returns, not daily firm-level. Adapting their finding to our case is a judgment call (see §3) |
| Migration to Hybrid is "incremental and reversible" | Asserted in §8 of parent memo; not yet validated by code — codex should challenge whether the shadow-path implementation is actually as clean as I claim |

### Low confidence (judgment calls; please challenge)

| Claim | Why I'm uncertain |
|---|---|
| "Practical apex for our regime is Level 0" | This is a strong empirical claim based on extrapolating DeMiguel/Michaud/Chopra-Ziemba from monthly equity-sector data to daily firm-level. I have NOT run a sim comparing Level 0 to Level 1 on our data. Codex's counter: it's possible that for THIS specific data + signal, Level 1 actually wins. The honest answer is "we don't know until we measure" — Phase 1 (shadow path) is what would tell us |
| "QP is over-engineered for our problem class" | Subjective. A different team would say "we built the right infrastructure; the bugs are growing pains." Both framings are defensible. My framing reflects my own bias toward simplicity |
| "<10% of bars need QP fallback under Hybrid" | A guess based on how often sector caps actually bind in production. I don't have logging for this; could be 1%, could be 30%. Phase 1 of migration would tell us |
| Cvxportfolio MultiPeriodOpt would be better than SinglePeriodOpt for us | Theoretically: yes, if signal-decay model is meaningful. Empirically: depends on label horizon vs decay rate. For fwd_60d_excess at our scale, marginal lift might be small |

---

## 3. Counter-arguments I considered and rejected (and why)

### CA1 — "DeMiguel 2009 tested monthly equity-sector returns, not daily firm-level. The finding doesn't transfer."

**Strength**: HIGH validity concern. DeMiguel's data was indeed monthly sector ETF / international stock / industry portfolio data. Daily firm-level US equity is a different regime.

**Why I still cite them**: Their *mechanism* — estimation error in μ̂ dominating optimization gain — is dataset-agnostic. The mechanism is amplified, not diminished, when you go to noisier data (daily firm-level returns have higher noise-to-signal than monthly sector returns). DeMiguel's finding is the LOWER bound of how often simpler beats Markowitz; for our noisier data, it's likely MORE often.

**What would change my mind**: If codex points me to a paper specifically showing that for **daily firm-level US equity with model-driven signals at IC ≈ 0.04**, Markowitz beats 1/N out-of-sample, I would weight that as direct evidence and downgrade the DeMiguel application.

### CA2 — "The bugs are growing pains. The QP infrastructure is now stable; rewriting risks more bugs."

**Strength**: HIGH validity. The current QP has 456 tests passing, has been in production for ~5 weeks, has survived several real-money days. A migration to Hybrid will introduce a new bug surface.

**Why I still recommend Hybrid**: My recommendation is *not* "rewrite QP" — it's "decompose decision-making so QP is only invoked rarely". The QP stays; it becomes a fallback. The new code is Stages 1-3, which is each simpler than what they replace (a single Task replacing a sub-tree of constraint-building Tasks). The shadow-path Phase 1 explicitly measures bug rate before live cutover.

**What would change my mind**: If Phase 1 reveals that the shadow Hybrid path produces materially different (worse) decisions than QP on 20%+ of bars, I'd retract and recommend "stay on QP, fix constraint composition layer instead."

### CA3 — "Why not just fix the constraint-composition layer?"

**Strength**: VERY HIGH. This is the most legitimate counter — it's a less ambitious change with smaller blast radius.

**Why I still recommend Hybrid**: Fixing constraint composition keeps us at Level 1 (the over-engineered level for our problem). It would take engineering effort to refactor `ComputeQPConstraintsTask` / `ApplyExposureScalingTask` / etc into a single coherent constraint-building module, and the result would be… a slightly better Level 1 implementation. Hybrid moves us toward Level 0 + Level 1 fallback, which has the theoretical advantage of less estimation-error sensitivity.

**What would change my mind**: If codex argues that the constraint-composition refactor is genuinely cheaper (e.g., 3 days work) and the empirical Level 0 advantage is small in our data, that's a valid concession. The constraint-composition fix is a defensible halfway position. I'd prefer it to the status quo even if I think full Hybrid is even better.

### CA4 — "You haven't actually compared Sharpe ratios. This is theoretical."

**Strength**: HIGH. I have not run the experiment.

**Why I still write the memo**: The whole point of the migration plan's Phase 1 (shadow path, 30 days) is to do exactly this comparison empirically. My memo says "here's why I think it'll be a win, here's how to validate." It would be irresponsible to ship Phase 3 (live cutover) without Phase 1 evidence; my memo argues for the SHADOW path, not the cutover.

**What would change my mind**: If Phase 1 shows shadow Hybrid Sharpe < QP Sharpe by more than 0.1 over 30 days, my recommendation flips to "stay on QP, fix the constraint composition layer instead."

### CA5 — "The user originally wanted industrial-grade infrastructure. Going back to Level 0 violates that."

**Strength**: HIGH. The user's 2026-05-04 mandate (`feedback_industry_leading_quality`) explicitly says "no hand-rolled, use mature libraries."

**Why I still recommend Hybrid**: My Level 0 recommendation is NOT hand-rolled. It cites:
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

If the actual rate is 1%, the migration is even better than I claim (QP becomes near-vestigial). If it's 30%, the maintenance burden of two paths is high. **Phase 1 of migration is exactly the measurement that resolves this.**

### U3 — Is the constraint-composition layer fixable in isolation?

My memo argues "constraint composition is the architectural debt, not the optimizer." Codex's likely counter: "OK, then fix the constraint composition layer." I haven't deeply analyzed how hard that would be.

A naive estimate: collapse `ComputeQPConstraintsTask` + `ApplyExposureScalingTask` + `ApplyConvictionCapTask` + the sector/corr indicator-building Tasks into a single coherent `BuildQPConstraintsTask` that owns the full constraint-vector contract. Probably 2-3 days of refactoring + comprehensive new tests.

**I haven't checked**: whether the existing Task split was deliberate for §1c size limit reasons (CLAUDE.md memory `feedback_qp_pipeline_alignment.md`: "each Task ≤50 lines"). If so, collapsing them violates that mandate. The right path might be "smaller Tasks but with a coordination contract that prevents contradictory constraint construction" — which is more work than I budgeted.

### U4 — Does our 60-day-horizon label actually have meaningful signal decay?

If `fwd_60d_excess` signal at t persists with high autocorrelation to fwd_60d_excess at t+1, then single-period and multi-period optimization give nearly identical answers. Then there's no reason to move to Level 2.

If signal decays significantly within a 60-day window, then Level 2 captures real value Level 1 misses.

**What we have**: We know from today's gate run that `label_autocorr_60` for BEAR = +0.213, for BULL_CALM = -0.040, for BULL_VOLATILE = +0.015, for CHOPPY = -0.024. **These autocorrelations are very low.** Signal-decay is FAST at this horizon — which makes Level 2 (multi-period planning) MORE useful, not less.

Updated view: I may have been too dismissive of Level 2 in §3 of the 3-questions addendum. With low label autocorr, signal decay IS material, and Level 2's planning-ahead value is real. Codex should challenge me on this.

### U5 — Does Hybrid Phase 1 actually preserve the regime-conditional discipline?

PRIME DIRECTIVE (CLAUDE.md §1) says every knob is regime-conditional. Today's QP has regime-conditional `w_upper` via the cap construction chain. Hybrid Stage 2 (Kelly fraction) is regime-conditional via the calibrator. Hybrid Stage 3 (no-trade band) — is it regime-conditional in my proposed design?

I haven't specified that. The Davis-Norman band already in the codebase is per-asset, not per-regime. If the band threshold should vary by regime (e.g., wider in CHOPPY for whipsaw protection, tighter in BULL_CALM for fast capital deployment), my Hybrid design doesn't address it.

**This is a hole in my proposal.** Phase 1 design needs to include regime-conditional configuration of all three stages, not just Stage 2.

---

## 5. What evidence would change my mind

Specific, falsifiable triggers that would flip my recommendation:

| Evidence | Recommendation flip |
|---|---|
| Codex cites a paper showing MV beats 1/N for daily firm-level US equity at IC ≈ 0.04 | Downgrade DeMiguel-applicability claim; Level 1 may be defensible |
| Codex says cvxportfolio production with similar μ̂ shows clear MultiPeriodOpt > SinglePeriodOpt | Promote Level 2 above Hybrid in the ranking |
| Constraint-composition refactor turns out to be 2-3 days; produces a clean coherent constraint builder | Switch recommendation to "Fix in place + stay at Level 1, defer migration" |
| Phase 1 shadow shows Hybrid Sharpe < QP Sharpe by > 0.1 over 30 days | Abandon Hybrid migration; investigate why QP's optimization gain is real for us |
| Phase 1 shadow shows Hybrid fallback rate > 30% | Hybrid is not the simplification I claim; reconsider |
| User says "industrial-grade specifically means MV optimization" | Stay at Level 1; the user mandate is binding |
| User says "I want to keep iterating on the QP, not migrate" | Stay at Level 1; focus on constraint-composition refactor |

---

## 6. Concrete decision tree

I'm not asking the user to make a binary "migrate or don't" choice. The decision tree:

```
                       ┌─ Codex argues Level 2 (MultiPeriodOpt) ──┐
                       │   for our case                            │
USER reads PR #125 ────┤                                           │
                       ├─ Codex argues constraint-composition ───┐ │
                       │   refactor as the right next step       │ │
                       │                                          │ │
                       └─ Codex agrees with Hybrid recommendation │ │
                                                                  │ │
                                                                  ▼ ▼
                                       ┌─── User decides ───┐
                                       │                    │
                                       ▼                    ▼
                          A. Authorize Phase 1     B. Defer migration,
                             (shadow path, 2 days     authorize constraint
                             eng + 30 days obs)       composition refactor
                                       │                    │
                                       ▼                    │
                          Hybrid shadow log shows           │
                          divergence from QP                │
                                       │                    │
                          ┌────────────┴───────────┐        │
                          ▼                        ▼        │
                  Hybrid ~= QP on              Hybrid       │
                  Sharpe + behaviour           materially   │
                                               worse        │
                                  ▼                ▼        │
                          C. Phase 2:           D. Abandon  │
                             sim verification      migration│
                                  ▼                         │
                          E. Phase 3:                       │
                             live cutover                   ▼
                                                  F. Stay at Level 1
                                                     with cleaner
                                                     constraint
                                                     composition
```

A, B, C, D, E, F are all valid endpoints. The honest answer is: "I don't know which is right without measurement." My memo argues that the **EXPECTED VALUE** of Phase 1 (cheap measurement) is high regardless of which way it points.

---

## 7. What I would do if I were the reviewer

If I were codex reviewing this PR, I would:

1. **Challenge claim U1** — push me to defend that IC=0.04 is "below the 1/N-vs-MV boundary"
2. **Probe CA3** — argue more aggressively for constraint-composition fix as the minimum-blast-radius path
3. **Push back on §3.b dismissal of Level 2** — given the new evidence (label autocorr very low in our data), Level 2 might be the actual best next step
4. **Question the migration timeline** — is Phase 1's 2 engineering days realistic given the depth of the existing QP integration?
5. **Ask for sim numbers** — refuse to authorize Phase 1 until I run the equivalent of a 1-cut WF sim with Hybrid and QP and produce a Sharpe delta

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
> The honest answer is "we don't know without measurement." The Phase 1
> shadow path is the cheapest experiment that resolves the uncertainty.**

I am 60% confident the Hybrid recommendation is right and 40% confident
codex's review will surface evidence shifting the recommendation toward
either (a) constraint-composition refactor in place at Level 1 or
(b) move to Level 2 (cvxportfolio MultiPeriodOpt) instead.

Any of (a), (b), or Hybrid is a strict improvement over the current
"keep iterating on Level 1 with growing technical debt" path. The user's
decision should be informed by codex's review of this analysis, not by
my prior alone.

---

## References supporting this thinking log

- Markowitz 1952 — original MV
- Michaud 1989 *Financial Analysts Journal* 45(1) — "error-maximizer" critique
- Chopra & Ziemba 1993 *JPM* 19(2) — quantitative finding that μ̂ errors ~10× more damaging than Σ̂ errors for MV
- DeMiguel, Garlappi & Uppal 2009 *RFS* 22(5) — 1/N beats MV across 14 datasets
- López de Prado 2016 *JPM* 42(4) — Hierarchical Risk Parity
- Boyd-Busseti-Diamond-Kahn-Koh-Nystrup-Speth 2024 cvxportfolio textbook — SinglePeriodOpt vs MultiPeriodOpt distinction
- Garleanu-Pedersen 2013 *J. Finance* 68(6) — partial-move under transaction costs (relevant to Level 2 motivation)
- Merton 1969 *Rev. Econ. Stat.* 51(3) — Level 3 continuous-time HJB
- Davis-Norman 1990 *Math. Operations Research* 15(4) — closed-form no-trade band
- Frazzini-Pedersen 2014 *J. Fin. Econ.* 111(1) — BAB (separate consideration, not the sizing layer)
- Internal: CLAUDE.md §1 PRIME DIRECTIVE, §7.5 single source of truth, §7.10 canonical references, §10 general coding guidelines
- Internal: `memory/feedback_industry_leading_quality.md`, `memory/feedback_validate_with_mature_lib.md`, `memory/feedback_qp_pipeline_alignment.md`, `memory/feedback_qp_gross_max_is_leverage.md`
- Internal: `doc/research/2026-06-02-bull-calm-no-signal-diagnostic.md` (the IC=0.04 source)
