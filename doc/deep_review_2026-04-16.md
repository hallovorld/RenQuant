# RenQuant Deep Review — 2026-04-16

## Scope

This review covered six areas:

1. Notebook vs LEAN logic parity for `renquant_103`
2. Test-suite comprehensiveness
3. Documentation alignment with current code
4. Design review, with emphasis on premature complexity and performance risk
5. A practical answer to whether the current system is trustworthy enough to keep extending
6. Reconstruction of what happened on 2026-04-16 and why orders were placed

I reviewed the active code paths in:

- `Notebooks/renquant_103.ipynb`
- `backtesting/renquant_103/main.py`
- `live/runner.py`
- `tests/`
- `doc/`
- `CLAUDE.md`
- `logs/daily_103/2026-04-16.log`
- `logs/live_103/2026-04-16-open.log`
- `logs/live_103/2026-04-16-preclose.log`

I also verified the current collected test count under the actual `renquant` interpreter: `405 tests collected`.

---

## Executive Summary

Short version: the repo is stronger than average for a personal quant project, but the central claim that notebook, LEAN, and live are effectively the same strategy is no longer true.

The biggest issue is not model quality. It is execution drift:

- Notebook simulation, LEAN backtest, and live runner now implement materially different trading systems.
- The tests are broad, but many of them test handwritten replicas of logic rather than the real sources of truth.
- Several docs still describe a cleaner, more unified architecture than what the code actually does.
- The design has become layered enough that it is now hard to know which component is adding signal and which component is only adding variance, maintenance cost, or false confidence.

If the goal is raw engineering judgment: the right move is not to add more filters. The right move is to simplify and force a single execution truth.

---

## 1. Notebook vs LEAN Parity

## Verdict

The notebook and LEAN are **not** identical with “only model differences”. They are close in some major branches, but there are important behavioral differences beyond model artifact loading.

## What is aligned

These areas are substantially aligned between the notebook simulation and LEAN:

- Sell priority order: trailing stop -> cumulative stop-loss -> single-day loss -> max hold -> model sell streak
- Transition uncertainty window exists in both
- Earnings filter exists in both
- SPY velocity gate exists in both
- SPY EMA50 gate exists in both
- Tiered thresholds exist in both
- Correlation guard exists in both
- Sector guard is effectively similar because LEAN appends newly bought names into `held_tickers`

Those are the areas the current alignment suite mostly targets, and that part of the repo is reasonably disciplined.

## Material mismatches

### A. Position sizing is not the same

This is the most important parity break I found.

In LEAN, regime confidence scales position sizing and cash reserve through `_rp()` for:

- `max_position_pct`
- `cash_reserve_pct`

Evidence:

- `backtesting/renquant_103/main.py`: `_rp()` scales those keys by `self._regime_confidence`
- `backtesting/renquant_103/main.py`: `_execute_buy()` uses `_rp("cash_reserve_pct")` and `_rp("max_position_pct")`

In the notebook simulation, position sizing uses raw regime parameters directly:

- `Notebooks/renquant_103.ipynb`: `max_pos_pct = rp.get("max_position_pct", 0.30)`
- `Notebooks/renquant_103.ipynb`: `cash_reserve = port_val * rp.get("cash_reserve_pct", 0.0)`

That means the notebook and LEAN can take different sizes even under the same regime and the same candidate ranking.

This is not a minor implementation detail. It changes portfolio exposure, cash drag, and compounding path.

### B. Notebook uses precomputed OOS signal series; LEAN computes runtime features and runtime scores

The notebook simulation buys and sells from precomputed objects such as:

- `results[ticker]["oos_signals"]`
- `results[ticker]["oos_raw_scores"]`

LEAN instead does runtime feature construction and runtime inference:

- `_build_feature_frame()`
- `_choose_action()`
- `_get_raw_model_score()`

This is more than “the model is different”. It is a different inference path and a different source of truth for scores.

Practical consequence:

- Notebook parity depends on the cached notebook-produced `results` object remaining semantically equivalent to current runtime inference.
- LEAN parity depends on the inline feature-engineering code staying equivalent to the notebook’s training/simulation features.

That is fragile.

### C. Notebook restricts the buy universe with `exportable`; LEAN does not use the same gate

Notebook candidate generation loops over:

- `exportable = {t for t, r in results.items() if r.get("passes_floor")}`

LEAN loops over loaded models in `self.models`.

Those sets may often overlap, but they are not literally the same gate and they can diverge operationally.

### D. BEAR defensive logic is implemented differently

Notebook BEAR branch:

- scans `DEFENSIVE_TICKERS & exportable`
- uses precomputed `oos_signals`
- ranks with `oos_raw_scores`

LEAN BEAR branch:

- scans `self._defensive`
- builds live features
- calls `_choose_action()`
- scores with `_get_raw_model_score()`

Again, this is not merely “same logic, different model file”. The execution path itself differs.

### E. Min-hold semantics are only accidentally aligned today

Notebook code still carries asymmetric knobs:

- `min_hold_profit_days`
- `min_hold_loss_days`

LEAN uses:

- `min_hold_days`

Today both notebook config values are `20`, so behavior happens to match. But the code paths are not the same design.

If those values diverge later, parity breaks immediately.

### F. Ranking weights are hardcoded in notebook and LEAN, but docs describe config-driven weights

Notebook combined rank:

- `0.5 * norm(model_score) + 0.5 * norm(rs_score)`

LEAN combined rank:

- `0.5 * norm(model_score) + 0.5 * norm(rs_score)`

Live runner:

- `w_rank * norm(rank_score) + w_rs * norm(rs_score)`
- reads `ranking.blend_weights`

So notebook and LEAN are aligned with each other here, but they are both misaligned with live and with the docs.

### G. Live is not running the full 103 regime engine

This matters because many docs imply a three-way equivalence.

`live/runner.py` explicitly says:

- `# Regime params — use BULL_CALM as default (runner doesn't have live GMM)`

The live runner does not implement the live GMM/Hurst/CUSUM regime state machine. It uses BULL_CALM defaults plus calibrated ranking.

That means:

- Notebook != LEAN != live
- Any claim that “only the model differs” is now inaccurate

## Bottom line on parity

The notebook and LEAN are aligned on several high-level guards, but the system is no longer a single strategy expressed in two places. It is now at least three related strategies:

1. Notebook simulation strategy
2. LEAN backtest strategy
3. Live runner strategy

That is the dominant technical risk in the repo.

---

## 2. Test Review

## What is good

The test suite is real and non-trivial.

Verified collection:

- `405 tests collected in 1.25s`

Current source-level distribution:

- `tests/test_policy_alignment.py`: 222 tests
- `tests/test_lean_policies.py`: 122 tests
- `tests/test_runner_ranking.py`: 42 tests
- `tests/test_simulation_policies.py`: 19 tests

Strengths:

- There is a real parity mindset
- Important policy boundaries are covered
- Regression tests exist for several previously found gaps
- Live ranking and calibration logic has dedicated tests
- There is a meta-test enforcing equal counts in `test_policy_alignment.py`

For a solo quant repo, this is above average.

## Where the suite is weaker than it looks

### A. Many parity tests validate handwritten replicas, not the actual code paths

`tests/test_policy_alignment.py` mostly defines pure helper replicas for notebook and LEAN behavior and compares those.

That catches conceptual drift only if the test author updates the replicas correctly.

It does **not** guarantee that:

- the notebook source still matches the notebook helper
- LEAN source still matches the LEAN helper
- docs still match either

This is the classic “the test proves the test author copied the same idea twice” problem.

### B. The suite did not catch the position-sizing confidence mismatch

The position-sizing alignment tests model raw `cash_reserve_pct` and raw `max_pos_pct` on both sides.

They do not exercise LEAN’s `_rp()` confidence scaling.

So the suite currently certifies parity in an area where real code is already divergent.

This is the strongest concrete example that the parity suite is not testing the actual behavior end to end.

### C. There is no golden-path cross-engine replay test

What is missing is a single synthetic scenario that drives:

- notebook simulation logic
- LEAN-like logic
- live runner logic

through the same 5 to 20 day market tape and checks a daily ledger of:

- regime
- candidate set
- scores
- selected names
- position size
- exits

Without that, the repo has branch tests, but not execution-equivalence tests.

### D. Live runner is under-tested relative to its importance

The live runner is where actual money moves, but it has only 42 tests and still contains obvious declared gaps such as:

- no trailing-stop state
- no live GMM regime engine
- BULL_CALM-default behavior for 103

That gap between production importance and test depth is too large.

### E. No test appears to enforce doc-to-code consistency

Given how often the docs claim cross-component equivalence, there should be at least lightweight checks for:

- test count claims
- ranking semantics
- config field usage
- existence of documented artifacts

Right now that discipline is manual.

## Test conclusion

The suite is **substantial but not comprehensive enough** for the claims the repo makes.

Recommended reframe:

- It is a good regression suite.
- It is not yet a trustworthy parity-proof suite.

---

## 3. Documentation Alignment Review

## Verdict

The docs are not fully aligned with current code. Some parts are good, but several important claims are stale or too strong.

## Confirmed mismatches

### A. Docs imply notebook, LEAN, and live share the same ranking semantics

Current reality:

- Notebook: raw model score + RS, hardcoded 50/50
- LEAN: raw model score + RS, hardcoded 50/50
- Live: calibrated `rank_score` + RS, config-driven weights from `ranking.blend_weights`

Stale or misleading docs include:

- `doc/logic_graph_103.md`
- `doc/renquant_103_design.md`
- `doc/architecture.md`
- `CLAUDE.md`

### B. Docs overstate live parity with notebook and LEAN

Current reality:

- Live runner does not run live GMM/Hurst/CUSUM regime detection
- Live runner explicitly uses BULL_CALM defaults
- Live runner does not implement trailing stop because it does not track per-position high-water marks

Any doc claiming the same exit priority in all three components is wrong today.

### C. Logic graph says min-hold path resets streak; LEAN does not do that there

`doc/logic_graph_103.md` says the min-hold blocked path sets `sell_streak[ticker] = 0`.

Actual LEAN code comments and behavior in the sell loop say “don’t touch streak” in that branch.

That mismatch matters because the logic graph is described as canonical.

### D. Logic graph and design docs imply config-driven blend weights in the 103 ranking path generally

Actual code:

- only live runner reads `ranking.blend_weights`
- notebook and LEAN still hardcode 0.5 / 0.5

### E. Architecture text still describes a cleaner 103 pipeline than what code executes

Examples of drift:

- some docs describe 103 as if volume scan remains part of the stock-selection path
- some docs describe simulation, LEAN, and live as all ranking off current model confidence in the same way
- some docs describe regime-adaptive execution more broadly than live actually supports

### F. Test-count claims are internally inconsistent in repo prose

Observed claims across docs:

- `403`
- `405`

Actual current collected count is `405`.

## Documentation conclusion

The docs are useful, but they currently overclaim architectural coherence.

That is dangerous in this repo because you are explicitly using docs and the logic graph as policy truth.

---

## 4. Design Review

## Core judgment

Your concern is correct: the current design has become premature in the specific sense that it is accumulating interacting mechanisms faster than it is accumulating trustworthy attribution.

You now have all of these layers affecting entry quality and portfolio behavior:

- per-symbol model tournament
- relative labels vs SPY
- raw score generation
- calibration into rank score
- logistic blend weight estimation
- regime detector
- transition window
- EMA50 gate
- SPY velocity gate
- earnings filter
- wash-sale guard
- min-hold
- sell streak
- tiered slot thresholds
- sector guard
- correlation guard
- trailing stop
- cumulative stop
- single-day loss gate
- drawdown circuit breaker

That is too many degrees of freedom for the current parity and attribution discipline.

## Why this is risky

### A. Too many moving parts means weak causal attribution

If performance changes, you cannot tell whether the cause was:

- the model
- calibration
- blend weights
- one of the macro gates
- candidate pruning
- portfolio construction
- execution timing

When a system gets here, development often becomes story-driven instead of evidence-driven.

### B. The system is no longer single-source-of-truth

This is the biggest architecture problem.

The research engine, LEAN engine, and live engine are not just different adapters. They are making strategy decisions differently.

That means:

- backtest confidence is weaker than it looks
- notebook tuning can optimize the wrong engine
- live outcomes are harder to explain post hoc

### C. Daily recalibration can dominate the strategy while hiding behind “meta logic”

On 2026-04-16, `recalibrate_scores.py` set:

- `ranking.blend_weights = [1.0, 0.0]`

That means RS was effectively zeroed out in live selection that day.

This is not necessarily wrong, but it means the meta-layer can silently redefine the live strategy day by day.

That is dangerous if:

- the sample is small
- coefficients are unstable
- docs/backtests still assume a 50/50 blend

The warning messages from sklearn during recalibration are another signal that this layer is numerically fragile enough to deserve much tighter controls.

### D. Live trading on daily bars is operationally risky when full runs are triggered intraday

On 2026-04-16 there were full runs at roughly:

- 08:44 PT
- 14:20 PT

If the intended design is “after close retrain then trade”, full intraday runs create ambiguity about whether you are trading on:

- prior completed daily bar
- partially formed current daily bar
- broker live prices mixed with daily feature history

This can easily create execution semantics that no backtest actually represents.

### E. Idempotency and run control are weak

There were multiple full runs in one day plus earlier failures. That increases the chance of:

- inconsistent state
- repeated recalibration
- different buy sets across reruns
- hard-to-explain orders

This is especially risky once live state, broker reconciliation, and config mutation are all happening in the same session.

### F. The live runner is strategically simpler than the backtest but financially more important

Live currently omits some of the complexity that docs emphasize, including:

- live regime engine
- trailing stop state

That means the richest design is not the one actually trading your capital.

This is a strong sign the design grew faster than the executable architecture.

---

## 5. How To Improve Performance Without Making the System More Fragile

This section is about improving strategy output and engineering reliability together.

## Recommendation 1: Force one execution truth

Best architecture-level move:

- Make the notebook produce a daily decision artifact per ticker, per date
- Let LEAN and live consume that artifact for validation/execution
- Stop recomputing materially different decision logic in three places

Example artifact fields:

- date
- ticker
- regime
- raw_score
- rank_score
- rs_score
- candidate_passed
- selected
- target_weight
- exit_reason if any

This does two things:

1. Makes live behavior explainable
2. Makes backtest/live parity auditable

If you want to keep LEAN self-contained, then the second-best option is:

- create a shared machine-readable policy spec
- generate notebook and LEAN checks from that spec

But right now handwritten duplication is costing too much.

## Recommendation 2: Decide what 103 actually is

Right now 103 is not one strategy. It is a family of overlapping implementations.

You should choose one of these paths explicitly:

### Path A. Keep raw-score 50/50 rank as the canonical strategy

Then:

- remove live-only calibration from selection
- use calibration only for monitoring and diagnostics
- keep notebook, LEAN, and live aligned around raw-score semantics

### Path B. Make calibrated rank score the canonical strategy

Then:

- port calibrated ranking into notebook simulation
- port calibrated ranking into LEAN
- test all three against the same score semantics

This is the path I would choose if you want mixed-model comparability.

But do not keep half the system on raw scores and the other half on calibrated probabilities.

## Recommendation 3: Cut layers that are not earning their complexity

Immediate candidates for ablation:

- RS blend if live repeatedly converges to `1.0 / 0.0`
- live-only logistic blend updates if the signal is unstable or usually collapses to one input
- any regime branch not implemented in live

Rule:

- every layer must justify itself with a tracked marginal improvement over a simpler baseline

If it cannot, remove it.

## Recommendation 4: Move from branch tests to ledger tests

Create one golden ledger test that replays a synthetic market tape and checks, per day:

- held names
- sell decisions
- candidate list
- ranked order
- selected names
- per-position size

Run it for:

- notebook simulation
- LEAN helper path
- live runner path

That one test will be more valuable than another 50 helper-replica unit tests.

## Recommendation 5: Enforce execution-time guards in live

For real-money safety:

- block buy phase unless market is closed or after a configured cutoff
- add a run-once-per-session lock for `daily_103.sh`
- persist and use per-position HWM so live can honor trailing stop
- require data freshness checks before any buy scan

These are not optional polish items. They are production controls.

## Recommendation 6: Separate research config from live mutable config

Right now `strategy_config.json` is being mutated by recalibration.

That is convenient, but it mixes:

- stable research assumptions
- daily live control state

Better split:

- `strategy_config.json`: structural strategy definition
- `runtime_state.json` or `daily_signal_state.json`: calibration and per-day weights

That makes provenance easier and avoids silent mid-session strategy mutation.

## Recommendation 7: Treat regime confidence scaling as a first-class design choice

Right now LEAN scales sizing by confidence but notebook does not.

This should be made explicit and tested as one of two choices:

- confidence scales size
- confidence does not scale size

Choose one and implement it everywhere.

## Recommendation 8: Simplify before optimizing signal quality

If the goal is better output performance, the highest expected-value path is:

1. unify semantics
2. remove execution drift
3. run ablations on a simplified stack
4. only then add new signal ideas

In quant systems, strategy simplicity plus measurement discipline usually beats adding another filter.

---

## 6. What Happened On 2026-04-16

## Timeline

### 00:25 PT full run failed

`logs/daily_103/2026-04-16.log` shows an early full run that:

- completed notebook export
- completed LEAN data export
- connected to Alpaca
- then failed with `requests.exceptions.ConnectionError` / `RemoteDisconnected`

The failure occurred during position retrieval in the live trading step.

This is consistent with the separate early-morning `live_103` open log, which also failed on Alpaca API position calls.

### 06:46 PT open sell-only run failed

`logs/live_103/2026-04-16-open.log` shows another Alpaca disconnect / connection-abort failure.

### 08:44 PT full run succeeded and bought BA

At 08:44 PT the full run completed all steps and placed a buy for BA.

Context logged by the runner:

- SPY > EMA50
- SPY 3-day velocity filter clear
- drawdown circuit breaker clear
- one existing holding: `AMZN`

Important live calibration event earlier in the same run:

- `ranking.blend_weights` updated to `[1.0, 0.0]`

So the buy ranking that run was effectively:

- 100% calibrated rank score
- 0% RS score

Candidate ranking at 08:44 PT:

1. BA rank `+0.4175`
2. CAT rank `+0.2797`
3. UNH rank `+0.2475`
4. LLY rank `+0.2271`
5. GLD rank `+0.1929`
6. UBER rank `+0.1748`
7. META rank `+0.1585`
8. XLV rank `+0.1117`

Why BA was bought:

- it was the highest-ranked candidate
- slot 1 threshold was `0.10`
- BA exceeded it comfortably
- available cash allowed `3` shares at about `$217.63`

Why no second buy at 08:44:

- slot 2 threshold was `0.30`
- every remaining candidate ranked below `0.30`

So only BA passed the slot-2 escalation rule.

### 12:49 PT preclose sell-only run held AMZN and BA

Preclose log shows two held positions:

- `AMZN`
- `BA`

Neither was sold because:

- stop-loss did not fire
- single-day-loss gate did not fire
- model sell was blocked by `min_hold=20d` with `held=0d`

### 14:20 PT second full run succeeded and bought UNH

A later full run again recalibrated blend weights to `[1.0, 0.0]` and rescanned.

Held at start:

- `AMZN`
- `BA`

Top ranked candidates that run:

1. CAT rank `+0.2914`
2. UNH rank `+0.2475`
3. LLY rank `+0.2271`
4. GLD rank `+0.1929`
5. GOOG rank `+0.1923`
6. NFLX rank `+0.1796`
7. UBER rank `+0.1748`
8. JPM rank `+0.1723`
9. META rank `+0.1585`
10. XLV rank `+0.1117`

Why CAT was not bought:

- invest size was about `$750`
- CAT price was about `$790.66`
- integer share sizing produced `0` shares
- runner logged `insufficient cash`

Why UNH was bought:

- it became the next candidate after CAT was rejected
- slot 1 threshold was still `0.10`
- UNH rank `+0.2475` passed
- two shares fit inside the available cash and sizing rules

Why no second buy after UNH:

- slot 2 threshold again jumped to `0.30`
- all remaining candidates ranked below `0.30`

End-of-run holdings:

- `AMZN`
- `BA`
- `UNH`

## Why AMZN was already there

The logs show `AMZN` being reconciled from Alpaca history as an existing holding with `entry_date=2026-04-16`, but I did not find a successful local log line in the current repo logs that shows the AMZN buy placement itself.

Most likely explanations:

- the order was placed during an earlier failed run before the local process crashed
- the order was placed outside the currently retained local logs
- the order came from a manual or other external execution path and was later reconciled from broker history

One oddity worth noting from `backtesting/renquant_103/live_state.json`:

- `AMZN` has `entry_dates["AMZN"] = "2026-04-16"`
- `last_sell_dates["AMZN"] = "2026-04-15"`

That is not automatically wrong, but it is another sign the live state and broker reconciliation layer is not fully clean or fully self-explanatory.

---

## 7. Direct Answer On Whether To Keep Building Here

If the question is whether this repo is salvageable and worth continuing: yes.

If the question is whether it is currently clean enough to trust without architectural cleanup: no.

My view:

- The repo has real structure.
- The author cared enough to add tests and explicit logic.
- The current problem is not incompetence.
- The current problem is uncontrolled divergence plus too many moving parts.

That is fixable, but only if the next phase is simplification and unification, not feature growth.

---

## 8. Priority Actions

If I were taking over the next hour of engineering work, I would do these in order:

1. Make a decision: canonical score semantics are either raw-score or calibrated rank-score, not both.
2. Make live either implement the real 103 regime engine or explicitly downgrade docs/backtests to match the simpler live reality.
3. Add a golden ledger parity test across notebook-like, LEAN-like, and live paths.
4. Add live trailing-stop state persistence.
5. Block full buy runs outside a defined post-close window.
6. Split mutable daily calibration state out of `strategy_config.json`.
7. Remove any layer that cannot show marginal value in an ablation table.

That sequence improves both engineering reliability and trading-process quality.

---

## Final Judgment

The repo is promising, but the main risk is no longer model alpha. The main risk is that different parts of the system are telling slightly different stories about what strategy is actually running.

Until that is fixed, every backtest, every doc, and every post-trade explanation is weaker than it should be.