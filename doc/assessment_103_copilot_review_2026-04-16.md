# Copilot Review of assessment_103.md

Date: 2026-04-16
Scope: Review Claude Code's assessment against the current renquant_103 notebook, LEAN strategy, live runner, tests, artifacts, and docs.

## Bottom Line

The assessment is directionally strong. I agree with its main thesis that the strategy has more engineering sophistication than statistical validation, and I agree that the live runner is materially behind the notebook and LEAN implementations.

That said, a few claims in the assessment are either stale, imprecise, or point to the wrong root cause. I also found two material engineering defects that the assessment missed:

1. LEAN GMM inference does not apply the saved scaler, even though the GMM was trained in standardized feature space.
2. The live runner can oversubscribe cash by sizing multiple buys from the same initial cash snapshot.

## What I Agree With

These findings are supported by the current codebase:

- Live runner is effectively hardcoded to BULL_CALM parameters.
  - Evidence: `live/runner.py:417-425`
- Live runner does not implement trailing stop behavior.
  - Evidence: `live/runner.py:573`
- CUSUM currently normalizes using the same test window's mean and standard deviation.
  - Evidence: `backtesting/renquant_103/main.py:569-593`
  - Evidence: `common/indicators/regime.py:74-97`
- CHOPPY has `max_hold_days = 10` while the strategy keeps `min_hold_days = 20`, so model-sell cannot become the active exit before max-hold in CHOPPY.
  - Evidence: `backtesting/renquant_103/strategy_config.json:36-37`
  - Evidence: `backtesting/renquant_103/strategy_config.json:94-101`
- XLV and UNH are currently present on disk with Sharpe below the export floor.
  - Evidence: `backtesting/renquant_103/models/XLV/XLV-policy-metadata.json:22`
  - Evidence: `backtesting/renquant_103/models/UNH/UNH-policy-metadata.json:28`
- Blend weights are currently `[1.0, 0.0]`.
  - Evidence: `backtesting/renquant_103/strategy_config.json:230-235`

## Corrections to the Assessment

### 1. Test Count Is Stale

The assessment says 405 tests. The current repo state is 407 tests collected.

- Assessment claim: `doc/assessment_103.md:12`
- Current docs: `CLAUDE.md:191`, `CLAUDE.md:226`, `README.md:191`
- Verified locally on 2026-04-16: `407 tests collected`

This is not a strategy defect, but it means the assessment should not be treated as fully current without revision.

### 2. XLV/UNH Root Cause Is Real, but the Attribution Is Wrong

The assessment blames the notebook's later "bootstrapped" display logic for the below-floor models remaining active. That is not the actual loading path used by LEAN or the live runner.

What is true:

- The notebook includes a later "bootstrapped models that exist on disk" section.
  - Evidence: `Notebooks/renquant_103.ipynb:2248-2268`
- XLV and UNH remain on disk even though they are below floor.
  - Evidence: `backtesting/renquant_103/models/XLV/XLV-policy-metadata.json:22`
  - Evidence: `backtesting/renquant_103/models/UNH/UNH-policy-metadata.json:28`
- LEAN and live both load models directly from per-symbol metadata if the file exists.
  - Evidence: `backtesting/renquant_103/main.py:1213-1231`
  - Evidence: `live/runner.py:280-305`

What that means:

- The real bug is not the notebook's chart-only bootstrap section.
- The real bug is that stale/below-floor model directories are not being purged for renquant_103 before LEAN/live load their watchlist models.

### 3. Some Numeric Claims Need Reproducible In-Repo Evidence

The assessment gives exact figures for:

- Sharpe tournament inflation (`1.21 annual SR units`)
- CUSUM false positive rate (`8.1%`)

Those may be plausible, but I did not find a checked-in script, notebook cell, or test in the repo that reproduces those values. They should be treated as external analysis until the simulation is added to the repo.

## Additional Issues the Assessment Missed

### 1. LEAN GMM Inference Ignores the Saved StandardScaler

This is the most important missed issue.

The notebook-side GMM implementation standardizes features before fitting:

- `common/indicators/regime.py:194`

The artifact saves:

- `scaler_mean`
- `scaler_scale`

Evidence:

- `common/indicators/regime.py:247-253`
- `backtesting/renquant_103/spy-gmm-regime.json:118-129`

But LEAN inference uses raw feature vector `x` directly against stored means/covariances:

- `backtesting/renquant_103/main.py:621-639`

I found no scaling step in LEAN before Mahalanobis scoring. That means the live backtest regime probabilities are not mathematically aligned with the trained GMM.

### 2. Live Runner Can Oversubscribe Cash Across Multiple Buys

The runner snapshots cash once:

- `live/runner.py:492-503`

Then each buy uses that same `cash_avail` snapshot:

- `live/runner.py:888-891`

But `cash_avail` is never reduced after a buy inside the same selection loop. If two or more buys pass filters on one run, each order can be sized as if the previous one never consumed cash.

This is a real execution bug.

### 3. Live Runner Misses More Than Just Regime Detection

The assessment correctly calls out the missing live regime logic, but the gap is broader:

- No transition uncertainty window in the live buy path.
- No earnings filter in the live buy path.
- No correlation guard in the live selection loop.
- Position sizing uses top-level `position_sizing`, not regime-specific `regime_params`.

Evidence:

- Config contains these controls:
  - `backtesting/renquant_103/strategy_config.json:47-49`
  - `backtesting/renquant_103/strategy_config.json:56`
  - `backtesting/renquant_103/strategy_config.json:59`
  - `backtesting/renquant_103/strategy_config.json:62`
  - `backtesting/renquant_103/strategy_config.json:66-108`
- Runner buy path does not consume them:
  - `live/runner.py:409-425`
  - `live/runner.py:744-926`

### 4. Live Runner Reconciliation Only Backfills Sells From Today

The assessment noticed the AMZN state inconsistency. The code path supports that concern.

Current state:

- `backtesting/renquant_103/live_state.json:5`
- `backtesting/renquant_103/live_state.json:14`

Reconciliation logic:

- `live/runner.py:526-546`

The runner only re-seeds `last_sell_dates` when `sell_day == today_str`. That means a restart or missing state can drop valid recent sell history from prior days, weakening the wash-sale guard.

### 5. Live Runner Uses a Hardcoded Sector ETF Map

The config has a `sector_etf_map`, but the runner recomputes a local hardcoded map for RS scoring instead of using config:

- Config source: `backtesting/renquant_103/strategy_config.json:140-149`
- Hardcoded runner map: `live/runner.py:802-805`

This is not the biggest issue, but it is a maintainability and parity problem.

### 6. Live Tests Are Not Covering the Highest-Risk Drift Areas

The repo is strong on notebook/LEAN parity tests, but weaker on the live path.

- Notebook/LEAN parity coverage exists in `tests/test_policy_alignment.py`
- LEAN-specific policy coverage exists in `tests/test_lean_policies.py`
- Live runner tests currently focus mainly on ranking and candidate logging in `tests/test_runner_ranking.py`

I did not find equivalent live-runner regression tests for:

- regime-specific sizing
- trailing stop persistence
- earnings filter
- correlation guard
- transition window
- multi-buy cash depletion

## Severity-Ordered Fix Plan

### Phase 1: Fix Execution Correctness First

1. Fix LEAN GMM scaling.
   - Apply `scaler_mean` / `scaler_scale` to the live feature vector before likelihood scoring.
   - Add a regression test that compares LEAN-side GMM probabilities against `RegimeGMM.predict()` on the same artifact and input rows.

2. Fix live cash accounting.
   - Reduce available cash after each accepted buy, or recompute from broker state after each order.
   - Add a runner test proving that two buys cannot each size off the original full cash snapshot.

3. Fix stale model artifact leakage.
   - Purge below-floor or removed-symbol model directories during notebook export or in a dedicated cleanup step.
   - Add a test that proves a below-floor symbol cannot remain loadable merely because its directory already exists.

4. Fix live wash-sale reconciliation.
   - Re-seed `last_sell_dates` from the latest historical sell regardless of whether it was today.
   - Add a restart/reconciliation regression test using historical sell data from prior sessions.

### Phase 2: Restore Live Runner Parity With 103 Logic

5. Port regime detection into the live runner.
   - Implement the same Hurst + CUSUM + GMM artifact path used by LEAN.
   - Persist transition countdown and regime confidence in state.

6. Port live trailing stop support.
   - Track per-position high-water mark in `live_state.json`.
   - Apply the same trailing stop priority as notebook and LEAN.

7. Port the missing live buy guards.
   - earnings filter
   - correlation guard
   - transition uncertainty window
   - BEAR defensive-only behavior if not already implicit in the final regime port

8. Remove hardcoded live RS config.
   - Use `sector_etf_map` from config rather than the local hardcoded map.

9. Align live sizing with regime params.
   - Use regime-specific `max_position_pct` and `cash_reserve_pct` instead of static top-level `position_sizing`.

### Phase 3: Tighten Statistical Validation

10. Add a reproducible research artifact for the assessment's numeric claims.
    - Sharpe uncertainty estimate
    - tournament selection inflation
    - CUSUM false-positive simulation

11. Revisit calibration methodology.
    - At minimum, document that `recalibrate_scores.py` calibrates on the current model's available history, not a fresh untouched holdout.
    - Prefer a cleaner split between model selection and calibration windows if you want the calibrated rank score to support stronger statistical claims.

12. Reassess regime signal robustness.
    - The Hurst/CUSUM concerns may still be valid, but they should be validated with an in-repo study rather than left as opinion.

### Phase 4: Clean Up Documentation and Audit Trail

13. Update `doc/assessment_103.md`.
    - Correct the test count.
    - Replace the XLV/UNH root-cause explanation with the actual stale-artifact problem.
    - Add the missed GMM scaling bug and live cash oversubscription bug.

14. Update stale docs that still say 405 tests.
    - `doc/renquant_103_design.md`
    - any other lingering review docs with the old count

## Recommended Implementation Order

If you want the fastest path to a safer system, do the work in this order:

1. LEAN GMM scaling bug
2. live cash oversubscription bug
3. live wash-sale reconciliation fix
4. stale model directory purge
5. live trailing stop support
6. live regime detection and regime-specific sizing
7. live earnings/correlation/transition guards
8. statistical validation and documentation refresh

## Practical Note

The highest-value boundary is still the same:

- first make execution correct
- then make live and backtest behavior match
- then make the statistical claims more defensible

Right now the biggest risks are execution drift and stale artifacts, not just weak Sharpe confidence intervals.