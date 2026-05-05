# wl183 Promotion — Deferred 2026-05-05 (with reproduction recipe)

Per CLAUDE.md §5.7, every failed experiment lands in the log with the
recipe. Two distinct bugs blocked wl183 promotion this session.

## Status snapshot

- **wl103 production** (2026-05-04 partial retrain + buy_floor adaptive):
  Sharpe 1.10, APY 13.27%, B2 holdout. e2e green tonight. SHIPPED.
- **wl183 candidate** (`strategy_config.wl183_daily_clean.json`):
  artifact training metrics LOOK BETTER (oos_mean_ic +0.0023,
  oos_std_ic −42%, train/oos gap negative — generalizing OOS better
  than train), but runtime path has unresolved bugs.

## Bug 1 — 30-min B2: NGBoost `set_μσ=0` every bar

```
Phase 2b: 90 candidates from 90 tickers
RealizedVolGate: dropped 33/90, 57 survive
ApplyNGBoostTask: n_cands=57, set_μσ=0, not_in_idx=57
ApplyScoresTask: panel scored 0/57 candidates
```

The inference matrix X.index does not include the candidate tickers
(or matrix is empty). Diag suggests ResolveInferenceFramesTask /
AssembleInferenceMatrixTask filter all 57 candidate tickers out.
Possible causes (untested):

- `panel_feature_frames` keyed by something different than ticker
  symbol (e.g. with suffix or after sector remapping)
- The 90 candidate tickers' frames are non-empty but `_pick_today_row`
  returns None for the sim_start date (frames have no row ≤
  2025-05-05)
- `ff_sub` filter dropping all because tickers absent from
  `panel_feature_frames` keys

## Bug 2 — 4-bar diag: TickerCandidateJob 0/90 cands

```
run_parallel: TickerCandidateJob 90 tickers 12 workers timeout=600
TickerCandidateJob DONE 0.01s    ← 100x faster than the 30-min sim
Phase 2b (buy scan): 0 candidates from 90 tickers
```

Each per-ticker job chain (EarningsFilter → WashSale → BuildFeatures
→ ScoreBuy → ScoreThreshold → RelativeStrength → AssembleCandidate)
short-circuits at the FIRST task. 5 ms total for 90 tickers in 12
workers means each chain returns False / raises an exception nearly
immediately. Likely candidates:

- `BuildFeaturesTask` finds no feature_cache for these tickers'
  windows (despite SimAdapter showing `feature_cache=191 tickers`)
- `EarningsFilterTask` flags all 90 (improbable — would need wl183
  earnings calendar to be entirely populated for this window)
- An exception is thrown and silently swallowed by run_parallel's
  error isolation

This bug appeared between two runs of the SAME side config + same
artifacts; difference is ONLY the addition of one log line to
`tasks_feature_matrix.py::AssembleInferenceMatrixTask`. The log line
is downstream of Phase 2b — cannot directly cause the regression.
Hypothesis: shared sim-DB state (`data/sim_runs.db` is cleared between
runs) interacts with TickerCandidateJob in a way I haven't traced.

## Reproduction recipe

```bash
# Bug 1 (30-min)
bash scripts/diagnose_funnel.sh wl183_promotion \
  --strategy-config-name strategy_config.wl183_daily_clean.json
# Look for: ApplyNGBoostTask: ... set_μσ=0, not_in_idx=N

# Bug 2 (4-bar)
python scripts/holdout_backtest.py --skip-train \
  --strategy-config-name strategy_config.wl183_daily_clean.json \
  --train-end 2025-05-04 --sim-start 2025-05-05 --sim-end 2025-05-09 \
  --output /tmp/wl183_diag.json
# Look for: Phase 2b: 0 candidates from 90 tickers; DONE 0.01s
```

## Next debug step (when revisited)

1. Add a print at the TOP of TickerCandidateJob's first task
   (EarningsFilterTask.run) to log ticker + return decision.
2. Run a SHORT sim with one bar.
3. Identify which task short-circuits + why.
4. Fix the per-ticker context construction for wl183.

## Why not now

User's directive 2026-05-05 00:00 PT: "long term no action needed,
focus on code quality." Promotion deferred; not blocking on it.
production wl103 (Sharpe 1.10, real-money e2e green tonight) is the
operating baseline.
