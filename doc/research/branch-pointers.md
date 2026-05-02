# Active experimental branches

Index of long-running experimental branches. Each branch isolates a major
architectural change from `main` so production stays clean. Merge back
to main only after the experiment validates (per CLAUDE.md §5.2 — sanity
check every claim) AND survives B2 hold-out (per roadmap §B2).

---

## `exp/wl500-and-sector-arch` — per-sector sub-model architecture

**Started:** 2026-05-01
**Branch base:** `main` @ `2874698` (M3 conformal Gate B fitted artifact)
**Worktree:** `/tmp/renquant-wl500-exp` (developer-local; recreate via
`git worktree add /tmp/renquant-wl500-exp exp/wl500-and-sector-arch`)

### Why
Watchlist expansion has been blocked by **failed-experiments E5 / E17 / E21**
(see `doc/research/failed-experiments-log.md`):

- Two independent expansion attempts (B1=227 mutual-fund-spec, wl178=quality-filter)
  both produced **negative or near-zero OOS IC** when retraining the existing
  cross-sectional rank-pairwise panel-LTR on heterogeneous universes.
- Diagnosis (E17): the rank-pairwise loss assumes universe homogeneity in
  feature distribution. Adding sectors (financials / industrials / energy /
  consumer) to a tech-heavy 103 ticker baseline breaks the rank ordering;
  even train IC drops, so it's not just generalization — model can't fit.
- Direct conclusion: **list expansion alone cannot work on the current
  architecture.** Closing the path forward requires per-sector or
  sector-aware modeling.

### What this branch is doing
1. Pulled broader universe (Russell 1000 = 1,009 tickers, via iShares IWB
   ETF holdings CSV) → `scripts/watchlist_universe.json`.
2. Designed per-sector sub-model architecture (one rank-pairwise ranker
   per GICS sector, inference routes by ticker-sector lookup, scores
   compared via per-sector percentile).
3. Built skeleton `kernel/panel_pipeline/` job + tasks for the sector-aware
   path. Default off (legacy single-panel path remains the production
   default until validation lands).
4. Will dispatch OHLCV backfill for new universe tickers in background
   (long-running, non-blocking).
5. Validates via B2 hold-out (`scripts/holdout_backtest.py` already on
   main) before any merge proposal.

### How to continue work on this branch
```bash
# From repo root
git worktree add /tmp/renquant-wl500-exp exp/wl500-and-sector-arch  # if worktree gone
cd /tmp/renquant-wl500-exp
# Work...
git push -u origin exp/wl500-and-sector-arch  # if remote tracking needed
```

### Merge criteria (do not merge until ALL of these clear)
- [ ] Per-sector ranker beats single-panel baseline on B2 hold-out by ≥ +2 APY pts
- [ ] OOS Spearman IC ≥ +0.0418 (current production with wl103) — must NOT
      regress; expansion only adds value if IC holds or improves
- [ ] Train IC ≥ +0.118 (rules out "model can't fit" failure mode that
      sank wl178)
- [ ] Full test suite green on the experimental branch
- [ ] Each sector-sub-model has at least 30 tickers or fall-back to a
      "general" bucket — under-populated sectors must not break inference
- [ ] Acceptance gates pass on production-config retrain with the new path

### Status
🔴 Branch created; design doc + skeleton in flight. No production impact.
Track progress via:
- `doc/research/per-sector-architecture-plan.md` (on branch)
- `git log exp/wl500-and-sector-arch` (commits)
