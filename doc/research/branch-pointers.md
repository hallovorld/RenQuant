# Active experimental branches

Index of long-running experimental branches. Each branch isolates a major
architectural change from `main` so production stays clean. Merge back
to main only after the experiment validates (per CLAUDE.md §5.2 — sanity
check every claim) AND survives B2 hold-out (per roadmap §B2).

**Convention:** branches under `exp/` are isolated worktrees, never
merged trivially. Each carries a design doc on its own branch and a
pointer here on main.

---

## `exp/wl500-and-sector-arch` — sector-aware multi-layer architecture (active)

**Started:** 2026-05-01
**Worktree:** `/tmp/renquant-wl500-exp`
**Lead doc on branch:** `doc/research/per-sector-architecture-plan.md`

### Why
Watchlist expansion past 178 tickers has been blocked twice by E17/E21
(OOS IC went negative on heterogeneous universes). Phase 0 diagnostic on
2026-05-01 confirmed via KS test that ALL 28 sector pairs in wl178 have
median KS ≥ 0.30 — Witter 2025 mechanism (rank-pairwise loss degrades on
heterogeneous distributions) is the actual cause.

### Architecture (post literature audit, design v2)

Five-layer pipeline; each layer flag-gated for independent A/B:

| Layer | Status | Reference |
|---|---|---|
| L1 — per-`(date, sector)` rank-norm | ✅ helper + Task wired (training + inference) | qlib `CSRankNorm` |
| L2 — sector as model conditioning | ⏳ next commit | Gu-Kelly-Xiu 2020 |
| L3 — Temporal Graph Convolution | ⏳ Phase C, NN backend | Feng et al. 2019 |
| L4 — Soft MoE (sector-group gating) | ⏳ Phase D, NN backend | MIGA 2024 |
| L5 — Empirical-Bayes shrinkage | ✅ helper landed | Robbins 1956, Efron-Morris 1973 |

Hard-split per-sector approach (the original v1 design) was **rejected**
based on Witter 2025 (JMS Vol. 10(3)): paper ran exactly that A/B and
showed pooled beats per-sector. Audit also surveyed qlib, alpha-mind,
MIGA, FSTGAT — none use hard splits.

### Commits on this branch (newest first)
- `dad3781` fix(audit): M1 — extract `RAW_FACTOR_COLS_FOR_NORM` shared constant
- `936aedd` feat(layer1): wire SectorRankNormalizeTask into inference path
- `c6b8047` feat(layer1): SectorRankNormalizeTask wired into PanelAssemblyJob
- `82c464f` feat(layer5): EB-shrinkage helpers + 17 tests
- `0917c14` feat(layer1): cross_sectional_rank_within_sector helper + 16 tests
- `ae9ecdb` feat(audit): wl178 failure diagnostic + universe builder + design v2

### Background tasks running
- **A/A retrain** (started 2026-05-01 21:43 PT) — both halves of wl178 random split. Half A in calibrator phase; half B in PanelFeatureJob. Logs at `/tmp/aa_half_a.log`, `/tmp/aa_half_b.log`. Confirms Witter mechanism if both halves recover ≈ +0.04 IC.
- **Universe OHLCV fetch** (started 2026-05-01 22:21 PT) — pulling 5y daily bars for 793 missing Russell-1000 tickers. Network-bound; no CPU contention with A/A. Log at `/tmp/universe_fetch.log`.

### Merge criteria (gates before PR back to main)
- [ ] Per-sector path beats single-panel baseline on B2 hold-out by ≥ +1 APY pt (Layer 1+2 alone), ≥ +2 (Layer 3 graph), ≥ +4 (Layer 4 MoE)
- [ ] OOS Spearman IC ≥ +0.0418 on the wl178 universe (rules out E17/E21 regression)
- [ ] Train IC ≥ +0.118 (rules out "model can't fit" — wl178 sat at +0.085)
- [ ] All hard acceptance gates green
- [ ] Full test suite green on this branch
- [ ] Sectors with <20 tickers route to general-pool fallback cleanly (no inference-time crash)

### Recovery
```bash
git worktree add /tmp/renquant-wl500-exp exp/wl500-and-sector-arch
mkdir -p /tmp/renquant-wl500-exp/data
ln -s $(pwd)/data/ohlcv /tmp/renquant-wl500-exp/data/ohlcv
```

---

## `exp/b2-baseline-wl103` — baseline B2 hold-out reference (active)

**Started:** 2026-05-01
**Worktree:** `/tmp/renquant-b2-baseline`
**Purpose:** establishes the apples-to-apples Layer-1-OFF baseline number that the sector-aware experiment compares against. Train wl103 (production) with `sample_end = 2024-12-31`; sim 2025-01-02 → 2026-04-30.

This baseline is **not yet dispatched** — waiting until A/A clears its
heaviest CPU phase (~10-20 min) so the two trainers don't oversubscribe
the M2 Pro cores.

### How to dispatch when ready
```bash
cd /tmp/renquant-b2-baseline
python scripts/holdout_backtest.py \
  --strategy-config-name strategy_config.golden.json \
  --train-end 2024-12-31 --sim-start 2025-01-02 --sim-end 2026-04-30 \
  > /tmp/b2_baseline.log 2>&1 &
```

Expected wallclock: ~30-60 min. Output JSON at
`/tmp/renquant-b2-baseline/data/holdout_results/2024-12-31.json`.

---

## Future planned branches (not yet created)

| Branch | Purpose | Trigger |
|---|---|---|
| `exp/per-sector-categorical` | Phase A.4 — `gics_sector` as XGBoost categorical | After A/A confirms diagnostic |
| `exp/graph-attention-nn` | Phase C — Feng 2019 TGC, NN backend | After Layer 1+2 validated |
| `exp/miga-moe` | Phase D — MIGA 2024 soft MoE, NN backend | After Layer 3 validated |
| `exp/macro-v3-isolation` | Quarantine the in-flight macro/embedding WIP currently uncommitted on main, get main green | Anytime — pure refactor |

---

## Process notes

- **Always create via `git worktree add`** — keeps main checkout clean.
- **Always `mkdir -p worktree/data && ln -s repo/data/ohlcv`** — data dir is gitignored, must be symlinked into the worktree.
- **Dispatch long-running training with `run_in_background`** and tail logs at `/tmp/<descriptor>.log`. Don't poll — wait for task notifications.
- **Each branch carries its own design doc** in `doc/research/`. The pointer here on main only summarizes; the source of truth is on the branch.
- **Pre-existing failures landscape (as of 2026-05-01 main)**: 13 tests fail on the macro v3 / asset-embeddings WIP that's currently uncommitted on main (`backtesting/.../macro_per_ticker.py`, `feature_matrix.py`, `tests/test_panel_bugfixes.py`). These are NOT introduced by recent commits; they came along with WIP that should be moved to its own branch.
