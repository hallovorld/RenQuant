# Pipeline Unification Plan

**Branch**: `pipeline-unification`  
**Goal**: Make notebook, LEAN, and live runner all run the same declared pipeline over a shared context — so strategy logic lives in exactly one place.

---

## Problem

Three components implement the same daily decision logic independently:

| Component | File | Lines of decision logic |
|-----------|------|------------------------|
| Notebook simulation | `renquant_103.ipynb` cell 657a4a6c | ~300 |
| LEAN engine | `main.py` `OnData()` | ~200 |
| Live runner | `pipeline/jobs/execution.py` | ~150 |

Every strategy change requires triple-updating. Parity is tested by mocks, not by structure.

There is also no declared training pipeline — training logic lives in `training/features.py`, `training/tournament.py`, and `training/export.py` called sequentially from the notebook, but not as a named pipeline.

---

## Solution

Two pipeline types, declared once, run everywhere.

### Inference Pipeline (per bar / per day)

```
RegimeJob → DrawdownJob → SellJob → BuyGatesJob → CandidateJob → RankingJob → SelectionJob
```

All 7 jobs live in `kernel/pipeline/jobs/` — Docker-safe, no `common/` imports. A single `InferenceContext` dataclass carries all shared state between jobs.

Each platform provides a thin **adapter** that normalizes its data into `InferenceContext`, then calls `InferencePipeline().run(ctx)`.

### Training Pipeline (once per retrain, notebook only)

```
DataFetchJob → RegimeFitJob → FeatureJob → TournamentJob → ExportJob → CorrelationJob → CalibrationJob
```

All 7 jobs live in `training/pipeline/jobs/` (can import `common/`, not Docker-constrained). A single `TrainingContext` carries shared training state.

---

## Target Directory Structure

```
backtesting/renquant_103/
│
├── kernel/
│   └── pipeline/                    ← NEW (Docker-safe)
│       ├── __init__.py
│       ├── base.py                  ← Job ABC + Pipeline class
│       ├── context.py               ← InferenceContext dataclass
│       └── jobs/
│           ├── regime.py            ← layers 1-3 + hard override + resolve + confidence
│           ├── drawdown.py          ← mark-to-market, HWM, circuit breaker, skip_buys
│           ├── sell.py              ← all 5 exits in priority order
│           ├── buy_gates.py         ← transition window, BEAR branch, velocity, EMA50
│           ├── candidates.py        ← scan + filter (wash-sale, earnings, model, min score)
│           ├── ranking.py           ← normalize + blend weights → combined_rank
│           └── selection.py         ← tiered threshold, sector, correlation, sizing, execute
│
├── training/
│   └── pipeline/                    ← NEW
│       ├── __init__.py
│       ├── base.py                  ← same Job/Pipeline primitives (or re-export from kernel)
│       ├── context.py               ← TrainingContext dataclass
│       └── jobs/
│           ├── data.py              ← DataFetchJob (fetch OHLCV + sector ETFs + SPY)
│           ├── regime_fit.py        ← RegimeFitJob (train GMM, save spy-gmm-regime.json)
│           ├── features.py          ← FeatureJob (build labelled frames per ticker)
│           ├── tournament.py        ← TournamentJob (4-model tournament, OOS Sharpe)
│           ├── export.py            ← ExportJob (save model artifacts to models/)
│           ├── correlation.py       ← CorrelationJob (pairwise matrix, save artifact)
│           └── calibration.py       ← CalibrationJob (isotonic/Platt, blend weights)
│
├── adapters/                        ← NEW (thin per-platform wiring, ~30 lines each)
│   ├── __init__.py
│   ├── notebook.py                  ← NotebookAdapter: ohlcv + config → InferenceContext
│   ├── lean.py                      ← LeanAdapter: QCAlgorithm self → InferenceContext
│   └── runner.py                    ← RunnerAdapter: broker + ohlcv → InferenceContext
│
├── renquant_103.ipynb               ← training cells: TrainingPipeline().run(ctx)
│                                       sim cell: for day: InferencePipeline().run(ctx)
├── main.py                          ← OnData(): LeanAdapter(self, slice).run()
└── pipeline/                        ← RETIRED after migration (replaced by kernel/pipeline/ + adapters/)
```

---

## Context Schemas

### InferenceContext

Inputs (set before pipeline runs each bar):

```python
@dataclass
class InferenceContext:
    # --- time ---
    today: datetime.date

    # --- market data (normalized, all platforms provide these) ---
    ohlcv: dict[str, pd.DataFrame]       # ticker → OHLCV DataFrame
    spy_returns: np.ndarray              # recent SPY daily log returns
    prev_closes: dict[str, float]        # yesterday's close per ticker

    # --- portfolio state ---
    holdings: dict[str, HoldingState]    # ticker → HoldingState (entry_price, entry_date, ...)
    cash: float
    portfolio_value: float
    hwm: float                           # high-water mark (updated by DrawdownJob)

    # --- persistent state (lives across bars) ---
    regime_state: RegimeState            # Hurst/CUSUM/transition countdown
    sell_streaks: dict[str, int]         # consecutive sell signal count per ticker
    last_sell_date: dict[str, date]      # wash-sale clock
    pos_hwm: dict[str, float]            # per-position high-water mark for trailing stop

    # --- artifacts (loaded once) ---
    gmm_artifact: dict
    corr_dict: dict[str, dict[str, float]]
    earnings_cal: dict[str, list[str]]
    models: dict[str, Any]              # loaded model objects per ticker

    # --- config (read-only) ---
    config: dict

    # --- outputs (written by jobs, read by subsequent jobs) ---
    regime: str                          # BULL_CALM / BULL_VOLATILE / CHOPPY / BEAR
    regime_confidence: float
    regime_params: dict                  # resolved rp dict for today's regime
    skip_buys: bool                      # set by DrawdownJob
    exit_actions: list[dict]             # sell decisions from SellJob
    candidates: list[CandidateResult]    # from CandidateJob
    ranked: list[CandidateResult]        # sorted by combined_rank, from RankingJob
    orders: list[dict]                   # buy orders from SelectionJob
```

### TrainingContext

```python
@dataclass
class TrainingContext:
    config: dict
    strategy_dir: Path
    today: str                           # ISO date string

    # filled by jobs:
    ohlcv: dict[str, pd.DataFrame]       # DataFetchJob
    feature_frames: dict[str, pd.DataFrame]  # FeatureJob
    tournament_results: dict[str, dict]  # TournamentJob
    exported: list[str]                  # ExportJob
    skipped: list[str]                   # ExportJob (below Sharpe floor)
```

---

## Adapter Contract

Each adapter normalizes platform-specific data into `InferenceContext`. The adapter is the ONLY place that knows about platform APIs (`History()`, broker calls, pandas slicing).

```python
# Notebook
ctx = NotebookAdapter(ohlcv, config, persistent_state).make_context(today)
InferencePipeline().run(ctx)
persistent_state.update_from(ctx)  # write back holdings, sell_streaks, etc.

# LEAN
def OnData(self, slice):
    ctx = LeanAdapter(self).make_context()
    InferencePipeline().run(ctx)
    LeanAdapter(self).apply_orders(ctx.orders, ctx.exit_actions)

# Live runner
ctx = RunnerAdapter(broker, ohlcv, persistent_state).make_context()
InferencePipeline().run(ctx)
RunnerAdapter.execute(ctx.orders, ctx.exit_actions, broker)
```

---

## Implementation Phases

### Phase 1 — Kernel pipeline infrastructure
**Files**: `kernel/pipeline/__init__.py`, `kernel/pipeline/base.py`, `kernel/pipeline/context.py`

- `Job` ABC: `.run(ctx: InferenceContext) -> None`, `.should_skip(ctx) -> bool`
- `Pipeline`: sequential orchestrator, calls each job's `should_skip` then `run`
- `InferenceContext`: full dataclass with all fields above

**Done when**: Can instantiate `InferencePipeline([])` with an empty job list and a valid context.

---

### Phase 2 — Inference jobs
**Files**: `kernel/pipeline/jobs/*.py` (7 files)

Extract from existing simulation cell (657a4a6c) and LEAN `OnData()`. Each job is a direct translation of one section of the logic graph.

| Job | Extracts from | Logic graph section |
|-----|--------------|---------------------|
| `RegimeJob` | sim cell lines 60–110 + LEAN `_detect_regime()` | REGIME DETECTION |
| `DrawdownJob` | sim cell lines 115–130 + LEAN mark-to-market | PORTFOLIO MARK-TO-MARKET + DRAWDOWN |
| `SellJob` | `kernel/exits.compute_exits()` (already extracted) | SELL LOOP |
| `BuyGatesJob` | sim cell buy gate checks | BUY GATE CHECKS |
| `CandidateJob` | sim cell candidate scan | CANDIDATE SCAN |
| `RankingJob` | `kernel/selection.score_candidates()` | RANKING |
| `SelectionJob` | `kernel/selection.run_selection_loop()` | SELECTION LOOP |

**Done when**: A synthetic test runs `InferencePipeline(all_7_jobs).run(ctx)` and produces correct `exit_actions` + `orders` on known input.

---

### Phase 3 — Thin adapters
**Files**: `adapters/notebook.py`, `adapters/lean.py`, `adapters/runner.py`

- Each adapter ~30 lines
- Notebook adapter manages persistent state across bars (sell_streaks, last_sell_date, pos_hwm, regime_state)
- LEAN adapter wraps `QCAlgorithm` API into normalized pandas/dict structures
- Runner adapter wraps broker + cached parquet

**Done when**: Notebook simulation cell replaced by:
```python
adapter = NotebookAdapter(ohlcv, config)
for today in bt_dates:
    ctx = adapter.make_context(today)
    InferencePipeline().run(ctx)
    adapter.commit(ctx)
equity_curve = adapter.equity_curve
trade_log = adapter.trade_log
```

---

### Phase 4 — Training pipeline
**Files**: `training/pipeline/context.py`, `training/pipeline/base.py`, `training/pipeline/jobs/*.py`

Wrap the existing `training/` module functions into declared jobs.

**Done when**: Notebook training cells replaced by:
```python
ctx = TrainingContext(config=CONFIG, strategy_dir=STRATEGY_DIR, today=TODAY)
TrainingPipeline().run(ctx)
# ctx.exported, ctx.skipped, ctx.tournament_results all populated
```

---

### Phase 5 — Update LEAN + live runner
**Files**: `main.py`, `live/runner.py`

- LEAN `Initialize()`: create + cache pipeline + initial context
- LEAN `OnData()`: adapter → pipeline → apply orders
- Live runner: swap old execution path for `RunnerAdapter + InferencePipeline`

**Done when**: All three components use `InferencePipeline`. Existing `pipeline/` directory can be retired.

---

### Phase 6 — Tests
**New test files**:
- `tests/test_inference_jobs.py` — unit test each of the 7 jobs in isolation
- `tests/test_training_jobs.py` — unit test each of the 7 training jobs
- `tests/test_pipeline_integration.py` — run full inference pipeline on synthetic 30-bar data, assert correct exits + buys
- `tests/test_adapter_parity.py` — run same context through notebook adapter and check output matches expected

**Retired**:
- `tests/test_policy_alignment.py` — parity is now structural, not tested by mocks

---

## What Gets Deleted / Simplified

| Before | After |
|--------|-------|
| Notebook sim cell: ~300 lines | ~10 lines (adapter loop) |
| LEAN `OnData()`: ~200 lines | ~20 lines (adapter + pipeline) |
| `pipeline/` high-level jobs | Replaced by `kernel/pipeline/` + `adapters/` |
| `test_policy_alignment.py` (222 tests by mock) | Job unit tests (structural parity) |
| Training cells: 3 × ~15 lines | 1 pipeline declaration |

---

## Progress

### Completed (commit `81012ad`, 2026-04-20)

| Phase | Status | Notes |
|-------|--------|-------|
| 1 — Kernel pipeline base | ✅ | `InferenceContext`, `Job`, `Pipeline` in `kernel/pipeline/` |
| 2 — 7 inference jobs | ✅ | All jobs in `kernel/pipeline/jobs/` — 32 unit tests passing |
| 3 — NotebookAdapter | ✅ | `adapters/notebook.py` + `InferencePipeline` convenience class |
| 4 — Training pipeline | ✅ | `training/pipeline/` with 7 jobs wrapping existing training/ modules |
| 5 — Wire consumers | ⏳ | **Next session** — see below |

### Phase 5 — What to do next session

**Step 5a: Validate NotebookAdapter produces identical output to current sim cell**

Before replacing anything, run both side-by-side on the same data and assert `equity_df` and `trade_log` match within tolerance:

```python
# In notebook or a test — run old sim cell, then run adapter, compare
old_equity = equity_df.copy()
old_trades = list(trade_log)

adapter  = NotebookAdapter(ohlcv, spy_daily_ret, results, corr_dict, CONFIG,
                            gmm_artifact, earnings_cal)
pipeline = InferencePipeline()
for today in bt_dates:
    ctx = adapter.make_context(today)
    pipeline.run(ctx)
    adapter.commit(ctx)

# Assert parity (allow small float diff)
pd.testing.assert_frame_equal(adapter.equity_df, old_equity, rtol=1e-4)
```

Add this as `tests/test_notebook_adapter_parity.py` before touching the live sim cell.

**Step 5b: Replace notebook simulation cell (657a4a6c)**

After parity confirmed, replace the ~300-line cell with:

```python
from adapters.notebook import NotebookAdapter, InferencePipeline

adapter  = NotebookAdapter(ohlcv, spy_daily_ret, results, corr_dict, CONFIG,
                            _gmm_artifact, _earnings_cal)
pipeline = InferencePipeline()

for today in spy_df.loc[BACKTEST_START:BACKTEST_END].index:
    ctx = adapter.make_context(today)
    pipeline.run(ctx)
    adapter.commit(ctx)

equity_df = adapter.equity_df
trade_log = adapter.trade_log
```

**Step 5c: Replace notebook training cells**

Replace cells 11 (features), 13 (tournament), 15 (export) with:

```python
from training.pipeline import TrainingPipeline, TrainingContext

ctx = TrainingContext(config=CONFIG, strategy_dir=STRATEGY_DIR, today=TODAY)
TrainingPipeline().run(ctx)

# Access results
results            = ctx.tournament_results
exported, skipped  = ctx.exported, ctx.skipped
corr_dict          = ctx.corr_dict
```

**Step 5d: Add LeanAdapter**

New file `adapters/lean.py` — wraps `QCAlgorithm` self into `InferenceContext`. Key mappings:
- `ohlcv`: `History(symbols, lookback, Resolution.Daily)` → normalize to dict of DataFrames
- `holdings`: `self.Portfolio` → build `{symbol: HoldingState}` from entry tracking dicts
- `action_fn` / `score_fn`: closures over loaded model artifacts (same as current LEAN code)
- After `pipeline.run(ctx)`: apply `ctx.exit_actions` via `self.Liquidate()` and `ctx.orders` via `self.SetHoldings()`

**Step 5e: Wire live runner**

`adapters/runner.py` — wraps broker + cached OHLCV into `InferenceContext`. Replaces `_run_once_multi_pipeline()` in `live/runner.py`.

### Key rule for Phase 5

**Run old and new in parallel first. Do not delete old code until parity test passes.**

## Resumption Notes

- **Branch**: `pipeline-unification` (off `main` at commit `96a5807`, current tip `81012ad`)
- **Phase 2 is complete** — all 7 jobs are implemented and unit-tested; `SellJob` and `SelectionJob` were the most complex
- **Do not touch `main.py` or the live sim cell until Step 5a parity test passes**
- The logic graph (`doc/logic_graph_103.md`) is the spec — every node maps to a job method
- `adapters/notebook.py` contains both `NotebookAdapter` and `InferencePipeline` for convenience
