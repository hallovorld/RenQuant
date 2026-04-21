# Pipeline Design — renquant_103

Canonical design for both training and inference pipelines.
Every structural decision is recorded here first; implementation follows.

---

## Core principle: per-ticker parallelism

The expensive work in both pipelines is **per-ticker and independent** — feature
building, model scoring, tournament training.  Cross-ticker coordination (regime
detection, portfolio-level drawdown, candidate ranking, correlation) is cheap and
must stay sequential.  Therefore both pipelines use the same two-phase pattern:

```
Phase 1  Global sequential   — shared context, one result for all tickers
Phase 2  Per-ticker parallel — one TickerContext per ticker, ThreadPoolExecutor
Phase 3  Global aggregation  — collect per-ticker results back into shared context
```

No per-ticker job may write to the shared global context — results are collected
after all threads complete.

---

## Training Pipeline

```
TrainingContext (global, shared)
        │
        ▼
Phase 1 ──────────────────────────────────────────────────
  DataFetchJob    ohlcv dict for watchlist + ETFs + SPY
  RegimeFitJob    hurst_series, cusum_series, final_regime, gmm artifact

Phase 2 ──────────────────────────────────────────────────  ← ThreadPoolExecutor
  per ticker:  TickerTrainingContext(ticker, ohlcv, config, strategy_dir)
    FeatureJob       feature_frame  (indicators + labels)
    TournamentJob    result         (best model + OOS Sharpe)
    ExportJob        exported flag  (writes models/ dir)
    CalibrationJob   calibration    (score_calibration metadata)

  collect → ctx.feature_frames, ctx.results, ctx.exported, ctx.calibration_summary

Phase 3 ──────────────────────────────────────────────────
  CorrelationJob  120-day return correlation → artifact
```

### TickerTrainingContext

```python
@dataclass
class TickerTrainingContext:
    ticker: str
    ohlcv: dict           # shared read-only reference
    config: dict
    strategy_dir: Path | None

    # outputs
    feature_frame: pd.DataFrame | None = None
    result: dict | None = None
    exported: bool = False
    calibration: dict | None = None
```

### Per-ticker Job chain

Each per-ticker job receives a `TickerTrainingContext`, not the global
`TrainingContext`.  Jobs are run in sequence within each ticker's worker thread:

```
FeatureJob(tc) → TournamentJob(tc) → ExportJob(tc) → CalibrationJob(tc)
```

If `FeatureJob` fails (sparse data etc.), the chain short-circuits for that
ticker — no tournament or export attempted.

---

## Inference Pipeline

```
InferenceContext (global, shared)
        │
        ▼
Phase 1 ──────────────────────────────────────────────────
  RegimeJob    detect regime → ctx.regime, ctx.confidence
  DrawdownJob  circuit breaker → ctx.hwm, ctx.skip_buys
  BuyGatesJob  market gates → ctx.buy_blocked, ctx.bear_only

Phase 2a ─────────────────────────────────────────────────  ← ThreadPoolExecutor
  per held ticker:  TickerInferenceContext(ticker, model, holding, prices, ohlcv, ...)
    TickerSellJob    exit_signal (ExitSignal | None)

  collect → ctx.exits, update ctx.holdings (streak + HWM)

Phase 2b ─────────────────────────────────────────────────  ← ThreadPoolExecutor
  (skipped if ctx.buy_blocked and not ctx.bear_only)
  per candidate ticker:  TickerInferenceContext(ticker, model, ohlcv, ...)
    TickerCandidateJob   candidate (CandidateResult | None)

  collect → ctx.candidates

Phase 3 ──────────────────────────────────────────────────
  RankingJob    blended sort → ctx.ranked
  SelectionJob  tiered selection → ctx.orders
```

### TickerInferenceContext

```python
@dataclass
class TickerInferenceContext:
    # inputs (read-only, set by orchestrator)
    ticker: str
    ohlcv: dict           # shared reference — all tickers' data
    model: dict           # model artifact for this ticker
    config: dict
    today: date
    regime: str
    regime_params: dict
    exit_params: dict     # pre-built from regime_params + config

    # sell-job inputs (None for candidate jobs)
    holding: HoldingState | None = None
    price: float = 0.0

    # outputs (written by job)
    exit_signal: ExitSignal | None = None   # SellJob output
    candidate: CandidateResult | None = None # CandidateJob output
```

### Orchestration in InferencePipeline.run()

```python
# Phase 1
RegimeJob().run(ctx)
DrawdownJob().run(ctx)
BuyGatesJob().run(ctx)

# Phase 2a — parallel sell evaluation
held_tctxs = [_make_sell_tctx(ctx, ticker) for ticker in ctx.holdings]
run_parallel(held_tctxs, TickerSellJob())
for tc in held_tctxs:
    ctx.holdings[tc.ticker] = tc.holding   # updated streak + HWM
    if tc.exit_signal and tc.exit_signal.should_exit:
        ctx.exits.append((tc.ticker, tc.exit_signal))

# Phase 2b — parallel candidate scoring
if not (ctx.buy_blocked and not ctx.bear_only):
    universe = _buy_universe(ctx)          # defensives if BEAR, else all models
    cand_tctxs = [_make_cand_tctx(ctx, t) for t in universe]
    run_parallel(cand_tctxs, TickerCandidateJob())
    ctx.candidates = [tc.candidate for tc in cand_tctxs if tc.candidate]

# Phase 3
RankingJob().run(ctx)
SelectionJob().run(ctx)
```

`run_parallel(tctxs, job, max_workers=8)` uses `ThreadPoolExecutor` with
`as_completed` for fault isolation — one ticker failure doesn't abort the batch.

---

## SellOnlyPipeline

Three-job subset for intraday exit-only runs (no buy phase):

```
RegimeJob → DrawdownJob → [parallel TickerSellJob per held ticker]
```

---

## File map

| File | Contents |
|------|----------|
| `kernel/pipeline/context.py` | `InferenceContext`, `TickerInferenceContext` |
| `kernel/pipeline/pipeline.py` | `Job`, `TickerJob`, `InferencePipeline`, `SellOnlyPipeline`, `run_parallel` |
| `kernel/pipeline/jobs/regime.py` | `RegimeJob` |
| `kernel/pipeline/jobs/drawdown.py` | `DrawdownJob` |
| `kernel/pipeline/jobs/gates.py` | `BuyGatesJob` |
| `kernel/pipeline/jobs/sell.py` | `TickerSellJob` (per-ticker) |
| `kernel/pipeline/jobs/candidates.py` | `TickerCandidateJob` (per-ticker) |
| `kernel/pipeline/jobs/ranking.py` | `RankingJob` |
| `kernel/pipeline/jobs/selection.py` | `SelectionJob` |
| `adapters/lean.py` | `LeanAdapter` |
| `adapters/runner.py` | `RunnerAdapter` |
| `training/pipeline.py` | `TrainingContext`, `TickerTrainingContext`, `TrainingPipeline`, all jobs |

---

## Isolation rules (unchanged)

- `kernel/` — stdlib + numpy + pandas only; no `common/`, no sklearn, no broker libs.
  Must run inside LEAN Docker.
- `training/` — can import `kernel/` and sklearn; not used in Docker.
- `adapters/` — can import `kernel/` and broker libs; not used in Docker.
