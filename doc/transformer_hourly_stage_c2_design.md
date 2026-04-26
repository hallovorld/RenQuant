# Transformer Hourly — Stage C-2 Wiring Design (2026-04-26)

**Stage C-1** (commit `d4ed1a4`): SCAFFOLD shipped — `kernel/intraday_wash.py`,
`training_panel/hourly_resolution_panel.py`. Pure functions + 35 unit tests.

**Stage C-2** (this doc): wire the hourly panel into `PanelTrainingPipeline`
so the transformer backend trains at hourly resolution when a config flag
flips. ~200-300 LOC change. Default OFF preserves daily-resolution training.

---

## Why we need this

Today's transformer Sunday sweep result: **CPCV mean OOS IC = -0.0029**
(vs XGBoost +0.0482). The transformer overfits because the panel has
**~2500 dates** — well below the Chen-Pelger-Zhu 2024 ship gate of
~5000 dates for transformer generalization on cross-sectional asset
pricing.

Hourly resolution: each ticker contributes ~6 rows per session × ~2500
sessions = **~15,000 rows per ticker**. Cross-section panel size grows
from ~225k to ~1.5M rows. Transformer should stop overfitting and
start generalizing.

## Dependencies (already shipped)

- ✅ `kernel.intraday_wash.wash_bars()` — outlier winsorize, sample
  weighting, hour-of-day cyclic encoding (Stage A + B)
- ✅ `training_panel.hourly_resolution_panel.build_hourly_resolution_panel()`
  — produces (ticker, datetime) long panel with hourly features +
  forward-return label (Stage C-1 scaffold)
- ✅ `kernel.intraday.HourlyBarStore` — parquet cache at
  `data/intraday/{SYM}/1h.parquet` (already populated for 83/101 tickers)

## Stage C-2 — concrete changes

### 1. Config flag

Add to `strategy_config.json` (default OFF):

```json
"panel_ltr": {
    "training_resolution": "daily",          // "daily" | "hourly"
    "hourly": {
        "label_horizon_bars": 7,             // ≈ 1 trading session forward
        "min_history_bars": 200,             // warmup
        "use_minute_bars": false             // future: 10-min resolution
    },
    ...
}
```

When `training_resolution == "daily"` (default), pipeline behaviour is
bit-for-bit unchanged. When `"hourly"`, dispatches to the new path.

### 2. New Task: `BuildHourlyPanelTask`

Sits in `pp_panel_training.py` parallel to `BuildPanelFeatureFramesTask`:

```python
class BuildHourlyPanelTask(PanelTask):
    """Build panel rows at hourly resolution (replaces daily aggregation)."""

    def run(self, ctx: PanelTrainingContext) -> bool | None:
        cfg = ctx.config.get("panel_ltr", {})
        if cfg.get("training_resolution", "daily") != "hourly":
            return True   # daily path active; this task is a no-op

        from kernel.intraday import HourlyBarStore
        from training_panel.hourly_resolution_panel import (
            build_hourly_resolution_panel,
        )

        store = HourlyBarStore(data_dir=...)
        bars_per_ticker = {t: store.load(t) for t in ctx.watchlist}
        bars_per_ticker = {t: b for t, b in bars_per_ticker.items() if b is not None}

        if not bars_per_ticker:
            log.warning("BuildHourlyPanelTask: no hourly bars — abort")
            return False

        benchmark_bars = bars_per_ticker.pop(ctx.benchmark, None)
        ctx.hourly_panel = build_hourly_resolution_panel(
            bars_per_ticker,
            label_horizon_bars=int(cfg.get("hourly", {}).get(
                "label_horizon_bars", 7)),
            benchmark_bars=benchmark_bars,
            apply_wash=True,
        )
        log.info("BuildHourlyPanelTask: panel rows = %d", len(ctx.hourly_panel))
        return True
```

### 3. Pipeline dispatch in `PanelTrainingPipeline.tasks`

```python
@property
def tasks(self):
    res = self.config.get("panel_ltr", {}).get("training_resolution", "daily")
    if res == "hourly":
        return [
            FetchOHLCVTask(),
            FetchHourlyBarsTask(),     # new — fetch + cache hourly bars
            BuildHourlyPanelTask(),     # new — produces ctx.hourly_panel
            CrossValidateTask(),         # uses ctx.hourly_panel directly
            FinalFitTask(),
            SaveArtifactTask(),
        ]
    # else: daily path (existing)
    return [...existing daily tasks...]
```

### 4. Adapt CrossValidateTask + FinalFitTask

Both currently read `ctx.panel`. Need to honor `ctx.hourly_panel` when
present.

```python
def _get_active_panel(ctx):
    return getattr(ctx, "hourly_panel", None) or ctx.panel
```

This isolates the change to one helper function. Each downstream
consumer calls `_get_active_panel(ctx)` instead of `ctx.panel`.

### 5. Adjust `group_sizes` for transformer

The transformer's date-group attention requires per-group sample counts.

- Daily: 1 group = 1 date × N tickers → group_size ≈ N
- Hourly: 1 group = 1 (date, hour) × N tickers → group_size ≈ N (still)
  but #groups grows ~6×

Code change: `group_sizes` calculation moves from `groupby('date')` to
`groupby(['date', 'hour'])`.

```python
# In _build_date_groups in transformer_model.py:
if 'hour' in panel.columns:
    group_key = panel[['date', 'hour']].apply(tuple, axis=1)
else:
    group_key = panel['date']
group_sizes = group_key.value_counts(sort=False).reindex(group_key.unique())
```

### 6. Label horizon adjustment

Daily path: `lookahead_days=10`, `forward_excess_return = (close[t+10]/close[t]-1) - benchmark[t+10]/benchmark[t]+1`

Hourly path: `label_horizon_bars=7` (1 session ≈ 7 hours). For longer
horizon transformer signal, use 35 (~5 sessions) or even 70 (~2 weeks).
Configurable.

### 7. Tests

```
tests/test_hourly_training_pipeline.py:
  - test_daily_path_unchanged_when_flag_off
  - test_hourly_path_produces_panel
  - test_hourly_panel_growth_factor (≥5×)
  - test_label_horizon_respected
  - test_group_sizes_reflect_hour_axis
  - test_cv_works_on_hourly_panel
  - test_save_load_artifact_round_trip
```

8 tests, ~250 LOC test code.

---

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| Hourly bars only cover 83/101 tickers — biased panel | Drop tickers with insufficient hourly history at start of pipeline; log skipped count |
| Memory: 1.5M rows × 12 features = ~150 MB float32; OK | Test on full universe before committing |
| Transformer might still underperform XGBoost | Sim validation gate same as for QP — needs to clear +2pt APY win on 27-mo OOS |
| Hourly fundamentals don't exist (only daily) | Forward-fill the daily fundamentals onto hourly grid (already-shipped pattern in `LoadFundamentalsTask`) |
| Time-of-day overfit (transformer learns "always sell at 15:30") | Hour-of-day cyclic encoding (Stage B) lets the model learn hour-specific patterns explicitly without categorical overfit |

## Roll-out

| Step | Scope | Validation |
|---|---|---|
| C-2.0 | Land code with `training_resolution: "daily"` default | All 35 hourly + 250 regression tests green |
| C-2.1 | Sunday sweep with `transformer + training_resolution: hourly` | Compare CPCV OOS IC vs daily-resolution transformer baseline |
| C-2.2 | If hourly transformer IC ≥ 0.10 (production-grade), promote | Update `golden_config_2026-04-23.md` with v4.3 entry |
| C-2.3 | If hourly IC > XGBoost daily IC, switch production backend | Sunday sweep re-tunes |

## What this DOES NOT include

- 10-min resolution path (would be Stage D, future)
- Hourly-resolution inference (LEAN/live still uses daily decisions
  even if training is hourly — orthogonal change)
- Cross-asset attention (transformer architecture change, not panel
  size change)

## Estimated work

- BuildHourlyPanelTask + FetchHourlyBarsTask: 80 LOC
- Pipeline dispatch + _get_active_panel helper: 40 LOC
- group_sizes adjustment: 20 LOC
- Tests: 250 LOC
- Doc updates (CLAUDE.md, ops runbook): 50 LOC
- **Total**: ~440 LOC + ~250 LOC tests, **~1.5 hours of focused work**
