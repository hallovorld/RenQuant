# Transformer-on-panel data requirements (research note)

**Date**: 2026-05-05
**Author**: research note for renquant_104 long-term roadmap
**Status**: pre-implementation; data preparation phase

---

## Why a Transformer

E27 walk-forward (2026-05-05) showed the current XGBoost rank:pairwise model has **regime-fragile alpha** — single-cut B2 Sharpe 0.68 collapses to mean Sharpe 0.21 ± 2.27 across 6-mo cuts, with all 3 cuts showing negative alpha vs SPY. The XGB model trained on 2.5y of 103 tickers (~77k panel rows) has hit a structural ceiling at our current data scale + label horizon.

A Transformer trained on the panel offers three potential improvements:

1. **Higher capacity** — millions of parameters can capture cross-asset and time-series interactions XGB can't (XGB sees one row at a time; Transformer sees an asset's history × peer asset cross-section).
2. **Richer label** — Transformer can target multi-horizon outcomes (5d / 20d / 60d returns simultaneously) via multi-head attention.
3. **Generalization** — with enough data, Transformer features generalize across regimes better than XGB tree splits which are inherently regime-conditional.

But all three depend on **having enough data**. The XGB ceiling we hit may simply be a 77k-rows ceiling, not a 100M-rows ceiling.

---

## Industry data scales for time-series Transformers

| Reference | Architecture | Train data | Notes |
|---|---|---|---|
| **TFT** (Lim et al. 2021, Google) | Temporal Fusion Transformer | 2-10M sequences | electricity, traffic, retail (60 features × 5y × 100k entities = ~2M rows) |
| **Informer** (Zhou et al. 2021, AAAI) | Sparse-attention | 1-5M time-points | univariate finance (S&P500, single ticker × 4000 days) — too small for our use |
| **Patch-TST** (Nie et al. 2023, ICLR) | Patch-based Transformer | 100k-1M patches | crucially uses 5-10y daily data per series |
| **Two Sigma** Halite/Lighthouse (industry, 2020-23) | proprietary | ~1B+ panel rows | 3000-5000 stocks × 20y daily × 50+ features |
| **AlphaResearch** Lopez de Prado (2020 book) | not strictly Transformer | 50k-500k rows minimum | for ANY ML model on financial panels |
| **Worldquant** Operator Library (industry) | Transformer + MLP heads | ~100M panel rows | 6000 stocks × 25y daily × multiple resolutions |

**Industry rule of thumb for financial Transformer:**

- **Minimum viable**: 1M panel rows (e.g. 500 tickers × 5y × 250 days × 1 row/day = 625k → bump to 1M)
- **Healthy**: 10M panel rows (e.g. 1000 tickers × 10y × 250d × ~4 features-per-day = 10M)
- **Production-grade**: 100M+ (e.g. 5000 tickers × 20y × 250d × ~4 = 100M)

**Hidden requirement: parameter-to-data ratio.** Transformers commonly have ≥1M parameters; rule of thumb is ≥10× more training rows than parameters → 10M+ rows for a small Transformer.

---

## Where renquant_104 currently sits

| | Value | Notes |
|---|---|---|
| Watchlist (model in production) | 103 | wl183 experiment failed (E26, E27) |
| Training years config | 2.5y | **THROTTLED — cache has 10y available** |
| Panel rows used in production training | ~77k | from wl103 × 2.5y |
| Label | binary "outperform-SPY 5d ahead" | XGB ceiling per E27 |
| Features | 21 | post macro/embedding ablations |

### OHLCV cache inventory (verified 2026-05-05 16:02)

```
total tickers in cache: 1006
   ≥10y history:        266 tickers (avg 2599 daily bars each)
   4-10y history:       571 tickers
   <4y history:          31 tickers (skip)

raw panel rows available (all tickers, full history): 1,436,499
```

**This is much better than I assumed before the inventory.** The current `training_years=2.5` setting is throttling us by ~4×. With minimal data prep (filter Tier-A 10y+, use full history) we can reach:

- Tier-A only (266 tickers × ~2599 bars) ≈ **691k panel rows**
- Tier-A+B (266 + 571 = 837 tickers × avg ~1500 bars) ≈ **1.2M panel rows**
- All cached (1006 × avg ~1430 bars) ≈ **1.4M panel rows**

**1.2M panel rows is at the "minimum viable" Transformer threshold.**

### Bottlenecks

1. ~~Years of history — 2.5y starves Transformer~~ **Solved by reading existing cache fully**
2. **Per-ticker quality**: of 1006, ~30% have <4y or low ADV — must filter
3. **Survivorship bias**: 1006 are CURRENTLY listed; need to add delisted tickers for honest backtest evaluation
4. **Sector skew**: cache is tech-biased (matches old wl103 expansions); GICS-balanced subset = harder to assemble

---

## Data prep plan (no bugs, no leaks)

### Phase 1 — Inventory + classify (1h, this session)

1. **Inventory existing OHLCV cache** — for each of 1006 tickers, record:
   - first/last date with valid close
   - number of trading days
   - average daily $-volume (for liquidity filtering)
   - sector classification (GICS via `sector_map`)

2. **Classify into 3 buckets:**
   - **Tier-A (production-quality)**: ≥10y history, avg ADV ≥ $50M
   - **Tier-B (training-quality)**: ≥5y history, avg ADV ≥ $10M
   - **Tier-C (skip)**: anything else

3. **Output:** `data/transformer_universe_inventory.json` — sorted list per tier

### Phase 2 — Data integrity audit (2h)

Per CLAUDE.md §5.6 ("definition of fixed = full 24h audit clean"), every datum that goes into a Transformer must pass:

1. **No-lookahead**: forward returns computed AFTER feature timestamp (use `shift(-N)` not `shift(+N)`)
2. **Survivorship-bias check**: include delisted tickers (verify yfinance/Alpaca cache contains them; backfill if missing)
3. **Stock-split / dividend adjustment**: cache uses adjusted-close (verify `close` column matches `adjclose` semantics)
4. **NaN propagation map**: any feature with NaN ratio > 5% gets imputed with cross-sectional median NOT zero (zero is a real value)
5. **Outlier clip**: returns > ±50% on a single bar = clip or flag (could be split mis-adjustment)
6. **Date-aligned across tickers**: all tickers share the same calendar (NYSE trading days); fill missing with NaN, NOT propagate

Output: `tests/test_transformer_data_integrity.py` — every check has a failing test BEFORE the audit and a passing test AFTER. Per CLAUDE.md §2 (every bug ships with regression test).

### Phase 3 — Multi-horizon label construction (1h)

Current label = `fwd_5d > threshold`. For Transformer:

- **fwd_5d_excess** = (close[t+5]/close[t]) / (spy[t+5]/spy[t]) − 1
- **fwd_20d_excess** = same with 20d horizon
- **fwd_60d_excess** = same with 60d horizon

Stack as 3-column target tensor for multi-head loss. Transformer learns shared representation, separate heads per horizon. Reference: TFT (Lim 2021) §III.B "Multi-Horizon Forecasting".

### Phase 4 — Feature scaling (30 min)

Transformer is more sensitive to scale than XGB. For each feature:
- z-score within ticker over rolling 252-day window (preserves cross-sectional ranking, kills baseline drift)
- clip to ±5σ
- replace NaN with cross-sectional median by date

Reference: Lopez de Prado (2018) "Advances in Financial Machine Learning" Ch. 3 "Labeling".

### Phase 5 — Train/val/test split (15 min)

**Critical**: walk-forward NOT random split. Per CLAUDE.md §5.2:
- Train: 2015-01-01 → 2022-12-31 (8 years)
- Val: 2023-01-01 → 2023-12-31 (1 year)
- Test: 2024-01-01 → 2025-12-31 (2 years)
- **Embargo**: 60 trading days between train/val and val/test (prevents leakage from labels at boundary)

Output: a single `panel_dataset.parquet` with columns: ticker, date, [features], [labels], split_label. Loaded by training script with `split_label == "train"` filter.

---

## Time + cost estimates

| Phase | Effort | Wall time | Risk |
|---|---|---|---|
| 1 — inventory | 1h | seconds (read parquet metadata) | low |
| 2 — integrity audit | 2h | ~15 min compute | medium (delisted tickers may need re-fetch) |
| 3 — multi-horizon labels | 1h | 1-2 min compute | low |
| 4 — scaling | 30 min | 5 min compute | low |
| 5 — split | 15 min | seconds | low |
| **Total** | **5h** | **~30 min** | — |

Plus actual model training: 30 min - 4 hours per Transformer (depending on size).

---

## Decision gates before training

1. **Data scale ≥ 1M rows** — count after Tier-A/B filtering; if <1M, expand universe via Russell 1000 fetch
2. **All integrity tests pass** — green light for training
3. **Multi-horizon labels stable** — fwd_60d in 2024 must finish AT LEAST 60d before "today" (no synthetic future)
4. **Walk-forward split is stable** — train/val gap = 60d embargo enforced

If any gate fails, document in `failed-experiments-log.md` and pause.

---

## Code organization (per CLAUDE.md §1c — every complex structure split into Tasks)

Phase 1-2 inventory + audit will live in:

```
backtesting/renquant_104/training_panel/transformer_data/
├── inventory.py              # InventoryUniverseTask
├── integrity_audit.py        # 6 checks, 1 task each
├── label_constructor.py      # MultiHorizonLabelTask
├── feature_scaler.py         # ZScoreNormalizeTask + ClipOutliersTask
└── split_builder.py          # WalkForwardSplitTask
```

Each Task ≤ 50 lines, single responsibility, dedicated test. Composed into `BuildTransformerDatasetJob` per CLAUDE.md §1c.

---

## What I will NOT do without explicit user approval

- Touch production models or artifacts
- Run training jobs that consume >1h CPU
- Auto-fetch new tickers (Alpaca rate limits)
- Modify `daily_104.sh` cron schedule

All Transformer work is research-track until walk-forward measurements justify promotion.
