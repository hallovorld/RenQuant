# Calibration Score Database — Plan + Design

**User spec (2026-04-26 round-5)**: 建立 calibrate 数据库，知道什么 score
value 是 top 5%。

## Problem statement

Today's decision tree uses **absolute panel_buy_floor=0.30** (just lowered
from 0.45). Issues:
- Calibrator drift: `pool_ic=0.001` means calibrated probabilities
  compress around base_rate 0.27. Threshold semantics shift over time.
- "Top 5%" is intuitive ("the strongest 5 of 100 cands"), but absolute
  threshold is brittle.
- No visibility into how thresholds map to rank percentiles.

**Solution**: persistent score distribution database. Decisions consult
empirical percentiles (e.g., "top 15%") instead of fixed absolute scores.

## Design

### Schema (extends `data/runs.db` SQLite)

#### Table 1: `score_distribution` (raw observations)

```sql
CREATE TABLE score_distribution (
    date          TEXT NOT NULL,            -- YYYY-MM-DD
    ticker        TEXT NOT NULL,
    raw_panel     REAL,                     -- pre-calibration scorer output
    rank_score    REAL,                     -- calibrated probability (post-cal)
    mu            REAL,                     -- NGBoost μ (optional)
    sigma         REAL,                     -- NGBoost σ (optional)
    regime        TEXT,                     -- BULL_CALM / etc.
    PRIMARY KEY (date, ticker)
);
CREATE INDEX idx_score_dist_date ON score_distribution(date);
```

#### Table 2: `score_percentiles_daily` (aggregated, fast lookup)

```sql
CREATE TABLE score_percentiles_daily (
    date          TEXT PRIMARY KEY,         -- YYYY-MM-DD
    n_cands       INTEGER NOT NULL,
    p01           REAL,
    p05           REAL,
    p10           REAL,
    p25           REAL,
    p50           REAL,
    p75           REAL,
    p85           REAL,                     -- "top 15%" threshold
    p90           REAL,                     -- "top 10%"
    p95           REAL,                     -- "top 5%"
    p99           REAL,
    score_min     REAL,
    score_max     REAL,
    regime        TEXT
);
CREATE INDEX idx_pctiles_date ON score_percentiles_daily(date);
```

#### Table 3: `score_distribution_meta` (calibrator drift tracking)

```sql
CREATE TABLE score_distribution_meta (
    date              TEXT PRIMARY KEY,
    calibrator_pool_ic REAL,                -- from artifact metadata
    scorer_oos_ic     REAL,                 -- ditto
    base_rate         REAL,
    threshold         REAL,
    n_features        INTEGER
);
```

### Pipeline integration

#### Phase 1: Collect (this session, ~30 min)

New Task `RecordScoreDistributionTask` runs **after PanelScoringJob**
(wherever rank_score is populated — both for cands and holdings):

```python
class RecordScoreDistributionTask(Task):
    """Persist today's score distribution + daily percentiles."""

    def run(self, ctx):
        if not ctx.candidates:
            return False
        rows = [
            (ctx.today.isoformat(), c.ticker,
             getattr(c, "panel_score", None),
             getattr(c, "rank_score", None),
             getattr(c, "mu", None),
             getattr(c, "sigma", None),
             ctx.regime)
            for c in ctx.candidates
        ]
        # INSERT OR REPLACE into score_distribution
        # then aggregate percentiles into score_percentiles_daily
        ...
```

Skip if `panel_buy_use_pctile == False` to keep collection opt-in.

#### Phase 2: Decision integration (next session)

New config:
```jsonc
"rotation": {
  "panel_buy_floor": 0.30,            // existing absolute
  "panel_buy_top_n": 3,                // existing rank fallback
  "panel_buy_pctile": 0.85,            // NEW: top 15% (lookback 5d default)
  "panel_buy_pctile_lookback_days": 5  // NEW: percentile rolling window
}
```

JointActionTask buy gate becomes:
```python
abs_pass    = score >= panel_buy_floor
top_n_pass  = rank_in_top_n AND score >= rank_floor
pctile_pass = score >= percentile_threshold(today, pctile=0.85, lookback=5d)
admit if any(abs_pass, top_n_pass, pctile_pass)
```

#### Phase 3: Validation (Sunday sweep)

A/B in Sunday training:
- Variant 1: absolute-only (current 0.30)
- Variant 2: rank-based-only (top-N from B fix)
- Variant 3: percentile-only (top 15%)
- Variant 4: union (A + B + percentile)

Compare on 27-mo OOS: APY, Sharpe, max DD, n_trades.

## Implementation effort

| Phase | Task | Time |
|---|---|---|
| 1 | Add 3 tables to runs.db schema (`kernel/persistence.py`) | 15 min |
| 1 | RecordScoreDistributionTask + wire into pipeline | 20 min |
| 1 | Tests (insert/aggregate/percentile lookup) | 15 min |
| 2 | percentile_threshold helper + JointAction integration | 20 min |
| 2 | Tests for percentile-based admission | 15 min |
| 3 | Backtest sweep variants 1-4 | 90 min (compute) |

**Phase 1 alone delivers visibility (auxiliary info, no decision change)**.
Phase 2+3 graduate to decision-impacting.

## Data growth

Daily cands ~50-100 tickers × ~250 trading days/year × 2 stores
→ ~25k rows/year score_distribution + 250 rows/year score_percentiles_daily.
Trivial SQLite size.

## Edge cases

- **Cold start (first day)**: percentile lookup returns NaN → falls back
  to absolute panel_buy_floor.
- **Sparse days**: n_cands < 20 → percentile is noisy → fall back to
  rolling 5-day union.
- **Regime transition**: separate percentiles per regime (already in
  schema column) so BEAR-day percentiles don't pollute BULL_CALM days.

## Tradeoffs

| Pro | Con |
|---|---|
| Adapts to calibrator drift | Cold-start needs ≥1 day of history |
| Intuitive ("top 5%") | Sensitive to N (≤20 cands → noisy) |
| Detects model degradation | Adds DB writes per bar |
| Audit-friendly history | Schema migration needed |

## Recommendation

**Phase 1 NOW** (auxiliary collection) + **Phase 2 next week** (after
verifying data integrity). Phase 3 is Sunday sweep.

If user agrees: I'll wire Phase 1 immediately (~50 min implementation).
