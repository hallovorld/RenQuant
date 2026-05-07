# Calibration

**What it is:** maps the panel scorer's raw output (LightGBM/XGBoost LTR rank score, or μ−λσ from NGBoost, or sklearn-LinearRegression dot-product for alpha158_linear) to a calibrated probability `P(outperform SPY by threshold% in lookahead_days)`. Lives at `artifacts/panel-rank-calibration.json` (production) or `panel-rank-calibration.alpha158_linear.json` (alpha158_linear path) and is fitted by `scripts/recalibrate_scores.py` or `scripts/fit_alpha158_linear_calibrator.py`.

The calibrator is **isotonic regression** by default; falls back to Platt scaling for small samples (per CLAUDE.md sample-size policy).

> **2026-05-07 status**: production runs the XGB-trained calibrator
> (`panel-rank-calibration.json`). `n_unique_prob_y=7 < 10` runtime
> floor → SOFT-WARN; refit when panel-LTR best_iter floor is bumped
> (P1 in roadmap). The alpha158_linear-trained calibrator
> (`panel-rank-calibration.alpha158_linear.json`) is fitted but only
> used when the alpha158_linear path is active.

## 1. Architecture

```
panel-ltr.json  →  raw scores  ─→  panel-rank-calibration.json (isotonic)  ─→  rank_score ∈ [0,1]
                                                                                    ↓
                                                                           used for tier admission,
                                                                           rotation tiebreaks,
                                                                           score_distribution analytics
```

**Implementation:** `kernel/panel_pipeline/scoring.py::ApplyGlobalCalibrationTask`.
**Fitter:** `kernel.calibration.fit_panel_calibrator` (called from `recalibrate_scores.py`).

---

## 2. Saturation issue (discovered 2026-04-26)

During e2e R7 score_distribution review, the production XGBoost calibrator was discovered to have **near-zero resolution** — six of the top-seven candidates collapsed to identical `rank_score = 0.34474`:

| Calibrator | unique y | y range | pool_ic | scorer_oos_ic | ratio |
|---|---:|---|---:|---:|---:|
| panel-rank-calibration.xgboost.bak | **6** | 0.239 → 0.345 | 0.0011 | 0.0482 | 44× |
| panel-rank-calibration.lightgbm.bak | **1** | 0.274 → 0.274 | 0.0097 | 0.0291 | 3× |
| panel-rank-calibration.lgbm.bak (Apr 23, healthy) | 33 | 0.0 → 1.0 | 0.0291 | 0.0269 | ~1 |

`pool_ic` is **50× lower** than `scorer_oos_mean_ic` for both XGBoost and LightGBM. The panel scores lose ~98% of their predictive power between CPCV eval and the calibrator fit pool.

### Why the gap exists

1. **Pool window mismatch** — calibrator pool may include very-recent bars where the scorer hasn't generalised, while CPCV uses purged + embargoed splits.
2. **Z-score parity** — calibrator may receive different neutralisation/factor z-scores than the scorer's training input.
3. **Forward-return label drift** — calibrator may use a different forward threshold or lookahead than CPCV's spearman target.
4. **Class imbalance** — base rate 0.274 + ~225k pool rows divided across 101 tickers = poor per-bucket SNR for isotonic.

### Why this didn't break R6/R7 buys

Decisions use **`panel_score` (raw) and `μ` (NGBoost)** for `net_alpha` ranking — not `rank_score`. Calibrated `rank_score` only feeds:
- `score_distribution` analytics (saturated percentiles, diagnostic-only)
- Rotation `swap_margin` veto (when both held + cand collapse to top tier, can't distinguish)

So **R6/R7 buys were driven by μ + raw panel_score, with rank_score as noisy tiebreak.**

---

## 3. Round-7 fix shipped (2026-04-26)

**Change:** `fit_global_calibrator` collapse-guard floor bumped from `< 3` to `< 5` (commit 483a84b).

**Why 5:** the production XGBoost calibrator at discovery had `n_unique_prob_y = 6`. A floor of 3 would have let it through; a floor of 5 still permits 6 but flags any further degradation. User's spec was explicitly "≥5 unique y values".

**Behavior:** any future fit with `n_unique_prob_y < 5` raises `ValueError("collapsed to N unique y values (need ≥5)")`. Operator must investigate (likely scorer signal below noise floor) before re-running the fit.

**Tests** (`tests/test_global_calibrator.py::TestCalibratorPoolDiagnostics`): boundary case (4-unique rejected), healthy passes, metadata correctness. **Reference:** Niculescu-Mizil & Caruana (2005). *Predicting Good Probabilities with Supervised Learning*, ICML.

### Action items still open

1. Audit calibrator fit pool vs scorer eval pool — `recalibrate_scores.py` should consume the same `panel-ltr.json` test indices the scorer reported `oos_mean_ic` on.
2. Document the pool/eval split contract in `panel-rank-calibration.json` metadata.
3. Add a CI guard: `pool_ic < 0.5 × scorer_oos_ic` → reject fit.
4. Re-evaluate base_rate threshold — `0.03` forward return at 10 lookahead days produces 27% positive rate; if too tight, isotonic loses signal. Try `threshold = 0.5 × er_std × √lookahead_days` or quantile-based.

**Operator action:** re-run `python scripts/recalibrate_scores.py --strategy renquant_104` to refit. If raises, investigate scorer signal quality before retrying.

---

## 4. Score Distribution DB (planned, user spec round-5)

**User spec:** *"建立 calibrate 数据库，知道什么 score value 是 top 5%"*.

**Why:** today's decision tree uses **absolute** `panel_buy_floor=0.30`. Issues:
- Calibrator drift makes absolute thresholds shift over time
- "Top 5%" is intuitive; absolute thresholds are brittle
- No visibility into how thresholds map to percentiles

**Solution:** persistent score-distribution database. Decisions consult empirical percentiles (e.g., "top 15%") instead of fixed absolute scores.

### 4.1 Schema (extends `data/runs.db`)

```sql
-- Raw observations: every (date, ticker) cand
CREATE TABLE score_distribution (
    date          TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    raw_panel     REAL,        -- pre-calibration scorer
    rank_score    REAL,        -- post-calibration probability
    mu            REAL,        -- NGBoost μ (optional)
    sigma         REAL,        -- NGBoost σ (optional)
    regime        TEXT,
    PRIMARY KEY (date, ticker)
);

-- Aggregated daily percentiles (fast lookup)
CREATE TABLE score_percentiles_daily (
    date     TEXT PRIMARY KEY,
    n_cands  INTEGER NOT NULL,
    p01, p05, p10, p25, p50, p75, p85, p90, p95, p99  REAL,
    score_min, score_max  REAL,
    regime   TEXT
);

-- Calibrator drift tracking
CREATE TABLE score_distribution_meta (
    date              TEXT PRIMARY KEY,
    calibrator_pool_ic REAL,
    scorer_oos_ic     REAL,
    base_rate         REAL,
    threshold         REAL,
    n_features        INTEGER
);
```

### 4.2 Pipeline integration

**Phase 1 — Collect** (auxiliary, no decision change): new `RecordScoreDistributionTask` after `PanelScoringJob`:

```python
rows = [(today, c.ticker, c.panel_score, c.rank_score, c.mu, c.sigma, regime)
        for c in ctx.candidates]
# upsert score_distribution + aggregate percentiles
```

**Phase 2 — Decision integration** (config-flag-gated):

```jsonc
"rotation": {
  "panel_buy_floor": 0.30,            // existing absolute
  "panel_buy_top_n": 3,                // existing rank fallback
  "panel_buy_pctile": 0.85,            // NEW: top 15%
  "panel_buy_pctile_lookback_days": 5
}
```

`JointActionTask` admission becomes `abs_pass OR top_n_pass OR pctile_pass`.

**Phase 3 — Validation** (Sunday sweep): 27-mo OOS A/B across {absolute / top-N / percentile / union} — APY, Sharpe, max DD, n_trades.

### 4.3 Effort + edge cases

| Phase | Effort |
|---|---|
| 1 (collect + tests) | ~50 min |
| 2 (decision integration + tests) | ~35 min |
| 3 (sweep) | ~90 min compute |

**Edge cases:** cold-start (no history → fallback to absolute floor); sparse days (n_cands < 20 → use 5-day rolling window); regime transition (separate percentile per regime, schema column already in place).

**Data growth:** ~25k rows/year score_distribution, ~250 rows/year percentiles_daily — trivial in SQLite.

---

## 5. Cross-references

- **Implementation**: [`scoring.py::ApplyGlobalCalibrationTask`](../../backtesting/renquant_104/kernel/panel_pipeline/scoring.py)
- **Fitter**: [`recalibrate_scores.py`](../../scripts/recalibrate_scores.py) + [`kernel/calibration.py`](../../backtesting/renquant_104/kernel/calibration.py)
- **Schema**: `panel-rank-calibration.json` (probability + expected_return maps)
- **Companion docs**:
  - [`panel-ltr.md`](panel-ltr.md) — primer for the consuming model
  - [`databases.md`](databases.md) — full runs.db schema
  - [`trade-evaluation.md`](trade-evaluation.md) — trade outcomes table consumes calibrated `rank_score`
- **Experiment record**: [`experiments/post-tier1-followups.md`](../experiments/post-tier1-followups.md)
- **Reference**: Niculescu-Mizil & Caruana (2005). *Predicting Good Probabilities with Supervised Learning*, ICML.
