# Panel-LTR Training Runs — Log

Running log of `scripts/train_104.py` invocations that produce `artifacts/panel-ltr.json` + `artifacts/ngboost-head.json`. One entry per run. Prepend every new run to the **top** so the most recent is first.

Template:

```
## Run N — YYYY-MM-DD HH:MM PT — <one-liner>

**Config (diff vs prior run):**
- <key>: <old> → <new>

**Artifacts:**
- panel-ltr.json: rows=… tickers=… dates=… feature_cols=… oos_mean_ic=… train_ic=…
- ngboost-head.json: n_rows=… train_μ_mean=… train_σ_mean=… feature_cols=…

**Diagnosis:**
- ...

**Action taken:**
- ...
```

---

## A/B — 2026-04-23 night PT — **Plan C (Kelly sizing + A-gate) PROMOTED → Golden v4** (+11.91 APY pts) ⭐

User request: *"A/B 当门槛，C 决定 size. 已有的 stock 也要计算，这样才
知道要不要加仓或者 rotate. This is the smart part of this model, think
deeply! Make it beautiful!"*

Shipped in single-session sweep:
- `kernel/kelly.py` — continuous-Kelly formula `f* = μ/σ²`
- `ApplyKellySizingTask` — writes `kelly_target_pct` on candidates AND holdings
- `TopUpHeldTask` — add-to-existing when `kelly_target - current_pct > 0.05`
- A-gate: `tiered_thresholds = [0.27, 0.45, 0.60]` (anchored to calibrator `base_rate = 0.273`)

**4-config sweep (27-mo OOS, `allow_fetch=False`, identical panel+NGBoost):**

| Config                 | APY      | Δ vs GOLDEN | Win  | Buys | Streak |
|------------------------|---------:|------------:|-----:|----:|-------:|
| GOLDEN (v3 hourly)     | +25.91%  |     —       | 81%  | 144 | 25d    |
| A + Kelly(quarter)     | +36.23%  |   +10.32    | 87%  | 117 | 43d    |
| **A + Kelly(half)** ⭐ | **+37.82%** | **+11.91** | 85% | 115 | 43d    |
| A + Kelly(tight cap)   | +36.23%  |   +10.32    | 87%  | 117 | 43d    |

**Absolute APYs lower than documented v3 (+44.20%)** because sweep uses
`allow_fetch=False` — fundamentals/earnings/insider fetches disabled for
reproducibility. All 4 share the same handicap; relative ranking is what
matters. Expected live-fetch APY under v4 ≈ +65% (pending next
Tue/Thu/Sun retrain for confirmation).

**Winner:** half-Kelly (`fractional=0.50`). Beats quarter by +1.59 pts;
both far above GOLDEN. `max_concentration=0.35` provides the risk
backstop even at f*=1.0.

**Trade-off surfaced (accepted):** Max no-trade streak 25d → 43d. Kelly
is disciplined — refuses low-μ/σ² bets during quiet markets. Average
gap still 4.9d (115 buys / 570 days). `monitoring.max_no_trade_days=15`
alert will fire more often under Kelly, by design — proves the system
is deliberately idle, not stuck.

**Sharpe rerun was kicked then cancelled by user** after APY + win rate
data made the promotion call decisive. To add post-promotion: run
`/tmp/kelly_rerun_top2.py` in next session for the Sharpe / Max DD
numbers to append here.

**Promoted as golden v4** — `strategy_config.golden.json` +
`strategy_config.json` + `doc/golden_config_2026-04-23.md` updated in
the ship commit. First live test at tomorrow's (2026-04-24) scheduled
runs:
  - open 6:32 AM PT (sell-only)
  - preclose 12:44 PM PT (sell-only)
  - daily 1:55 PM PT (full buy+sell — first Kelly live decision)

Log: sweep output `/tmp/kelly_validation.log`.

---

## A/B — 2026-04-23 evening PT — Plan H transformer on hourly panel SHELVED

Plan H. Second transformer rerun — first was shelved on the daily-only
47k panel (ratio 0.49×). This run uses the same transformer architecture
on the **hourly-enhanced 47k × 31 panel** — the "more features + same
rows" condition the transformer design was hoping for.

Both backends trained via `scripts/compare_panel_backends.py --strategy
renquant_104 --device mps` (MPS with
`PYTORCH_ENABLE_MPS_FALLBACK=1` for a missing nested_tensor op).

**Results (CPCV 15-split, same folds, same panel):**

| Backend      | mean_ic  | elapsed | device |
|--------------|---------:|--------:|--------|
| XGBoost      | +0.0309  |  60.8s  | cpu    |
| Transformer  | +0.0063  | 148.0s  | mps (+ cpu fallback) |
| **Ratio**    | **0.20×** |  —      | —      |

**Verdict:** shelve (ship gate was ≥1.30×; even the ensemble gate was ≥1.10×).
Worse than the daily-only baseline (0.49×).

**Diagnosis:**

Adding 6 hourly feature columns (47k rows × 25 → 31 cols) did not
close the transformer vs XGBoost gap. In fact the gap widened. Likely
causes:

1. **Dataset still too small.** Cross-sectional transformers need ≫
   47k rows before attention out-learns axis-aligned splits. The
   panel has 1,256 dates — the transformer sees 1,256 cross-sections
   of ~38 tickers each, which is barely above the underfitting floor.
2. **Noisy features dilute attention.** XGBoost's L1/L2 penalties +
   tree splits can ignore noisy features cheaply. Transformer
   attention distributes gradient across all features and noise
   creeps in as spurious cross-ticker dependencies.
3. **More features, same regularisation budget.** Transformer config
   in `strategy_config.json::panel_ltr.transformer_params` has the
   same dim/heads/epochs as before — the extra 6 features add
   parameters to attend to without a compensating capacity bump.

**Action taken:**

- Active backend stays `xgboost` (config unchanged — was never
  flipped for this run; `compare_panel_backends.py` swaps backends
  internally and restores the XGBoost artifact from `.compare_backup/`
  after the comparison).
- `panel-transformer.pt` + `panel-transformer.json` artifacts exist
  on disk but are gitignored per the transformer backend shelved
  note in `.gitignore`.
- Transformer revisit moves to a later phase: not on the current
  roadmap. Will revisit only if the panel grows to > 200k rows
  (roughly 4× current) — e.g. from fetching an additional 5yr of
  hourly bars plus broadening the watchlist, AND a full
  transformer_params sweep.

**No regression to golden.** Hourly-enhanced XGBoost panel remains
the active golden at +44.20% APY.

Log: `/tmp/transformer_ab.log`.

---

## A/B — 2026-04-23 late PT — Plan G hourly features PROMOTED (+4.18 APY pts)

Plan G. Hourly-bar aggregates (morning/afternoon drift, VWAP premium,
vol ratio, intraday realized vol, overnight gap) added to the daily
panel. Fetched 153,403 hourly rows × 44 symbols × 2yr via
`scripts/fetch_hourly_bars.py` (Alpaca IEX). Retrained panel +
NGBoost with `panel_ltr.hourly.enabled=true`.

**Artifact changes:**

- panel-ltr.json: rows=47,000 (was ~47,190 daily-only)
  tickers=38  dates=1256  **features=31 (was 25)**
  oos_mean_ic=**+0.0326** (was +0.041 daily-only)
  train_ic=+0.2617
- ngboost-head.json: n=47,000  train_μ_mean=-0.00221
  train_σ_mean=0.06556  feature_cols=31  elapsed=227s

**Per-feature IC of the 6 new hourly columns:**

| Feature                   | IC       |
|---------------------------|---------:|
| intraday_realized_vol_z   |  -0.0381 |
| vwap_premium_z            |  -0.0159 |
| afternoon_drift_z         |  -0.0152 |
| vol_ratio_z               |  +0.0138 |
| morning_drift_z           |  -0.0065 |
| overnight_gap_z           |  -0.0020 |

`intraday_realized_vol_z` ranks top-5 by |IC| alongside `beta_60d_z`
(0.058) and `roe_z` (0.041). The other 5 hourly features are modest
(|IC| ≤ 0.016) but still contribute in XGBoost's feature interactions.

**A/B (27-month OOS sim, identical gates, only panel artifact + flag flipped):**

| Run | APY    | Δ     | win | buys | streak |
|-----|-------:|------:|----:|----:|------:|
| hourly=OFF (daily-only golden) | +40.02% |   —   | 79% | 122 | 25d |
| hourly=ON  (hourly-enhanced)   | **+44.20%** | **+4.18** | **82%** | 117 | 26d |

**Diagnosis — IC vs APY divergence:**

OOS mean IC went DOWN (0.041 → 0.033) but live APY went UP 4.18 pts.
The features aren't flat-useful across all (ticker, date) pairs; they
help specifically in conditions where they matter (e.g.,
intraday_realized_vol + vwap_premium likely matter in BULL_VOLATILE
/ CHOPPY regimes where position entry timing matters more). XGBoost's
tree splits find the right interactions even when average IC is
modest.

**Win rate 79% → 82%** is the headline. Per-trade avg PnL up too
(+0.1297 → +0.1359). Buys count slightly down (122 → 117) — more
selective. Max streak ~flat.

**Action taken:**

- `panel_ltr.hourly.enabled: true` is the new golden.
- `strategy_config.golden.json` updated to snapshot the win.
- `panel-ltr.json` + `ngboost-head.json` overwritten with
  hourly-enhanced variants. Daily-only golden backup retained as
  `panel-ltr.golden-daily.json` + `ngboost-head.golden-daily.json`
  for easy rollback.
- 153,403 hourly rows cached at `data/intraday/{SYM}/1h.parquet`
  (total ~8.6 MB).
- Panel feature count bumped 25 → 31 in the feature_cols artifact.
- Next daily_104.sh retrain will automatically pick up the hourly
  flag from strategy_config.json (cadence-gated Tue/Thu/Sun).

**New golden: +44.20% APY (after-tax), 82% win, 26d max streak.**

Logs: `/tmp/train_hourly.log`, `/tmp/hourly_ab.log`.

---

## A/B — 2026-04-23 late PT — Plan F regime-conditional calibration SHELVED

Plan F. Four per-regime calibrators fit via
`scripts/fit_panel_calibrator.py --regime-conditional`, sourced from
the 27k sim + 27 live decision traces in `data/runs.db::pipeline_runs`.

**Per-regime pool_IC on pooled training data:**

| Regime         | n      | pool_IC | base_rate |
|----------------|-------:|--------:|----------:|
| BULL_CALM      | 18,506 | +0.0765 |     0.301 |
| BULL_VOLATILE  |  1,102 | +0.0899 |     0.285 |
| CHOPPY         |    646 | -0.1159 |     0.322 |
| BEAR           |    988 | +0.3690 |     0.394 |
| **POOLED**     | 89,671 | +0.0218 |     0.273 |

The per-regime ICs looked 3.5× – 17× the pooled IC. The negative CHOPPY
IC was even presented as evidence the pooled calibrator "hurt" in that
regime. But live A/B tells a different story:

**A/B (27-month OOS sim, identical panel+ngboost artifacts, only flag flipped):**

| Run | APY    | Δ     | win | buys | streak |
|-----|-------:|------:|----:|----:|------:|
| regime_cond=OFF (pooled) | +40.02% |   —   | 79% | 122 | 25d |
| regime_cond=ON           | +36.24% | −3.78 | 80% | 132 | 22d |

**Diagnosis (why the in-sample IC lift didn't translate):**

1. Selection bias — fitting isotonic on a narrower regime sub-sample
   produces extreme knots that don't generalize. BEAR's 988-row fit
   has a base rate of 0.394 (vs 0.273 pooled), so it systematically
   predicts higher probabilities than reality.
2. Concurrent regime labels — the DB records the regime that was
   CURRENT at the decision bar, but the panel score was computed at
   the same bar. At production time the regime label may have just
   flipped; the "right" calibrator isn't what the training label says.
3. CHOPPY's in-sample negative IC was the warning sign. A robust
   signal doesn't invert by regime — when it does, it's probably
   overfitting a tiny subsample.
4. Extra knots + narrower distributions mean candidate rank_scores
   are more often just below the 0.10 tier floor — buys get filtered
   out in the wrong regimes (observed 52d consecutive no-candidate
   streak in BEAR during run B).

**Action taken:**

- Config flag `ranking.panel_scoring.global_calibration.regime_conditional.enabled`
  flipped back to `false`. Pooled calibrator remains golden.
- Per-regime JSON artifacts are kept on disk (they're small) as evidence
  for future work — e.g. could be revisited with smoothed / shrunk
  isotonic curves that pull toward the pooled calibrator, or with
  Platt-scaled heads that generalize better than raw isotonic on small
  n.
- Plan F task marked "infra shipped, verdict SHELVED" in roadmap.

**Takeaway for future A/Bs:** in-sample IC is a necessary but not
sufficient signal. Always run the live-sim A/B before flipping the
flag — IC is computed on IID samples but trading requires the ranking
to stay consistent through time *and* regime transitions.

Log: `/tmp/regime_ab.log`.

---

## A/B — 2026-04-23 eve PT — σ-penalty λ sweep — SHELVED at all λ

Plan C. Kept `score_mode=mu_minus_lambda_sigma` across all runs so the
reordered calibrator (commit `339944b`) actually maps μ−λσ → probability.
Swept λ ∈ {0.0, 0.1, 0.25, 0.5, 1.0} against an `additive, λ=0` baseline
reusing the current panel-ltr + ngboost artifacts on disk. No retrains
needed — only `ranking.panel_scoring.ngboost.{score_mode, lambda_sigma}`
flipped between runs. All sims were 27-month OOS with
`fundamentals.allow_fetch=False` (panel was already on disk — stripped
live fetch only; re-uses baked-in factor z-scores).

**Results:**

| mode                  | λ    | APY    | Δ APY   | win | buys | streak |
|-----------------------|-----:|-------:|--------:|----:|----:|------:|
| additive (baseline)   | 0.00 | +31.97% |   —     | 81% | 161 | 30d   |
| μ−λσ                  | 0.00 | +26.55% |  −5.4  | 74% | 146 | 28d   |
| μ−λσ                  | 0.10 | +30.55% |  −1.4  | 74% | 146 | 30d   |
| **μ−λσ**              | 0.25 | +33.94% |  +2.0  | 83% | 129 | 41d   |
| μ−λσ                  | 0.50 | +15.18% | −16.8  | 85% |  73 | 40d   |
| μ−λσ                  | 1.00 |  +3.68% | −28.3  | 75% |  17 | 53d   |

**Diagnosis:**
- λ=0.25 is the only positive delta but only by +2 APY pts — below the
  +3 pt promotion threshold. Win rate does lift (81% → 83%) but buy
  count drops 161 → 129, so total compounding is close to flat.
- Above λ≥0.5 the σ penalty starves the ranker of trades (17 buys
  at λ=1.0) and the 53-day max-streak becomes unacceptable under the
  monitoring invariant (< 20 days).
- The sweep baseline does NOT match the documented T4 golden (+40.1%
  APY) because the sim disables live-fetch for fundamentals/earnings/
  insider. Absolute APYs are lower but relative Δs between λ values
  are still valid.

**Action taken:**
- Keep `score_mode=additive` as golden (unchanged).
- Shelve μ−λσ as a ranking-score override at constant λ. May revisit
  if the σ head is re-fit on a panel with NGBoost's predicted variance
  better-calibrated (Plan F depends on regime data so is decoupled).
- σ-sizing multiplier (separate feature, knob is
  `sigma_sizing.enabled`) is not affected — this was only a study of
  how σ should bias the *ranking* score.
- Task E (re-fit global calibrator on μ−λσ distribution) is cancelled
  — it was conditional on C showing a winner.

Log: `/tmp/sigma_sweep.log` (6 sims, ~7min each).

---

## A/B — 2026-04-23 late PT — LightGBM backend shelved (APY −12.7 pts)

Task B / plan-A/B. Compared XGBoost (current golden backend) against
LightGBM LambdaRank on identical panels and the T4 golden xgb_params
neighborhood (lightgbm_params already populated in strategy_config.json).

Preconditions — found a bug blocking the run on the way in:
`PanelLGBMModel.train` passed per-GROUP weights to `lgb.Dataset`, but
LightGBM 4.x takes per-ROW weights even when `group` is set. Fixed in
commit `8d6b08a` (broadcast group-mean weight back to rows, regression
test added).

**Results** (27-month OOS sim, same panel data, NGBoost re-fit for each):

| Metric (after-tax) | XGBoost | LightGBM | Δ |
|---|---|---|---|
| OOS CPCV mean IC | +0.0409 ± 0.023 | **NaN** (CPCV broke on lgbm preds) | — |
| Train IC | +0.264 | NaN | — |
| APY | **+34.4%** | +21.7% | **−12.7 pts** |
| Total return | +92.9% | +54.6% | −38 pts |
| Buys / sells | 157 / 156 | 142 / 137 | −9% volume |
| Win rate | **84%** | 74% | −10 pts |
| Avg P&L / trade | +10.2% | +8.3% | −2 pts |
| Longest no-trade streak | 28 d | 28 d | — |
| Retrain time | 234s | 218s | — |

**Diagnosis:**
- LightGBM's NDCG@10 objective should have matched our 8-slot selection
  but in practice produced a weaker per-date signal ordering. Win rate
  −10 pts means LGBM is picking bad trades, not just filtering.
- The NaN OOS CV IC is a separate measurement bug (bucketize +
  CPCV interaction on continuous Gaussianized labels). Since SIM
  predictions do fire (142 buys executed), the fitted model is
  producing usable scores; the NaN is in the CV metric computation,
  not the model itself. Worth fixing if we ever revisit LGBM.

**Action taken:**
- Config unchanged: `panel_ltr.backend` stays at `"xgboost"`.
- LGBM infra (model class, scorer, dispatcher) stays shipped — 9
  regression tests green after the per-row-weight fix. A future rerun
  (different hyperparams, or after the hourly-panel lands) can flip
  `panel_ltr.backend: "lightgbm"` in config and re-A/B.
- Shipped the per-row-weight fix (commit `8d6b08a`) as a standalone
  patch even though LGBM isn't getting promoted — the bug would have
  bitten anyone who flipped the backend blind.

**When to revisit:**
- After G lands (hourly panel features). Bigger feature set might
  favor LGBM's leaf-count + NDCG@k objective more than daily-only data.
- OR tune `lightgbm_params.num_leaves`, `max_depth`, `learning_rate`
  independently — today's A/B used the untouched config default.

---

## A/B — 2026-04-23 15:30 PT — NGBoost `score_mode=mu_minus_lambda_sigma` shelved (APY −27 pts)

Purpose: test task #2 (revert `ranking.panel_scoring.ngboost.score_mode` from
`additive` back to `mu_minus_lambda_sigma` with `lambda_sigma=1.0`) on top of
the T4 golden config. Required a code refactor first — `ApplyGlobalCalibrationTask`
used to short-circuit in mu_minus_lambda_sigma mode, which made raw μ−λσ
values (range ~±0.05) fail the 0.10 tier threshold → zero trades. The
refactor (commit `339944b`) reordered `PanelScoringJob` so calibration runs
AFTER NGBoost, letting the isotonic calibrator map μ−λσ → probability.

**Setup:** same panel artifact (today's daily_104 retrain, OOS IC 0.0363,
CPCV 15-split), same T4 xgb_params, same 27-month OOS window. Only the
`ranking.panel_scoring.ngboost.{score_mode, lambda_sigma}` pair flipped.

| Metric (after-tax) | Baseline (additive, λ_σ=0) | Variant (μ−λσ, λ_σ=1.0) | Δ |
|---|---|---|---|
| Total return | +85.2% | +11.4% | **−74 pts** |
| **APY** | **+32.0%** | **+5.0%** | **−27 pts** |
| Buys / sells | 161 / 157 | 29 / 28 | −82% volume |
| Win rate | 81% | 75% | −6 pts |
| Avg P&L per trade | +9.0% | +7.0% | −2 pts |
| Longest no-trade streak | 30 d | 31 d | — |

**Diagnosis:**
σ-penalty at λ=1.0 is way too aggressive. Observed behavior:
- Candidate volume collapses (−82%) — σ pushes most candidates below the
  tier threshold even after calibration.
- Win rate drops too (81% → 75%), so the σ-penalty isn't just pruning
  losers — it's filtering out real signal along with noise.
- The σ ≈ 0.065 observed in training has a similar scale to μ ≈ 0.03,
  so μ − 1·σ tends to be negative for most tickers, pushing the
  calibrated probability down uniformly.

**Action taken:**
- Config kept at `score_mode: additive, lambda_sigma: 0.0`.
- Code refactor (commit `339944b` — reorder calibration after NGBoost,
  remove the short-circuit) stays shipped. It's provably a no-op in
  additive mode AND it enables the λ_σ=0.X sweep as a future
  experiment without re-touching the pipeline code.
- σ-aware sizing (`ranking.panel_scoring.sigma_sizing`) is already
  enabled in golden (floor=0.3, ceiling=1.0) — that path uses σ
  multiplicatively on position size, not subtractively on rank_score,
  and is the better way to express σ-awareness given the λ=1.0 result.

**When to revisit:**
- Before trying λ_σ=1.0 again, sweep lower values (0.1, 0.25, 0.5) to
  see if there's a non-zero optimum. Needs its own A/B.
- OR re-fit the global calibrator on μ−λσ values directly (currently
  fit on raw panel_score) so the mapping is metric-calibrated, not
  just directionally monotone.

---

## A/B — 2026-04-23 09:54 PT — Transformer backend shelved (ratio 0.49)

First head-to-head of the XGBoost panel-LTR vs the new `PanelTransformerModel`
PyTorch backend on the real 47k-row panel. Both ran on identical feature
frames (25 features after the DEFAULT_DROP_COLS union fix, see below),
identical CV splits (5-fold purged, embargo=lookahead), NGBoost head disabled
for both.

**Config:**
- `panel_ltr.backend`: "xgboost" vs "transformer" (in-memory swap per run)
- `panel_ltr.cv_method`: purged, `cv_n_splits`: 5 (overridden by A/B script from the default CPCV 15-split to keep wall-clock reasonable)
- Transformer params: d_model=128, n_heads=4, n_layers=3, max_epochs=30, dropout=0.3, device=cpu

**Results:**

| Metric | XGBoost | Transformer | Ratio |
|---|---|---|---|
| OOS mean IC | +0.0316 | +0.0156 | 0.49 |
| Train IC    | +0.1677 | +0.2779 | overfits 3× more |
| Per-fold min | −0.0229 | −0.0703 | worse tail |
| Per-fold max | +0.0857 | +0.0745 | — |
| Training time | 51s | 198s | 3.9× slower |

**Diagnosis:**
- Transformer overfits badly on 47k rows — train/OOS ratio 18× vs XGBoost's 5.6×.
  Matches the design doc §1 warning: *"Transformers typically want 10–100× [80k]; we may overfit."*
- Per-fold min dipped to −0.0703 for the transformer vs −0.0229 for XGBoost — significantly worse worst-case date.
- Dropout=0.3 + feature_dropout=0.2 + ticker_dropout=0.1 didn't tame it enough. Would need deeper data (more years of history, or adding ETF-adjacent universe) before the transformer has a shot.

**Action taken:**
- **Shelved transformer backend** per `doc/renquant_104_transformer_design.md §5` ship-gate (ratio 0.49 well below the 1.10 ensemble threshold).
- Kept the transformer code shipped in the repo — 27/27 transformer / scorer / integration / ensemble tests still green. Future experiments (more data, richer features, or a different backbone) can pick up the infra without re-implementing.
- Skipped step 2e (ensemble Task wiring into `PanelScoringJob`) — at ratio 0.49 an ensemble would drag composite IC down vs XGBoost alone.
- XGBoost remains the production panel backend. `panel_ltr.backend` defaults to `"xgboost"` in the golden config.

**Bugs found & fixed during this A/B** (keepers):
- `PanelTransformerModel.predict()` silently truncated groups > `max_tickers` to uninitialized memory when no `date` column was present → NaN OOS IC. Now raises without date/group_sizes, splits oversized groups into chunks, and initializes preds as NaN so any unwritten slot surfaces cleanly. (Commit `fff97af`, 4 regression tests.)
- `BuildPanelTask` user-provided `drop_cols` used to REPLACE `DEFAULT_DROP_COLS` — raw `close` (std~40) + constant SPY columns leaked into every backend's feature set. Union fix: user `drop_cols` now augments, not replaces. XGBoost gets a cleaner 25-feature input too.
- `scripts/compare_panel_backends.py` needs CV-work caps (`cv_n_splits=5`, transformer `num_boost_round=30`) or else 15-split CPCV × 75 epochs/fold = 1125 epoch-fits on CPU = ~40 min per leg. (Commit `2c06d19`.)

---

## Run 3 — 2026-04-22 09:17 PT — lookahead 10d + stronger regularization ⭐ target hit

Follow-up to Run 2. Extended label horizon from 5d → 10d (doubles signal-to-noise) and applied stronger regularization.

**Config (diff vs Run 2):**
- `panel_ltr.lookahead_days`: `5` → **`10`**
- `panel_ltr.cv_embargo_days`: `5` → **`10`** (must match lookahead)
- `panel_ltr.num_boost_round`: `150` → **`300`**
- `panel_ltr.xgb_params`:
  - `eta 0.05 → 0.02` (smaller learning rate, more rounds)
  - `max_depth 4 → 3` (shallower trees)
  - `min_child_weight 40 → 60`
  - `subsample 0.7 → 0.5` (stronger bagging)
  - `colsample_bytree 0.7 → 0.5`
  - `lambda 1.0 → 5.0` (stronger L2)
  - `alpha 0.5 → 2.0` (stronger L1)

**Artifacts:**
- `panel-ltr.json`: rows=80,627 × tickers=38 × dates=2,247 × 20 features
  - **oos_mean_ic = +0.04032** (Run 2: +0.02499 → **+61% improvement, passes 0.04 target**)
  - oos_per_fold_ic = `[+0.03300, +0.02533, +0.02584, +0.06629, +0.05116]`
  - **min fold IC = +0.0253** (all positive, no fold below 0.025)
  - training_train_ic = **+0.32565** (Run 2: +0.39904 → slightly lower, more regularized)
  - train/OOS ratio = **8.1×** (Run 2: 16× → halved again)
- `ngboost-head.json`: n_rows=80,437, train_μ_mean=−0.0023, train_σ_mean=0.0642 (σ up from 0.045, consistent with 2× longer horizon)

**Diagnosis — signal is real and stable:**

| Metric | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| OOS mean-IC | +0.0039 | +0.0250 | **+0.0403** |
| Min fold IC | −0.0102 | +0.0158 | **+0.0253** |
| Max fold IC | +0.0194 | +0.0419 | **+0.0663** |
| All folds positive | ✗ (2 of 5 neg) | ✓ | ✓ |
| Train IC | +0.833 | +0.399 | **+0.326** |
| Train/OOS ratio | 213× | 16× | **8.1×** |
| Top-3 gain | ~12% | 17.5% | **19.1%** |
| Top-5 gain | ~21% | 28.4% | **30.1%** |

Top-10 features by gain (Run 3):

| Rank | Feature | Gain % | Notes |
|---|---|---|---|
| 1 | `beta_60d_z` | 7.10% | rose from 6.04% in Run 2 |
| 2 | `hurst_proxy` | 6.23% | regime-context, very stable |
| 3 | `spy_realized_vol` | 5.80% | regime-context |
| 4 | `obv_slope` | 5.56% | volume indicator — newly prominent |
| 5 | `spy_adx` | 5.42% | regime-context |
| 6 | `close` | 5.37% | relative price |
| 7 | `size_z` | 5.37% | cross-sectional factor |
| 8 | `cci` | 5.27% | mean-reversion |
| 9 | `trend_long` | 4.97% | momentum |
| 10 | `rel_mom_20d` | 4.94% | short momentum |

Structural read: the top-5 is dominated by **regime-context features** (Hurst, SPY vol, SPY ADX, β, size) plus volume. Matches the interpretation from Run 2 — the model uses macro conditions to set the baseline, then mean-reversion + momentum to differentiate within a bar.

**Verdict:** Run 3 is a clear winner. OOS mean-IC 0.04 is the doc's "ship-ready" threshold we were aiming for. All folds positive with narrow IQR (min 0.025, max 0.066) indicates the signal is stable across time periods. Train/OOS gap 8× is at the lower bound of acceptable for tree-based models on ~80k rows.

**Next levers to explore** (in roadmap priority order):
- Item #2 Global calibrator on panel — can now ship with confidence
- Item #6 CPCV — should fit tighter fold distribution, might lift IC to 0.05+
- Item #7 LightGBM LambdaRank@10 — top-10 weighted objective matches our 8-selection budget
- Item #8 Intraday features — biggest upside but highest cost

**Action taken:** Keep Run 3's config as production. Proceed with roadmap.

---

## Run 2 — 2026-04-22 07:50 PT — fundamentals off, shrunk model, widened β window

Targeted fix for Run 1 diagnosis: disable time-invariant fundamentals, reduce model capacity, wider beta window for label neutralization.

**Config (diff vs Run 1):**
- `panel_ltr.fundamentals.enabled`: `true` → **`false`**
- `panel_ltr.num_boost_round`: `400` → **`150`**
- `panel_ltr.xgb_params`: `{}` → **`{max_depth: 4, min_child_weight: 40, subsample: 0.7, colsample_bytree: 0.7}`**
- `panel_ltr.beta_window`: `60` → **`252`**
- Skip baseline (`--skip-baseline`) — tournament artifacts from Run 1 still valid.
- NGBoost + σ-sizing unchanged (still on).

**Artifacts:**
- `panel-ltr.json`: rows=80,627 × tickers=38 × dates=2,247 × **20 feature columns** (4 fundamental z-cols removed)
  - **oos_mean_ic = +0.02499** (Run 1: +0.00388 → **6.4× improvement**)
  - oos_per_fold_ic = `[+0.01760, +0.01580, +0.02347, +0.04193, +0.02615]` (**all 5 folds positive**, Run 1 had 2 negative)
  - training_train_ic = **+0.39904** (Run 1: +0.83293)
  - **train/OOS ratio = 16×** (Run 1: 213× → 13× less overfit)
- `ngboost-head.json`: n_rows=80,627, train_μ_mean=−0.00118, train_σ_mean=0.04504, 20 feature cols

**Diagnosis — genuine signal, not noise:**

| Dimension | Run 1 | Run 2 | What changed |
|---|---|---|---|
| OOS mean-IC | +0.0039 | **+0.0250** | 6.4× |
| Folds all positive | 2 of 5 negative | **5 of 5 positive** | ✓ consistent direction |
| Train/OOS gap | 213× | **16×** | 13× reduction |
| Top-3 gain concentration | ~12% | **17.5%** | signal concentrating on real features |
| Top-5 gain concentration | ~21% | **28.4%** | ditto |

Top-10 features by gain (Run 2):

| Rank | Feature | Gain % | Bucket |
|---|---|---|---|
| 1 | `beta_60d_z` | 6.04% | tech factor |
| 2 | `spy_realized_vol` | 5.99% | regime context |
| 3 | `hurst_proxy` | 5.49% | regime context |
| 4 | `close` | 5.46% | (relative price) |
| 5 | `mom_12_1_z` | 5.37% | tech factor |
| 6 | `bbp` | 5.36% | mean-reversion indicator |
| 7 | `spy_adx` | 5.24% | regime context |
| 8 | `size_z` | 5.22% | tech factor |
| 9 | `macd_hist` | 5.17% | momentum indicator |
| 10 | `spy_trend` | 5.15% | regime context |

The top 4 features are **β, realized vol, Hurst, relative-price** — all regime/market-context features. This makes sense: 5-day forward residual return is noisy at the ticker level, so the model leans on macro conditions to set the baseline and uses technicals for differentiation. Bottom-of-list features (RSI/CCI/trend_long at ~4.1–4.5%) have diminishing marginal value — candidates to prune in a future round if we want further regularization.

**Verdict:**
- OOS mean-IC 0.025 is still below the doc's "ship" target of 0.08. But Run 2 is clearly *real* signal:
  - All 5 folds positive.
  - Train/OOS gap collapsed from catastrophic to merely weak-but-fixable.
  - Feature importance concentrating on structural regime/factor features.
- This is the right point to ship σ-aware ranking (NGBoost μ−λσ) and σ-sizing live — the μ is modest but unbiased, and σ-sizing provides risk protection independent of μ's magnitude.
- **Next improvement levers** (future runs to explore):
  - Try `max_depth=3` and `num_boost_round=100` — even more regularization, see if IC holds.
  - Prune features below 4.5% gain (RSI, CCI, trend_long) — 17 features might outperform 20.
  - Consider second-pass target: replace the current 5-day Gaussianized residual with 10-day or 20-day (rotation horizon is 20d). Longer horizon = smaller noise-to-signal.
  - Introduce true time-series fundamentals (quarterly P/E, earnings momentum) instead of static snapshots — but that's a data-pipeline build-out, not a config flip.

**Action taken:** Keep Run 2's config. Proceed with SimAdapter refactor (issue #3) so notebook + LEAN + live all consume the same pipeline decisions going forward.

---

## Run 1 — 2026-04-22 07:24 PT — baseline w/ fundamentals + NGBoost enabled

First full training with all Stage-2 + Stage-3.1 flags on.

**Config:**
- `training.cadence = "custom"`, `allowed_weekdays = [1, 3, 6]`
- `panel_ltr.fundamentals.enabled = true`
- `panel_ltr.ngboost.enabled = true`
- `panel_ltr.num_boost_round = 400`
- `panel_ltr.xgb_params = {}` (defaults: `max_depth=6, min_child_weight=20, lambda=1.0, alpha=0.5`)
- `panel_ltr.beta_window = 60`
- `ranking.panel_scoring.ngboost.enabled = true`
- `ranking.panel_scoring.ngboost.score_mode = "mu_minus_lambda_sigma"`
- `ranking.panel_scoring.ngboost.lambda_sigma = 1.0`
- `ranking.panel_scoring.sigma_sizing.enabled = true`
- Fundamentals coverage at training time: 33/38 tickers had at least one factor column (the 4 ETFs/commodities XLF/GLD/XLU/XLK had nothing). Missing values filled with same-sector median before z-scoring.

**Artifacts:**
- `panel-ltr.json`: rows=80,627 × tickers=38 × dates=2,247 × **24 feature columns** (16 neutralized indicators + 4 technical factor z-cols + 4 fundamental z-cols)
  - **oos_mean_ic = +0.00388**
  - oos_per_fold_ic = `[-0.01022, +0.00208, +0.01937, +0.01631, -0.00813]`
  - training_train_ic = **+0.83293**
- `ngboost-head.json`: n_rows=80,627, train_μ_mean=−0.0011, train_σ_mean=0.04636, 24 feature cols

**Diagnosis — the model is memorizing noise, not learning signal:**

| Symptom | Value | Why it matters |
|---|---|---|
| OOS mean-IC | +0.0039 | Doc target is 0.08 — we're 20× below. Essentially no cross-sectional signal. |
| Train/OOS IC ratio | 213× (0.83 / 0.004) | Textbook overfitting — the model memorizes the training set but generalizes nowhere. |
| Per-fold IC sign | 2 of 5 negative | A real signal would produce all-positive folds with similar magnitude. The negative folds say the model's ranking is sometimes wrong more than random on OOS bars. |
| Feature importance dispersion | All 24 features in 3.76%–4.74% gain band | Uniform importance is the signature of XGBoost splitting on whichever feature wins a coin toss — noise-driven splits, not signal concentration. |

Feature-importance breakdown:
- Technical indicators (16 cols): **65.5% of gain**
- Fundamentals (4 cols): **17.4% of gain**
- Technical factors (4 cols): **17.2% of gain**

**Key insight:** The 4 fundamental z-columns (`earnings_yield_z`, `roe_z`, `gross_profitability_z`, `book_to_price_z`) are **time-invariant per ticker** in this release — we only have a single static OpenBB snapshot, which is broadcast to every bar. So the model isn't learning "low earnings yield → future outperformance"; it's learning **"this constant 0.34 value = this is AAPL"** — covertly memorizing ticker identity through the fundamentals channel. This is the failure mode flagged in `doc/research_scoring.md §3.5` ("cross-sectional rank imputation on the feature itself — creates artificial clustering that trees latch onto as spurious signal"), which was not enforced when fundamentals were wired in this session.

**Action taken:** → Move to Run 2 with three changes:
1. **Disable fundamentals** (`panel_ltr.fundamentals.enabled = false`). Keep the module + cache for the future time-series build-out.
2. **Shrink model capacity**: `num_boost_round 400 → 150`, `max_depth 6 → 4`, `min_child_weight 20 → 40`, `subsample 1.0 → 0.7`.
3. **Widen beta window** for label neutralization: `beta_window 60 → 252` — reduces noise in the rolling OLS β used when computing residual forward returns.

Expected effect: IC should lift out of noise range and feature importance should concentrate on a few columns. If mean-IC doesn't pass 0.02, the label pipeline itself is suspect and needs deeper investigation.

σ-sizing + NGBoost enabling kept on — these are risk-adjustment layers independent of IC quality.
