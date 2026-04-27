# Rust Transformer → Production Promotion Plan

Living plan for promoting the Rust panel transformer (v5
hourly-era only, val_IC = +0.0519, beats Python LightGBM +0.0372 by
+39.5%) into the renquant_104 production stack.

**Status: PAUSED (2026-04-25 PT, late-session) — Phase A2 result kills
the promotion case.** Production LightGBM on same panel + same split
= +0.0850 (vs transformer +0.0519). LightGBM wins by 64%. No promotion.
Plan retained as scratchpad for future improvement attempts.

## Why this matters

Single-split val_IC of +0.0519 on the hourly-era panel beats the
production LightGBM panel-LTR by 39.5% on the same data window. If
that holds under cross-validation AND translates to better APY in the
sim, the transformer should replace LightGBM as the panel backend.
Conservative fallback: keep LightGBM, blend transformer in as an
ensemble third backend (NGBoost is the second).

## Headline numbers (must keep updated)

| Run                                            | val_IC  | Notes |
|------------------------------------------------|--------:|-------|
| Production LightGBM (full 1251-date panel, prod params) | +0.0372 | shipped baseline (older comparison, NOT apples-to-apples) |
| Naive LightGBM quick (491-date hourly-era, default params) | -0.0359 | NOT prod params, sanity only |
| Rust transformer v5 (491-date hourly-era, ListNet 200ep) | +0.0519 | first healthy real-data curve |
| **Production LightGBM (491-date hourly-era, DEFAULT_PARAMS)** | **+0.0850** | A2 — the real apples-to-apples baseline |
| Rust transformer v5 + LightGBM ensemble (TBD)            | TBD     | to be measured (would only narrow the LGBM lead) |
| Rust transformer CPCV mean (TBD)                          | TBD     | (deprioritized — the single-split number already lost A2) |

## Step-by-step plan (doing them in this order)

### Phase A — Rigor (validate the +39.5% number)

- [x] **A1 — Confirm reproducibility.** Re-run v5 with a different seed.
  *Acceptance:* val_IC within ±0.005 of +0.0519. Otherwise we got lucky
  on the seed and need bigger ensembles.
- [ ] **A2 — Fair LGBM apples-to-apples.** Run `training_panel/lgbm_ltr.py::PanelLGBMModel`
  with production hyperparams (DEFAULT_PARAMS) on `/tmp/real_panel_hourly_era.csv`,
  same chronological 80/20 split. Naive LGBM quick said -0.0359; the prod-params
  LGBM number is the actual head-to-head we should report.
  *Acceptance:* report number to user. If LGBM-prod-on-491 ≥ +0.0519, transformer
  hasn't actually won. If < +0.0519, headline holds.
- [ ] **A3 — CPCV cross-validation on transformer.** Use `rust/transformer_scorer/src/cv.rs`
  with `n_splits=15` on the same hourly-era panel. Report mean & std of val_IC
  across folds.
  *Acceptance:* CPCV mean IC ≥ +0.040, std ≤ +0.030. (If std too high, the
  +0.0519 was a single-fold luck-out and we shouldn't promote.)
- [ ] **A4 — Notebook 3-way compare cell.** Build a notebook cell at
  `backtesting/renquant_104/renquant_104.ipynb` that loads the same panel,
  runs (a) production LightGBM, (b) Rust transformer (via subprocess →
  safetensors), (c) ensemble blend (50/50, then optimal-weight blend by
  regression). Visualize per-date IC distributions side-by-side.

### Phase B — Production wiring (no behavior change yet)

- [ ] **B1 — Add `TransformerPanelScorer` Python adapter.** Mirror
  `kernel/panel_pipeline/load_scorer_task.py` to load safetensors via
  the existing PyO3 binding (`rust/transformer_scorer/python/`). Keep
  the LightGBM path as the default; add `panel_ltr.backend: "transformer"`
  selector.
- [ ] **B2 — Calibrate transformer scores.** Run `scripts/recalibrate_scores.py`
  with the transformer backend selected — produce `panel-rank-calibration.json`
  + per-regime calibrators in transformer score space. Verify the calibrated
  rank_score has roughly the same value distribution as LGBM's so the
  tier-threshold logic doesn't need re-tuning.
- [ ] **B3 — Tests.** Add `tests/test_panel_transformer_backend.py` with:
  parity-with-Rust-CLI test, alignment-with-LGBM-shape test, regression
  test that `LoadScorerTask` correctly dispatches based on backend.
- [ ] **B4 — Sim runtime parity.** `sim/runner.py::run_backtest` with
  backend="transformer" must produce a SimResult with the same row
  counts and order types as LGBM (only the panel_score and downstream
  rotations should differ). Snapshot test.

### Phase C — Sim & golden config

- [ ] **C1 — Full sim with transformer backend.** Run the OOS sim
  (2024-04-25 → today) with backend="transformer" and compare APY,
  Sharpe, max_dd, n_trades, exit_reasons against the current golden
  v4.1 baseline.
  *Acceptance:* APY ≥ golden v4.1 APY (currently +39.82% on 27-mo OOS,
  ~+65% expected live). Per CLAUDE.md §2a, theoretically-clean wins
  ship even at small positive margin if the mechanism is correct.
- [ ] **C2 — Ensemble sim (50/50 blend).** Same sim with
  panel_score = 0.5 × LGBM_score + 0.5 × transformer_score (after
  same-cal calibration). Hedges against single-model drift.
- [ ] **C3 — Golden v5 promotion.** If C1 or C2 wins:
  - Update `backtesting/renquant_104/strategy_config.golden.json`
  - Update `doc/golden_config_2026-04-23.md` history table
  - Frozen artifact: `artifacts/panel-transformer.golden-v5.safetensors`
  - Test count update in CLAUDE.md
- [ ] **C4 — Cron wiring.** If transformer becomes the active backend,
  ensure `daily_104.sh` + `retrain_panel.sh` retrain it (Rust binary
  invocation). Otherwise the cron retrain only refreshes LightGBM and
  the transformer artifact will become stale.

### Phase D — Docs + reports

- [ ] **D1 — Papers report.** Write `doc/papers_implemented.md`
  catalogue:
  - Vaswani 2017 — base transformer architecture
  - Poh-Lim-Zohren-Roberts 2020 — listwise > pairwise on cross-sec ranking
  - Burges 2005 — RankNet pairwise BCE
  - "On Evaluating Loss Functions for Stock Ranking" CIKM 2025
  - ApxML transformer regularization survey
  - López de Prado AFML ch.7 — CPCV
  - Frisch-Waugh-Lovell — label residualization
  - Howard Hinnant date algorithm (civil_to_days)
  - sparsity-aware NaN handling (XGBoost/LightGBM)
- [ ] **D2 — Update CLAUDE.md** with: transformer backend mention in
  renquant_104 spec + golden v5 reference + new test counts.
- [ ] **D3 — Update `doc/rust_transformer_ic_baseline.md`** with final
  CPCV + ensemble + sim numbers (replacing the single-split +0.0519
  headline).

## Parallel: continue audit work

Per CLAUDE.md §2b, the prod-vs-test divergence finding should keep us
suspicious of similar issues elsewhere. Continue auditing for:

- [ ] Other train/val distribution shifts (e.g. fundamental factors
  added mid-history, sector ETF changes, watchlist additions).
- [ ] More dead-code calls: AUDIT-PROD-IMPUTATION-DEAD-CODE found
  `add_missingness_indicators` defined but never called. Find more by
  greping defined-but-not-imported public API.
- [ ] Any silent-zero-fills in the Python pipeline that mirror our
  Rust 0.0-fill bug (look for `.fillna(0)` calls, `pd.fillna` in
  inference paths).

## Risks / gotchas

1. **Single-split overfit.** +0.0519 might not survive CPCV. If A3
   shows mean IC drops below LightGBM's +0.0372, we should NOT promote.
2. **IC-vs-APY divergence.** Higher IC doesn't always mean higher APY
   — the rotation gate / tier thresholds / sector guard can equalize
   small IC differences.
3. **Maintenance cost.** Adding a Rust backend means daily_104 needs
   to call the Rust binary. If the box has no `cargo` available, the
   cron will silently fail. Alternative: pre-built binary in
   `rust/target/release/` checked in or auto-built by daily_104.sh.
4. **Train data window mismatch.** Production LightGBM trains on full
   1251-date panel; transformer-v5 trains on 491-date hourly-era only.
   If the LightGBM signal degrades over time as features change, this
   becomes a wash.

## Decision tree

```
A1 reproducible (val_IC ≈ +0.052) → A2 LGBM-prod-on-491
  ├─ LGBM-prod-on-491 ≥ +0.052 → STOP. Just match production.
  └─ LGBM-prod-on-491 < +0.052 → A3 CPCV
       ├─ CPCV mean < +0.040 → STOP. Single-fold luck.
       └─ CPCV mean ≥ +0.040 → B1-B4 wiring → C1 sim
            ├─ Transformer-only sim ≥ golden v4.1 APY → C3 golden v5
            ├─ Ensemble sim ≥ golden v4.1 APY → C3 golden v5 (ensemble)
            └─ Both lose → B5 leave as opt-in backend, no golden change
```

## Time budget (current session)

- A1, A2, A4 — TODAY (next 30 min)
- A3 — TODAY (CPCV is in cv.rs already, just needs wiring)
- B1-B4 — Next session (will refactor LoadScorerTask + tests)
- C1-C4 — Next session
- D1-D3 — Concurrent

Update this file as steps complete (check the boxes!).
