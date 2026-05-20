# 2026-05-18 — Live inference data path diverges from training (CRITICAL)


> **📅 Historical snapshot — content below reflects state at the date in filename/header.**
> Verify against current code per CLAUDE.md §1 "code is the source of truth" before acting on
> present-tense claims. For current state see `doc/roadmap.md` § "📍 Current state" +
> `CLAUDE.md` § "🗂 Current state".

## Finding

The model is HEALTHY on training data. The bug is in the LIVE inference data construction.

| Test | μ̂ std | μ̂ range |
|---|---|---|
| **Model on TRAINING panel** (2026-02-05, 142 wl200 tickers) | **0.20** | [-0.36, +0.48] |
| Model on training panel WITH sentiment ZEROED | 0.20 | similar (sentiment is <2% of variance) |
| Model on training panel WITH sentiment+PEAD+SUE ALL ZEROED | 0.21 | similar |
| **Model on LIVE INFERENCE** (today's 69 candidates) | **0.012** | [+0.0003, +0.0439] |

Live is **17× tighter** than training on otherwise-identical model + same regime (BULL_CALM-ish in early 2026 too).

## Why this matters

The calibrator was trained on the wide training-data μ̂ distribution. Its x-domain is `[-0.59, +0.56]`. Today's live μ̂ values cluster in `[+0.0003, +0.0439]` — a SINGLE flat segment of the isotonic curve. All 69 candidates → same probability → IQR=0 → tie-broken selection → MCD picked at random.

This isn't a "calibrator needs more flexibility" issue. The calibrator is FINE for properly-distributed inputs. The bug is that LIVE produces 17× tighter inputs.

## Hypothesized causes

Three candidates to investigate:

1. **Alpha158 computation divergence** — `compute_alpha158_at` (live) may produce different values than `build_alpha158_qlib.py` (training). Direct column-by-column comparison on AAPL/2026-02-05 showed 158/158 columns differing with abs_diff up to 2.5 (but the comparison framing may itself be wrong — the training panel stores ALREADY-NORMALIZED features, while live computes raw then normalizes via `artifact.feature_means/stds`).

2. **Feature coverage at inference** — the live path may be zeroing/imputing more features than training due to missing fundamentals, missing earnings_surprise, missing sentiment, etc. The existing FEATURE-HEALTH check logs warnings but doesn't quantify variance loss.

3. **Normalization path** — training applies z-score at panel-build time AND a second normalization at scorer-load. Live applies normalization via `artifact.feature_means/stds`. If these means/stds were fit on POST-z-score values, live's RAW input would land in the wrong region.

## Why existing test passes but this still happens

`tests/test_alpha158_e2e_inference.py::test_aapl_inference_matches_dataset` PASSES. But it:
- Tests the **LINEAR** scorer (`panel-ltr.alpha158_linear.json`), not the prod XGB one (`panel-ltr.alpha158_fund.json`)
- Tolerance is **0.5 absolute** on a single score — way too loose to catch a 17× variance change
- Tests ONE date for ONE ticker — no distribution check
- Doesn't include the new sentiment/PEAD/SUE merge path used in prod

This is exactly the failure mode CLAUDE.md §5.13.1 warns about ("test fixtures lie" — call through ACTUAL prod path).

## Safety nets shipped TODAY (compounding)

Until root cause fix:

1. `min_reentry_days = 5` (commit 7e17e65): blocks ANY same-ticker rebuy within 5 days regardless of P/L sign. Anti-churn at the lot-history layer.

2. `abstain_on_calibrator_saturation = true` (commit 91e188a): when calibrator IQR < 0.05, block NEW buys (existing holdings can still sell). Conviction gate at the QP layer.

3. LIVE MCD order canceled (Alpaca API, 14:00 PT).

These two guards mean: **even if the underlying data bug persists, tomorrow's cron won't churn**. The model will hold cash on "no signal" days rather than tie-break-and-buy.

## Followup TASKS (next session priority)

P0:
1. Write proper `tests/test_inference_distribution_matches_training.py`:
   - Score the prod XGB model on both (a) panel parquet rows and (b) `compute_alpha158_at` live-path output for the SAME 50 tickers × 3 dates
   - Assert μ̂ std within 20% of training-panel μ̂ std
   - Assert column-by-column raw-feature L2 norm < 0.1 between training and live

2. Add a HARD preflight check `P-INFERENCE-DIST` that fires before LIVE trade:
   - Compute μ̂ std on today's 69+ candidates
   - If std < 0.05: HARD FAIL preflight (cron skips entire run)
   - Catches the bug at every cron firing, not just incident discovery

3. Diagnose root cause (which of 3 hypotheses):
   - 3a: instrument compute_alpha158_at to log per-feature stats
   - 3b: compare side-by-side raw alpha158 dict for AAPL 2026-02-05
   - 3c: trace normalization pipeline end-to-end

P1:
4. Add `tests/test_panel_build_inference_equivalence.py`: tighter assertion via the actual scorer path

5. Consider switching from isotonic to a smoother monotone calibrator (B-spline / Platt) so flat regions don't appear naturally

## Honest answer to user's question

> "is this finding deep enough? is there data issue? is there issue in model?"

- **Data issue**: YES — live inference produces 17× tighter input distribution than training. Affects ALL recent decisions, not just MCD.
- **Model issue**: NO — model trained correctly; on training data its predictions are properly varied (std=0.20).
- **Architectural issue**: YES — no preflight check on inference-distribution health, no equivalence test pinning live-vs-training std ratio.

The MCD incident exposed THREE problems:
- Symptom: same-day rebuy after sell (anti-churn missing) ✅ fixed
- Mechanism: calibrator saturation tie-broke the rebuy decision (abstain missing) ✅ fixed
- Root cause: live inference inputs are wildly different from training inputs (UNFIXED — proper P0 next session)

The two SAFETY NETS shipped today are necessary but not sufficient. The root cause investigation is the real P0.
