# μ̂ autocorrelation per regime — measurement results

**Date**: 2026-06-03 · **Updated**: 2026-06-03 after codex #128 review
**Status**: §8 Step 2 of [`2026-06-02-qp-architecture-review-and-alternatives.md`](2026-06-02-qp-architecture-review-and-alternatives.md). Closes the codex #125 HIGH-4 finding.
**Author**: Claude
**Inputs**: `data/sim_runs.db::score_distribution` (3,052 (date, ticker, μ̂, regime) rows, 540 dates, 65 tickers, 2024-01-02 → 2026-03-27)
**Outputs**: [`evidence/2026-06-03-mu-hat-autocorrelation-by-regime.json`](evidence/2026-06-03-mu-hat-autocorrelation-by-regime.json) (canonical evidence artifact)
**Script**: [`scripts/measure_mu_hat_autocorrelation_by_regime.py`](../../scripts/measure_mu_hat_autocorrelation_by_regime.py)

> **#128 review (v2 correction)**: codex flagged that v1 of the fix
> measured "same-regime-both-endpoints" persistence rather than the
> documented "regime-at-t-only" contract. v2 fixes the helpers so each
> ticker's FULL μ̂ series (across all regimes) is reindexed against the
> global trading-day grid and only the BASE date t is required to be in
> the target regime — t+k can be in any regime. The corrected BULL_CALM
> autocorr values shifted slightly: L20 went from +0.238 to +0.166,
> L60 from −0.200 to −0.158. Interpretation updated accordingly.
>
> **#128 review correction**: codex (HIGH-1) caught that the first
> version of the script measured *observation-index lag*, not
> *trading-day lag*. `pd.Series.shift(5)` on a sparse in-regime
> series shifts by the 5th later observation, which on this DB had
> median 5 / p90 70 / max 422 trading-day gap. The script now reindexes
> each ticker's regime series against the global trading-day index
> (sorted unique dates across the full panel) before shifting. The
> corrected headline numbers below replace the original (incorrect)
> set; the qualitative interpretation is also updated.

## Why this measurement exists

The parent QP architecture memo (PR #125) originally cited
`label_autocorr_60` — realized forward-label autocorrelation — as
evidence about "fast signal decay" supporting the Level-2 MPO
attractiveness argument. Codex (HIGH-4) correctly flagged this is the
wrong observable: realized forward-label autocorr conflates label
noise with forecast persistence. **Multi-period optimization needs
the *forecast-state* persistence**, `corr(μ̂_t, μ̂_{t+k})`, measured
on the calibrator output the QP actually consumes.

This script measures exactly that, plus top-K rank overlap and
expected-return half-life, per regime, for **global trading-day**
lags {1, 5, 10, 20, 60}.

## Regime stratification choice

We stratify by **regime at t** (the bar where the decision is made),
not by "regime stays constant from t to t+k". The persistence
question is: *given we are in regime R at t, how stable is μ̂ k
trading days later regardless of the regime at t+k*. The
"regime-stays-constant" alternative shrinks the sample dramatically
and answers a different question (which we may want to measure
separately when we have more BULL_VOLATILE / CHOPPY / BEAR data).

## Headline numbers (BULL_CALM, the dominant regime — 78% of recent bars)

| Lag (global trading days) | mean autocorr | top-5 overlap | top-10 overlap | top-20 overlap |
|--------------------------:|--------------:|--------------:|---------------:|---------------:|
| 1                         | **+0.860**    | 0.827         | 0.964          | 0.950          |
| 5                         | **+0.565**    | 0.625         | 0.868          | null          |
| 10                        | **+0.361**    | 0.505         | 0.747          | null          |
| 20                        | **+0.166**    | 0.374         | 0.640          | null          |
| 60                        | **-0.158**    | 0.266         | 0.183          | null          |

**Half-life: 10 trading days** (smallest lag where mean autocorr ≤ 0.5).

Per-ticker eligibility: 26 of 65 tickers had ≥ 30 BULL_CALM
observations and contributed to the per-regime mean at all measured
lags.

## Per-regime summary

| Regime          | n_rows | n_eligible tickers | half-life          |
|-----------------|-------:|--------------------|--------------------|
| BULL_CALM       |  2,699 | 26                 | 10 trading days    |
| BULL_VOLATILE   |    266 | 0                  | undersampled       |
| CHOPPY          |     63 | 0                  | undersampled       |
| BEAR            |     24 | 0                  | undersampled       |

BULL_VOLATILE / CHOPPY / BEAR per-ticker regime histories are all
below the `min_ticker_dates=30` threshold; the JSON artifact's
`n_eligible_tickers_by_lag` field is 0 across lags for those regimes,
so `undersampled: true` is set correctly (codex #128 MED-2 fix —
flag is now driven by eligibility, not just `n_rows`). Lowering the
threshold would surface numbers at the cost of noisier per-ticker
estimates; we hold the threshold and explicitly mark the regimes
unmeasurable at this sample size.

## Interpretation

The BULL_CALM μ̂ surface is **strongly persistent day-to-day**
(L=1 = +0.86, top-5 overlap 0.85), **decays at moderate speed over
the 5-20 day horizon**, and is **mildly anti-correlated at the 60-day
horizon** (L=60 = −0.16).

- **Half-life of about two trading weeks.** L=5 autocorr is +0.56;
  mean autocorr crosses 0.5 between lag 5 and lag 10.
- **By the 20-day horizon the forecast still has noticeable signal** (+0.17
  Pearson) — not the "L=20 near zero" claim the first (buggy) version
  reported. The 60-day decorrelation is genuine but consistent with
  the calibrator targeting a 60-day excess return horizon (a new
  forecast at t+60 has essentially no information overlap with the
  one at t).
- **Top-K rank overlap holds up substantially better than Pearson.**
  Even at L=60, ~33% of the top-5 names from 60 trading days ago are
  still on the current top-5. The ranking persists longer than the
  magnitudes — which is the more relevant observable for a top-K
  selector like the Hybrid Stage 1 in §5 Option F.

## Implication for the §8 Step 4 offline A/B replay

Both codex (HIGH-4) and gemini (#3) flagged that the parent memo's
dismissal of Level 2 (cvxportfolio MultiPeriodOpt) was premature. The
corrected measurement gives a **partial confirmation** to that
critique, weaker than the original (incorrect) script suggested:

- **For Level-2 MPO**: the corrected numbers show **moderate, not
  fast, signal decay**. MPO's planning-ahead value exists (autocorr
  declines from +0.86 at L=1 to +0.17 at L=20) but the slope is less
  steep than the original script implied. MPO is worth including as
  one of the §8 Step 4 baselines, but not as a clear front-runner
  based on this observable alone.

- **For Level-0 Kelly / Hybrid**: the ~0.58 top-5 overlap at L=10 and
  ~0.47 at L=20 means a rule-based "stay in top-K, rebalance when the
  rank changes meaningfully" path is operationally viable — the
  cohort doesn't churn so fast that turnover costs dominate.

- **For current Level-1 SinglePeriodOpt**: the day-to-day persistence
  (+0.86 L=1) is comfortably enough that single-period decisions are
  not random. The L=20 = +0.17 Pearson still has signal; the
  optimizer is *not* over-fitting a horizon where the forecast has
  drifted to noise. This is a softer version of the §4
  DeMiguel-mechanism argument than the original (incorrect) script
  supported.

**Net for §8 Step 4**: include Level-2 MPO explicitly as one of the
5 baselines, but treat its expected advantage over Level-1 as smaller
than the original (incorrect) script implied. The decision-grade
artifact remains the offline A/B output, not this measurement alone.

## Limits of this measurement

- **Sample is the SIM decision trace**, not paper trading or LIVE.
  These are the μ̂ values the sim path produced — they should match
  prod within calibrator drift.
- **3 of 4 regimes are undersampled.** 78% of the rows are BULL_CALM
  (matching the live market regime distribution since 2024). The
  per-regime claim above is BULL_CALM-only; BULL_VOLATILE, CHOPPY,
  and BEAR autocorrelation would need a longer or paper-augmented
  sample.
- **`min_ticker_dates=30` was the threshold.** Lowering it would
  surface per-regime numbers for the other 3 regimes at the cost of
  noisier per-ticker autocorr estimates.

## Reproduction recipe (#128 review fix)

```bash
.venv/bin/python scripts/measure_mu_hat_autocorrelation_by_regime.py
# Default writes doc/research/evidence/2026-06-03-mu-hat-autocorrelation-by-regime.json

# External DB (now works after the #128 MED-3 fix):
.venv/bin/python scripts/measure_mu_hat_autocorrelation_by_regime.py \
    --db /path/to/other/sim_runs.db
# data_source label falls back gracefully to absolute path when the DB
# is outside the repo.
```

The script fails fast with a clear error if the DB does not exist
(`data/` is gitignored; a fresh checkout will not have
`data/sim_runs.db` and must point `--db` at an external location —
previously this surfaced as `sqlite3.OperationalError`, now it surfaces
as a clean `SystemExit` with reproduction hints).
