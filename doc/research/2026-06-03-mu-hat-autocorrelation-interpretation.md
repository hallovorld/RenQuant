# μ̂ autocorrelation per regime — measurement results

**Date**: 2026-06-03
**Status**: §8 Step 2 of [`2026-06-02-qp-architecture-review-and-alternatives.md`](2026-06-02-qp-architecture-review-and-alternatives.md). Closes the codex #125 HIGH-4 finding.
**Author**: Claude
**Inputs**: `data/sim_runs.db::score_distribution` (3,052 (date, ticker, μ̂, regime) rows, 540 dates, 65 tickers, 2024-01-02 → 2026-03-27)
**Outputs**: [`evidence/2026-06-03-mu-hat-autocorrelation-by-regime.json`](evidence/2026-06-03-mu-hat-autocorrelation-by-regime.json) (canonical evidence artifact)
**Script**: [`scripts/measure_mu_hat_autocorrelation_by_regime.py`](../../scripts/measure_mu_hat_autocorrelation_by_regime.py)

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
expected-return half-life, per regime, for trading-day lags
{1, 5, 10, 20, 60}.

## Headline numbers

| Regime         | n_rows | L=1     | L=5     | L=10    | L=20    | L=60    | Half-life |
|----------------|--------|---------|---------|---------|---------|---------|-----------|
| BULL_CALM      | 2,699  | +0.783  | +0.352  | +0.145  | −0.046  | −0.078  | **5 d**   |
| BULL_VOLATILE  |   266  | —       | —       | —       | —       | —       | (insufficient per-ticker history) |
| CHOPPY         |    63  | —       | —       | —       | —       | —       | (undersampled) |
| BEAR           |    24  | —       | —       | —       | —       | —       | (undersampled) |

(BULL_VOLATILE / CHOPPY / BEAR per-regime ticker histories are too
short under the default `min_ticker_dates=30` threshold; lowering it
to ~10 would surface a number but with wide standard error. We
explicitly flag those as undersampled in the JSON and do not draw a
quantitative conclusion from them.)

## Top-K rank overlap (BULL_CALM, the dominant regime)

| K  | L=1   | L=5   | L=10  | L=20  | L=60  |
|----|-------|-------|-------|-------|-------|
|  5 | 0.83  | 0.63  | 0.53  | 0.37  | 0.28  |
| 10 | 0.96  | 0.86  | 0.76  | 0.58  | 0.38  |
| 20 | 0.95  | —     | —     | —     | —     |

(K=20 only has enough valid pairs at L=1 — most bars don't have 20+
candidates with valid μ̂.)

## Interpretation

The BULL_CALM μ̂ surface is **persistent day-to-day** (L=1 = +0.78,
top-5 overlap 0.83) but **decays fast on the multi-day timescale**:

- **Half-life of about a week.** L=5 autocorr is +0.35; mean-autocorr
  crosses 0.5 between lag 1 and lag 5.
- **By 2 weeks (L=10) the marginal forecast value is small** (+0.14
  Pearson), and by a month (L=20) the forecast is essentially
  uncorrelated with its earlier self.
- **Top-K rank overlap holds up better than Pearson** — even at L=60,
  ~28% of the top-5 names from a month-and-a-half ago are still on
  the current top-5. The ranking persists longer than the magnitudes.

## Implication for the §8 Step 4 offline A/B replay

Both codex (HIGH-4) and gemini (#3) flagged that the parent memo's
dismissal of Level 2 (cvxportfolio MultiPeriodOpt) was premature.
This measurement narrows down the question:

- **For Level-2 MPO**: fast Pearson decay (half-life ~5 days, L=20
  near zero) is exactly the regime MPO mathematically exploits —
  spreading Δw over multiple bars while alpha decays. Boyd-cvxportfolio
  is explicit that MPO's value is in alpha-decay × friction balance.
  **BULL_CALM may be a stronger MPO use case than the parent memo
  claimed**.

- **For Level-0 Kelly / Hybrid**: persistent top-K rank (overlap 0.5
  at L=10 trading days) means a rule-based "stay in top-K, update
  weights when the rank changes" path is operationally viable — the
  cohort doesn't churn so fast that turnover costs dominate.

- **For current Level-1 SinglePeriodOpt**: the day-to-day persistence
  (+0.78 L=1) is *enough* that single-period decisions are not
  random, but the L=20 near-zero autocorr suggests the optimizer is
  over-fitting the *current* μ̂ to a horizon where the forecast has
  already drifted. This is consistent with the §4 DeMiguel-mechanism
  argument that estimation error dominates the optimization gain.

**The offline A/B replay (§8 Step 4) should include Level-2 MPO
explicitly as one of the 5 baselines** — the autocorr profile makes
it a real candidate, not (as the original parent memo wrote) a
"defensive infrastructure" addition.

## Limits of this measurement

- **Sample is the SIM decision trace**, not paper trading or LIVE.
  These are the μ̂ values the sim path produced — they should match
  prod within calibrator drift.
- **3 of 4 regimes are undersampled.** 78% of the rows are BULL_CALM
  (matching the live market regime distribution since 2024). The
  per-regime claim above is BULL_CALM-only; BULL_VOLATILE, CHOPPY,
  and BEAR autocorrelation would need a longer or paper-augmented
  sample. This is consistent with the regime-stratified diagnostic
  ([`2026-06-02-bull-calm-no-signal-diagnostic.md`](2026-06-02-bull-calm-no-signal-diagnostic.md))
  finding: 78% of recent bars are BULL_CALM, and that is the only
  regime where we currently have enough rows for stable per-regime
  inference of any kind.
- **`min_ticker_dates=30` was the threshold.** Lowering it would
  surface per-regime numbers for the other 3 regimes at the cost of
  noisier per-ticker autocorr estimates. We can re-run with a lower
  threshold if codex thinks the trade-off is worth it.

## Reproduction recipe

```bash
.venv/bin/python scripts/measure_mu_hat_autocorrelation_by_regime.py
# Writes doc/research/evidence/2026-06-03-mu-hat-autocorrelation-by-regime.json
# Console summary as in the table above.
```

The script reads `data/sim_runs.db::score_distribution`. Pass `--db
<other-db>` to point at a paper-broker or alpaca decision-trace DB
once we have those tables populated; the schema is the same.
