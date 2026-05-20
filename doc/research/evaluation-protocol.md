# Industry-Leading Strategy Evaluation Protocol (2026-05-12)

Mandatory protocol for comparing any candidate strategy/config against
the production baseline. Replaces the prior 6-window-mean-APY method
which the 2026-05-12 audit showed was statistically underpowered and
methodologically flawed (overlapping windows, mixed lengths, regime
variance dominating signal variance).

## What was wrong with the previous method

1. **Mixed window lengths** — W3 was 12 months, others 4 months. APY
   annualization amplifies 4mo noise 3× more than 12mo. Mixing in one
   mean = comparing apples to oranges.
2. **Heavy window overlap** — W3 ⊇ W2 ∪ W6 ∪ half of W1. Same calendar
   period counted multiple times → "n=6" was effectively n≈2-3.
3. **Regime variance hides strategy variance** — baseline alone ranged
   −19.8% (W6) to +91.0% (W4). With per-window σ ≈ 25pt and n=6, the
   minimum detectable effect at 80% power is ~20pt. Any candidate
   showing ±5pt mean Δ is in noise. Forbidden under §5.13.4.
4. **No paired-bar comparison** — every config was compared as
   independent APY estimates, ignoring that baseline and candidate
   trade the SAME daily price moves. A paired comparison removes
   market noise.
5. **No HAC SE / no bootstrap** — naive t-stats assume i.i.d. returns
   which is wrong for autocorrelated portfolio P&L.
6. **No multiple-comparison correction at the right K** — K_trials in
   DSR was based on per-batch counts, but the cumulative across batches
   over the session is much larger.

## Canonical references this protocol follows

- **Lopez de Prado 2018** *Advances in Financial Machine Learning*
  (Wiley) — Chapter 11 "Backtesting Through Cross-Validation"; Chapter
  14 "Backtest Statistics"; Chapter 15 "Understanding Strategy Risk".
- **Bailey & López de Prado 2014** "The Deflated Sharpe Ratio" *J.
  Portfolio Mgmt* 40(5):94 — DSR formula.
- **Bailey, Borwein, López de Prado & Zhu 2015** "The Probability of
  Backtest Overfitting" *J. Computational Finance* 20(4):39 — PBO via
  CSCV.
- **Newey & West 1987** "A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix"
  *Econometrica* 55(3):703 — HAC SE.
- **Politis & Romano 1994** "The stationary bootstrap" *J. Amer.
  Statistical Assoc.* 89(428):1303 — block bootstrap.
- **Politis-Romano-Wolf 2008** "Subsampling for nonstationary time
  series with a network of randomly chosen times" *J. Empirical Fin.*
  15:319 — bootstrap for Sharpe.
- **Harvey, Liu & Zhu 2016** "...and the Cross-Section of Expected
  Returns" *Rev. Financial Studies* 29(1):5 — multiple testing.
- **Cohen 1988** *Statistical Power Analysis for the Behavioral
  Sciences* 2e — effect size (Cohen's d).
- **Andrews 1991** "Heteroskedasticity and autocorrelation consistent
  covariance matrix estimation" *Econometrica* 59:817 — Newey-West
  lag selection: L = floor(4 × (n/100)^(2/9)).

## Window panel (mandatory)

**8 non-overlapping 3-month windows** spanning 2024-04-01 → 2026-04-01
(24 months total). Constraints:
- 4-month windows are too short for HAC SE convergence at typical
  daily-return autocorrelation (lag ≈ 5-15 days); 3-month gives
  ~63 trading days per window, just above the threshold.
- 24-month total = 504 trading days = enough cross-window pooled n
  for ~5pt detection floor (Cohen's d=0.5 at α=0.05, power=0.80).

Future expansion to 12 or 16 windows requires extending the OOS span
back to 2023 (constrained by walkforward manifest's first cutoff
2024-01-01 + 60-day label lookahead).

| Win | Start | End |
|---|---|---|
| P1 | 2024-04-01 | 2024-07-01 |
| P2 | 2024-07-01 | 2024-10-01 |
| P3 | 2024-10-01 | 2025-01-01 |
| P4 | 2025-01-01 | 2025-04-01 |
| P5 | 2025-04-01 | 2025-07-01 |
| P6 | 2025-07-01 | 2025-10-01 |
| P7 | 2025-10-01 | 2026-01-01 |
| P8 | 2026-01-01 | 2026-04-01 |

## Comparison protocol

For each candidate config, run BOTH baseline and candidate over each of
the 8 windows. Emit daily equity curves via `--equity-json`.

**Per-window analysis:**
1. Daily log-return: r_t = log(NAV_t / NAV_{t-1})
2. Paired daily delta: d_t = r_candidate[t] - r_baseline[t]
3. Mean Δ: μ_d = Σ d_t / n_days
4. **Newey-West HAC SE** with Andrews 1991 lag:
   - L = floor(4 × (n/100)^(2/9))
   - SE_NW = √(γ_0 + 2 × Σ_{k=1..L} (1 - k/(L+1)) × γ_k)
   - γ_k = sample autocovariance at lag k
5. t-statistic: t_window = μ_d / SE_NW
6. Annualized Sharpe of Δ: SR_d = (μ_d / σ_d) × √252
7. **Probabilistic Sharpe Ratio** (Bailey-LdP 2012): PSR = Φ((SR_d × √(n-1)) / √(1 - skew × SR_d + ((kurt-1)/4) × SR_d²))

**Cross-window aggregation:**
1. Pooled n = Σ n_days across 8 windows
2. Pooled mean Δ: μ_pool = Σ (μ_d_i × n_days_i) / Σ n_days_i (sample-size weighted)
3. Pooled SE_NW: HAC on the concatenated daily series (preserves
   within-window autocorrelation; between-window blocks are
   independent by construction)
4. **Pooled t**: t_pool = μ_pool / SE_NW_pool
5. **Stationary block bootstrap** (Politis-Romano 1994):
   - Block length = optimal under Politis-White 2004
   - B = 5000 bootstrap samples
   - Empirical 95% CI on μ_pool
6. Cohen's d: d = μ_pool / σ_pool (effect size)

**Multiple-comparison correction:**
- K_trials = cumulative number of variants tested across sessions; must
  be tracked (use `runs.db::training_runs` row count as proxy, or `mlruns/`
  if MLflow is wired). DO NOT hardcode a specific count.
- DSR computed with this K (Bailey-López de Prado 2014)
- PBO via CSCV (Bailey-Borwein-LdP-Zhu 2015) as complement
- Bonferroni-Holm at α=0.01 OR BH-FDR at q=0.05

## Promotion tiers (rewritten)

### Tier 1 — REJECT (hard)

ANY of:
- t_pool < -1.0 (clear loss signal)
- μ_pool × 252 < -2% (annualized > 2pt mean loss)
- > 60% of windows have negative t_window

### Tier 2 — SCREEN (NOT live-promotable)

ALL of:
- t_pool > 1.5
- > 60% of windows positive (t_window > 0)
- Cohen's d > 0.20 (small-to-medium effect)
- Bootstrap 95% CI lower bound on μ_pool > 0
- Δα-SPY ≥ 0 (does not lose ground vs passive benchmark)

→ Soft candidate. Run multi-seed expansion + tighter sensitivity sweep.
NOT eligible for live config flip.

### Tier 3 — CONFIRMED (live-promotable)

Tier 2 + ANY of (per `doc/research/promotion-methodology.md` authoritative spec, CLAUDE.md §5.13.4a):
- **DSR > 0.5** (probability true SR > 0 exceeds 50% AFTER selection-bias correction with K_trials, Bailey-López de Prado 2014)
- **PBO < 0.5** (probability of overfitting below random, Bailey-Borwein-LdP-Zhu 2015 CSCV)
- **n ≥ 30 AND t > 3.0** (large-sample t-test, Harvey-Liu-Zhu 2016 multi-test threshold)

→ Live-promotable. Flip prod config in same commit; pin via regression test.

(Earlier draft of this section required ALL three conditions with AND — that
was inconsistent with `promotion-methodology.md` which is the canonical source.
Reconciled 2026-05-20 to OR per CLAUDE.md ground truth.)

**Also recommended (Tier 2+ soft requirements)**:
- Newey-West p-value < 0.01
- Pre-registered hypothesis match
- |Cohen's d| > 0.50

---

## Walk-forward cuts (RenQuant-specific protocol)

In addition to the generic ML walk-forward protocol above (8 non-overlapping 3-month windows for paired daily comparison), RenQuant uses 5 named cuts in `kernel/walk_forward_splits.py::build_default_cuts()` for IC walk-forward evaluation:

| Cut | Val window | Regime intent |
|---|---|---|
| `cut1_covid` | 2020-01-01 → 2020-04-30 | BEAR-heavy / COVID crash |
| `cut2_fed` | 2022-01-01 → 2022-04-30 | Fed rate hike tech selloff (trending) |
| `cut3_inflpk` | 2022-10-01 → 2022-12-31 | Inflation peak (transitional) |
| `cut4_svb` | 2023-01-01 → 2023-04-30 | SVB / regional bank shock |
| `cut5_unwind` | 2024-06-01 → 2024-09-30 | Late-cycle unwind |

PRIME DIRECTIVE compliance: each val window deliberately contains at least one SPIKED regime. Models that win in BULL_CALM only will fail at least one cut. Min-regime IC across these cuts (not pooled mean) is the canonical selection metric in the HF Trainer refactor (`scripts/patchtst_hf.py::PerRegimeICCallback`).

## Pre-registration (mandatory)

BEFORE running any new variant batch, write a brief pre-registration
document at `doc/experiments/pre-reg/<date>_<name>.md`:

1. **Hypothesis**: what theoretical reason predicts this works?
2. **Predicted direction**: ΔAPY positive/negative? Magnitude range?
3. **Decision rule**: under what criteria do we promote? reject?
4. **Pre-committed K_trials**: how many config variants WILL be run?

Per Nosek-2015 *Royal Society Open Science* "Promoting an open
research culture" — distinguishes confirmatory from exploratory work.

## Implementation

- `scripts/eval_paired_returns.py` — paired daily-returns analyzer,
  reads `--equity-json` outputs, emits per-window + pooled stats.
- `scripts/run_sim_104.py --equity-json PATH` — emits daily NAV JSON.
- `kernel/metrics/hac_se.py` — Newey-West HAC SE.
- `kernel/metrics/block_bootstrap.py` — Politis-Romano stationary
  bootstrap.
- `kernel/metrics/deflated_sharpe.py` — DSR (already exists).
- `kernel/metrics/pbo.py` — PBO via CSCV (already exists).

## What gets re-evaluated

All variant candidates from the 2026-05-12 session that the broken
analyzer surfaced as "interesting" must be re-run under this protocol:

1. **Baseline** (lambda_000) — establish reference
2. **vt15** (E43 vol-target=0.15) — appeared to win at W1, "rejected"
   by broken methodology; needs proper test
3. **GK IC=0.094** (Grinold-Kahn) — best W1 result of session
4. **GK IC=0.05, 0.15** — sensitivity sweep
5. **vt10, vt20** (vol-target sweep) — completeness

These re-runs use the 8 non-overlapping windows AND emit daily equity
JSON. The prior 36-config TIER1_REJECT verdicts from the broken
analyzer are now flagged as "INCONCLUSIVE under old method" and not
acted on without rerun.
