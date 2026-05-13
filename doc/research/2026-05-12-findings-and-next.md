# 2026-05-12 — Session findings + forward research plan

Captures the structural findings from the 2026-05-12 dead-path audit +
industry-grade evaluation rebuild. Pinned here so the conditional-feature
discoveries and the prerequisites to act on them are not lost.

## Session arc (2026-05-12)

1. Built methodology infrastructure (commits `555f5b1`, `a63e133`, `ef52039`):
   - Industry-grade evaluator: paired daily returns + Newey-West HAC SE
     (statsmodels) + stationary block bootstrap (arch package) + DSR + PBO
   - Regime-stratified analyzer using SPY-derived trend × vol regime labels
   - Replaces broken 6-window mean-APY method (overlapping windows, mixed
     lengths, regime variance dominating signal). See
     `doc/research/evaluation-protocol.md`.
2. Identified 2 dead-path bugs (`555f5b1`):
   - Vol-targeting (Moskowitz-Ooi-Pedersen 2012) nested inside dormant
     Kelly path → never reached QP optimizer
   - DD-Kelly (Grossman-Zhou 1993) — same root cause
3. Identified 1 NGBoost-class bug (`7bc9b56`):
   - QP's `_BuildMuVectorTask` fallback chain: NGB OFF → panel_score
     (z-score ±2 scale), NGB ON → NGBoost μ (~1e-3 scale). The QP's
     λ_risk and tax weights are calibrated to panel_score scale, so
     swapping signal source breaks the optimizer's risk-return tradeoff.
   - Implemented two fixes (both behind feature flags, OFF by default):
     - Option A: `ForceMuSourceTask` (use panel_score even with NGB on)
     - Option C: `ApplyGrinoldKahnTransformTask` (μ = IC × σ × z(score))
4. Built 4-config × 8-window non-overlapping evaluation panel:
   - Baseline / vt15 / GK094 / GK15 each across 8 × 3-month windows
   - 496 daily paired observations (vs 6 noisy APY estimates in old method)

## Verdicts under industry-grade method

| Config | Pooled t | mean Δ ann | 95% CI | Cohen's d | Cons | Tier |
|---|---:|---:|---|---:|---:|---|
| baseline ≡ baseline (sanity) | +0.00 | +0.00% | [0,0] | 0.00 | 0/8 | NEITHER ✓ |
| vt15 vs baseline | +0.75 | +0.84% | [−1.0, +3.4] | +0.03 | 2/8 | NEITHER |
| GK094 vs baseline | +0.50 | +2.66% | [−7.6, +13.0] | +0.02 | 6/8 | NEITHER |
| GK15 vs baseline | (pending) | | | | | |

**Key:** all 3 candidates pooled to NEITHER. The OLD framework said all
3 were "REJECT". Both verdicts are now superseded by the regime-conditional
finding below.

## Major finding: GK094 is regime-conditional (commit `ef52039`)

Stratifying 496 paired daily Δ by SPY-derived regime (60d Sharpe × 20d vol
percentile) reveals **non-uniform structure**:

| Regime (SPY 60d Sharpe / 20d vol pct) | n | mean Δ ann | t | p | d |
|---|---:|---:|---:|---:|---:|
| **HIGH_CALM** (strong trend, calm vol) | 123 | **+17.90%** | **+1.67** | 0.094 | +0.13 |
| **HIGH_SPIKED** (strong trend, vol spikes) | 53 | **−31.82%** | **−1.95** | 0.051 | −0.26 |
| LOW_NORMAL | 37 | +29.66% | +1.31 | 0.191 | +0.22 |
| MED_SPIKED | 60 | +6.48% | +0.32 | 0.748 | +0.04 |
| LOW_SPIKED | 79 | +0.32% | +0.04 | 0.969 | +0.00 |
| HIGH_NORMAL | 52 | −1.30% | −0.08 | 0.937 | −0.01 |
| MED_NORMAL | 46 | +1.20% | +0.10 | 0.922 | +0.01 |
| MED_CALM | 41 | −13.44% | −0.83 | 0.405 | −0.13 |
| LOW_CALM | 5 | (skipped, n<8) | | | |

**Mechanism interpretation:**

- GK094 = `μ_QP = IC × σ × z(panel_score)`. The z-score normalization
  compresses extreme panel_score values, producing a flatter μ distribution
  across candidates. Net effect: the QP optimizer takes LESS-concentrated
  bets on top-ranked names.
- In **HIGH_CALM** (smooth uptrend), risk-controlled bet sizing wins.
  Concentrated bets bring drawdown without proportional upside.
- In **HIGH_SPIKED** (uptrend interrupted by sharp drawdowns),
  diversified-across-top-N exposure gets caught when the spike comes.
  Concentrated baseline bets recover faster because the leaders lead
  the rebound.
- This is the trend-momentum literature pattern (Moskowitz-Ooi-Pedersen
  2012, Asness-Moskowitz-Pedersen 2013): factor returns depend on
  realized trend/vol regime.

**Why this is NOT yet a Tier 3 conditional-deployment promotion:**

1. **Borderline p-values.** t=+1.67 (p=0.09) HIGH_CALM, t=−1.95 (p=0.05)
   HIGH_SPIKED. With 9 regimes tested, Bonferroni-Holm adjusted p needs
   < 0.0056 to claim significance. We're 10-20× above that bar.
2. **Sample size.** n=53 in HIGH_SPIKED → barely enough for HAC SE
   convergence. Need n ≥ 200 per regime for Tier 3 power.
3. **Regime detector unreliability.** The strategy's internal regime
   detector labels 95% of these 496 days BULL_CALM regardless of true
   conditions. Even if we know HIGH_SPIKED is bad for GK094, prod can't
   tell when it's in HIGH_SPIKED.

## Blockers to actionable conditional deployment

### Blocker 1: regime detector too sticky (P0)

Current GMM-based regime detector (`kernel/pipeline/job_regime.py`)
labels every day in our 24-month OOS as BULL_CALM. The SPY-derived
labels in our analyzer (objective 60d Sharpe + 20d vol percentile) show
clear HIGH_SPIKED periods (2024-08 carry-trade unwind, 2024-12 Fed
hawkish pivot, 2025-04 tariff shock). The internal detector misses
all of these.

**Fix:** add SPY-derived trend/vol features to the regime decision.
Either:
- (a) Augment GMM with these features (retrain regime model)
- (b) Add a parallel `SpyRegimeLabelTask` that writes `ctx.spy_regime`
  alongside `ctx.regime`; downstream tasks read the appropriate one
- (c) Replace GMM regime detector entirely with rule-based SPY labels

Estimated effort: 4-6 hours including walkforward retrain of GMM.

### Blocker 2: 24-mo OOS too short for regime stratification (P0)

To get n ≥ 200 per regime cell (needed for HAC SE convergence + Bonferroni
correction at α=0.05 / 9 cells), we need ~36-48 months OOS. Current
walkforward manifest starts 2024-01-01 (constrained by fwd_60d label
requirements + first cutoff date).

**Fix:** extend training data + walkforward manifest back to 2022-01-01,
re-train 39 → ~50-60 walkforward heads. This is ~4 hours of compute
(panel rebuild + LGBM/XGB retrain at each cutoff).

### Blocker 3: regime-conditional ranking config doesn't exist yet (P1)

Current architecture: `regime_params` block supports per-regime
risk-knob overrides (max_position_pct, stop_loss_pct, drawdown_halt_pct).
Ranking config (`ranking.alpha_to_mu`, `ranking.kelly_sizing`,
`exposure_scaling`) is currently GLOBAL.

**Fix:** add `regime_overrides` sub-block to ranking sections that
the new tasks (ApplyGrinoldKahnTransformTask, ApplyExposureScalingTask)
read from before applying defaults. ~2-3 hours implementation + tests.

## Forward research plan

Priority order (gated by blockers above):

| # | Task | Effort | Unlocks |
|---|---|---|---|
| **P0-A** | Fix sticky regime detector — add SPY-derived signals or replace GMM | 4-6h | conditional deployment possible |
| **P0-B** | Extend walkforward manifest 2022-2024 → 36-48mo OOS | 4h compute | regime power n≥200 per cell |
| **P1** | Regime-conditional ranking config (`ranking.X.regime_overrides`) | 2-3h | architectural clean path |
| **P2** | Re-evaluate GK094 + vt15 + GK15 on extended OOS, stratified | 1 day | real Tier 3 verdict on conditional |
| **P3** | If GK conditional wins survive, ship GK-in-HIGH_CALM live config | 1-2h | first regime-conditional production feature |

Per CLAUDE.md §5.13.10: do NOT add the `regime_overrides` config block
until the regime detector is fixed; otherwise it's dead code by definition.
Order is P0-A → P0-B → P1 → P2 → P3.

## Files / commits this session

- `doc/research/evaluation-protocol.md` — industry-grade protocol spec
- `doc/AUDIT_2026-05-12_dead_paths.md` — vol-target + DD-Kelly + NGBoost audits
- `scripts/eval_paired_returns.py` — pooled paired-daily analyzer
- `scripts/eval_regime_stratified.py` — regime-conditional analyzer
- `kernel/metrics/hac_se.py` — Newey-West (statsmodels wrapper)
- `kernel/metrics/block_bootstrap.py` — Politis-Romano (arch wrapper)
- `kernel/portfolio_qp/tasks.py` — ApplyExposureScalingTask, ApplyGrinoldKahnTransformTask, ForceMuSourceTask
- Commits: `555f5b1` `7bc9b56` `8e87a48` `ef52039` `a63e133`

## What we shipped to prod TODAY

**Nothing.** Production baseline unchanged. All new feature flags default
to OFF.

This is the correct outcome — premature deployment of borderline signals
is exactly what the 3-tier framework + DSR + PBO are designed to prevent.
