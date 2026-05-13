# Pre-registration: Track 6 — Multi-horizon ensemble retest

**Date**: 2026-05-13
**Pre-registered BEFORE experiment.**

## Hypothesis

**H0**: Single fwd_60d-trained model performs equal or better than
ensemble of fwd_5d / fwd_20d / fwd_60d trained models on 16-window
paired-daily panel.

**H1**: Multi-horizon ensemble (mean of 3 model scores) yields
**+1-3pt mean ΔAPY** with t_pool > 1.5 vs fwd_60d baseline.

## Theoretical basis

López de Prado *AFML* §17 (meta-labeling, multi-resolution): combining
predictions across horizons captures complementary information.
fwd_5d captures short-term reversal; fwd_20d captures momentum;
fwd_60d captures longer-term mean-reversion-to-fundamentals.

Empirically (Qlib benchmarks):
- Multi-horizon ensembles: +5-15% IC over single horizon
- AQR factor research uses multi-horizon stacking by default

**Prior work**: E42 single-window test showed fwd_60d single-window
+3.3pt vs other horizons. **But that was on broken 6-window methodology.**
Under proper 16-window paired analysis, retest is needed.

## Implementation plan

1. **Build 3 horizon-specific configs** (~10 min): use existing artifacts
   `panel-ltr.fwd5d.json`, `panel-ltr.fwd20d.json`, `panel-ltr.fwd60d.json`
   (already trained, in `artifacts/sim/E42_walkforward_horizons/`)
2. **Ensemble task** (~1h code): new `EnsembleScoresTask` that averages
   scores from 3 panel-LTR models, behind config flag
3. **Side configs**: `sim_horizon_5d_ext.json`, `sim_horizon_20d_ext.json`,
   `sim_horizon_60d_ext.json`, `sim_horizon_ensemble_ext.json`
4. **Smoke test** (~17 min): 1 window Q08 each variant
5. **Full panel** (~70 min): 16-window batch × 4 configs = 64 sims
6. **Analysis** (~10 min)

**Compute: ~3-4h**

## Pre-committed evaluation criteria

Standard `doc/research/evaluation-protocol.md` Tier criteria:
- Tier 1 REJECT: t_pool < −1.0 OR mean_ann < −2% OR cons < 40%
- Tier 2 SCREEN: t_pool > 1.5, cons ≥ 60%, CI_lo > 0, d > 0.20
- Tier 3 PROMOTE: + t_pool > 3.0 + DSR > 0.5 + p < 0.01 + |d| > 0.50

K_trials = 9 (5 prior candidates + 4 horizon variants in this batch)

## Auto-promote target if Tier 3

Update `strategy_config.golden.json`:
- Set `ranking.ensemble.enabled = true`
- Set `ranking.ensemble.horizons = ["5d","20d","60d"]`
- Set `ranking.ensemble.weights = [0.33, 0.33, 0.34]` (or fitted weights)
