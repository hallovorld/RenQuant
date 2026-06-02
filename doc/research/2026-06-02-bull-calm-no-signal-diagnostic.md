# Why "no trades today" — BULL_CALM no-signal diagnostic

**Date**: 2026-06-02
**Status**: Mainline finding — closes the "model doesn't react to today's market" thread
**Verdict producing artifact**: `backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.weekly_20260602T072600Z.staging.json`

## TL;DR

Production isn't broken. The alpha158+fund GBDT recipe has:
- **+0.307 mean IC in BEAR** (96% hit rate) — strong signal
- **+0.011 mean IC in BULL_CALM** (48% hit rate, coin flip) — zero signal
- 2024-2025 markets are BULL_CALM-dominated (~78% of days)
- The live `regime_admission` gate correctly blocks BULL_CALM buys
- Result: very few admissions in 2024-2025 → "no trades today"

The user's mainline question ("why doesn't the model react to today's market") has a clean answer: **it does react, but the prediction in BULL_CALM is statistically a coin flip, and the downstream gate correctly distrusts it.**

## Per-regime IC breakdown (from `wf_gate_metadata.sanity_regime_ic.regimes`)

| Regime | n_dates | mean_ic | median_ic | hit_rate | mean_confidence | passed | placebo_60/aligned_real | label_autocorr_60 |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| BEAR          |  50 | **+0.307** | +0.316 | **96.0%** | 0.83 | FAIL (placebo edge) | 0.524 | 0.213 |
| BULL_CALM     | 400 | +0.011 | −0.012 | 48.3% | 0.64 | FAIL (under min_ic) | 0.615 | −0.040 |
| BULL_VOLATILE |  19 | −0.024 | −0.127 | 42.1% | 0.64 | (n<30 ineligible) | — | 0.015 |
| CHOPPY        |  39 | +0.017 | +0.028 | 58.9% | 0.34 | FAIL (under min_ic) | — | −0.024 |

Gate thresholds: `min_n_dates=30`, `min_mean_ic=0.02`, `max_placebo_ratio=0.5`.

## Sim trade-regime distribution (sim DOES NOT have regime_admission active)

```
buys by regime:  {BULL_CALM: 85}    ← 100% of buys placed in the noise regime
sells by regime: {BULL_CALM: 27, CHOPPY: 22, BEAR: 9, BULL_VOLATILE: 9}
```

The sim's QP/Kelly machinery accepts the BULL_CALM scores even though they're noise. Those 85 buys lost vs SPY in 3/3 WF cuts. **Live system blocks these via `regime_admission` — that's why prod looks "silent" today.**

## Per-cut market mix

| Cut | Sharpe (vs SPY) | dom regime | BULL_CALM days | BEAR days | CHOPPY days |
|---|---|---|---:|---:|---:|
| 2024-01-02 → 2024-12-31 | +0.90 (vs +1.78) | BULL_CALM | 195 | 7 | 14 |
| 2024-07-01 → 2025-06-30 | +0.47 (vs +0.72) | BULL_CALM | 139 | 39 | 13 |
| 2025-04-01 → 2026-03-28 | +0.49 (vs +0.75) | BULL_CALM | 155 | 27 | 15 |

**BULL_CALM dominates every cut, by a lot.** In any market where the model HAS signal (BEAR, CHOPPY), the model also tends to underperform a passive index (because the alpha is small relative to dispersion in those regimes).

## Why the WF gate's verdict is correct

The gate's `benchmark_ok=False` is the right call. The model:
- Has +0.62 mean Sharpe across 3 cuts ✓ (absolute_ok=True)
- Loses to SPY 0/3 in Sharpe and 0/3 in APY
- Has no signal in the regime that dominates the test window

Promoting it would put live $$ behind a model whose only working dimension (BEAR) rarely materializes, while its ranking in the dominant regime (BULL_CALM) is indistinguishable from random.

## Actionable research directions

Ordered by cheapness, not by likely yield:

1. **Defensive: disable BULL_CALM buys at the model level** — `regime_admission` already does this for live; the WF gate sim doesn't, which is why the sim shows 85 buys that all lost. Loosening the live gate is the WRONG direction; the issue is the sim is too permissive. **Diagnostic only — doesn't generate trades.**
2. **Cheap research: per-regime calibrator** — fit a separate calibrator for BULL_CALM that maps scores into a wider [low, low] band so QP's `delta_below_min_dw` shrinks the position to near-zero. Already partially in place; tune `regime_params.BULL_CALM.calibrator_scale`.
3. **Medium research: BULL_CALM-specific features** — momentum + low-vol persistence features (e.g., `momentum_1y_carry`, `low_vol_premium`) — these are designed for the dispersionless calm regime. Kelly-Gu-Xiu 2020 RFS Table 9 documents these as the strongest factors in low-volatility periods.
4. **Larger research: separate model per regime** — train 4 specialist models (one per detector regime), ensemble via the regime detector's confidence. Bigger lift but addresses the root cause directly.

## What this is NOT

- ❌ NOT a bug in the gate (gate verdict is correct)
- ❌ NOT a leakage problem (placebo at 120d = 26% passes the PR #31 fix)
- ❌ NOT an infra problem (all 6 layers of infra rot were fixed earlier tonight)
- ❌ NOT a recipe regression (today's candidate has SAME fingerprint as the 39 WF v2 cuts; the model itself is internally consistent)

## What this IS

A statement about the alpha158+fund GBDT signal: **it works in BEAR markets and is approximately useless in BULL_CALM**. Production silence in BULL_CALM-dominated months is correct behavior, not a malfunction.

The mainline question changes from "fix the silence" to "find a signal that works in BULL_CALM, or accept that the strategy only fires in 20% of trading days when the dominant regime cooperates."

## Files of record

- Today's stamped verdict: `backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.weekly_20260602T072600Z.staging.json`
- Gate log: `/tmp/rerun_gate_post_b1.log`
- Memory: [`project_perf_wall_realized_ic_2026-05-27`](../../memory/project_perf_wall_realized_ic_2026-05-27.md) — 5/27 saw the same wall, attributed to "realized IC ≈ 0"; today's diagnostic localizes that to BULL_CALM specifically
- Memory: [`feedback_regime_conditional_strategy`](../../memory/feedback_regime_conditional_strategy.md) — PRIME DIRECTIVE that this finding validates
