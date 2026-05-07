# Regime classifier mis-fire analysis (CHOPPY under-firing)

**Date**: 2026-05-02
**Script**: `scripts/diagnose_regime_classifier.py`
**Report data**: `data/audit/regime_diagnosis_<run>.json`
**Status**: Finding only. No production change yet — proposed tuning needs user review.

---

## Headline finding

The current classifier (Hurst + CUSUM + GMM) **massively under-fires CHOPPY** relative to a forward-looking heuristic ground truth:

| | Classifier | Ground truth (60d-fwd heuristic) |
|---|---:|---:|
| BULL_CALM     | 79.7% | 39.3% |
| BULL_VOLATILE | 12.1% |  6.4% |
| BEAR          |  7.5% | 14.5% |
| **CHOPPY**    | **0.8%** | **39.8%** |

Overall agreement = 34.7% (n=1560 days, 2020-01-02 → 2026-04-30).

**Most common single mis-classification: BULL_CALM → CHOPPY (514 days, ~33% of all trading days)** — the classifier calls BULL_CALM in periods that, in retrospect, were sideways/range-bound markets.

CHOPPY episodes when they DO fire: only 3 episodes total in 6 years, average 4 days, max 5 days. Functionally CHOPPY is "never on" in the live system.

---

## Why this matters (macro impact)

Conformal Gate B (M3, fitted 2026-05-01) thresholds are regime-conditional:
- BULL_CALM: τ = 0.090 (n=71,805 historical candidates)
- CHOPPY: τ = 0.020 (n=1,700)
- BULL_VOLATILE / BEAR: insufficient samples → fall back to static

If 33% of trading days are *actually* CHOPPY but called BULL_CALM, **Gate B uses τ=0.090 (high bar) when it should use τ=0.020 (low bar)** for a third of the year. This systematically blocks candidates whose edge-Sharpe is between 0.02 and 0.09 — most CHOPPY-appropriate trades.

Estimated macro impact: hard to quantify without an A/B, but if even 10% of those blocked candidates were profitable, the alpha cost is meaningful (multi-percent APY).

---

## Why CHOPPY under-fires — mechanism

Classifier resolution order (`detect_regime`, kernel/regime.py:443):
1. **Hard BEAR override** if vol > 35% or 20d-return < −8% → BEAR.
2. **GMM BEAR** if `gmm_probs[BEAR] > 0.5` → BEAR.
3. **Hurst MOMENTUM** (Hurst > 0.65) → BULL_CALM.
4. **Hurst REVERSION** (Hurst < 0.52) → **CHOPPY**. ← only firing path
5. **Else** → `dominant_gmm` (or BULL_VOLATILE if dominant=BEAR).

The only path to CHOPPY is step 4: Hurst < 0.52. In sideways markets Hurst is typically ~0.50 (random walk) — *near* 0.52 but not reliably below it. Step 4 misses → falls through to step 5 (GMM-driven), which gravitates to BULL_CALM (most-common training regime).

**Root cause: Hurst threshold is too strict to catch real-world choppy markets.** A Hurst of 0.50 (textbook random walk) doesn't trigger a < 0.52 cutoff because the rolling-window estimator is noisy.

---

## Three tuning experiments (in priority order, none shipped yet)

### Experiment A — relax Hurst reversion threshold

```yaml
regime.hurst_reversion_threshold:  0.52 → 0.55
```

Predicted effect: ~20-30% more CHOPPY days, classification distribution moves toward ground truth. Risk: may over-fire CHOPPY into mild bull periods. Tunable via A/B.

**Sanity check before shipping**: re-run diagnostic with new threshold; verify CHOPPY% lands somewhere between 5% and 25% (not jumping to 80%). Verify classifier-vs-ground agreement improves.

### Experiment B — add explicit range-bound detector

When neither Hurst MOMENTUM nor BEAR fires, AND SPY 20d cumulative return is in [-2%, +2%] AND vol is in [10%, 25%], directly classify CHOPPY (skip GMM fallback).

Predicted effect: catches sideways markets that Hurst's noisy window misses. Lower false-positive risk than (A) because the cumulative-return band is well-defined.

### Experiment C — retrain GMM to penalize BULL_CALM bias

Current GMM artifact (`spy-gmm-regime.json`) was trained when BULL_CALM dominated the training data — so its prior favors BULL_CALM in ambiguous cases. Retrain with class-balanced weighting.

Cost: high (full GMM retrain + acceptance gate run). Probably overkill for this fix.

---

## Caveats — ground truth is heuristic, not oracle

The "60d-forward return + vol" ground-truth label is a useful proxy but not perfect:
- Lookahead by definition (uses future data unavailable to the classifier in real time).
- May over-call CHOPPY: any 60d window with absolute return < 5% gets CHOPPY even if there was clear short-term momentum.
- Doesn't account for path: a +5% / -5% / +5% choppy oscillation that ends at 0 looks like CHOPPY here, but might still admit profitable swing trades.

The 33% mis-classification number is suggestive, not proven. The right next step is **A/B Experiment A** with a small tweak (0.52 → 0.55) and measure the classifier-vs-ground agreement *change*, not the absolute number.

---

## Recommended next step

Run **Experiment A** with the threshold tweak under a side strategy_config:
```bash
# Build side config with hurst_reversion_threshold=0.55
python -c "
import json
cfg = json.load(open('backtesting/renquant_104/strategy_config.json'))
cfg['regime']['hurst_reversion_threshold'] = 0.55
json.dump(cfg, open('backtesting/renquant_104/strategy_config.choppy_tuned.json','w'), indent=2)
"
# Re-run diagnostic
python scripts/diagnose_regime_classifier.py \
    --strategy-config-name strategy_config.choppy_tuned.json \
    --out data/audit/regime_choppy_tuned.json

# Compare against current classifier — agreement should improve from 34.7%
# CHOPPY% should land in [10%, 25%], BULL_CALM% should drop from 79.7%
```

If agreement crosses 45%, ship the tweak as a golden config update. If it doesn't, try Experiment B.
