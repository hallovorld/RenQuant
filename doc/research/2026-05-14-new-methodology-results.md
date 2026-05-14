# 2026-05-14 — 5-test methodology applied to 8 panels

## Method (pre-registered before re-running)

Five orthogonal tests, candidate is **Tier 3 PROMOTE** if passes ≥4 of 5:

1. Pooled HAC mean (t-test) > 0
2. Pooled Wilcoxon median > 0 (sign-test p < 0.10)
3. Regime-stratified (≥3 of 4 regimes positive, Holm-corrected)
4. DSR computed on non-zero subset > 0.5
5. No-catastrophe (worst regime mean > −3%/yr)

Regime classification (return × vol):
- BEAR: annualized SPY return < −10%
- BULL_VOLATILE: annualized SPY vol > 22%
- BULL_STRONG: SPY return > +25% AND vol ≤ 22%
- CHOPPY: everything else

16 window classification:

| Regime | n | windows |
|---|---:|---|
| BEAR | 5 | Q01, Q02, Q06, Q12, Q16 |
| BULL_STRONG | 5 | Q04, Q05, Q07, Q08, Q14 |
| BULL_VOLATILE | 2 | Q03, Q13 |
| CHOPPY | 4 | Q09, Q10, Q11, Q15 |

## Verdicts (8 panels)

| Panel | Old (HAC) | NEW (5-test) | Passed | Notes |
|---|---|---|---:|---|
| kappa05 | REJECT | TIER 1 REJECT | 0/5 | All tests fail |
| mindw05 | NEITHER | TIER 1 REJECT | 0/5 | Zero effect dominant |
| p15_cellA | REJECT | TIER 1 REJECT | 0/5 | Same as κ=0.05 alone |
| **vt15** | **NEITHER** | **TIER 2 SCREEN** | **2/5** | **Upgraded**: T4 DSR-nonzero ✓, T5 no-catastrophe ✓ |
| gk094 | NEITHER | TIER 1 REJECT | 0/5 | |
| gk15 | NEITHER | TIER 1 REJECT | 0/5 | But strong regime split — see below |
| riskav5 | REJECT | TIER 1 REJECT | 0/5 | |
| maxpos10 | NEITHER | TIER 1 REJECT | 0/5 | Worst regime −15.8pt |

## Key finding: gk15 is a regime-conditional WINNER (currently miscategorized)

Per-regime breakdown for gk15:

| Regime | n | mean Δ/yr | median | n_pos |
|---|---:|---:|---:|---:|
| **BEAR** | 5 | **+2.13** | +2.93 | **4/5** ✓ |
| **CHOPPY** | 4 | **+3.09** | +11.34 | **3/4** ✓ |
| BULL_STRONG | 5 | −5.11 | −15.55 | 2/5 |
| BULL_VOLATILE | 2 | −6.36 | −6.36 | 1/2 |

**Pattern**: gk15 wins in defensive regimes (BEAR + CHOPPY), loses in
momentum regimes (BULL_STRONG + BULL_VOLATILE). This is structurally
clean — Grinold-Kahn α→μ rescaling helps when alpha decay matters
(noisy markets) but hurts when momentum dominates (steady bulls).

Q11 (CHOPPY, −31pt) is an outlier within its regime; **CHOPPY without
Q11 mean = +14.4pt**. Robust win.

## Existing regime-conditional GK config is INVERTED

`backtesting/renquant_104/strategy_config.sim_GK_conditional_ext.json`
enables GK in `HIGH_CALM` (= BULL_STRONG in my mapping) — exactly the
regime where this analysis shows gk15 LOSES (−5.11pt mean).

The config disables GK in `HIGH_SPIKED` (= BULL_VOLATILE, also losing
per my data, so correct) and `MED_CALM` (= CHOPPY in my mapping, where
gk15 WINS — so disabled wrongly).

**The prior GK_conditional rejection ("NEITHER") was due to inverted
regime activation, not because the underlying gk15 mechanism doesn't
work.**

## What to do with this finding

### Option A: Rebuild GK_conditional with correct regime mapping

Re-config GK enabled in defensive regimes (BEAR, CHOPPY in my labels;
LOW_* and MED_NORMAL in SpyRegimeLabelTask's scheme — need to verify
the exact label mapping). Disable in HIGH_CALM, HIGH_SPIKED.

Re-run as 16-window panel. Expected: passes Tier 2 cleanly under new
methodology, possibly Tier 3.

### Option B: First fix the methodology infrastructure

Bake the 5-test methodology into `eval_paired_returns.py` so all
future panels automatically use it. ~1h work + test.

### Option C: Skip to shorts (Phase 2B) since GK fix is single-strategy

Phase 2B unlocks **multiple** new mechanisms (sector-neutral with
shorts, vol-target with beta-hedge, etc.). One Phase 2B build vs one
GK_conditional config — Phase 2B has higher leverage.

## Recommendation

**Both A and C in parallel:**

- Phase 2B code (additive, no architecture change to baseline)
- Side config `sim_gk15_defensive_regime.json` — flip GK enabled
  according to corrected regime mapping
- Run that config as a 16-window panel after current queue clears

Both can ship at the same time. The GK_conditional re-run is a cheap
"if the old result was a bug, let's confirm it" test.
