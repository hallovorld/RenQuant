# 2026-05-15 — Conditional Shorts (P1a overlay) Verdict: NEITHER

## TL;DR

First production application of the per-regime overlay (P1a, commit
`054a572`): enable shorts only in CHOPPY + BULL_VOLATILE regimes.

**Mechanism works.** Resolver fires correctly. Q04 BULL_VOLATILE-labeled
days activated shorts and bigger position sizing.

**Alpha does NOT replicate.** Headline +2.29pt vs GMM baseline is
single-window-driven. Q04 alone contributes +35.02pt. Remaining 15
windows: +0.11pt mean. **NEITHER per 5-test methodology.**

## 5-test results (16-window paired panel)

| Test | Result | Threshold | Verdict |
|---|---|---|---|
| Pooled mean Δ_APY paired t | +2.29pt, t=1.04, p=0.316 | p<0.10 | ❌ NOT SIG |
| Wilcoxon median (non-zero n=7) | median=+0.00pt, p=0.375 | p<0.10 | ❌ NEITHER |
| Regime stratified | BULL_CALM +11.9pt (Q04 driven); BEAR +1.1pt; rest ~0 | 3W needed | ⚠️ 2W/1L |
| Without Q04 outlier (n=15) | mean=+0.11pt σ=1.38 | edge proof | ❌ flat |
| No-catastrophe | 0 windows ≤ -5pt (max neg = -1.57pt Q13) | 0 cat | ✅ PASS |

**Aggregate: 1 PASS, 1 marginal, 3 FAIL → NEITHER.**

## Per-window detail

```
Q    regime         GMM     MA200    CondShorts   ΔvsGMM
=========================================================
Q01  BEAR        -24.76%  -24.76%    -24.76%    +0.00pt
Q02  BEAR        -23.45%  -20.18%    -20.18%    +3.27pt  ← MA200 win preserved
Q03  CHOPPY      +14.96%  +11.78%    +11.78%    -3.18pt
Q04  BULL_CALM   +19.05%  +11.41%    +54.06%   +35.02pt  ← outlier
Q05  BULL_CALM   +28.97%  +28.97%    +28.97%    +0.00pt
Q06  BEAR        -18.26%  -18.26%    -18.26%    +0.00pt
Q07  BULL_STRONG +50.80%  +50.80%    +50.80%    +0.00pt
Q08  BULL_VOL    +26.51%  +26.51%    +27.23%    +0.72pt
Q09  CHOPPY      -24.59%  -24.59%    -24.59%    +0.00pt
Q10  BULL_STRONG  -6.04%   -6.04%     -6.04%    +0.00pt
Q11  BULL_STRONG +54.13%  +54.13%    +54.13%    +0.00pt
Q12  BULL_CALM   -23.46%  -22.79%    -22.79%    +0.67pt
Q13  BULL_STRONG +26.75%  +26.75%    +25.18%    -1.57pt
Q14  BULL_VOL     +0.80%   +0.80%     +0.80%    +0.00pt
Q15  BULL_VOL     +3.50%   +3.50%     +3.50%    +0.00pt
Q16  BULL_STRONG -17.98%  -16.32%    -16.32%    +1.67pt
```

## Q04 outlier dissection

Q04 (2023 Q1 recovery): MA200 baseline +11.41% → CondShorts +54.06%.
Mechanism:
- Regime distribution: 45 BULL_CALM / 14 BULL_VOLATILE / 3 BEAR
- Overlay activated shorts on the 14 BULL_VOLATILE days
- Two short trades fired: EQIX (~$5K notional), MSFT (~$3K notional)
- But $43K alpha emerged — far beyond what 2 shorts can deliver
- True driver: `regime_params.BULL_VOLATILE.max_position_pct = 0.20`
  (vs BULL_CALM's 0.15) → 33% bigger longs during BULL_VOLATILE days
- Strategy bought more aggressively on a recovery day → captured the rally

**The Q04 win is regime-conditional POSITION SIZING, not shorts.** Same
mechanism would fire if BULL_VOLATILE detection were just enabled
elsewhere, regardless of shorts.

## Why this fails the methodology

5-test framework explicitly designed to flag single-window-outlier wins:

1. Pooled t-test high σ (8.83) reflects the Q04 single shot
2. Wilcoxon median = 0 — most windows unchanged
3. Regime-stratified: BULL_CALM win is n=3, with Q04 dominating
4. **Sans-outlier mean = +0.11pt** — directly shows fragility

This is exactly the kind of "single-shot lottery" Bailey-Lopez de Prado
2014 (Deflated Sharpe) and Bailey et al 2015 (PBO via CSCV) penalize.

## What we learned

1. **P1a overlay mechanism works.** First production application: fires
   correctly, picks up per-regime config.
2. **Conditional shorts in CHOPPY/BULL_VOL is NOT robust alpha.** Same
   conclusion as longshort_clean panel (2026-05-14), now with the
   correct detector (MA200 gate). The regime-conditional shorts result
   doesn't beat baseline.
3. **The Q04 effect is BULL_VOLATILE max_position_pct=0.20 firing**,
   not shorts. Worth investigating: does enabling MORE BULL_VOLATILE
   detection (without shorts) reproduce the Q04 alpha across other
   recovery windows?

## Production decision

**DO NOT promote conditional shorts to live.** Pattern matches the
single-shot lottery 5-test rejects. Memory entry:
`feedback_promotion_methodology.md` Tier 1 REJECT.

`strategy_config.golden.json` unchanged.

## Next experiments (per roadmap.md P1c-h, ordered by literature support)

1. **Moreira-Muir 2017 vol-targeting** (CANONICAL) — `c/σ²_t` continuous
   formula. Published alpha +4.86%/yr, beta 0.6. Highest-confidence
   bet for actual regime-conditional alpha.

2. **Kaminski-Lo 2014 regime-conditional stop_loss** — TIGHT in CHOPPY
   (~0.08), WIDE in BULL_VOL (~0.20). Theory directly supports.

3. **Defensive ticker basket sub-types** — add SHV/BIL/DBC for
   stagflation BEARs (2022 GLD/TLT failed). Needs OHLCV fetch first.

4. **Per-regime kappa (risk aversion)** — Ang-Bekaert 2002 direction
   supported; specific values exploratory.

## Commit (this experiment)

```
e822106  sim configs: conditional shorts — P1a overlay activation in CHOPPY+BULL_VOL
```

Result file: `data/logs/sim_2026-05-15_conditional_shorts/equity/Q01..Q16.json`.
