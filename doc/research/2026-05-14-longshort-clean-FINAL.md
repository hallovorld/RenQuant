# Long-short clean panel (no leverage) — FINAL 16-window verdict

**Date:** 2026-05-14 evening
**Config:** `strategy_config.sim_longshort_clean.json` — `long_short.enabled=true`,
`max_gross_exposure=1.00` (NO implicit leverage from short proceeds),
`max_short_pct=0.05`, `max_shorts=3`, `short_decile=0.10`.
**Baseline:** `sim_2026-05-12_phase2/sim_baseline_ext` (16-window paired).
**Treatment artifacts:** `data/logs/sim_2026-05-14_longshort_clean/equity/Q01..Q16.json`.

## TL;DR

Removing the implicit leverage (max_gross_exposure: 1.30 → 1.00) from the
v7 long-short config shrinks the headline ΔAPY from **+13.69pt** to
**+6.23pt** and pushes the result into **NEITHER** territory under the
robust 5-test methodology. **DO NOT auto-promote.**

The v7 +13.69pt win was ~55% leverage / ~45% real shorts-alpha by point-
contribution. The shorts-alpha component is regime-conditional, not
uniform.

## 5-test results (n=16 paired windows)

| Test | Verdict | Headline number |
|---|---|---|
| Pooled mean Δ_APY paired t | NEITHER | +6.23pt, t=1.24, p=0.234, 95% CI [−4.5, +16.9] |
| Wilcoxon signed-rank | NEITHER | median +7.61pt, p=0.32 |
| Regime-stratified mean | WIN (3W / 2L) | BEAR +22.0pt, CHOPPY +14.1pt, BULL_VOL +12.6pt vs BULL_CALM −7.8pt, BULL_STRONG −1.8pt |
| Deflated Sharpe (PSR) | NEITHER | treat SR=0.32, PSR=0.903 (below 0.95) |
| No-catastrophe | **LOSE** | Q07 Δ=−20.5pt, Q11 Δ=−26.1pt — 2 catastrophes |

**Aggregate: 1 WIN / 1 LOSE / 3 NEITHER → NEITHER**

## Per-window detail (16-window paired)

```
window  regime        b_apy    t_apy    Δ_apy_pt   Δ_sh    Δ_dd_pt
Q01     BEAR         -24.76%  -11.81%   +12.95     +0.70   -2.31    win
Q02     BEAR         -23.45%  -14.36%    +9.09     +0.59   -1.40    win
Q03     CHOPPY       +14.96%  +23.36%    +8.40     +0.31   +0.70    win
Q04     BULL_CALM    +19.05%   +8.82%   -10.23     -0.56   +3.57    LOSE
Q05     BULL_CALM    +28.97%  +20.13%    -8.85     -0.50   -0.30    LOSE
Q06     BEAR         -18.26%  +25.60%   +43.85     +2.88   -2.19    HUGE WIN
Q07     BULL_STRONG  +50.80%  +30.28%   -20.52     -0.99   -0.00    catastrophe
Q08     BULL_VOL     +26.51%  +40.80%   +14.29     +0.44   -0.50    win
Q09     CHOPPY       -24.59%   -4.70%   +19.89     +1.77   -1.07    win
Q10     BULL_STRONG   -6.04%  +39.98%   +46.03     +1.78   -4.16    HUGE WIN
Q11     BULL_STRONG  +54.13%  +28.02%   -26.11     -0.80   -0.37    catastrophe
Q12     BULL_CALM    -23.46%  -27.87%    -4.41     -0.26   +0.75    lose
Q13     BULL_STRONG  +26.75%  +26.53%    -0.21     -0.10    0.00    flat
Q14     BULL_VOL      +0.80%  +17.53%   +16.73     +0.98   -2.76    win
Q15     BULL_VOL      +3.50%  +10.31%    +6.81     +0.28   -1.93    win
Q16     BULL_STRONG  -17.98%  -26.07%    -8.09     -0.81   +0.88    lose

mean                                     +6.23     +0.42   -0.84
median                                   +7.61
```

## Decomposing v7 (+13.69pt with leverage) vs clean (+6.23pt no leverage)

Per-regime comparison:

| Regime | v7 (gross=1.30) | clean (gross=1.00) | leverage share |
|---|---|---|---|
| BEAR | +13.64pt | **+21.96pt** | **clean BEATS v7** — real shorts alpha |
| BULL_STRONG | +15.30pt | **−1.78pt** | leverage was the ENTIRE win |
| BULL_VOL | +19.66pt | +12.61pt | ~36% leverage / ~64% shorts |
| CHOPPY | +14.71pt | +14.15pt | ~0% leverage — basically pure shorts |
| BULL_CALM | n/a (not in v7) | −7.83pt | shorts ALONE lose in BULL_CALM |

**Insight:** In BULL_STRONG markets, v7's win came entirely from MORE
GROSS LONG EXPOSURE (leverage), not from shorts. Shorts in BULL_STRONG
actively hurt (Q07 −20.5pt, Q11 −26.1pt). Conversely in BEAR markets,
removing the leverage REDUCES exposure to the falling longs and lets the
shorts contribute MORE.

## Verdict & recommendation

**DO NOT auto-promote longshort_clean.** Per `feedback_auto_promote_to_prod.md`:
- Shorts are explicitly in the auto-promote exclusion list (risk-loosening
  change, exposes to short-squeeze risk).
- Pooled-mean test not significant (p=0.23) — fails Tier 2 statistical bar.
- Catastrophes in 2 BULL_STRONG windows fail the no-catastrophe gate.

**Strategic implications:**
1. The original +13.69pt v7 finding is **half real, half leverage**.
2. Pure shorts-alpha at gross_max=1.00 is **regime-conditional** (BEAR /
   CHOPPY / BULL_VOL only).
3. **The leverage knob (max_gross_exposure) is independent of shorts** —
   it could be tuned alone to capture the BULL_STRONG leverage win
   without taking on short-squeeze risk.

## Next steps (for user decision)

| Option | Description | Risk |
|---|---|---|
| **A. Reject shorts entirely** | Keep `long_short.enabled=false` in prod | Lowest risk; no upside |
| **B. Leverage-only sweep** | Sweep `max_gross_exposure ∈ {1.05, 1.10, 1.15, 1.20, 1.30}` WITHOUT shorts | Tests the leverage hypothesis cleanly; needs ~16 sims × 5 configs = ~80 sims |
| **C. Conditional shorts** | Deploy shorts only when regime ∈ {BEAR, CHOPPY, BULL_VOL} | Blocked on regime detector (currently 95% BULL_CALM) |
| **D. Tighter shorts** | `max_short_pct=0.03` and add `disable_shorts_in_regime=BULL_STRONG` | Adds new task; needs new panel |

Recommend **B** first (cleanest test, no new code), then **D** if leverage
also fails.

## Methodology references

- `feedback_eval_robust_methodology.md` — 5-test framework
- `feedback_qp_gross_max_is_leverage.md` — leverage confound discovery
- `feedback_auto_promote_to_prod.md` — auto-promote exclusion list
- `doc/research/promotion-methodology.md` — 3-tier promotion gating
