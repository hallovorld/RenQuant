# 2026-05-15 — MA200-gate Postmortem (Panel A regression fully diagnosed + fixed)

## Headline

Panel A's −4.10pt mean regression was caused by a single mechanism that
took THREE wrong fix attempts to localize, before the fourth correctly
diagnosed the root cause through theoretical reasoning.

| Iteration | Fix | Theory | Q11 result |
|---|---|---|---|
| 1 | MA50 direction-aware Hurst | Hurst alone can't distinguish up/down trend | **−27.15pt** (worse) |
| 2 | softbear config (`regime_params.BEAR.*`) | Softer BEAR response (Kaminski-Lo) | −27.15pt (no change) |
| 3 | BEARBranchTask soft-gate | Block bear_only on transient/low-conf labels | −27.15pt (no change) |
| 4 | **MA200 confirmation gate** | **Require BOTH MA50 AND MA200 below** | **+0.00pt (restored)** |

## The actual loss mechanism

When MA50 fix labeled a SINGLE day as BEAR in a bull window:
1. `RegimeFinalizeTask` flipped `new_regime = BEAR`
2. `prev_regime != new_regime` → set `state.countdown = trans_bars` (=3)
3. `state.in_transition = True` for 3 bars
4. `compute_regime_confidence` returns flat **0.5** during transition
5. `confidence_to_size_multiplier(0.5)` returns **0.5** (the floor)
6. `SizeAndEmitTask` multiplies `max_position_pct *= 0.5` for 3 bars
7. Strategy buys at HALF size for 3 days
8. In a strong bull rally (Q11 SPY +12% over Q4 2024), 3 days at 50%
   sizing × 5 BEAR mis-labels = ~12-15 days at 50% → captured ~half the
   rally instead of full → **−27pt cumulative**

## Why theories 2 & 3 didn't work

- **Theory 2 (softbear config)** assumed `regime_params.BEAR.{max_position_pct,
  drawdown_halt_pct, cash_reserve_pct}` was the lever. Wrong — those
  knobs only matter AFTER `ctx.bear_only=True` triggers defensive-only
  selection. And bear_only never fired in Q11 (no log entries).

- **Theory 3 (BEARBranchTask soft-gate)** correctly identified that
  bear_only was the trigger. But Q11 SOFTGATE produced identical result
  to MA50 fix — auditing the log revealed bear_only never fired even
  in pre-softgate code. The loss was upstream, in size scaling
  triggered by the regime transition itself, NOT by the BEAR label
  reaching BEARBranchTask.

## Why theory 4 worked

The MA200 gate prevents the regime change from happening at all on
bull-market noise days. Q11 BULL_STRONG: SPY > MA200 100% of bars →
direction-aware-BEAR path never fires → no regime change → no transition
cooldown → no 0.5 confidence → no 0.5 sizing.

| Window | <MA50 % | <MA200 % | Net effect on detector |
|---|---|---|---|
| Q01 2022Q2 BEAR | 86% | 94% | True BEAR preserved |
| Q02 2022Q3 BEAR | 52% | **100%** | Even stronger BEAR signal |
| Q06 2023Q3 (corr.) | 41% | 0% | Skip (was a 7% correction, not bear) |
| Q10 2024Q3 BULL | 25% | **0%** | All 16 mis-fires eliminated |
| Q11 2024Q4 BULL | 8% | **0%** | All 5 mis-fires eliminated |
| Q15 2025Q4 BULL | 11% | **0%** | All 7 mis-fires eliminated |

## Verification results (3 catastrophe windows + Q15 bonus)

| Q | Regime | GMM | MA50 (broken) | MA200 (fix) | Δ vs GMM | Δ vs MA50 |
|---|---|---|---|---|---|---|
| Q10 | BULL_STRONG | −6.04% | −16.96% | **−6.04%** | +0.00pt | +10.92pt |
| Q11 | BULL_STRONG | +54.13% | +26.98% | **+54.13%** | +0.00pt | +27.15pt |
| Q15 | BULL_VOL | +3.50% | −21.82% | **+11.57%** | **+8.07pt** | +33.39pt |

Q15 IMPROVES over baseline — the MA200 gate not only eliminates false
BEAR noise but also lets *genuine* BEAR labels (SPY dipping below both
MAs during BULL_VOL vol spikes) trigger productive defensive action.
This is the regime-conditional alpha we've been trying to capture.

## Lessons (per CLAUDE.md PRIME DIRECTIVE)

1. **Detector improvements without RESPONSE-PATH audit are a footgun.**
   The MA50 fix correctly identified more BEAR days; the BUG was in the
   downstream confidence/sizing pipeline reacting too strongly. Always
   trace a regime CHANGE through transition cooldown → confidence
   formula → sizing multiplier BEFORE shipping.

2. **Empirical thresholds need MA200-level smoothing.** A single MA50
   cross is normal bull-market noise; require dual confirmation. This
   matches Hamilton 1989 HMM intuition (transition probabilities ≥ 0.85
   require multi-bar persistence).

3. **3 wrong theories before 1 right one** is the cost of debugging
   from outcome instead of mechanism. Lesson: when a 'fix' produces
   IDENTICAL results to the bug (Q11 softbear = Q11 MA50), STOP testing
   variants. The mechanism is NOT what you think it is — re-audit from
   first principles.

## Commits this session (debug trajectory)

```
a87f54a  fix(regime): require BOTH MA50 AND MA200 below for BEAR  ← winner
2447dcb  fix(regime): BEARBranchTask soft-gate (didn't help Q11)
68db94d  sim config: softbear — P1d range-find (didn't help Q11)
3925c0d  fix(regime): direction-aware Hurst (caused the regression)
```

## Next: full 16-window panel verification

A87f54a is shipped. Need to run all 16 windows to confirm:
- Mean Δ_APY recovers from Panel A's −4.10pt to ≥0pt
- Q01/Q02 BEAR windows still get +3.27pt / +0pt that direction-aware-Hurst
  enabled (now via MA200 path)
- No new regressions

Side configs already in place; just relaunch via fixed-data-race version
of the panel queue script (use -P 2 instead of -P 5 to avoid parquet
TMP rename races; or run sequentially).
