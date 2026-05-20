# 2026-05-14 — longshort_v1 = TIER 3 PROMOTE candidate (5/5 tests pass)


> **📅 Historical snapshot — content below reflects state at the date in filename/header.**
> Verify against current code per CLAUDE.md §1 "code is the source of truth" before acting on
> present-tense claims. For current state see `doc/roadmap.md` § "📍 Current state" +
> `CLAUDE.md` § "🗂 Current state".

## Final v6 panel result (after 6 bug fixes in short-execution chain)

### Pooled HAC
- Mean Δ annualised: **+13.69%/yr** (NW SE 0.018% daily, lag=6)
- t-statistic: **+2.99** (p=0.0028)
- 95% bootstrap CI: [+5.05%, +22.51%] (both ends positive)
- Sharpe of Δ: +1.53 (CI [+0.56, +2.48])
- DSR (K=100): +1.000 (max)
- Window consistency: 14/16 positive (88%)

### 5-test methodology (pre-registered)
| Test | Value | Pass |
|---|---|---|
| T1 pooled mean t | mean=+15.18, t=+4.82 | ✓ |
| T2 Wilcoxon/sign | median=+18.09, wilcoxon p=0.0005 | ✓ |
| T3 regime stratification | 4/4 regimes positive | ✓ |
| T4 DSR-nonzero | 0.595 | ✓ |
| T5 no-catastrophe | worst regime +13.64 | ✓ |

Per regime: BEAR +13.64, BULL_STRONG +15.30, BULL_VOLATILE +19.66, CHOPPY +14.71.

## Per-window deltas

| Q | regime | base | v6 | Δ | t |
|---:|---|---:|---:|---:|---:|
| 01 | BEAR | -24.76 | -11.39 | +13.37 | +1.26 |
| 02 | BEAR | -23.45 | -11.81 | +14.16 | +1.54 |
| 03 | BULL_VOL | +14.96 | +30.78 | +15.81 | +0.70 |
| 04 | BULL_STR | +19.05 | +5.78 | -13.26 | -0.86 |
| 05 | BULL_STR | +28.97 | +60.99 | +32.02 | +0.82 |
| 06 | BEAR | -18.26 | +12.70 | **+30.95** | +2.06* |
| 07 | BULL_STR | +50.80 | +62.89 | +12.09 | +0.94 |
| 08 | BULL_STR | +26.51 | +49.99 | +23.48 | +0.53 |
| 09 | CHOPPY | -24.59 | -4.23 | +20.36 | +1.40 |
| 10 | CHOPPY | -6.04 | +15.33 | +21.37 | +1.20 |
| 11 | CHOPPY | +54.13 | +78.67 | +24.53 | +0.59 |
| 12 | BEAR | -23.46 | -16.86 | +6.61 | +0.52 |
| 13 | BULL_VOL | +26.75 | +50.25 | +23.50 | **+2.15*** |
| 14 | BULL_STR | +0.80 | +22.96 | +22.16 | +0.99 |
| 15 | CHOPPY | +3.50 | -3.94 | -7.46 | -0.61 |
| 16 | BEAR | -17.98 | -7.26 | +10.72 | +0.72 |

* significant at α=0.05

## Confounds and caveats

1. **`gross_max=1.30` leverage**: longshort_v1 lets the QP go up to 130% gross.
   When few shorts fire (just 2-5 per bar), the budget effectively allows
   ~125% net long. Part of the +13.69 is leverage, not pure short alpha.
   Leverage_only control panel running now to disentangle.

2. **Source-map override side effect**: Phase 2B fix makes short candidates
   override long candidates in the source map. This REMOVES bottom-decile
   tickers from the long pool, freeing the QP to concentrate on top-decile
   names. Q06's WDC concentration (15 buys vs baseline 9) is an example.

3. **Backtest sample size 16 windows**: not enough to definitively rule out
   regime-specific overfit. DSR penalty applied via K_trials=100.

4. **Borrow cost minimal**: all 103 tickers ETB at ~0% rate per Alpaca live
   API. Real-world borrow drag negligible.

## Six bugs fixed to reach this result

1. ShortCandidateSelectionTask over-aggressive `ctx.candidates` exclusion
2. `_BuildSourceMapTask` long-wins-on-tie precedence
3. ShortCandidateSelectionTask emitted raw (positive) panel_score → no short signal
4. `_emit_qp_sell` bailed for non-held tickers → no short-open orders
5. `commit()` dedupe dropped qp_short_open with qp_close
6. SimAdapter had no `_apply_short_open` path

Plus 2 audit-found follow-ups (entry-price cross-zero, borrow-charge overdraw).

## Decision gate

**Do NOT flip live yet.** Architecture change exclusion per
`feedback_auto_promote_to_prod.md`. Requires:

1. leverage_only control panel to isolate shorts alpha (~80min)
2. §1233 wash-sale on shorts code (not yet)
3. Alpaca borrow/locate pre-trade check (not yet)
4. Reg-T margin pre-trade check (not yet)
5. 2-week paper test on alpaca-paper
6. User explicit OK

If leverage_only shows the +13.69 is mostly leverage (delta < 5pt with
shorts removed), this is not a "shorts win" — it's a "we discovered
130% gross leverage helps" finding which is a separate experiment.

If leverage_only shows the +13.69 is mostly shorts (delta > 5pt of pure
shorts alpha), then Phase 2D-2F work to land it cleanly in live.
