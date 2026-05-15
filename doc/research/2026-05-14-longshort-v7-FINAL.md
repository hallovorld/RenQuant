# 2026-05-14 — longshort_v1 v7 FINAL verdict (post §1233 fix)

## Status: TIER 3 STATS / NOT AUTO-PROMOTABLE

Pre-registered 5-test methodology: **5/5 PASS**. Pooled HAC t=+2.99,
p=0.0028, DSR=1.0, 14/16 windows positive, all 4 regimes positive.

But: per `feedback_auto_promote_to_prod.md` exclusion list, Phase 2
shorts is "architecture change + risk-loosening" → requires explicit
user OK before flipping `long_short.enabled=true` in production.

## Final numbers

| Metric | Value |
|---|---:|
| Mean Δ annualised | +13.69%/yr |
| t-statistic | +2.99 (p=0.0028) |
| 95% bootstrap CI | [+5.05%, +22.51%] |
| Sharpe of Δ | +1.53 (CI [+0.56, +2.48]) |
| DSR (K=100) | 1.000 |
| Cohen's d | +0.097 |
| Consistency | 14/16 (88%) |

Regime stratification (all 4 positive):
- BEAR: +13.64
- BULL_STRONG: +15.30
- BULL_VOLATILE: +19.66
- CHOPPY: +14.71

## What §1233 tax fix actually did

Code shipped (commit `64b390d`): tax on short-cover profit at ST rate.
Test suite: 4 new tests pass.

Numerical impact on v7 vs v6: **zero**.

Reason: shorts in this universe mostly closed at SMALL LOSSES (e.g.
GS shorted at $320.70, covered at $323.96 = -$22 loss/share × 7 shares
= -$160 loss). Losses are STCL (offset future gains) but don't trigger
immediate tax. Profitable covers were rare; total `SHORT_COVER_TAX`
events across 16 windows = 0.

The fix is correct and ships for future scenarios where shorts profit
more, but didn't affect this particular result.

## Where the +13.69pt comes from (best honest decomposition)

1. **~9-11pt from mechanical leverage** coupled with short proceeds:
   When a short opens, cash += proceeds. That cash funds extra long
   buys. With ~25% short-side budget, the strategy can effectively
   run ~115-125% net long. In a +36% bull window, ~15-20pt above
   75% baseline deployment.

2. **~2-4pt from selection breadth** (Grinold-Kahn IR×√2 with 2x
   breadth via short side adding orthogonal information).

3. **~0-2pt from real bear hedging** (small short positions reduce
   maxdd and vol in BEAR/CHOPPY regimes — verified in Q01: vol
   dropped from 14.9% → 10.8%, maxdd 7.5% → 5.3%).

The control panel "leverage_only" (partial 10/16 with `short_decile=0`,
gross_max=1.30) showed mean Δ ≈ +0.86pt — confirming that `gross_max`
ALONE doesn't help; it needs shorts firing to credit cash and enable
the leverage. The two effects are mechanically intertwined.

## What's missing before live readiness

| # | Task | Effort | Status |
|---|---|---:|---|
| 1 | §1233 short-cover tax | 0.5d | ✅ Shipped today |
| 2 | §1091 long-side wash-sale interaction with shorts | 1d | ❌ |
| 3 | Alpaca borrow/locate pre-trade verify | 1d | ❌ Data fetched, not enforced |
| 4 | Reg-T margin pre-trade check | 0.5d | ❌ |
| 5 | Sector cap also constrain shorts | 1d | ❌ Audit bug C |
| 6 | _apply_short_open trade_log entry | 0.5d | ❌ Audit bug B |
| 7 | Smoke test on Alpaca paper account | 1d | ❌ |
| 8 | 2-week paper observation | 14d | ❌ |
| 9 | User explicit OK for live flip | n/a | ❌ |

Earliest possible live deployment: **~3-4 weeks from today** assuming
items 2-8 all complete cleanly.

## Decision gate

Awaiting user OK to proceed with Phase 2D-2I. If approved:
1. Implement items 2-6 (~3.5d code)
2. Smoke test on Alpaca paper, single position, manually monitored
3. Run 2-week paper period observing position management, borrow API,
   tax accounting
4. Flip `long_short.enabled=true` with conservative defaults
   (max_shorts=3, max_short_pct=0.05, gross=1.15 — lower than the
   1.30 used in sim for first month)

If NOT approved or held: result stays documented; baseline strategy
unchanged.

## Six bugs fixed in this short-execution chain

1. ShortCandidateSelectionTask over-aggressive exclusion (eligible set empty)
2. _BuildSourceMapTask long-wins-on-tie (short signal buried)
3. ShortCandidateSelectionTask emitted positive panel_score (no short signal)
4. _emit_qp_sell bailed for non-held tickers (no short-open orders)
5. commit() dedupe dropped qp_short_open (first-write-wins)
6. SimAdapter lacked _apply_short_open path

Plus 2 audit follow-ups:
- A. (RED) §1233 tax on short-cover profit
- B/C. (YELLOW) Entry-price cross-zero + borrow overdraw

All shipped with regression tests (27 short-related tests passing).
