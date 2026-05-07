# Watchlist 200 v2 — quality-first selection plan

**Date:** 2026-04-28 (post B1 regression)
**Status:** Plan only; no retrain triggered yet.

## Why v1 (mutual-fund spec) failed

The 2026-04-27/28 attempt was **B1 = current 103 + VPMAX/FCNTX top holdings + miscellaneous mega-caps → 227 tickers**.
Result: paired CPCV OOS IC = +0.0234 vs golden +0.0418 — a **−44%** regression.

Root cause hypothesis (3 in priority order):
1. **Heterogeneity dilution** — adding 124 new tickers diluted the cross-sectional rank signal because the new names had different return-distribution shapes (different volatility regimes, different sector betas) than the 103 the model had been tuned on.
2. **Liquidity spread** — the 227 set spanned a wider dollar-volume range; the LTR ranker isn't liquidity-aware so it scored low-liquidity names equally with mega-caps.
3. **Selection by holdings ≠ selection by signal quality** — VPMAX/FCNTX hold names for fundamental reasons; that's not the same as "tickers our model can rank well."

## v2 selection criteria

**Goal:** add ~50–100 high-quality candidates that the model has the best chance of ranking. Quality = (a) sufficient liquidity, (b) interpretable risk profile, (c) similar return-distribution shape to the existing 103.

### Filters (in order)

1. **Liquidity floor**: median 1y dollar-volume ≥ \$50M.
   Eliminates micro-caps where market microstructure (bid-ask spread, latency) dominates fundamentals.

2. **History floor**: ≥ 504 trading days (~2 years) of clean OHLCV.
   Matches the panel's `min_history_days=252` plus a safety margin so the new ticker has enough history for CPCV folds.

3. **Risk-distribution similarity to current panel**: realized 1y annualized vol ∈ [0.15, 0.85].
   - The current 103 panel has σ ∈ [12%, 127%] with median 43%. Restricting to [15%, 85%] keeps tickers whose vol regime matches the bulk of the panel and excludes the long-tail (NVTS at 127% would be excluded — and we now have empirical evidence that the model handles such extreme-vol names but they generate fat-tail risk we can't rebalance fast enough at 30-min cron cadence).

4. **Sharpe quality**: 1y realized Sharpe ≥ +0.5.
   Mild quality floor; not aggressive top-decile selection (that would re-introduce selection bias). Preserves a reasonable cross-section.

5. **Diversification**: cap each sector at +20% of new additions.
   The current 103 is tech-heavy; v2 should not double-down on tech.

### Candidate pool (probed 2026-04-28)

From local OHLCV cache (235 tickers available, 103 already in watchlist), 30 candidates that pass filters 1–4 sorted by 1y Sharpe:

| Ticker | Sharpe | Ret | Vol | Med-DV | Sector hint |
|---|---|---|---|---|---|
| C    | +2.42 | +67.8% | 28% | $1297M | financials |
| FDX  | +2.40 | +64.5% | 27% | $442M  | industrials |
| VLO  | +2.33 | +80.1% | 34% | $450M  | energy |
| CSX  | +2.30 | +51.8% | 22% | $476M  | industrials |
| PH   | +2.17 | +51.9% | 24% | $491M  | industrials |
| ROST | +2.09 | +51.4% | 25% | $401M  | consumer disc |
| MS   | +2.09 | +52.9% | 25% | $840M  | financials |
| XLI / XLE / XLK | sector-ETF surrogates | | | | sector |

(Full list in `scripts/probe_watchlist_v2.py` output — landed alongside this doc.)

### Validation protocol (mandatory before deployment)

This is what made B1 fail (we deployed without paired CPCV pre-validation):

1. **Per-ticker IC contribution test**: for each candidate, retrain on (current 103 + this 1 ticker), measure paired CPCV OOS IC delta. Reject any ticker that *individually* regresses IC.
2. **Greedy forward selection**: rank candidates by individual IC contribution; add greedily until marginal IC delta < +0.001.
3. **Final paired CPCV** on the resulting set vs golden 103 baseline. Promote only if paired t > +1.5 across CV folds.
4. **A/A sanity test** (per CLAUDE.md principle 5.2): shuffle labels, retrain, confirm IC ≈ 0.

Estimated cost: ~3h compute (224 single-ticker retrain runs + final paired CPCV).

### Rollback plan

The 2026-04-28 audit fix to `auto_revert_b1_regression.sh` (per-file dest paths + SHA verify + post-revert config-consistency check) means **a v2 rollback now actually works end-to-end**. Rollback was rehearsed on a tmp dir (see audit commit `bd9c413`).

## Open questions

- **Listing-date heterogeneity**: candidates with different IPO dates have different recency-weighting profiles. We may want to weight new candidates' contribution at half-strength until they have ≥ 1 full year of in-panel history. Defer to v2.1 if v2 ships clean.
- **Liquidity-aware ranker**: long-term, the LTR loss should be liquidity-weighted (heavier weight on names a real portfolio could actually trade). Not in v2 scope; tracked separately on roadmap.

## Decision gate

Don't start v2 until:
- Z9 (broker-side stops) shipped
- M2 horizon blender either shipped or shelved
- 24h-audit clean window holds (per principle 5.6)
