# Watchlist quality curve — admission discipline

**Branch**: `exp/wl500-and-sector-arch`
**Date**: 2026-05-02
**Status**: design (not yet implemented)

## Why

`wl178` expansion produced **−65% CPCV IC** (E17/E21). The naive
expansion mode (Russell 1000 minus filters → list) brought too many
tickers whose feature distributions don't match the wl103 mechanism.
L1+L2 sector arch can rescue only ~30% of the loss. The real fix is
**admission discipline before training**.

## Falsified prior hypothesis

E_LAYER1_PLUS_2_ON_WL103 (2026-05-02): L1+L2 hurts on homogeneous
universes (−21.5%). Confirms the lift on wl178 is *treating heterogeneity*,
not adding signal. **Therefore: admission ≠ "filter for L1+L2 to work";
admission = "filter for the shared pricing-mechanism to apply".**

## Three-stage funnel

### Stage 1 — Mechanical (cheap, deterministic)

Drop the obvious unfit before any compute. Pre-filter the Russell 1000
universe (1009 candidates) to ~400 by these mechanical checks:

```
admit IFF
   ADV_60d_USD            >= 10_000_000        # liquidity
   AND age_in_market_yrs  >= 3.0               # sufficient history
   AND price_60d_min      >= 5.0               # no penny stocks
   AND not is_etf                              # ETFs skew sector pools (E21 finding)
   AND has_sector_in_map                       # sector_map coverage required for L1+L2
```

Drop rate: ~60% (1009 → ~400). Cost: 1 SQL query on metadata. No retrain.

### Stage 2 — Distributional similarity (cheap, statistical)

For each Stage-1 survivor, compute KS distance vs the wl103 pool on
each of the 27 production feature columns. Reject candidates whose
**median KS across features > 0.20**. Tighter than Witter 2025's 0.30
because we're not relying on L1+L2 to rescue.

Algorithm:
1. Build the wl103 reference distribution per feature: pool all wl103
   ticker-day rows, compute the empirical CDF per feature col.
2. For candidate ticker T, build the same per-feature CDF on T's last
   3 years.
3. Compute KS_T_f = KS(wl103_pool_f, T_f) for f in features.
4. Reject if median(KS_T_*) > 0.20.

Drop rate: ~50% of Stage-1 survivors (~400 → ~200). Cost: vectorised
scipy.stats.ks_2samp on cached feature frames; <5 min for 400 candidates.
**No retrain.**

### Stage 3 — Greedy IC-additive batch admission (expensive but bounded)

Stage 2's survivors are admitted to a *candidate pool*, not the
training set. Greedy batch admission:

```
universe ← wl103
candidates ← stage_2_survivors                  # ~200
batch_size ← 5                                  # 5 tickers/round
acceptance_threshold ← -0.0020                  # -2bp tolerance per batch

WHILE candidates is non-empty AND |universe| < target:
   batch ← random sample of 5 from candidates    # OR by sector to balance
   universe' ← universe ∪ batch
   train CPCV with universe'                     # ~5 min per retrain
   delta ← mean_ic(universe') - mean_ic(universe)
   IF delta > acceptance_threshold:
      universe ← universe'                       # accept batch
   candidates ← candidates - batch
```

**Cost**: ~40 retrains × 5 min = ~3.5 hours sequential to grow wl103
to ~wl200. Doubled when sanity-checked against an A/A control.

**Why batches not single-ticker greedy?**
- Pure single-ticker greedy is O(N²) ≈ 400 retrains; intractable.
- Batch noise: 5 random tickers' joint contribution averages individual
  noise. Rejection bias drops.
- Realistic operator workflow: insider/PEAD/microstructure data fetches
  also batch; admission cadence syncs.

## Acceptance metrics (per batch)

A batch is admitted iff **all** of:
- `delta_mean_ic > -0.0020` (−2bp absolute, per CPCV mean)
- `train_ic does NOT degrade by > 5bp` (catches train-time fit collapse)
- `best_iter remains > 5` (eval set still discriminating)
- No new ticker triggers `min_sector_size` fallback to global (would mean
  L1 path can't help even if we re-enabled it)

## Hard rejects (skip Stage 3 entirely)

Reject candidates that match any of:
- ETF (XLE/XLI/XLK/XLY/IYR — E21 finding)
- Holding company / closed-end fund (BRK.A, BRK.B, AZN — separate spec)
- Recent IPO (< 3 yrs OHLCV) — already filtered Stage 1, but double-check
- Earnings cadence < 4 reports/yr (broken PEAD signal)
- Reverse-split or M&A in last 12 months (price discontinuity)

## Predicted yield

- Stage 1: 1009 → ~400 (60% drop)
- Stage 2: 400 → ~200 (50% drop)
- Stage 3: 200 candidates → ~100 admitted (50% accept rate over 40 batches)
- **Final wl: ~200 tickers, IC ≥ wl103 baseline +0.0418** (no degradation)

If Stage 3 accept rate is < 30% after 20 batches, **stop and revisit
Stage 2 thresholds** — likely too loose (KS 0.20 not strict enough).

## Where this differs from wl178

| | wl178 (failed) | wl-curated (planned) |
|---|---|---|
| Source | Russell 1000 minus crude filters | Same source |
| Stage 1 (mechanical) | Yes (similar liquidity/age) | Same |
| Stage 2 (distributional) | **MISSING** | KS test gate |
| Stage 3 (IC-additive) | **MISSING** — added everything at once | Greedy batch admission |
| L1+L2 needed? | Yes, to claw back heterogeneity loss | No (homogeneous by construction) |
| Predicted IC | wl178: +0.007 raw, +0.015 with L1+L2 | wl-curated: ≥ +0.040 (no degradation) |

## Implementation sequencing

1. **D-1** Stage 1 + Stage 2 scripts (no retrain). Output: `wl_candidates_stage2.json`. Cost: 1-2 hours dev.
2. **D-2** Greedy batch admission script with retrain hook. Cost: 1 day dev.
3. **D-3** Run Stage 3 (one weekend, 4-6 hours wallclock). Cost: 1 weekend.
4. **D-4** Verdict log + DB rows for each accept/reject batch.

## Open questions

1. **Sector balancing in batches**: pure random sample biases toward
   the largest sector groups. Stratified random by sector? Probably
   yes — open to test both.
2. **Refresh cadence**: once curated, how often do we re-screen?
   Quarterly? Annually? Tickers that drift can poison the pool over
   time. Suggest: re-Stage-2 quarterly, re-Stage-3 only on triggered
   underperformance.
3. **Compatibility with insider/PEAD/microstructure**: these features
   should be added BEFORE Stage 3 retrains, so the admission criterion
   evaluates the full feature set. Sequencing: insider/PEAD/hourly
   first → re-baseline wl103 → then run Stage 3.

## Related

- `doc/research/failed-experiments-log.md::E17` — wl178 quality-filter expansion
- `doc/research/failed-experiments-log.md::E21` — wl174 (ETF removal didn't fix)
- `doc/research/failed-experiments-log.md::E_LAYER1_PLUS_2_ON_WL103` — L1+L2 not universal
- `doc/research/per-sector-architecture-plan.md` — Witter 2025 KS analysis
- Lopez de Prado, *Advances in Financial Machine Learning*, ch. 8 — feature importance + ticker screening
