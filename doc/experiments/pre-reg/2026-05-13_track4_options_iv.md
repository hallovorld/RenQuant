# Pre-registration: Track 4 — Options-implied volatility features


> **📅 Historical snapshot — content below reflects state at the date in filename/header.**
> Verify against current code per CLAUDE.md §1 "code is the source of truth" before acting on
> present-tense claims. For current state see `doc/roadmap.md` § "📍 Current state" +
> `CLAUDE.md` § "🗂 Current state".

**Date**: 2026-05-13
**Pre-registered BEFORE experiment. STATUS: planning, execution deferred (needs new data pipeline).**

## Hypothesis

**H0**: Adding options-implied volatility features (IV term structure,
skew, put-call ratio) to the 169-feature panel produces equal or worse
OOS IC.

**H1**: IV features yield **+5-15bp IC** and **+1-3pt mean ΔAPY**.

## Theoretical basis

**Goyal & Saretto 2009** *Journal of Financial Economics* "Cross-section
of option returns and stock returns" — IV-realized vol spread predicts
future stock returns (the "vol-mispricing premium").

**Ang & Liu 2007** *Annals of Finance* — IV skew predicts crash risk and
informs cross-section ranking.

**An, Ang, Bali & Cakici 2014** *Journal of Finance* — options-implied
features add +5-10bp IC over technical features in stock cross-section.

**Common features** (from Bali-Engle-Murray 2016 ch.14):
- IV30 (30-day ATM IV)
- IV term structure: IV90 - IV30
- IV skew: 25-delta-put-IV - 25-delta-call-IV
- Put-call open interest ratio
- IV-realized vol spread (signed)

## Implementation plan (DEFERRED — needs data pipeline)

1. **Data fetch** (~3-5 days eng):
   - OptionMetrics / IvyDB Historical (paid)
   - Polygon.io Options (~$200/mo)
   - Tradier API (~$10/mo) — for live
   - Local cache builder mirroring ohlcv structure
2. **Feature extractor** (~2 days): compute IV30/skew/term/realized-spread
3. **Panel integration** (~1 day): append 5-7 new IV features to alpha158
4. **Retrain** (~30 min compute): 74-cutoff walkforward on extended panel
5. **Full panel** (~70 min): 16-window paired analysis

**Total: ~1-2 weeks eng + 1 day compute. Defer.**

## Pre-committed evaluation criteria

Same Tier framework. Additional gate:
- Feature importance from XGB must rank ≥ 1 IV feature in top 30 of 174
  → if IV features are unused by model, no point

## Why deferred

- Data acquisition is the blocker
- Cost: $200-500/mo data subscription
- Engineering: 1-2 weeks
- Lower-priority than tabular model class swaps (LGBM)
