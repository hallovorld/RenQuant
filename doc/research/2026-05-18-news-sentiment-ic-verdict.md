# 2026-05-18 — News sentiment IC eval verdict

## TL;DR

**SCREEN candidate** (Tier 2 per CLAUDE.md §5.13.4a). News sentiment shows real forward predictive power on fwd_60d_excess, but news is partially endogenous to recent returns. Best feature: `sentiment_pos_share × fwd_60d_excess` IC = **+0.046** with 5/5 WF cuts same sign and ts-30 placebo collapsing to +0.015 (near shuffle noise +0.017).

NOT auto-promotable; integrate into panel as one of N features, let model down-weight if not additive to current 169-feat panel.

## Setup

- Panel: `data/alpha158_291_fundamental_dataset.parquet` (292 tickers, 2016-01 → 2026-02-10)
- Sentiment: `data/news_sentiment_alpaca/` (103 tickers, 2025-05-18 → 2026-05-17, 16,793 daily rows)
- Merge: 10,669 rows (1.5% panel coverage — overlap window is only Aug 2025 – Feb 2026, ~6 months panel × sentiment)
- IC: Spearman cross-sectional, averaged across dates
- §5.2 sanity battery: shuffle-label, time-shift ±30d, A/A split, walk-forward 5-cut

## Headline results

| Feature × Label | raw IC | placebo shuffle max | ts+30 placebo | ts-30 placebo | A/A even / odd | WF sign-consistent | Verdict |
|---|---|---|---|---|---|---|---|
| **sentiment_pos_share × fwd_60d_excess** | **+0.046** | +0.017 | +0.062 | **+0.015** | +0.040 / +0.052 | **5/5** | **SCREEN** |
| sentiment_dispersion × fwd_60d_excess | +0.052 | +0.026 | +0.059 | +0.023 | +0.050 / +0.053 | 4/5 | SCREEN |
| mean_sentiment × fwd_60d_excess | +0.044 | +0.007 | +0.090 | +0.028 | +0.041 / +0.047 | 3/5 | weak |
| n_articles × fwd_60d_excess | +0.035 | +0.025 | +0.056 | +0.022 | +0.035 / +0.034 | 4/5 | placebo-overlap |
| sentiment_neg_share × fwd_60d_excess | +0.006 | +0.022 | -0.038 | +0.003 | -0.003 / +0.015 | 3/5 | NULL — drop |

**Best aggregate feature**: `sentiment_pos_share` (% articles with FinBERT signed score > +0.20).

## Critical interpretation: time-shift placebo

The +30d placebo is HIGHER than raw IC for most features:

| Feature × fwd_20d_excess | raw IC | ts+30 (past returns) |
|---|---|---|
| mean_sentiment | +0.040 | **+0.108** |
| sentiment_pos_share | +0.040 | **+0.077** |

**Why**: ts+30 shifts the sentiment timestamp 30 days *backwards*. The panel's `fwd_60d_excess` at the new (earlier) date represents the return from `D+1 to D+60` where D is 30 days before the article date — so we're correlating the article with the PAST 30 days + future 30 days of returns.

The high IC at ts+30 = **news lags recent returns**. Classic Tetlock 2007 finding: journalists write about what already moved.

**Reassurance**: ts-30 (shift sentiment FORWARD 30 days; correlate with returns 31-90 days POST-news) drops to placebo noise (~0.015). The forward predictive component (days 1-30) is real but mixed with autocorrelation from days -30 to 0.

## Integration plan (per CLAUDE.md PRIME DIRECTIVE)

1. **Add the 4 surviving features** to the alpha158 panel:
   - `sentiment_pos_share` (✓ 5/5 WF, clean ts-30)
   - `sentiment_dispersion` (✓ 4/5 WF)
   - `mean_sentiment` (weak but additive)
   - `n_articles` (low IC, but news flow as activity proxy)
   - **DROP** `sentiment_neg_share` (NULL — overlap with shuffle)

2. **Treat as conditional feature, not universal**:
   - 8.5 months of sentiment coverage means most of the 2016-2025 training set will have NULL values
   - Option A: extend Alpaca News backfill to 2020+ (Alpaca history allegedly back to 2015; ~3-4 hours API time)
   - Option B: train two-headed model — one path with sentiment, one without; ensemble at inference
   - Option C: use sentiment ONLY for re-ranking or weight tilt, not as a panel feature

   **Recommendation: Option A** — backfill 2020-2025 too (~2.6M article rows estimated; 5 years × ~500k/year). Disk ~150 MB. FinBERT scoring ~30 min.

3. **Acceptance gate for retrained panel-LTR**:
   - val_IC ≥ current baseline +0.0294 (per §5.13.15 quality gate)
   - Per-feature SHAP/gain on sentiment columns must be > 0 (else feature is dead-weight in model)
   - WF 3-cut sign-consistency on retrained model ≥ 2/3
   - DSR > 0.5 OR PBO < 0.5

4. **Promote**: weekly Saturday WF promote path (already enforced).

## Decision: not auto-promote, requires extended backfill

Integration with only 8 months sentiment history is risky:
- Training set 95% NULL sentiment → model learns "sentiment available = recent data" rather than "sentiment value predicts return"
- Possible data leak via "is_recent" implicit feature

Before integration, backfill sentiment to 2020-01-01 (~5 years; gives model years of varied regimes including 2020 COVID, 2022 bear, 2023-24 AI bull). Then retrain.

ETA backfill: ~3.5 hours API + ~30 min FinBERT (MPS).
