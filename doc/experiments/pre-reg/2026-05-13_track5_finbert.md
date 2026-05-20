# Pre-registration: Track 5 — FinBERT sentiment features


> **📅 Historical snapshot — content below reflects state at the date in filename/header.**
> Verify against current code per CLAUDE.md §1 "code is the source of truth" before acting on
> present-tense claims. For current state see `doc/roadmap.md` § "📍 Current state" +
> `CLAUDE.md` § "🗂 Current state".

**Date**: 2026-05-13
**Pre-registered BEFORE experiment. STATUS: planning, execution deferred (needs NLP pipeline + news data).**

## Hypothesis

**H0**: Adding FinBERT-derived sentiment scores to the 169-feature panel
produces equal or worse OOS IC.

**H1**: FinBERT features yield **+5-15bp IC** and **+1-3pt mean ΔAPY**.

## Theoretical basis

**Araci 2019** *FinBERT: Financial Sentiment Analysis with Pre-trained
Language Models* — FinBERT (BERT fine-tuned on financial corpus) beats
general BERT by ~10 points F1 on financial sentiment classification.

**Bali, Engle, Hou 2020** *JFE* "Capital market views and momentum" —
news sentiment predicts 1-week to 1-month returns with ~5-10bp IC.

**Tetlock 2007** *Journal of Finance* "Giving Content to Investor
Sentiment" — media tone explains cross-section.

**Common features**:
- Daily sentiment score per ticker (mean of news article scores)
- Sentiment dispersion (std across articles)
- Sentiment trend (5-day vs 60-day mean)
- News volume (count of articles)
- Earnings-call transcript sentiment

## Implementation plan (DEFERRED)

1. **News data pipeline** (~3-5 days):
   - Source: Polygon.io news endpoint, NewsAPI, or RavenPack (paid)
   - Coverage requirement: ≥80% of wl103 must have daily articles
2. **FinBERT inference pipeline** (~2 days):
   - HuggingFace `ProsusAI/finbert` model, ~440MB
   - Batch inference for backfill (~10k articles / day on CPU)
3. **Feature aggregation** (~1 day):
   - Compute per-ticker per-day sentiment summary
4. **Panel integration + retrain** (~2 days)
5. **Full panel** (~70 min)

**Total: 1-2 weeks eng + 1 day compute. Defer.**

## Pre-committed evaluation criteria

Same Tier framework. Plus:
- Feature importance gate: ≥1 sentiment feature must rank top 30
- Sanity: shuffled-news placebo test must give IC ≈ 0

## Why deferred

- News data 1-2 weeks fetch + cache build
- FinBERT inference 1-2 days for full panel backfill
- Engineering 1-2 weeks total
- Lower priority than data-already-available improvements
